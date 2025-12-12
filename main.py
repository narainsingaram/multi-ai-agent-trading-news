from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from groq import Groq
import os
from dotenv import load_dotenv
import json
import asyncio
from datetime import datetime
from market_data import (
    get_market_analysis,
    format_market_data_for_agent,
    get_chart_data,
    run_simple_backtest,
    get_economic_calendar,
    get_earnings_calendar,
    scan_chart_patterns,
    scan_market_for_criteria,
)
from typing import List, Dict, Any, Optional

# AI Learning System imports
from database import get_db
from pattern_recognition import get_pattern_recognition
from recommendation_tracker import get_tracker
from confidence_calibrator import get_calibrator

# Intelligent Query System imports
from query_router import get_router, QueryIntent
from dynamic_scanner import get_scanner
from quick_response import get_quick_handler
from enhanced_prompts import (
    BullAgentOutput,
    ConsensusAgentOutput,
    ENHANCED_BULL_PROMPT,
    ENHANCED_CONSENSUS_PROMPT
)

# Advanced News Analysis
from advanced_news_analysis import get_news_analyzer

# ---------- Setup ----------
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
USE_LANGCHAIN_PIPELINE = os.getenv("USE_LANGCHAIN_PIPELINE", "1").lower() not in {"0", "false", "no"}


def parse_ticker_list(raw: str) -> List[str]:
    """Parse comma/space separated tickers into clean list."""
    if not raw:
        return []
    parts = [p.strip().upper() for p in raw.replace("–", ",").replace("—", ",").split(",")]
    # Also split on whitespace for cases like "TSLA NVDA"
    tickers: List[str] = []
    for part in parts:
        if not part:
            continue
        tickers.extend([t for t in part.split() if t])
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return deduped

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev; tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "status": "ok",
        "app": "Sigma Trade Terminal API",
        "docs": "/docs",
    }

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

# ---------- Agent Prompts ----------
# These are now more "agentic": each has a clear role, inputs & required outputs.

AGENT_PROMPTS = {
    "PLANNER": (
        "You are the PLANNER agent in a multi-agent trading system.\n"
        "Goal: Read the user's news / headline and define a clear trading objective.\n"
        "The user may also provide explicit tickers and a horizon; respect them and keep them in the output list.\n\n"
        "You MUST:\n"
        "1. Extract tickers mentioned (if none, leave empty array). If user_tickers are provided, merge them into tickers.\n"
        "2. Classify event_type (earnings, macro, guidance, product_launch, regulatory, other).\n"
        "3. Infer sentiment (bullish, bearish, mixed, unclear).\n"
        "4. Determine intended horizon (scalp, intraday, swing, position, long_term). If user_horizon provided, set horizon accordingly.\n"
        "5. Decide if this news is TRADEABLE or NOT_TRADEABLE.\n"
        "6. Summarize the thesis in plain English (what the trader is betting on).\n"
        "7. Provide explicit constraints (e.g. 'no options', 'small size', 'high risk tolerance').\n\n"
        "Respond ONLY with a valid JSON object with keys:\n"
        "{\n"
        "  \"tickers\": [\"...\"],\n"
        "  \"event_type\": \"...\",\n"
        "  \"sentiment\": \"bullish|bearish|mixed|unclear\",\n"
        "  \"horizon\": \"scalp|intraday|swing|position|long_term\",\n"
        "  \"tradeable\": \"TRADEABLE|NOT_TRADEABLE\",\n"
        "  \"thesis_summary\": \"...\",\n"
        "  \"constraints\": [\"...\", \"...\"],\n"
        "  \"handoff_note\": \"One line for NEWS on what context to gather (drivers, symbols)\"\n"
        "}"
    ),
    "NEWS": (
        "You are the NEWS & FUNDAMENTALS agent.\n"
        "Inputs: headline, planner output, recent news articles, and fundamental snapshot.\n"
        "Goal: Build a concise but thorough news/context briefing for downstream agents.\n\n"
        "You MUST:\n"
        "1) Extract core drivers from the headline + articles (earnings, guidance, macro, regulatory, product, M&A).\n"
        "2) Summarize sentiment with justification tied to the articles.\n"
        "3) Identify catalysts (upcoming earnings/events) and immediate risks.\n"
        "4) Bridge to fundamentals: valuation (PE/PEG/PB), quality (ROE/margins/FCF), and balance sheet (debt/equity).\n"
        "5) Provide 3-5 bullet implications for trading the main ticker.\n"
        "6) Cite sources: include an array of {title, publisher, url} for each article used.\n\n"
        "Respond ONLY with JSON:\n"
        "{\n"
        "  \"headline_summary\": \"...\",\n"
        "  \"sentiment\": \"bullish|bearish|mixed|unclear\",\n"
        "  \"drivers\": [\"...\", \"...\"],\n"
        "  \"catalysts\": [\"...\"],\n"
        "  \"fundamental_take\": {\"valuation\": \"...\", \"quality\": \"...\", \"balance_sheet\": \"...\"},\n"
        "  \"risks\": [\"...\"],\n"
        "  \"trade_implications\": [\"...\"],\n"
        "  \"sources\": [{\"title\": \"...\", \"publisher\": \"...\", \"url\": \"...\"}],\n"
        "  \"handoff_note\": \"One line for DATA_CONTEXT on key themes/levels to inspect\"\n"
        "}"
    ),
    "DATA_CONTEXT": (
        "You are the DATA_CONTEXT agent with access to REAL market data.\n"
        "You will receive actual stock prices, technical indicators, fundamentals, curated news context, and a target_ticker.\n\n"
        "Given the planner output, NEWS agent briefing, fundamentals, and REAL MARKET DATA below, you MUST for the target_ticker:\n"
        "1. Analyze trend using SMA/EMA/price action and cite levels.\n"
        "2. Assess momentum using the real RSI, MACD values provided.\n"
        "3. Evaluate volatility using ATR and Bollinger Bands data.\n"
        "4. Blend fundamentals (valuation/quality) with technicals to identify 3–5 key drivers; call out valuation (PE/PEG/PB), balance sheet (debt/equity), and profitability markers.\n"
        "5. Provide bullish and bearish scenarios grounded in actual technical + news catalysts.\n"
        "6. Include a compact indicator digest: price vs SMA20/50/200, RSI state, MACD bias, BB position, ATR dollar value, volume vs 20d avg.\n\n"
        "IMPORTANT: Use the REAL data provided. Reference specific price levels, RSI values, support/resistance, and valuation markers.\n\n"
        "Respond ONLY with JSON:\n"
        "{\n"
        "  \"primary_ticker\": \"...\",\n"
        "  \"current_price\": 123.45,\n"
        "  \"trend\": \"uptrend|downtrend|sideways|strong_uptrend|strong_downtrend\",\n"
        "  \"volatility\": \"low|moderate|high\",\n"
        "  \"sector_mood\": \"bullish|bearish|mixed|unclear\",\n"
        "  \"key_drivers\": [\"...\", \"...\"],\n"
        "  \"technical_summary\": \"Brief summary of RSI, MACD, moving averages\",\n"
        "  \"support_levels\": [price1, price2],\n"
        "  \"resistance_levels\": [price1, price2],\n"
        "  \"bullish_scenario\": \"...\",\n"
        "  \"bearish_scenario\": \"...\",\n"
        "  \"handoff_note\": \"One line for STRATEGY: the clearest setup and decisive levels\"\n"
        "}"
    ),
    "STRATEGY": (
        "You are the STRATEGY agent with access to REAL technical analysis.\n"
        "You receive planner + real market data context for a specific target_ticker and design ONE clear strategy.\n"
        "This is EDUCATIONAL ONLY – do not present as financial advice.\n\n"
        "Given the inputs with REAL technical indicators, you MUST:\n"
        "1. Choose a strategy_type (stock_momentum, stock_mean_reversion, covered_call, call_spread,\n"
        "   put_spread, straddle, strangle, cash_secure_put, other_stock, other_options).\n"
        "2. Describe entry_logic using SPECIFIC price levels from the real data (e.g., 'enter if price breaks above $150 with RSI > 50').\n"
        "3. Describe exit_logic with ACTUAL support/resistance levels (e.g., 'take profit at $160, stop loss at $145').\n"
        "4. Define risk_management using real ATR for stop placement.\n"
        "5. Explicitly reference MULTIPLE technical confluences available in data_context: RSI, MACD (and histogram), EMA/SMA stack, VWAP, Bollinger position, ATR, volume vs average, Stochastic %K/%D, OBV/MFI flow, and any pattern_scan levels. Do NOT rely on just RSI/MACD.\n"
        "6. Use institutional/pattern levels if provided (support/resistance, order blocks, liquidity pools) when setting entries/stops/targets.\n"
        "7. Add 2–3 implementation_notes based on volatility regime and volume/flow context.\n\n"
        "Respond ONLY with JSON:\n"
        "{\n"
        "  \"strategy_type\": \"...\",\n"
        "  \"directional_bias\": \"bullish|bearish|neutral\",\n"
        "  \"entry_price\": 123.45,\n"
        "  \"entry_logic\": \"Specific conditions with real price levels and indicators\",\n"
        "  \"target_price\": 130.00,\n"
        "  \"stop_loss\": 120.00,\n"
        "  \"exit_logic\": \"Specific exit conditions with real levels\",\n"
        "  \"risk_management\": \"...\",\n"
        "  \"position_sizing_guidelines\": \"...\",\n"
        "  \"implementation_notes\": [\"...\", \"...\"],\n"
        "  \"handoff_note\": \"One line for RISK on the biggest vulnerabilities or gaps\"\n"
        "}"
    ),
    "RISK": (
        "You are the RISK agent.\n"
        "You only think about HOW THIS CAN GO WRONG.\n\n"
        "Given planner + context + strategy JSON, and a list of upcoming_events (macro + earnings), you MUST:\n"
        "1. List at least 4 concrete risks.\n"
        "2. For each risk, propose a mitigation.\n"
        "3. Explicitly mention if any upcoming macro/earnings events create gap/volatility risk.\n"
        "3. Rate overall risk_level: low, moderate, high, speculative.\n"
        "4. Suggest what a cautious trader might do instead (e.g., paper trade).\n\n"
        "Respond ONLY with JSON:\n"
        "{\n"
        "  \"risk_level\": \"low|moderate|high|speculative\",\n"
        "  \"risks\": [\n"
        "    {\"risk\": \"...\", \"mitigation\": \"...\"},\n"
        "    {\"risk\": \"...\", \"mitigation\": \"...\"}\n"
        "  ],\n"
        "  \"cautious_alternative\": \"...\",\n"
        "  \"handoff_note\": \"One line for CRITIC on approval vs. required revisions\"\n"
        "}"
    ),
    "CRITIC": (
        "You are the CRITIC agent.\n"
        "You evaluate whether the strategy is coherent and aligned with the planner + context.\n\n"
        "You MUST:\n"
        "1. Decide if the strategy is APPROVED or NEEDS_REVISION.\n"
        "2. Give 2–4 bullet-point critiques.\n"
        "3. If NEEDS_REVISION, give explicit instructions for how to fix it.\n\n"
        "Respond ONLY with JSON:\n"
        "{\n"
        "  \"status\": \"APPROVED|NEEDS_REVISION\",\n"
        "  \"critiques\": [\"...\", \"...\"],\n"
        "  \"revision_instructions\": \"... or empty string if approved\",\n"
        "  \"handoff_note\": \"One line for SUMMARY on what to emphasize or adjust\"\n"
        "}"
    ),
    "BULL": (
        "You are the BULL agent - the optimistic advocate in a trading debate.\n"
        "Your role: Build the STRONGEST possible bullish case using real data.\n\n"
        "Given the strategy, market data, and context, you MUST:\n"
        "1. Identify ALL bullish technical confluences (trend, momentum, volume, institutional support).\n"
        "2. Highlight positive fundamental catalysts and valuation support.\n"
        "3. Reference institutional support: Order Blocks, OBV rising, VWAP position, MFI.\n"
        "4. Cite SPECIFIC price levels and indicator values as evidence (e.g., 'RSI 45', 'VWAP $244.75').\n"
        "5. Provide 3-5 concrete reasons why THIS is a high-probability long trade.\n"
        "6. Assign confidence score (0-100) to your bullish thesis based on evidence strength.\n\n"
        "Be aggressive but evidence-based. Use actual numbers from the data. Don't ignore bearish signals, but explain why bullish factors dominate.\n\n"
        "Respond ONLY with JSON:\n"
        "{\n"
        "  \"thesis\": \"Concise 1-2 sentence bullish argument\",\n"
        "  \"evidence\": [\"RSI 45 + Stoch oversold = reversal setup\", \"OBV rising 15% = institutional accumulation\", \"ADX 32 = strong uptrend\", ...],\n"
        "  \"key_levels\": {\"entry\": 245.00, \"target\": 260.00, \"stop\": 240.00},\n"
        "  \"catalysts\": [\"Earnings beat expected\", \"Bullish Order Block at $242-245\", \"Sector rotation into tech\", ...],\n"
        "  \"confidence\": 75,\n"
        "  \"timeframe\": \"3-5 days\",\n"
        "  \"handoff_note\": \"One line for BEAR on what to challenge\"\n"
        "}"
    ),
    "BEAR": (
        "You are the BEAR agent - the pessimistic advocate in a trading debate.\n"
        "Your role: Build the STRONGEST possible bearish case and challenge the BULL's thesis.\n\n"
        "Given the strategy, market data, Bull's thesis, and context, you MUST:\n"
        "1. Identify ALL bearish technical signals (overbought, divergences, resistance, distribution).\n"
        "2. Highlight negative fundamental risks and valuation concerns.\n"
        "3. Reference distribution signals: OBV falling, MFI overbought, Fair Value Gaps, liquidity pools.\n"
        "4. DIRECTLY CHALLENGE Bull's evidence with counter-evidence and alternative interpretations.\n"
        "5. Provide 3-5 concrete reasons why this trade could FAIL or reverse.\n"
        "6. Assign confidence score (0-100) to your bearish thesis based on evidence strength.\n\n"
        "Be skeptical and rigorous. Point out what Bull is missing or misinterpreting. Use specific data.\n\n"
        "Respond ONLY with JSON:\n"
        "{\n"
        "  \"thesis\": \"Concise 1-2 sentence bearish argument\",\n"
        "  \"evidence\": [\"Stoch 85 overbought\", \"Price at FVG resistance $246-248\", \"MFI 72 = distribution\", \"Recent rejection at $252\", ...],\n"
        "  \"challenges_to_bull\": [\"Bull ignores overbought Stoch - high reversal risk\", \"Bull's OBV rising but MFI shows distribution = divergence\", ...],\n"
        "  \"key_levels\": {\"short_entry\": 248.00, \"target\": 235.00, \"stop\": 252.00},\n"
        "  \"risks\": [\"Gap down risk on earnings\", \"Liquidity pool at $253 = stop hunt zone\", \"Sector weakness\", ...],\n"
        "  \"confidence\": 65,\n"
        "  \"timeframe\": \"2-4 days\",\n"
        "  \"handoff_note\": \"One line for DEVILS_ADVOCATE on what both sides missed\"\n"
        "}"
    ),
    "DEVILS_ADVOCATE": (
        "You are the DEVIL'S ADVOCATE - the critical challenger who finds flaws in ALL arguments.\n"
        "Your role: Challenge BOTH Bull and Bear. Identify blind spots, biases, and overlooked factors.\n\n"
        "Given Bull thesis, Bear thesis, and all market data, you MUST:\n"
        "1. Identify what BOTH Bull and Bear are overlooking, misinterpreting, or cherry-picking.\n"
        "2. Point out confirmation bias - are they seeing what they want to see?\n"
        "3. Highlight conflicting signals that neither side adequately addressed.\n"
        "4. Question assumptions about timeframes, catalysts, support/resistance levels.\n"
        "5. Suggest alternative interpretations of the same data.\n"
        "6. Rate the quality of both arguments (0-100) based on evidence rigor and logic.\n\n"
        "Your job is to make the final decision BETTER by exposing weaknesses. Be ruthlessly objective.\n\n"
        "Respond ONLY with JSON:\n"
        "{\n"
        "  \"bull_weaknesses\": [\"Ignores Stoch overbought signal\", \"Overestimates OBV significance - only 10-day trend\", \"Assumes earnings beat without evidence\", ...],\n"
        "  \"bear_weaknesses\": [\"Underestimates ADX 32 trend strength\", \"Ignores bullish Order Block support at $242\", \"Overweights short-term overbought\", ...],\n"
        "  \"overlooked_factors\": [\"Earnings announcement in 3 days = high gap risk\", \"Fed meeting next week\", \"Sector correlation breakdown\", ...],\n"
        "  \"conflicting_signals\": [\"RSI neutral (55) but Stoch overbought (85)\", \"OBV rising but MFI flat = mixed volume flow\", \"VWAP bullish but FVG resistance ahead\", ...],\n"
        "  \"bull_argument_quality\": 70,\n"
        "  \"bear_argument_quality\": 65,\n"
        "  \"recommendation\": \"Wait for clarity\" | \"Lean bullish\" | \"Lean bearish\" | \"No trade\",\n"
        "  \"critical_factor\": \"The most important factor both sides should weigh\",\n"
        "  \"handoff_note\": \"One line for CONSENSUS on what to prioritize in final decision\"\n"
        "}"
    ),
    "CONSENSUS": (
        "You are the CONSENSUS agent - the final decision synthesizer after debate.\n"
        "Your role: Weigh Bull, Bear, and Devil's Advocate arguments to reach a balanced, evidence-based decision.\n\n"
        "Given all debate outputs, market data, and upcoming events, you MUST:\n"
        "1. Summarize the strongest points from Bull and Bear (2-3 each).\n"
        "2. Incorporate Devil's Advocate concerns and overlooked factors.\n"
        "3. Assign weights to Bull and Bear arguments based on evidence quality and Devil's critique.\n"
        "4. Make a CLEAR directional call: LONG, SHORT, or NO_TRADE.\n"
        "5. Provide precise entry, target, and stop levels with justification.\n"
        "6. Give overall confidence score (0-100) reflecting debate outcome.\n"
        "7. Acknowledge dissenting opinion and conditions that would invalidate the thesis.\n\n"
        "Be decisive but acknowledge uncertainty. Explain your weighting logic clearly.\n\n"
        "Respond ONLY with JSON:\n"
        "{\n"
        "  \"decision\": \"LONG\" | \"SHORT\" | \"NO_TRADE\",\n"
        "  \"confidence\": 72,\n"
        "  \"bull_weight\": 0.60,\n"
        "  \"bear_weight\": 0.40,\n"
        "  \"rationale\": \"Bull's institutional accumulation (OBV +15%) and trend strength (ADX 32) outweigh Bear's overbought concerns. However, Fair Value Gap at $248 is valid resistance. Devil's Advocate correctly notes earnings risk in 3 days.\",\n"
        "  \"recommended_action\": {\n"
        "    \"entry\": 244.75,\n"
        "    \"entry_logic\": \"Wait for pullback to VWAP support, don't chase at FVG resistance $246-248\",\n"
        "    \"target\": 258.00,\n"
        "    \"target_logic\": \"Next resistance level, conservative given earnings risk\",\n"
        "    \"stop\": 241.50,\n"
        "    \"stop_logic\": \"Below bullish Order Block and liquidity pool at $242\",\n"
        "    \"position_size\": \"50-60% of normal (moderate confidence, earnings risk)\"\n"
        "  },\n"
        "  \"key_factors_weighted\": [\n"
        "    {\"factor\": \"OBV rising (institutional accumulation)\", \"weight\": \"high\", \"direction\": \"bullish\"},\n"
        "    {\"factor\": \"ADX 32 (strong trend)\", \"weight\": \"high\", \"direction\": \"bullish\"},\n"
        "    {\"factor\": \"Stoch 85 overbought\", \"weight\": \"medium\", \"direction\": \"bearish\"},\n"
        "    {\"factor\": \"FVG resistance $246-248\", \"weight\": \"medium\", \"direction\": \"bearish\"},\n"
        "    {\"factor\": \"Earnings in 3 days\", \"weight\": \"high\", \"direction\": \"risk\"}\n"
        "  ],\n"
        "  \"dissenting_view\": \"Bear's overbought Stoch concern is valid for 1-2 day pullback. Short-term traders should wait.\",\n"
        "  \"invalidation_conditions\": [\"Break below $241 Order Block\", \"Bearish earnings surprise\", \"Stoch bearish crossover with volume spike\"],\n"
        "  \"timeframe\": \"3-7 days (before earnings)\",\n"
        "  \"handoff_note\": \"One line for RISK on event-based risks that could invalidate consensus\"\n"
        "}"
    ),
    "STRATEGY_FINAL": (
        "You are the STRATEGY_FINAL agent acting as a comparator.\n"
        "You receive multiple ticker strategies and must pick the BEST single trade for the brief horizon.\n\n"
        "Given planner, per-ticker contexts, and strategies:\n"
        "1. Pick best_ticker and best_strategy (one of the provided strategies).\n"
        "2. Give 2–3 selection reasons comparing momentum, setup quality, and risk, and explicitly mention technical confluences (EMA/VWAP/RSI/MACD/ATR/volume/OBV/Stoch/patterns) that drove the choice.\n"
        "3. If no strategy is workable, set best_ticker to \"NONE\" and explain why.\n\n"
        "Respond ONLY with JSON:\n"
        "{\n"
        "  \"best_ticker\": \"TSLA\",\n"
        "  \"selection_reasons\": [\"...\", \"...\"],\n"
        "  \"best_strategy\": { ...strategy schema... }\n"
        "}"
    ),
    "SUMMARY": (
        "You are the FINAL_SUMMARY agent.\n"
        "You have the planner, multi-ticker contexts, strategies, final pick, Bull vs Bear debate, consensus, risk, and critic outputs.\n"
        "Your job is to write a polished trading plan summary for EDUCATIONAL PURPOSES ONLY.\n\n"
        "Structure it as:\n"
        "- title\n"
        "- thesis\n"
        "- market_context\n"
        "- debate_summary: Concise summary of Bull case, Bear case, Devil's Advocate concerns, and Consensus decision\n"
        "- strategy (include key technical confluences: EMA/SMA stack, RSI/MACD, VWAP, volume/OBV/MFI, Stoch, ATR, Bollinger, notable patterns/levels)\n"
        "- execution_plan\n"
        "- risk_checklist (bullet list)\n"
        "- disclaimer (must clearly say: educational, NOT financial advice, no real-time prices).\n"
        "- comparison_notes: short bullets comparing the tickers.\n\n"
        "Respond ONLY with JSON:\n"
        "{\n"
        "  \"title\": \"...\",\n"
        "  \"thesis\": \"...\",\n"
        "  \"market_context\": \"...\",\n"
        "  \"debate_summary\": {\n"
        "    \"bull_case\": \"2-3 sentence summary of Bull's strongest points\",\n"
        "    \"bear_case\": \"2-3 sentence summary of Bear's strongest points\",\n"
        "    \"devils_advocate\": \"1-2 sentence summary of key concerns raised\",\n"
        "    \"consensus_decision\": \"LONG|SHORT|NO_TRADE with confidence score and rationale\"\n"
        "  },\n"
        "  \"strategy\": \"...\",\n"
        "  \"execution_plan\": \"...\",\n"
        "  \"risk_checklist\": [\"...\", \"...\"],\n"
        "  \"disclaimer\": \"...\",\n"
        "  \"comparison_notes\": [\"...\", \"...\"]\n"
        "}"
    ),
    "WOLF": (
        "You are Sigma Wolf, a friendly and concise trading assistant. "
        "Answer the user's question based on the provided analysis context. "
        "Use the planner, data_context, strategies, and summary to give helpful explanations. "
        "Keep your answers under 3 sentences and conversational. "
        "If you don't have enough information, just say so briefly. "
        "This is for educational purposes only, not financial advice."
    ),
}

# ---------- Core LLM Call ----------
async def agent_llm_call(agent_key: str, payload: dict):
    """
    Generic helper that:
    - Sends a system prompt + JSON payload to the model
    - Forces JSON output via response_format (except for WOLF)
    - Parses JSON on the backend (or returns plain text for WOLF)
    """
    system_prompt = AGENT_PROMPTS[agent_key]

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": json.dumps(payload, indent=2)
        },
    ]

    try:
        # WOLF agent returns plain text, others return JSON
        if agent_key == "WOLF":
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                temperature=0.7,
                max_tokens=300,
            )
            return completion.choices[0].message.content.strip()
        else:
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                temperature=0.2,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
            response_content = completion.choices[0].message.content
            return json.loads(response_content)
    except json.JSONDecodeError:
        return {
            "error": "JSONDecodeError",
            "raw_response": completion.choices[0].message.content
        }
    except Exception as e:
        return {
            "error": "APIError",
            "message": str(e)
        }

async def agent_llm_call_streaming(agent_key: str, payload: dict):
    """
    Streaming version of agent_llm_call.
    Yields tokens as they arrive and returns the full parsed JSON at the end.
    
    Yields:
        dict: {"type": "token", "content": str} for each token
        dict: {"type": "complete", "content": dict} for final parsed JSON
        dict: {"type": "error", "content": str} if error occurs
    """
    system_prompt = AGENT_PROMPTS[agent_key]

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": json.dumps(payload, indent=2)
        },
    ]

    accumulated_content = ""

    try:
        stream = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.2,
            max_tokens=1024,
            response_format={"type": "json_object"},
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                accumulated_content += token
                yield {"type": "token", "content": token}
        
        # Parse the accumulated JSON
        try:
            parsed_json = json.loads(accumulated_content)
            yield {"type": "complete", "content": parsed_json}
        except json.JSONDecodeError:
            yield {
                "type": "error",
                "content": f"JSONDecodeError: {accumulated_content}"
            }
    
    except Exception as e:
        yield {
            "type": "error",
            "content": f"APIError: {str(e)}"
        }

# ---------- Pipeline Route ----------
@app.post("/run-pipeline")
async def run_pipeline(request: Request):
    """
    Non-streaming pipeline (legacy) now supports optional tickers/horizon for multi-ticker mode.
    """
    body = await request.json()
    headline = (body.get("news_text") or "").strip()
    user_tickers = parse_ticker_list(body.get("tickers", ""))
    user_horizon = body.get("horizon")

    if not headline and not user_tickers:
        raise HTTPException(status_code=400, detail="news_text or tickers required")
    if not headline and user_tickers:
        headline = f"Multi-ticker analysis for: {', '.join(user_tickers)}"

    # This dict will store all agent outputs
    pipeline_state = {}

    # Shared context passed to each agent
    shared_payload = {
        "headline": headline,
        "user_tickers": user_tickers,
        "user_horizon": user_horizon,
        "note": (
            "All agents are collaborating in a multi-step pipeline. "
            "This is for educational purposes only and does NOT use real-time market data."
        ),
    }

    try:
        # ---- 1) PLANNER ----
        planner_input = {**shared_payload}
        planner_output = await agent_llm_call("PLANNER", planner_input)
        pipeline_state["planner"] = planner_output

        effective_tickers = user_tickers or planner_output.get("tickers", []) if isinstance(planner_output, dict) else []

        macro_events = get_economic_calendar()
        shared_payload["upcoming_events"] = {
            "macro": macro_events,
            "earnings": {},
        }
        shared_payload["pattern_scan"] = {}

        # ---- 2) DATA_CONTEXT per ticker ----
        data_outputs = []
        strategies = []

        for ticker in effective_tickers[:3]:  # guardrail for demo
            market_data_formatted = ""
            fundamentals = None
            earnings_events = []
            patterns = []
            institutional_levels = {}
            try:
                market_data = get_market_analysis(ticker, period="3mo")
                fundamentals = market_data.get("fundamentals")
                earnings_events = get_earnings_calendar(ticker)
                patterns = scan_chart_patterns(ticker, period="6mo")
                
                # Get institutional levels
                from market_data import get_chart_data
                chart_result = get_chart_data(ticker, period="6mo")
                if "error" not in chart_result:
                    institutional_levels = chart_result.get("institutional_levels", {})
                
                market_data_formatted = format_market_data_for_agent(market_data, institutional_levels)
            except Exception as e:
                print(f"Error fetching market data for {ticker}: {e}")
                market_data_formatted = f"Unable to fetch real market data for {ticker}"
                earnings_events = []
                patterns = []
                institutional_levels = {}

            shared_payload["upcoming_events"]["earnings"][ticker] = earnings_events
            shared_payload["pattern_scan"][ticker] = patterns

            data_input = {
                **shared_payload,
                "planner": planner_output,
                "target_ticker": ticker,
                "market_data": market_data_formatted,
                "fundamentals": fundamentals,
                "note": "Use the REAL MARKET DATA provided above to inform your analysis. Reference specific technical indicator values.",
                "upcoming_events": shared_payload["upcoming_events"],
                "pattern_scan": patterns,
            }
            data_output = await agent_llm_call("DATA_CONTEXT", data_input)
            if isinstance(data_output, dict):
                data_output["primary_ticker"] = ticker
                
                # Inject Chart Data & Institutional Levels
                try:
                    chart_result = get_chart_data(ticker, period="6mo")
                    if "error" not in chart_result:
                        data_output["chart_data"] = chart_result.get("data", [])
                        data_output["institutional_levels"] = chart_result.get("institutional_levels", {})
                except Exception as e:
                    print(f"Error injecting chart data for {ticker}: {e}")

            data_outputs.append(data_output)

            # ---- STRATEGY per ticker ----
            strategy_input_v1 = {
                **shared_payload,
                "planner": planner_output,
                "data_context": data_output,
                "target_ticker": ticker,
                "market_data": market_data_formatted,
                "fundamentals": fundamentals,
                "note": "Use the REAL MARKET DATA to set specific entry/exit prices, stop losses based on ATR, and reference actual technical levels.",
                "upcoming_events": shared_payload["upcoming_events"],
                "pattern_scan": patterns,
            }
            strategy_v1_output = await agent_llm_call("STRATEGY", strategy_input_v1)
            if isinstance(strategy_v1_output, dict):
                strategy_v1_output["ticker"] = ticker
            strategies.append(strategy_v1_output)

        pipeline_state["data_context"] = data_outputs
        pipeline_state["strategy_v1"] = strategies

        # ---- 3) PICK BEST via STRATEGY_FINAL prompt ----
        comparator_input = {
            **shared_payload,
            "planner": planner_output,
            "data_context_all": data_outputs,
            "strategies": strategies,
            "instruction": "Compare the per-ticker strategies and choose the best single trade. Return best_strategy and explain the choice.",
            "upcoming_events": shared_payload["upcoming_events"],
            "pattern_scan": shared_payload["pattern_scan"],
        }
        strategy_final_output = await agent_llm_call("STRATEGY_FINAL", comparator_input)
        pipeline_state["strategy_final"] = strategy_final_output

        # ---- 3.5) BULL vs BEAR DEBATE ----
        best_strategy = strategy_final_output.get("best_strategy", strategy_final_output) if isinstance(strategy_final_output, dict) else {}
        
        # BULL agent - optimistic case
        bull_input = {
            **shared_payload,
            "planner": planner_output,
            "data_context": data_outputs,
            "strategy": best_strategy,
            "strategy_final": strategy_final_output,
            "upcoming_events": shared_payload["upcoming_events"],
            "pattern_scan": shared_payload["pattern_scan"],
        }
        bull_output = await agent_llm_call("BULL", bull_input)
        pipeline_state["bull"] = bull_output

        # BEAR agent - pessimistic case, challenges Bull
        bear_input = {
            **shared_payload,
            "planner": planner_output,
            "data_context": data_outputs,
            "strategy": best_strategy,
            "strategy_final": strategy_final_output,
            "bull_thesis": bull_output,
            "upcoming_events": shared_payload["upcoming_events"],
            "pattern_scan": shared_payload["pattern_scan"],
        }
        bear_output = await agent_llm_call("BEAR", bear_input)
        pipeline_state["bear"] = bear_output

        # DEVILS_ADVOCATE - challenges both Bull and Bear
        devils_advocate_input = {
            **shared_payload,
            "planner": planner_output,
            "data_context": data_outputs,
            "strategy": best_strategy,
            "bull_thesis": bull_output,
            "bear_thesis": bear_output,
            "upcoming_events": shared_payload["upcoming_events"],
            "pattern_scan": shared_payload["pattern_scan"],
        }
        devils_advocate_output = await agent_llm_call("DEVILS_ADVOCATE", devils_advocate_input)
        pipeline_state["devils_advocate"] = devils_advocate_output

        # CONSENSUS - synthesizes debate into final recommendation
        consensus_input = {
            **shared_payload,
            "planner": planner_output,
            "data_context": data_outputs,
            "strategy": best_strategy,
            "bull_thesis": bull_output,
            "bear_thesis": bear_output,
            "devils_advocate": devils_advocate_output,
            "upcoming_events": shared_payload["upcoming_events"],
            "pattern_scan": shared_payload["pattern_scan"],
        }
        consensus_output = await agent_llm_call("CONSENSUS", consensus_input)
        pipeline_state["consensus"] = consensus_output

        # Update best_strategy with consensus recommendation
        if isinstance(consensus_output, dict) and "recommended_action" in consensus_output:
            best_strategy = consensus_output.get("recommended_action", best_strategy)

        # ---- 3.6) AI LEARNING SYSTEM INTEGRATION ----
        # Get tracker and calibrator instances
        tracker = get_tracker()
        calibrator = get_calibrator()
        pattern_detector = get_pattern_recognition()
        
        # Prepare market data for tracking
        best_ticker = strategy_final_output.get("best_ticker", effective_tickers[0] if effective_tickers else "UNKNOWN")
        
        # Get market data for best ticker
        try:
            market_data_for_tracking = get_market_analysis(best_ticker, period="3mo")
            chart_data = get_chart_data(best_ticker, period="6mo")
            
            # Detect patterns in current data
            if "historical_data" in chart_data and chart_data["historical_data"] is not None:
                detected_patterns = pattern_detector.detect_all_patterns(chart_data["historical_data"])
                pipeline_state["detected_patterns"] = detected_patterns
                
                print(f"🔍 Detected {len(detected_patterns)} pattern(s) in {best_ticker}")
                if detected_patterns:
                    print(f"   Top pattern: {detected_patterns[0]['type']} (quality: {detected_patterns[0].get('quality_score', 0):.2f})")
            else:
                detected_patterns = []
            
            # Apply confidence calibration to all agents
            if isinstance(bull_output, dict) and "confidence" in bull_output:
                original_bull_conf = bull_output["confidence"]
                bull_output["confidence"] = calibrator.calibrate_confidence("BULL", original_bull_conf)
                bull_output["original_confidence"] = original_bull_conf
            
            if isinstance(bear_output, dict) and "confidence" in bear_output:
                original_bear_conf = bear_output["confidence"]
                bear_output["confidence"] = calibrator.calibrate_confidence("BEAR", original_bear_conf)
                bear_output["original_confidence"] = original_bear_conf
            
            if isinstance(consensus_output, dict) and "confidence" in consensus_output:
                original_consensus_conf = consensus_output["confidence"]
                consensus_output["confidence"] = calibrator.calibrate_confidence("CONSENSUS", original_consensus_conf)
                consensus_output["original_confidence"] = original_consensus_conf
            
            # Log recommendation to database
            rec_id = tracker.log_recommendation(
                pipeline_state=pipeline_state,
                market_data={
                    "ticker": best_ticker,
                    "quote": market_data_for_tracking.get("quote", {}),
                    "technicals": market_data_for_tracking.get("technicals", {}),
                    "historical_data": chart_data.get("historical_data"),
                },
                ticker=best_ticker
            )
            
            pipeline_state["recommendation_id"] = rec_id
            pipeline_state["learning_enabled"] = True
            
            print(f"✅ AI Learning: Logged recommendation #{rec_id} with {len(detected_patterns)} patterns")
            
        except Exception as e:
            print(f"⚠️  AI Learning integration error: {e}")
            pipeline_state["learning_enabled"] = False
            pipeline_state["detected_patterns"] = []

        # ---- 4) RISK on consensus pick ----
        best_strategy = strategy_final_output.get("best_strategy", strategy_final_output) if isinstance(strategy_final_output, dict) else {}
        risk_input = {
            **shared_payload,
            "planner": planner_output,
            "data_context": data_outputs,
            "strategy": best_strategy,
            "upcoming_events": shared_payload["upcoming_events"],
            "pattern_scan": shared_payload["pattern_scan"],
        }
        risk_output = await agent_llm_call("RISK", risk_input)
        pipeline_state["risk"] = risk_output

        # ---- 5) CRITIC on best pick ----
        critic_input = {
            **shared_payload,
            "planner": planner_output,
            "data_context": data_outputs,
            "strategy": best_strategy,
            "risk": risk_output,
            "upcoming_events": shared_payload["upcoming_events"],
            "pattern_scan": shared_payload["pattern_scan"],
        }
        critic_output = await agent_llm_call("CRITIC", critic_input)
        pipeline_state["critic"] = critic_output

        # ---- 6) FINAL SUMMARY ----
        summary_input = {
            **shared_payload,
            "planner": planner_output,
            "data_context": data_outputs,
            "strategies": strategies,
            "strategy_final": strategy_final_output,
            "bull": bull_output,
            "bear": bear_output,
            "devils_advocate": devils_advocate_output,
            "consensus": consensus_output,
            "risk": risk_output,
            "critic": critic_output,
            "upcoming_events": shared_payload["upcoming_events"],
            "pattern_scan": shared_payload["pattern_scan"],
        }
        summary_output = await agent_llm_call("SUMMARY", summary_input)
        pipeline_state["summary"] = summary_output
        pipeline_state["events"] = shared_payload["upcoming_events"]
        pipeline_state["patterns"] = shared_payload["pattern_scan"]

        return {
            "pipeline_state": pipeline_state,
            "meta": {
                "disclaimer": (
                    "This multi-agent system is for educational/demo purposes only. "
                    "It does NOT fetch real-time market data and is NOT financial advice."
                )
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")


# ---------- Streaming Pipeline Route (SSE) ----------
@app.get("/stream-pipeline")
async def stream_pipeline(news_text: str | None = None, request: Request = None, tickers: str | None = None, horizon: str | None = None):
    """
    Streaming version of the pipeline using Server-Sent Events (SSE).
    Supports optional user-specified tickers and horizon for multi-ticker mode.
    """
    headline = (news_text or "").strip()

    user_tickers = parse_ticker_list(tickers or "")
    user_horizon = horizon.strip() if horizon else None

    if not headline and not user_tickers:
        raise HTTPException(status_code=400, detail="news_text or tickers required")

    # If no headline provided, synthesize one from tickers
    if not headline and user_tickers:
        headline = f"Multi-ticker analysis for: {', '.join(user_tickers)}"

    from streaming_pipeline import stream_pipeline_events as legacy_stream
    langchain_ready = False
    langchain_stream = None
    if USE_LANGCHAIN_PIPELINE:
        try:
            from langchain_pipeline import (
                stream_pipeline_events_langchain,
                LANGCHAIN_READY,
            )

            langchain_ready = LANGCHAIN_READY
            if langchain_ready:
                langchain_stream = stream_pipeline_events_langchain
        except Exception:
            langchain_ready = False

    async def event_generator():
        try:
            chosen_stream = None
            if USE_LANGCHAIN_PIPELINE and langchain_ready and langchain_stream:
                chosen_stream = langchain_stream(headline, user_tickers, user_horizon, AGENT_PROMPTS)
            else:
                chosen_stream = legacy_stream(headline, user_tickers, user_horizon)

            async for chunk in chosen_stream:
                # Stop streaming if the client disconnects to avoid socket.send errors
                if await request.is_disconnected():
                    print("SSE client disconnected; stopping stream.")
                    break
                yield chunk
        except asyncio.CancelledError:
            print("SSE stream cancelled; client likely closed connection.")
        except Exception as e:
            err = {
                "agent": "SYSTEM",
                "error": f"Stream crashed: {str(e)}"
            }
            yield f"event: error\ndata: {json.dumps(err)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )



# ---------- Market Data Endpoint ----------
@app.get("/market-data/{ticker}")
async def get_market_data_endpoint(ticker: str, period: str = "3mo"):
    """
    Get real-time market data and technical indicators for a ticker.
    
    Args:
        ticker: Stock symbol (e.g., AAPL, TSLA)
        period: Historical data period (1d, 5d, 1mo, 3mo, 6mo, 1y)
    
    Returns:
        Comprehensive market analysis with quote, technicals, and performance
    """
    try:
        market_data = get_market_analysis(ticker.upper(), period=period)
        return market_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch market data: {str(e)}")

@app.get("/market-data/{ticker}/chart")
async def get_chart_data_endpoint(ticker: str, period: str = "3mo", limit: int = 60):
    """
    Get candlestick chart data with technical indicators.
    
    Args:
        ticker: Stock symbol (e.g., AAPL, TSLA)
        period: Historical data period (1d, 5d, 1mo, 3mo, 6mo, 1y)
        limit: Maximum number of data points to return (default: 60)
    
    Returns:
        OHLCV data with technical indicators formatted for charting
    """
    try:
        chart_data = get_chart_data(ticker.upper(), period=period, limit=limit)
        return chart_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch chart data: {str(e)}")

# ---------- Sigma Wolf Chat Endpoint ----------
@app.post("/wolf-chat")
async def wolf_chat(request: Request):
    """
    Lightweight helper to answer questions via Sigma Wolf using existing pipeline context.
    Input JSON:
    {
      "question": "...",
      "pipeline": { ... optional pipeline state ... }
    }
    """
    body = await request.json()
    question = (body.get("question") or "").strip()
    pipeline = body.get("pipeline") or {}

    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    payload = {
        "question": question,
        "planner": pipeline.get("planner"),
        "data_context": pipeline.get("data_context"),
        "strategies": pipeline.get("strategy_v1") or pipeline.get("strategies"),
        "summary": pipeline.get("summary"),
        "strategy_final": pipeline.get("strategy_final"),
    }
    try:
        reply = await agent_llm_call("WOLF", payload)
        return {"answer": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Wolf chat failed: {str(e)}")


# ---------- Backtesting Endpoint ----------
@app.post("/backtest")
async def backtest_strategy(request: Request):
    """
    Simple historical backtest placeholder.
    Input JSON: { "ticker": "AAPL", "start": "2022-01-01", "end": "2024-01-01", "strategy": {...optional...} }
    """
    body = await request.json()
    ticker = (body.get("ticker") or "").strip().upper()
    start = (body.get("start") or "2022-01-01").strip()
    end = (body.get("end") or datetime.now().strftime("%Y-%m-%d")).strip()
    strategy = body.get("strategy")

    if not ticker:
        raise HTTPException(status_code=400, detail="ticker is required")

    try:
        result = run_simple_backtest(ticker, start, end, strategy)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")


@app.post("/scan-market")
async def scan_market(request: Request):
    """
    Market scanner endpoint.
    Input JSON: { "query": "RSI < 30 and positive sentiment", "universe": [...optional...], "limit": 40 }
    """
    body = await request.json()
    query = (body.get("query") or "").strip()
    universe = body.get("universe")
    limit = body.get("limit", 40)

    if not query:
        raise HTTPException(status_code=400, detail="query is required for market scans")

    try:
        tickers = universe if isinstance(universe, list) else None
        result = scan_market_for_criteria(query, universe=tickers, limit=int(limit))
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Market scan failed: {str(e)}")


# ---------- Ticker Extraction Endpoint ----------
@app.post("/extract-tickers")
async def extract_tickers_endpoint(request: Request):
    """
    Extract stock ticker symbols from natural language text using AI.
    
    Input JSON: { "text": "What's happening with Apple and Tesla?" }
    Returns: { "tickers": ["AAPL", "TSLA"] }
    """
    from ticker_extractor import extract_tickers
    
    body = await request.json()
    text = body.get("text", "").strip()
    
    if not text:
        return {"tickers": []}
    
    try:
        tickers = extract_tickers(text)
        return {"tickers": tickers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ticker extraction failed: {str(e)}")


# ---------- Analysis Storage (In-Memory for Demo) ----------
# In production, replace with database (PostgreSQL, MongoDB, etc.)
analysis_storage: Dict[str, Dict[str, Any]] = {}

def generate_analysis_id() -> str:
    """Generate a unique ID for saved analyses."""
    import uuid
    return str(uuid.uuid4())[:8]  # Short ID for easy sharing

@app.post("/save-analysis")
async def save_analysis(request: Request):
    """
    Save an analysis with auto-generated ID.
    
    Input JSON: {
        "query": "...",
        "pipeline": {...},
        "summary": {...},
        "dataContext": {...},
        "tickers": ["AAPL", "TSLA"],
        "horizon": "swing"
    }
    
    Returns: { "analysis_id": "abc123", "url": "http://..." }
    """
    body = await request.json()
    
    analysis_id = generate_analysis_id()
    timestamp = datetime.now().isoformat()
    
    # Store the complete analysis
    analysis_storage[analysis_id] = {
        "id": analysis_id,
        "timestamp": timestamp,
        "query": body.get("query", ""),
        "pipeline": body.get("pipeline", {}),
        "summary": body.get("summary"),
        "dataContext": body.get("dataContext"),
        "tickers": body.get("tickers", []),
        "horizon": body.get("horizon", "auto"),
    }
    
    # Generate shareable URL (adjust domain for production)
    share_url = f"http://localhost:3000?analysis={analysis_id}"
    
    return {
        "analysis_id": analysis_id,
        "url": share_url,
        "timestamp": timestamp
    }

@app.get("/analysis/{analysis_id}")
async def get_analysis(analysis_id: str):
    """
    Retrieve a saved analysis by ID.
    
    Returns the complete analysis data or 404 if not found.
    """
    if analysis_id not in analysis_storage:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return analysis_storage[analysis_id]

@app.get("/analysis/history")
async def get_analysis_history():
    """
    Get list of all saved analyses (metadata only).
    
    Returns: [
        {
            "id": "abc123",
            "timestamp": "2024-01-01T12:00:00",
            "query": "...",
            "tickers": ["AAPL"],
            "horizon": "swing"
        },
        ...
    ]
    """
    # Return metadata only (not full pipeline data)
    history = [
        {
            "id": item["id"],
            "timestamp": item["timestamp"],
            "query": item["query"],
            "tickers": item["tickers"],
            "horizon": item["horizon"],
            "sentiment": item.get("pipeline", {}).get("planner", {}).get("sentiment", "unknown")
        }
        for item in analysis_storage.values()
    ]
    
    # Sort by timestamp descending (newest first)
    history.sort(key=lambda x: x["timestamp"], reverse=True)
    
    return {"history": history, "count": len(history)}

@app.delete("/analysis/{analysis_id}")
async def delete_analysis(analysis_id: str):
    """
    Delete a saved analysis.
    
    Returns: { "success": true, "message": "..." }
    """
    if analysis_id not in analysis_storage:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    del analysis_storage[analysis_id]
    
    return {
        "success": True,
        "message": f"Analysis {analysis_id} deleted successfully"
    }


# ---------- Options Chain Endpoint ----------
@app.get("/options/{ticker}")
async def get_options_chain_endpoint(ticker: str, expiration: str | None = None):
    """
    Get real options chain data for a ticker.
    
    Args:
        ticker: Stock symbol (e.g., AAPL, TSLA)
        expiration: Optional expiration date (YYYY-MM-DD)
    
    Returns:
        Options chain with calls, puts, put/call ratios, unusual activity, and heatmap
    """
    from options_data import get_options_chain
    
    try:
        options_data = get_options_chain(ticker.upper(), expiration)
        
        if "error" in options_data:
            raise HTTPException(status_code=400, detail=options_data["error"])
        
        return options_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch options data: {str(e)}")


# ---------- Social Sentiment Endpoints ----------
@app.get("/sentiment/trending")
async def get_trending_sentiment(limit: int = 20):
    """
    Get trending stocks from Reddit with sentiment analysis.
    
    Args:
        limit: Number of trending tickers to return (default: 20)
    
    Returns:
        Trending tickers with mention counts, sentiment scores, and discussion highlights
    """
    from sentiment_data import get_trending_stocks
    
    try:
        sentiment_data = get_trending_stocks(limit=limit)
        
        if "error" in sentiment_data:
            raise HTTPException(status_code=400, detail=sentiment_data["error"])
        
        return sentiment_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch sentiment data: {str(e)}")


@app.get("/sentiment/{ticker}")
async def get_ticker_sentiment_endpoint(ticker: str):
    """
    Get detailed sentiment analysis for a specific ticker.
    
    Args:
        ticker: Stock symbol (e.g., AAPL, TSLA)
    
    Returns:
        Sentiment score, discussions, and sentiment distribution
    """
    from sentiment_data import get_ticker_sentiment
    
    try:
        sentiment_data = get_ticker_sentiment(ticker.upper())
        
        if "error" in sentiment_data and sentiment_data.get("mention_count", 0) == 0:
            # Return empty data instead of error for no mentions
            return sentiment_data
        
        return sentiment_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch ticker sentiment: {str(e)}")


# ========== AI LEARNING SYSTEM API ENDPOINTS ==========

@app.get("/api/recommendations/history")
async def get_recommendations_history(
    ticker: Optional[str] = None,
    days: int = 30,
    outcome: Optional[str] = None,
    limit: int = 100
):
    """
    Get historical recommendations with outcomes.
    
    Query params:
        ticker: Filter by ticker (optional)
        days: Number of days to look back (default: 30)
        outcome: Filter by outcome: WIN, LOSS, PENDING, etc. (optional)
        limit: Max results (default: 100)
    """
    try:
        tracker = get_tracker()
        history = tracker.db.get_recommendations_history(ticker, days, outcome, limit)
        
        # Parse JSON fields
        for rec in history:
            if rec.get('technical_snapshot'):
                try:
                    rec['technical_snapshot'] = json.loads(rec['technical_snapshot'])
                except:
                    pass
            if rec.get('pattern_detected'):
                try:
                    rec['pattern_detected'] = json.loads(rec['pattern_detected'])
                except:
                    pass
        
        return {
            "total": len(history),
            "recommendations": history,
            "filters": {
                "ticker": ticker,
                "days": days,
                "outcome": outcome
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch recommendations: {str(e)}")


@app.get("/api/recommendations/{rec_id}")
async def get_recommendation_detail(rec_id: int):
    """Get detailed information about a specific recommendation."""
    try:
        tracker = get_tracker()
        rec = tracker.db.get_recommendation(rec_id)
        
        if not rec:
            raise HTTPException(status_code=404, detail=f"Recommendation #{rec_id} not found")
        
        # Parse JSON fields
        if rec.get('technical_snapshot'):
            try:
                rec['technical_snapshot'] = json.loads(rec['technical_snapshot'])
            except:
                pass
        if rec.get('pattern_detected'):
            try:
                rec['pattern_detected'] = json.loads(rec['pattern_detected'])
            except:
                pass
        
        return rec
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch recommendation: {str(e)}")


@app.post("/api/recommendations/{rec_id}/update")
async def update_recommendation_outcome(
    rec_id: int,
    actual_exit_price: float,
    mfe: Optional[float] = None,
    mae: Optional[float] = None
):
    """
    Update a recommendation with actual outcome.
    
    Body:
        actual_exit_price: Actual exit price
        mfe: Maximum Favorable Excursion (optional)
        mae: Maximum Adverse Excursion (optional)
    """
    try:
        tracker = get_tracker()
        tracker.update_outcome(rec_id, actual_exit_price, mfe=mfe, mae=mae)
        
        # Get updated recommendation
        updated_rec = tracker.db.get_recommendation(rec_id)
        
        return {
            "success": True,
            "recommendation_id": rec_id,
            "outcome": updated_rec.get('outcome'),
            "pnl_percent": updated_rec.get('pnl_percent')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update recommendation: {str(e)}")


@app.get("/api/patterns/library")
async def get_pattern_library(
    pattern_type: Optional[str] = None,
    ticker: Optional[str] = None,
    limit: int = 100
):
    """
    Get pattern library with success rates.
    
    Query params:
        pattern_type: Filter by pattern type (optional)
        ticker: Filter by ticker (optional)
        limit: Max results (default: 100)
    """
    try:
        tracker = get_tracker()
        patterns = tracker.db.get_pattern_library(pattern_type, ticker, limit)
        
        # Parse JSON fields
        for pattern in patterns:
            if pattern.get('pattern_data'):
                try:
                    pattern['pattern_data'] = json.loads(pattern['pattern_data'])
                except:
                    pass
            if pattern.get('technical_context'):
                try:
                    pattern['technical_context'] = json.loads(pattern['technical_context'])
                except:
                    pass
        
        return {
            "total": len(patterns),
            "patterns": patterns,
            "filters": {
                "pattern_type": pattern_type,
                "ticker": ticker
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch pattern library: {str(e)}")


@app.get("/api/agent/performance")
async def get_agent_performance(
    agent_name: str,
    days: int = 90
):
    """
    Get performance statistics for an agent.
    
    Query params:
        agent_name: Agent name (BULL, BEAR, CONSENSUS, DEVILS_ADVOCATE)
        days: Number of days to look back (default: 90)
    """
    try:
        if agent_name not in ['BULL', 'BEAR', 'CONSENSUS', 'DEVILS_ADVOCATE']:
            raise HTTPException(status_code=400, detail="Invalid agent name")
        
        tracker = get_tracker()
        stats = tracker.db.get_agent_stats(agent_name, days)
        
        return {
            "agent_name": agent_name,
            "period_days": days,
            "stats": stats
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch agent performance: {str(e)}")


@app.get("/api/agent/calibration")
async def get_agent_calibration(agent_name: str):
    """
    Get calibration report for an agent showing confidence accuracy.
    
    Query params:
        agent_name: Agent name (BULL, BEAR, CONSENSUS, DEVILS_ADVOCATE)
    """
    try:
        if agent_name not in ['BULL', 'BEAR', 'CONSENSUS', 'DEVILS_ADVOCATE']:
            raise HTTPException(status_code=400, detail="Invalid agent name")
        
        calibrator = get_calibrator()
        report = calibrator.get_calibration_report(agent_name)
        
        return report
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch calibration report: {str(e)}")


@app.post("/api/agent/calibration/update")
async def update_agent_calibration(agent_name: Optional[str] = None):
    """
    Manually trigger calibration update for an agent or all agents.
    
    Query params:
        agent_name: Agent name (optional, updates all if not specified)
    """
    try:
        calibrator = get_calibrator()
        
        if agent_name:
            if agent_name not in ['BULL', 'BEAR', 'CONSENSUS', 'DEVILS_ADVOCATE']:
                raise HTTPException(status_code=400, detail="Invalid agent name")
            calibrator.update_calibration(agent_name)
            return {"success": True, "updated": [agent_name]}
        else:
            calibrator.update_all_calibrations()
            return {"success": True, "updated": ['BULL', 'BEAR', 'CONSENSUS', 'DEVILS_ADVOCATE']}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update calibration: {str(e)}")


@app.get("/api/performance/stats")
async def get_performance_stats(
    ticker: Optional[str] = None,
    days: int = 90
):
    """
    Get overall performance statistics.
    
    Query params:
        ticker: Filter by ticker (optional)
        days: Number of days to look back (default: 90)
    """
    try:
        tracker = get_tracker()
        stats = tracker.get_performance_stats(ticker, days)
        
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch performance stats: {str(e)}")


@app.get("/api/patterns/detected/{ticker}")
async def get_detected_patterns(ticker: str, period: str = "6mo"):
    """
    Detect patterns in a ticker's chart right now.
    
    Path params:
        ticker: Stock ticker
    Query params:
        period: Time period (default: 6mo)
    """
    try:
        pattern_detector = get_pattern_recognition()
        
        # Get chart data
        chart_data = get_chart_data(ticker.upper(), period=period)
        
        if "error" in chart_data:
            raise HTTPException(status_code=404, detail=chart_data["error"])
        
        # Detect patterns
        if "historical_data" in chart_data and chart_data["historical_data"] is not None:
            patterns = pattern_detector.detect_all_patterns(chart_data["historical_data"])
            summary = pattern_detector.get_pattern_summary(patterns)
            
            return {
                "ticker": ticker.upper(),
                "period": period,
                "patterns_detected": len(patterns),
                "patterns": patterns,
                "summary": summary
            }
        else:
            return {
                "ticker": ticker.upper(),
                "period": period,
                "patterns_detected": 0,
                "patterns": [],
                "summary": "No chart data available"
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to detect patterns: {str(e)}")


# ========== INTELLIGENT QUERY ENDPOINT ==========

@app.post("/query")
async def intelligent_query(request: Request):
    """
    Intelligent query endpoint that handles ANY natural language input.
    Routes queries dynamically based on intent classification.
    
    Supported intents:
    - single_ticker: "Should I buy TSLA?" → Full pipeline
    - market_scan: "Find oversold tech stocks" → Dynamic scanner
    - market_overview: "How's the market?" → Quick overview
    - educational: "What is RSI?" → Educational response
    - conversational: "Tell me more" → Context-aware response
    
    Example requests:
    {"query": "Should I buy Tesla?"}
    {"query": "Find me oversold stocks with high volume"}
    {"query": "What's happening in the market today?"}
    {"query": "What is the MACD indicator?"}
    """
    try:
        data = await request.json()
        user_query = data.get("query", "").strip()
        
        if not user_query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        print(f"\n{'='*60}")
        print(f"🔍 INTELLIGENT QUERY: {user_query}")
        print(f"{'='*60}\n")
        
        # Step 1: Route query using intelligent router
        router = get_router()
        routing = router.classify_query(user_query)
        
        print(f"📍 Intent: {routing.intent.value}")
        print(f"📊 Confidence: {routing.confidence:.2f}")
        print(f"💭 Reasoning: {routing.reasoning}\n")
        
        # Step 2: Execute appropriate workflow based on intent
        
        if routing.intent == QueryIntent.SINGLE_TICKER:
            # SINGLE TICKER: Run full pipeline
            print("→ Routing to FULL PIPELINE\n")
            
            tickers = routing.entities.tickers
            if not tickers:
                return {
                    "error": "No ticker found in query",
                    "suggestion": "Please specify a stock ticker (e.g., 'Should I buy TSLA?')"
                }
            
            # Use first ticker
            ticker = tickers[0]
            
            # Create headline from reformulated query
            headline = routing.reformulated_query or user_query
            
            # Run full pipeline (reuse existing logic)
            pipeline_data = {
                "headline": headline,
                "user_tickers": [ticker]
            }
            
            # Import and run pipeline
            from main import run_full_pipeline_logic
            result = await run_full_pipeline_logic(pipeline_data)
            
            return {
                "type": "full_analysis",
                "intent": routing.intent.value,
                "ticker": ticker,
                "result": result
            }
        
        elif routing.intent == QueryIntent.MARKET_SCAN or routing.intent == QueryIntent.CUSTOM_SCAN:
            # MARKET SCAN: Use dynamic scanner
            print("→ Routing to DYNAMIC SCANNER\n")
            
            scanner = get_scanner()
            
            # Parse query into structured criteria
            criteria = scanner.parse_scan_query(user_query)
            
            # Execute scan
            results = scanner.execute_scan(criteria)
            
            return {
                "type": "scan_results",
                "intent": routing.intent.value,
                "query": user_query,
                "criteria": criteria.dict(),
                "results_count": len(results),
                "results": [r.dict() for r in results],
                "reformulated_query": routing.reformulated_query
            }
        
        elif routing.intent == QueryIntent.MARKET_OVERVIEW:
            # MARKET OVERVIEW: Quick response
            print("→ Routing to QUICK OVERVIEW\n")
            
            quick_handler = get_quick_handler()
            response = quick_handler.handle_market_overview()
            
            return {
                "type": "quick_response",
                "intent": routing.intent.value,
                "response": response.dict()
            }
        
        elif routing.intent == QueryIntent.EDUCATIONAL:
            # EDUCATIONAL: Quick educational response
            print("→ Routing to EDUCATIONAL RESPONSE\n")
            
            quick_handler = get_quick_handler()
            response = quick_handler.handle_educational(user_query)
            
            return {
                "type": "educational",
                "intent": routing.intent.value,
                "response": response.dict()
            }
        
        elif routing.intent == QueryIntent.CONVERSATIONAL:
            # CONVERSATIONAL: Context-aware response
            print("→ Routing to CONVERSATIONAL RESPONSE\n")
            
            quick_handler = get_quick_handler()
            context = router.get_context()
            response = quick_handler.handle_follow_up(user_query, context)
            
            return {
                "type": "conversational",
                "intent": routing.intent.value,
                "response": response.dict()
            }
        
        else:
            return {
                "error": "Intent not yet implemented",
                "intent": routing.intent.value,
                "suggestion": "Try asking about a specific stock, market scan, or market overview"
            }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Query error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.get("/query/examples")
async def get_query_examples():
    """
    Get example queries for each intent type.
    Helps users understand what they can ask.
    """
    return {
        "single_ticker": [
            "Should I buy TSLA?",
            "Analyze NVDA for potential trade",
            "Is AAPL a good buy right now?",
            "MSFT earnings analysis"
        ],
        "market_scan": [
            "Find oversold tech stocks",
            "Show me stocks breaking out",
            "Find dividend stocks with good fundamentals",
            "Stocks with high volume today"
        ],
        "custom_scan": [
            "Find tech stocks with RSI < 30 and bullish MACD",
            "Show me large cap stocks with P/E under 15",
            "Dividend stocks with revenue growth > 10%"
        ],
        "market_overview": [
            "How's the market today?",
            "What's happening in tech sector?",
            "Market summary",
            "Is the market bullish or bearish?"
        ],
        "educational": [
            "What is RSI?",
            "Explain MACD indicator",
            "How to read Bollinger Bands?",
            "What are support and resistance levels?"
        ],
        "conversational": [
            "Tell me more about that",
            "What are the risks?",
            "Can you explain that in simpler terms?",
            "What should I do next?"
        ]
    }


# ========== ADVANCED NEWS ANALYSIS ENDPOINTS ==========

@app.post("/news/analyze")
async def analyze_news_comprehensive(request: Request):
    """
    Comprehensive news analysis endpoint.
    Pulls from multiple sources and performs deep analysis.
    
    Request body:
    {
      "ticker": "TSLA",
      "headline": "Tesla announces record deliveries",  // optional
      "days_back": 7  // optional, default 7
    }
    
    Returns deep analysis with:
    - Multi-source news aggregation
    - Sentiment analysis with confidence
    - Entity extraction
    - Management tone analysis
    - Market-moving insights
    - Trade implications
    """
    try:
        data = await request.json()
        ticker = data.get("ticker", "").upper()
        headline = data.get("headline", f"News analysis for {ticker}")
        days_back = data.get("days_back", 7)
        
        if not ticker:
            raise HTTPException(status_code=400, detail="ticker is required")
        
        print(f"\n{'='*60}")
        print(f"📰 ADVANCED NEWS ANALYSIS: {ticker}")
        print(f"{'='*60}\n")
        
        # Get news analyzer
        analyzer = get_news_analyzer()
        
        # Step 1: Fetch news from multiple sources
        print("🔍 Fetching news from multiple sources...")
        articles = analyzer.fetch_news_multi_source(ticker, headline, days_back)
        
        if not articles:
            return {
                "ticker": ticker,
                "error": "No news articles found",
                "suggestion": "Try increasing days_back or check if ticker is correct"
            }
        
        # Step 2: Get fundamentals for context
        from market_data import get_market_analysis
        market_data = get_market_analysis(ticker, period="3mo")
        fundamentals = market_data.get("fundamentals")
        
        # Step 3: Perform deep analysis
        print("🧠 Performing deep analysis with LLM...")
        analysis = analyzer.analyze_news_deep(ticker, headline, articles, fundamentals)
        
        print(f"\n✅ Analysis complete!")
        print(f"   Sentiment: {analysis.overall_sentiment} ({analysis.sentiment_score:.2f})")
        print(f"   Confidence: {analysis.confidence:.2f}")
        print(f"   Articles analyzed: {analysis.articles_analyzed}")
        print(f"   Key drivers: {len(analysis.key_drivers)}")
        print(f"   Entities: {len(analysis.entities_mentioned)}\n")
        
        return {
            "ticker": ticker,
            "analysis": analysis.dict(),
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ News analysis error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"News analysis failed: {str(e)}")


@app.get("/news/fetch/{ticker}")
async def fetch_news_multi_source(ticker: str, days_back: int = 7):
    """
    Fetch news from multiple sources without deep analysis.
    Useful for quick news lookup.
    
    Query params:
    - days_back: Number of days to look back (default: 7)
    
    Returns:
    - List of articles from multiple sources
    - Deduplication applied
    - Sorted by relevance and recency
    """
    try:
        ticker = ticker.upper()
        
        print(f"📰 Fetching news for {ticker} (last {days_back} days)")
        
        analyzer = get_news_analyzer()
        articles = analyzer.fetch_news_multi_source(ticker, days_back=days_back)
        
        return {
            "ticker": ticker,
            "articles_count": len(articles),
            "articles": [a.dict() for a in articles],
            "sources": list(set([a.source for a in articles])),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch news: {str(e)}")


@app.post("/news/sentiment")
async def analyze_sentiment_only(request: Request):
    """
    Quick sentiment analysis without full deep analysis.
    Faster response time.
    
    Request body:
    {
      "ticker": "TSLA",
      "text": "Tesla announces record deliveries beating expectations"
    }
    
    Returns sentiment score and classification.
    """
    try:
        data = await request.json()
        ticker = data.get("ticker", "").upper()
        text = data.get("text", "")
        
        if not text:
            raise HTTPException(status_code=400, detail="text is required")
        
        # Use LLM for quick sentiment
        prompt = f"""Analyze the sentiment of this news about {ticker}:

"{text}"

Respond with ONLY valid JSON:
{{
  "sentiment": "bullish|bearish|mixed|unclear",
  "score": <number from -1 to 1>,
  "confidence": <number from 0 to 1>,
  "reasoning": "brief explanation"
}}"""
        
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=256,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(completion.choices[0].message.content)
        
        return {
            "ticker": ticker,
            "text": text,
            "sentiment": result,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {str(e)}")


# ---------- Run locally ----------
# uvicorn main:app --reload --port 8000
