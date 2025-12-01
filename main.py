from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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
)
from typing import List, Dict, Any

# ---------- Setup ----------
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


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
        "  \"constraints\": [\"...\", \"...\"]\n"
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
        "  \"sources\": [{\"title\": \"...\", \"publisher\": \"...\", \"url\": \"...\"}]\n"
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
        "  \"bearish_scenario\": \"...\"\n"
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
        "  \"implementation_notes\": [\"...\", \"...\"]\n"
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
        "  \"cautious_alternative\": \"...\"\n"
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
        "  \"revision_instructions\": \"... or empty string if approved\"\n"
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
        "You have the planner, multi-ticker contexts, strategies, final pick, risk, and critic outputs.\n"
        "Your job is to write a polished trading plan summary for EDUCATIONAL PURPOSES ONLY.\n\n"
        "Structure it as:\n"
        "- title\n"
        "- thesis\n"
        "- market_context\n"
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
            try:
                market_data = get_market_analysis(ticker, period="3mo")
                fundamentals = market_data.get("fundamentals")
                market_data_formatted = format_market_data_for_agent(market_data)
                earnings_events = get_earnings_calendar(ticker)
                patterns = scan_chart_patterns(ticker, period="6mo")
            except Exception as e:
                print(f"Error fetching market data for {ticker}: {e}")
                market_data_formatted = f"Unable to fetch real market data for {ticker}"
                earnings_events = []
                patterns = []

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

        # ---- 4) RISK on best pick ----
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

    from streaming_pipeline import stream_pipeline_events
    
    async def event_generator():
        try:
            async for chunk in stream_pipeline_events(headline, user_tickers, user_horizon):
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





# ---------- Run locally ----------
# uvicorn main:app --reload --port 8000
