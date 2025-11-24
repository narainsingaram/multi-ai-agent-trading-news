from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from groq import Groq
import os
from dotenv import load_dotenv
import json
import asyncio
from datetime import datetime
from market_data import get_market_analysis, format_market_data_for_agent, get_chart_data

# ---------- Setup ----------
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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
        "Goal: Read the user's news / headline and define a clear trading objective.\n\n"
        "You MUST:\n"
        "1. Extract tickers mentioned (if none, leave empty array).\n"
        "2. Classify event_type (earnings, macro, guidance, product_launch, regulatory, other).\n"
        "3. Infer sentiment (bullish, bearish, mixed, unclear).\n"
        "4. Determine intended horizon (scalp, intraday, swing, position, long_term).\n"
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
        "You will receive actual stock prices, technical indicators, fundamentals, and curated news context.\n\n"
        "Given the planner output, NEWS agent briefing, fundamentals, and REAL MARKET DATA below, you MUST:\n"
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
        "You receive planner + real market data context and design ONE clear strategy.\n"
        "This is EDUCATIONAL ONLY – do not present as financial advice.\n\n"
        "Given the inputs with REAL technical indicators, you MUST:\n"
        "1. Choose a strategy_type (stock_momentum, stock_mean_reversion, covered_call, call_spread,\n"
        "   put_spread, straddle, strangle, cash_secure_put, other_stock, other_options).\n"
        "2. Describe entry_logic using SPECIFIC price levels from the real data (e.g., 'enter if price breaks above $150 with RSI > 50').\n"
        "3. Describe exit_logic with ACTUAL support/resistance levels (e.g., 'take profit at $160, stop loss at $145').\n"
        "4. Define risk_management using real ATR for stop placement.\n"
        "5. Reference the actual RSI, MACD, Bollinger Bands in your strategy logic.\n"
        "6. Add 2–3 implementation_notes based on current volatility and volume.\n\n"
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
        "Given planner + context + strategy JSON, you MUST:\n"
        "1. List at least 4 concrete risks.\n"
        "2. For each risk, propose a mitigation.\n"
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
    "SUMMARY": (
        "You are the FINAL_SUMMARY agent.\n"
        "You have the planner, context, final strategy, risk, and critic outputs.\n"
        "Your job is to write a polished trading plan summary for EDUCATIONAL PURPOSES ONLY.\n\n"
        "Structure it as:\n"
        "- title\n"
        "- thesis\n"
        "- market_context\n"
        "- strategy\n"
        "- execution_plan\n"
        "- risk_checklist (bullet list)\n"
        "- disclaimer (must clearly say: educational, NOT financial advice, no real-time prices).\n\n"
        "Respond ONLY with JSON:\n"
        "{\n"
        "  \"title\": \"...\",\n"
        "  \"thesis\": \"...\",\n"
        "  \"market_context\": \"...\",\n"
        "  \"strategy\": \"...\",\n"
        "  \"execution_plan\": \"...\",\n"
        "  \"risk_checklist\": [\"...\", \"...\"],\n"
        "  \"disclaimer\": \"...\"\n"
        "}"
    ),
}

# ---------- Core LLM Call ----------
async def agent_llm_call(agent_key: str, payload: dict):
    """
    Generic helper that:
    - Sends a system prompt + JSON payload to the model
    - Forces JSON output via response_format
    - Parses JSON on the backend
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
    body = await request.json()
    headline = body.get("news_text", "").strip()

    if not headline:
        raise HTTPException(status_code=400, detail="news_text is required")

    # This dict will store all agent outputs
    pipeline_state = {}

    # Shared context passed to each agent
    # We KEEP EVERYTHING so later agents see the full story.
    shared_payload = {
        "headline": headline,
        "note": (
            "All agents are collaborating in a multi-step pipeline. "
            "This is for educational purposes only and does NOT use real-time market data."
        ),
    }

    try:
        # ---- 1) PLANNER ----
        planner_input = {
            **shared_payload,
        }
        planner_output = await agent_llm_call("PLANNER", planner_input)
        pipeline_state["planner"] = planner_output

        # If planner decides it's not tradeable, we still run others, but they will see this flag.
        # ---- 2) DATA_CONTEXT ----
        # Fetch real market data for the primary ticker
        market_data = None
        market_data_formatted = ""
        
        if isinstance(planner_output, dict) and planner_output.get("tickers"):
            tickers = planner_output.get("tickers", [])
            if tickers and len(tickers) > 0:
                primary_ticker = tickers[0]
                try:
                    market_data = get_market_analysis(primary_ticker, period="3mo")
                    market_data_formatted = format_market_data_for_agent(market_data)
                except Exception as e:
                    print(f"Error fetching market data for {primary_ticker}: {e}")
                    market_data_formatted = f"Unable to fetch real market data for {primary_ticker}"
        
        data_input = {
            **shared_payload,
            "planner": planner_output,
            "market_data": market_data_formatted,
            "note": "Use the REAL MARKET DATA provided above to inform your analysis. Reference specific technical indicator values.",
        }
        data_output = await agent_llm_call("DATA_CONTEXT", data_input)
        pipeline_state["data_context"] = data_output

        # ---- 3) STRATEGY (v1) ----
        strategy_input_v1 = {
            **shared_payload,
            "planner": planner_output,
            "data_context": data_output,
            "market_data": market_data_formatted,
            "note": "Use the REAL MARKET DATA to set specific entry/exit prices, stop losses based on ATR, and reference actual technical levels.",
        }
        strategy_v1_output = await agent_llm_call("STRATEGY", strategy_input_v1)
        pipeline_state["strategy_v1"] = strategy_v1_output

        # ---- 4) RISK ----
        risk_input = {
            **shared_payload,
            "planner": planner_output,
            "data_context": data_output,
            "strategy": strategy_v1_output,
        }
        risk_output = await agent_llm_call("RISK", risk_input)
        pipeline_state["risk"] = risk_output

        # ---- 5) CRITIC ----
        critic_input = {
            **shared_payload,
            "planner": planner_output,
            "data_context": data_output,
            "strategy": strategy_v1_output,
            "risk": risk_output,
        }
        critic_output = await agent_llm_call("CRITIC", critic_input)
        pipeline_state["critic"] = critic_output

        # ---- 6) Optional STRATEGY REVISION ----
        strategy_final_output = strategy_v1_output
        if isinstance(critic_output, dict) and critic_output.get("status") == "NEEDS_REVISION":
            revision_input = {
                **shared_payload,
                "planner": planner_output,
                "data_context": data_output,
                "previous_strategy": strategy_v1_output,
                "critic_feedback": critic_output,
                "instruction": (
                    "Revise the previous strategy to address ALL critiques above. "
                    "Keep the same output JSON schema as STRATEGY."
                ),
            }
            strategy_final_output = await agent_llm_call("STRATEGY", revision_input)

        pipeline_state["strategy_final"] = strategy_final_output

        # ---- 7) FINAL SUMMARY ----
        summary_input = {
            **shared_payload,
            "planner": planner_output,
            "data_context": data_output,
            "strategy_final": strategy_final_output,
            "risk": risk_output,
            "critic": critic_output,
        }
        summary_output = await agent_llm_call("SUMMARY", summary_input)
        pipeline_state["summary"] = summary_output

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
async def stream_pipeline(news_text: str):
    """
    Streaming version of the pipeline using Server-Sent Events (SSE).
    Sends real-time updates as each agent processes.
    """
    headline = news_text.strip()

    if not headline:
        raise HTTPException(status_code=400, detail="news_text is required")

    from streaming_pipeline import stream_pipeline_events
    
    return StreamingResponse(
        stream_pipeline_events(headline),
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

# ---------- Run locally ----------
# uvicorn main:app --reload --port 8000
