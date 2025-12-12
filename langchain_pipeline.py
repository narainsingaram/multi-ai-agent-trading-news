"""
LangChain-powered streaming pipeline that stitches agents with a shared state.
Falls back gracefully if LangChain is not installed.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.runnables import RunnableLambda
    from langchain_groq import ChatGroq

    LANGCHAIN_READY = True
except Exception:
    # The server will emit a clear error event if LangChain is missing
    LANGCHAIN_READY = False


def _format_sse(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def _build_chain(agent_name: str, agent_prompt: str, model: ChatGroq):
    """
    Build a tiny LCEL chain per agent: prompt -> model -> JSON parser.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                agent_prompt
                + "\n\nUse the provided CONTEXT to reason and respond with JSON only.",
            ),
            (
                "human",
                "CONTEXT:\n{context}\n\nUSER:\n{user_block}",
            ),
        ]
    )
    parser = JsonOutputParser()
    return prompt | model | parser


async def _run_chain_stream(chain, payload: dict, agent: str, ticker: str | None, holder: dict):
    """
    Stream tokens + final JSON output from a LangChain chain, yielding SSE events.
    The final parsed output is stored in holder["output"] for the caller.
    """
    yield _format_sse(
        "agent_start",
        {"agent": agent, "ticker": ticker, "timestamp": datetime.now().isoformat()},
    )
    try:
        async for event in chain.astream_events(payload):
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk and chunk.content:
                    yield _format_sse(
                        "agent_token",
                        {"agent": agent, "ticker": ticker, "token": chunk.content},
                    )
            if event["event"] == "on_chain_end":
                output = event["data"]["output"]
                holder["output"] = output
                yield _format_sse(
                    "agent_complete",
                    {
                        "agent": agent,
                        "ticker": ticker,
                        "output": output,
                        "timestamp": datetime.now().isoformat(),
                    },
                )
    except Exception as e:
        yield _format_sse(
            "error",
            {"agent": agent, "ticker": ticker, "error": str(e)},
        )


async def stream_pipeline_events_langchain(
    headline: str,
    user_tickers: Optional[List[str]] = None,
    user_horizon: Optional[str] = None,
    agent_prompts: Optional[Dict[str, str]] = None,
):
    """
    LangChain-driven SSE pipeline. It shares a state dict so each agent receives
    upstream outputs and curated data (market/fundamentals/patterns).
    """
    if not LANGCHAIN_READY:
        yield _format_sse(
            "error",
            {
                "agent": "SYSTEM",
                "error": "LangChain is not installed. Install langchain-core, langchain-groq to use this pipeline.",
            },
        )
        return

    from market_data import (
        get_market_analysis,
        format_market_data_for_agent,
        get_economic_calendar,
        get_earnings_calendar,
        scan_chart_patterns,
    )

    state: Dict[str, Any] = {
        "headline": headline,
        "user_tickers": user_tickers or [],
        "user_horizon": user_horizon,
        "note": "LangChain multi-agent pipeline (educational).",
    }

    agent_prompts = agent_prompts or {}
    model = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.1-70b-versatile",
        temperature=0.2,
        streaming=True,
    )

    planner_chain = _build_chain("PLANNER", agent_prompts.get("PLANNER", ""), model)
    planner_ctx = json.dumps(
        {
            "headline": headline,
            "user_tickers": user_tickers,
            "user_horizon": user_horizon,
        },
        indent=2,
    )
    planner_holder: Dict[str, Any] = {}
    async for chunk in _run_chain_stream(
        planner_chain,
        {"context": planner_ctx, "user_block": "Plan the trade."},
        agent="PLANNER",
        ticker=None,
        holder=planner_holder,
    ):
        yield chunk
    planner_output = planner_holder.get("output")
    if planner_output is None:
        return
    state["planner"] = planner_output

    effective_tickers = user_tickers or []
    if not effective_tickers and isinstance(planner_output, dict):
        effective_tickers = planner_output.get("tickers", []) or []
    effective_tickers = effective_tickers[:3]

    # Pre-fetch data/tools so downstream agents share the same context.
    market_bundle = []
    for ticker in effective_tickers:
        try:
            raw = get_market_analysis(ticker, period="3mo")
            fundamentals = raw.get("fundamentals")
            earnings = get_earnings_calendar(ticker)
            patterns = scan_chart_patterns(ticker, period="6mo")
            
            # Get institutional levels
            from market_data import get_chart_data
            chart_result = get_chart_data(ticker, period="6mo")
            institutional_levels = chart_result.get("institutional_levels", {}) if "error" not in chart_result else {}
            
            market_fmt = format_market_data_for_agent(raw, institutional_levels)
        except Exception as e:  # pragma: no cover - defensive
            raw = {}
            fundamentals = None
            market_fmt = f"Unable to fetch market data: {e}"
            earnings = []
            patterns = []
            institutional_levels = {}
        market_bundle.append(
            {
                "ticker": ticker,
                "raw": raw,
                "fundamentals": fundamentals,
                "formatted": market_fmt,
                "earnings": earnings,
                "patterns": patterns,
                "institutional_levels": institutional_levels,
            }
        )

    macro_events = get_economic_calendar()
    state["events"] = {
        "macro": macro_events,
        "earnings": {b["ticker"]: b["earnings"] for b in market_bundle},
    }
    state["patterns"] = {b["ticker"]: b["patterns"] for b in market_bundle}

    # NEWS agent with fundamentals + aggregated articles
    news_chain = _build_chain("NEWS", agent_prompts.get("NEWS", ""), model)
    news_ctx = json.dumps(
        {
            "headline": headline,
            "articles": sum([b["raw"].get("news", []) for b in market_bundle], []),
            "fundamentals": market_bundle[0]["fundamentals"] if market_bundle else None,
            "upcoming_events": state["events"],
            "planner": planner_output,
        },
        indent=2,
        default=str,
    )
    news_holder: Dict[str, Any] = {}
    async for chunk in _run_chain_stream(
        news_chain,
        {"context": news_ctx, "user_block": "Summarize news + catalysts."},
        agent="NEWS",
        ticker=None,
        holder=news_holder,
    ):
        yield chunk
    news_output = news_holder.get("output")
    state["news"] = news_output

    data_context_outputs = []
    strategy_outputs = []

    data_chain = _build_chain("DATA_CONTEXT", agent_prompts.get("DATA_CONTEXT", ""), model)
    strat_chain = _build_chain("STRATEGY", agent_prompts.get("STRATEGY", ""), model)

    for bundle in market_bundle:
        ticker = bundle["ticker"]
        data_ctx = json.dumps(
            {
                "ticker": ticker,
                "market_data": bundle["formatted"],
                "fundamentals": bundle["fundamentals"],
                "planner": planner_output,
                "news_briefing": news_output,
                "upcoming_events": state["events"],
                "patterns": bundle["patterns"],
            },
            indent=2,
            default=str,
        )
        data_holder: Dict[str, Any] = {}
        async for chunk in _run_chain_stream(
            data_chain,
            {"context": data_ctx, "user_block": f"Analyze {ticker} with real indicators."},
            agent="DATA_CONTEXT",
            ticker=ticker,
            holder=data_holder,
        ):
            yield chunk
        data_output = data_holder.get("output")
        if isinstance(data_output, dict):
            data_output["primary_ticker"] = ticker
        data_context_outputs.append(data_output)
        state.setdefault("data_context", []).append(data_output)

        strat_ctx = json.dumps(
            {
                "ticker": ticker,
                "data_context": data_output,
                "planner": planner_output,
                "fundamentals": bundle["fundamentals"],
                "market_data": bundle["formatted"],
                "news_briefing": news_output,
                "upcoming_events": state["events"],
                "patterns": bundle["patterns"],
            },
            indent=2,
            default=str,
        )
        strat_holder: Dict[str, Any] = {}
        async for chunk in _run_chain_stream(
            strat_chain,
            {"context": strat_ctx, "user_block": f"Draft a strategy for {ticker}."},
            agent="STRATEGY",
            ticker=ticker,
            holder=strat_holder,
        ):
            yield chunk
        strat_output = strat_holder.get("output")
        if isinstance(strat_output, dict):
            strat_output["ticker"] = ticker
        strategy_outputs.append(strat_output)
        state.setdefault("strategy_v1", []).append(strat_output)

    # STRATEGY_FINAL comparator
    comparator_chain = _build_chain("STRATEGY_FINAL", agent_prompts.get("STRATEGY_FINAL", ""), model)
    cmp_ctx = json.dumps(
        {
            "planner": planner_output,
            "data_context_all": data_context_outputs,
            "strategies": strategy_outputs,
            "upcoming_events": state["events"],
            "patterns": state.get("patterns"),
        },
        indent=2,
        default=str,
    )
    strategy_final_holder: Dict[str, Any] = {}
    async for chunk in _run_chain_stream(
        comparator_chain,
        {"context": cmp_ctx, "user_block": "Pick the best trade idea and justify."},
        agent="STRATEGY_FINAL",
        ticker=None,
        holder=strategy_final_holder,
    ):
        yield chunk
    strategy_final_output = strategy_final_holder.get("output")
    state["strategy_final"] = strategy_final_output

    best_strategy = (
        strategy_final_output.get("best_strategy", strategy_final_output)
        if isinstance(strategy_final_output, dict)
        else {}
    )

    # RISK
    risk_chain = _build_chain("RISK", agent_prompts.get("RISK", ""), model)
    risk_ctx = json.dumps(
        {
            "planner": planner_output,
            "data_context": data_context_outputs,
            "strategy": best_strategy,
            "news_briefing": news_output,
            "upcoming_events": state["events"],
            "patterns": state.get("patterns"),
        },
        indent=2,
        default=str,
    )
    risk_holder: Dict[str, Any] = {}
    async for chunk in _run_chain_stream(
        risk_chain,
        {"context": risk_ctx, "user_block": "List concrete risks and mitigations."},
        agent="RISK",
        ticker=None,
        holder=risk_holder,
    ):
        yield chunk
    risk_output = risk_holder.get("output")
    state["risk"] = risk_output

    # CRITIC
    critic_chain = _build_chain("CRITIC", agent_prompts.get("CRITIC", ""), model)
    critic_ctx = json.dumps(
        {
            "planner": planner_output,
            "data_context": data_context_outputs,
            "strategy": best_strategy,
            "risk": risk_output,
            "upcoming_events": state["events"],
        },
        indent=2,
        default=str,
    )
    critic_holder: Dict[str, Any] = {}
    async for chunk in _run_chain_stream(
        critic_chain,
        {"context": critic_ctx, "user_block": "Critique coherence; require fixes if needed."},
        agent="CRITIC",
        ticker=None,
        holder=critic_holder,
    ):
        yield chunk
    critic_output = critic_holder.get("output")
    state["critic"] = critic_output

    # SUMMARY
    summary_chain = _build_chain("SUMMARY", agent_prompts.get("SUMMARY", ""), model)
    summary_ctx = json.dumps(
        {
            "planner": planner_output,
            "data_context": data_context_outputs,
            "strategies": strategy_outputs,
            "strategy_final": strategy_final_output,
            "risk": risk_output,
            "critic": critic_output,
            "upcoming_events": state["events"],
        },
        indent=2,
        default=str,
    )
    summary_holder: Dict[str, Any] = {}
    async for chunk in _run_chain_stream(
        summary_chain,
        {"context": summary_ctx, "user_block": "Produce final concise trading plan."},
        agent="SUMMARY",
        ticker=None,
        holder=summary_holder,
    ):
        yield chunk
    summary_output = summary_holder.get("output")
    state["summary"] = summary_output

    yield _format_sse(
        "pipeline_complete",
        {"pipeline_state": state, "timestamp": datetime.now().isoformat()},
    )
