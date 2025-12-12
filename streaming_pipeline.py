"""
Streaming Pipeline Endpoint - SSE Implementation

This file contains the streaming version of the pipeline that sends
Server-Sent Events (SSE) to the frontend for real-time updates.
"""

import asyncio
import json
from datetime import datetime
from typing import List, Optional


async def stream_pipeline_events(headline: str, user_tickers: List[str] | None = None, user_horizon: Optional[str] = None):
    """
    Generator function that yields SSE-formatted events for the pipeline.
    
    Event types:
    - agent_start: Agent begins processing
    - agent_token: Token from streaming LLM
    - agent_complete: Agent finished with parsed output
    - pipeline_complete: Entire pipeline finished
    - error: Error occurred
    """
    
    def format_sse(event_type: str, data: dict) -> str:
        """Format data as Server-Sent Event"""
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
    
    pipeline_state = {}
    shared_payload = {
        "headline": headline,
        "user_tickers": user_tickers or [],
        "user_horizon": user_horizon,
        "note": "Multi-agent trading system for educational purposes only.",
    }
    
    try:
        # ---- 1) PLANNER ----
        yield format_sse("agent_start", {
            "agent": "PLANNER",
            "timestamp": datetime.now().isoformat()
        })
        
        planner_input = {**shared_payload}
        planner_output = None
        
        # Import the streaming function
        from main import agent_llm_call_streaming
        
        async for event in agent_llm_call_streaming("PLANNER", planner_input):
            if event["type"] == "token":
                yield format_sse("agent_token", {
                    "agent": "PLANNER",
                    "token": event["content"]
                })
            elif event["type"] == "complete":
                planner_output = event["content"]
                pipeline_state["planner"] = planner_output
                yield format_sse("agent_complete", {
                    "agent": "PLANNER",
                    "output": planner_output,
                    "timestamp": datetime.now().isoformat()
                })
            elif event["type"] == "error":
                yield format_sse("error", {
                    "agent": "PLANNER",
                    "error": event["content"]
                })
                return
        
        # Determine target tickers (respect user provided list first)
        effective_tickers = user_tickers or []
        if not effective_tickers and isinstance(planner_output, dict):
            effective_tickers = planner_output.get("tickers", [])

        # Pre-fetch market data per ticker (used by downstream agents)
        from market_data import get_market_analysis, format_market_data_for_agent, get_economic_calendar, get_earnings_calendar, scan_chart_patterns, get_chart_data
        market_data_bundle = []
        for ticker in effective_tickers[:3]:  # cap for demo safety
            try:
                market_data_raw = get_market_analysis(ticker, period="3mo")
                fundamentals = market_data_raw.get("fundamentals")
                news_articles = market_data_raw.get("news", [])
                earnings_events = get_earnings_calendar(ticker)
                patterns = scan_chart_patterns(ticker, period="6mo")
                
                # Get institutional levels
                chart_result = get_chart_data(ticker, period="6mo")
                institutional_levels = chart_result.get("institutional_levels", {}) if "error" not in chart_result else {}
                
                market_data_formatted = format_market_data_for_agent(market_data_raw, institutional_levels)
            except Exception as e:
                market_data_raw = {}
                fundamentals = None
                news_articles = []
                market_data_formatted = f"Unable to fetch market data for {ticker}: {str(e)}"
                earnings_events = []
                patterns = []
                institutional_levels = {}
            market_data_bundle.append({
                "ticker": ticker,
                "raw": market_data_raw,
                "fundamentals": fundamentals,
                "news": news_articles,
                "formatted": market_data_formatted,
                "earnings": earnings_events,
                "patterns": patterns,
                "institutional_levels": institutional_levels,
            })

        macro_events = get_economic_calendar()
        shared_payload["upcoming_events"] = {
            "macro": macro_events,
            "earnings": {b["ticker"]: b["earnings"] for b in market_data_bundle},
        }
        pipeline_state["events"] = shared_payload["upcoming_events"]
        pipeline_state["patterns"] = {b["ticker"]: b["patterns"] for b in market_data_bundle}

        # ---- 2) NEWS ----
        yield format_sse("agent_start", {
            "agent": "NEWS",
            "timestamp": datetime.now().isoformat()
        })

        news_input = {
            **shared_payload,
            "planner": planner_output,
            "headline": headline,
            "articles": sum([b["news"] for b in market_data_bundle], []),
            "fundamentals": market_data_bundle[0]["fundamentals"] if market_data_bundle else None,
            "upcoming_events": shared_payload["upcoming_events"],
        }
        news_output = None

        async for event in agent_llm_call_streaming("NEWS", news_input):
            if event["type"] == "token":
                yield format_sse("agent_token", {
                    "agent": "NEWS",
                    "token": event["content"]
                })
            elif event["type"] == "complete":
                news_output = event["content"]
                pipeline_state["news"] = news_output
                yield format_sse("agent_complete", {
                    "agent": "NEWS",
                    "output": news_output,
                    "timestamp": datetime.now().isoformat()
                })
            elif event["type"] == "error":
                yield format_sse("error", {
                    "agent": "NEWS",
                    "error": event["content"]
                })
                return

        # ---- 3) DATA_CONTEXT per ticker ----
        data_outputs = []
        for bundle in market_data_bundle:
            ticker = bundle["ticker"]
            yield format_sse("agent_start", {
                "agent": "DATA_CONTEXT",
                "ticker": ticker,
                "timestamp": datetime.now().isoformat()
            })

            data_input = {
                **shared_payload,
                "planner": planner_output,
                "target_ticker": ticker,
                "market_data": bundle["formatted"],
                "fundamentals": bundle["fundamentals"],
                "news_briefing": news_output,
                "upcoming_events": shared_payload["upcoming_events"],
                "pattern_scan": bundle["patterns"],
            }
            data_output = None
            
            async for event in agent_llm_call_streaming("DATA_CONTEXT", data_input):
                if event["type"] == "token":
                    yield format_sse("agent_token", {
                        "agent": "DATA_CONTEXT",
                        "ticker": ticker,
                        "token": event["content"]
                    })
                elif event["type"] == "complete":
                    data_output = event["content"]
                    if isinstance(data_output, dict):
                        data_output["primary_ticker"] = ticker
                    data_outputs.append(data_output)
                    pipeline_state.setdefault("data_context", []).append(data_output)
                    yield format_sse("agent_complete", {
                        "agent": "DATA_CONTEXT",
                        "ticker": ticker,
                        "output": data_output,
                        "timestamp": datetime.now().isoformat()
                    })
                elif event["type"] == "error":
                    yield format_sse("error", {
                        "agent": "DATA_CONTEXT",
                        "ticker": ticker,
                        "error": event["content"]
                    })
                    return
        
        # ---- 4) STRATEGY per ticker ----
        strategies = []
        for idx, bundle in enumerate(market_data_bundle):
            ticker = bundle["ticker"]
            context_for_ticker = data_outputs[idx] if idx < len(data_outputs) else None

            yield format_sse("agent_start", {
                "agent": "STRATEGY",
                "ticker": ticker,
                "timestamp": datetime.now().isoformat()
            })
            
            strategy_input = {
                **shared_payload,
                "planner": planner_output,
                "data_context": context_for_ticker,
                "target_ticker": ticker,
                "market_data": bundle["formatted"],
                "fundamentals": bundle["fundamentals"],
                "news_briefing": news_output,
                "upcoming_events": shared_payload["upcoming_events"],
                "pattern_scan": bundle.get("patterns"),
            }
            strategy_output = None
            
            async for event in agent_llm_call_streaming("STRATEGY", strategy_input):
                if event["type"] == "token":
                    yield format_sse("agent_token", {
                        "agent": "STRATEGY",
                        "ticker": ticker,
                        "token": event["content"]
                    })
                elif event["type"] == "complete":
                    strategy_output = event["content"]
                    if isinstance(strategy_output, dict):
                        strategy_output["ticker"] = ticker
                    strategies.append(strategy_output)
                    pipeline_state.setdefault("strategy_v1", []).append(strategy_output)
                    yield format_sse("agent_complete", {
                        "agent": "STRATEGY",
                        "ticker": ticker,
                        "output": strategy_output,
                        "timestamp": datetime.now().isoformat()
                    })
                elif event["type"] == "error":
                    yield format_sse("error", {
                        "agent": "STRATEGY",
                        "ticker": ticker,
                        "error": event["content"]
                    })
                    return
        
        # ---- 5) STRATEGY_FINAL comparator ----
        yield format_sse("agent_start", {
            "agent": "STRATEGY_FINAL",
            "timestamp": datetime.now().isoformat()
        })

        comparator_input = {
            **shared_payload,
            "planner": planner_output,
            "data_context_all": data_outputs,
            "strategies": strategies,
            "instruction": "Compare the per-ticker strategies and choose the best single trade. Return best_strategy and explain the choice.",
            "upcoming_events": shared_payload["upcoming_events"],
            "pattern_scan": pipeline_state.get("patterns"),
        }
        strategy_final_output = None

        async for event in agent_llm_call_streaming("STRATEGY_FINAL", comparator_input):
            if event["type"] == "token":
                yield format_sse("agent_token", {
                    "agent": "STRATEGY_FINAL",
                    "token": event["content"]
                })
            elif event["type"] == "complete":
                strategy_final_output = event["content"]
                pipeline_state["strategy_final"] = strategy_final_output
                yield format_sse("agent_complete", {
                    "agent": "STRATEGY_FINAL",
                    "output": strategy_final_output,
                    "timestamp": datetime.now().isoformat()
                })
            elif event["type"] == "error":
                yield format_sse("error", {
                    "agent": "STRATEGY_FINAL",
                    "error": event["content"]
                })
                return

        # ---- 5.5) BULL vs BEAR DEBATE ----
        best_strategy = strategy_final_output.get("best_strategy", strategy_final_output) if isinstance(strategy_final_output, dict) else {}
        
        # BULL agent
        yield format_sse("agent_start", {
            "agent": "BULL",
            "timestamp": datetime.now().isoformat()
        })
        
        bull_input = {
            **shared_payload,
            "planner": planner_output,
            "data_context": data_outputs,
            "strategy": best_strategy,
            "strategy_final": strategy_final_output,
            "upcoming_events": shared_payload["upcoming_events"],
            "pattern_scan": pipeline_state.get("patterns"),
        }
        bull_output = None
        
        async for event in agent_llm_call_streaming("BULL", bull_input):
            if event["type"] == "token":
                yield format_sse("agent_token", {
                    "agent": "BULL",
                    "token": event["content"]
                })
            elif event["type"] == "complete":
                bull_output = event["content"]
                pipeline_state["bull"] = bull_output
                yield format_sse("agent_complete", {
                    "agent": "BULL",
                    "output": bull_output,
                    "timestamp": datetime.now().isoformat()
                })
            elif event["type"] == "error":
                yield format_sse("error", {
                    "agent": "BULL",
                    "error": event["content"]
                })
                return

        # BEAR agent
        yield format_sse("agent_start", {
            "agent": "BEAR",
            "timestamp": datetime.now().isoformat()
        })
        
        bear_input = {
            **shared_payload,
            "planner": planner_output,
            "data_context": data_outputs,
            "strategy": best_strategy,
            "strategy_final": strategy_final_output,
            "bull_thesis": bull_output,
            "upcoming_events": shared_payload["upcoming_events"],
            "pattern_scan": pipeline_state.get("patterns"),
        }
        bear_output = None
        
        async for event in agent_llm_call_streaming("BEAR", bear_input):
            if event["type"] == "token":
                yield format_sse("agent_token", {
                    "agent": "BEAR",
                    "token": event["content"]
                })
            elif event["type"] == "complete":
                bear_output = event["content"]
                pipeline_state["bear"] = bear_output
                yield format_sse("agent_complete", {
                    "agent": "BEAR",
                    "output": bear_output,
                    "timestamp": datetime.now().isoformat()
                })
            elif event["type"] == "error":
                yield format_sse("error", {
                    "agent": "BEAR",
                    "error": event["content"]
                })
                return

        # DEVILS_ADVOCATE agent
        yield format_sse("agent_start", {
            "agent": "DEVILS_ADVOCATE",
            "timestamp": datetime.now().isoformat()
        })
        
        devils_advocate_input = {
            **shared_payload,
            "planner": planner_output,
            "data_context": data_outputs,
            "strategy": best_strategy,
            "bull_thesis": bull_output,
            "bear_thesis": bear_output,
            "upcoming_events": shared_payload["upcoming_events"],
            "pattern_scan": pipeline_state.get("patterns"),
        }
        devils_advocate_output = None
        
        async for event in agent_llm_call_streaming("DEVILS_ADVOCATE", devils_advocate_input):
            if event["type"] == "token":
                yield format_sse("agent_token", {
                    "agent": "DEVILS_ADVOCATE",
                    "token": event["content"]
                })
            elif event["type"] == "complete":
                devils_advocate_output = event["content"]
                pipeline_state["devils_advocate"] = devils_advocate_output
                yield format_sse("agent_complete", {
                    "agent": "DEVILS_ADVOCATE",
                    "output": devils_advocate_output,
                    "timestamp": datetime.now().isoformat()
                })
            elif event["type"] == "error":
                yield format_sse("error", {
                    "agent": "DEVILS_ADVOCATE",
                    "error": event["content"]
                })
                return

        # CONSENSUS agent
        yield format_sse("agent_start", {
            "agent": "CONSENSUS",
            "timestamp": datetime.now().isoformat()
        })
        
        consensus_input = {
            **shared_payload,
            "planner": planner_output,
            "data_context": data_outputs,
            "strategy": best_strategy,
            "bull_thesis": bull_output,
            "bear_thesis": bear_output,
            "devils_advocate": devils_advocate_output,
            "upcoming_events": shared_payload["upcoming_events"],
            "pattern_scan": pipeline_state.get("patterns"),
        }
        consensus_output = None
        
        async for event in agent_llm_call_streaming("CONSENSUS", consensus_input):
            if event["type"] == "token":
                yield format_sse("agent_token", {
                    "agent": "CONSENSUS",
                    "token": event["content"]
                })
            elif event["type"] == "complete":
                consensus_output = event["content"]
                pipeline_state["consensus"] = consensus_output
                yield format_sse("agent_complete", {
                    "agent": "CONSENSUS",
                    "output": consensus_output,
                    "timestamp": datetime.now().isoformat()
                })
            elif event["type"] == "error":
                yield format_sse("error", {
                    "agent": "CONSENSUS",
                    "error": event["content"]
                })
                return

        # Update best_strategy with consensus recommendation
        if isinstance(consensus_output, dict) and "recommended_action" in consensus_output:
            best_strategy = consensus_output.get("recommended_action", best_strategy)

        # ---- 6) RISK ----
        yield format_sse("agent_start", {
            "agent": "RISK",
            "timestamp": datetime.now().isoformat()
        })
        
        best_strategy = strategy_final_output.get("best_strategy", strategy_final_output) if isinstance(strategy_final_output, dict) else {}
        risk_input = {
            **shared_payload,
            "planner": planner_output,
            "data_context": data_outputs,
            "strategy": best_strategy,
            "fundamentals": market_data_bundle[0]["fundamentals"] if market_data_bundle else None,
            "news_briefing": news_output,
            "upcoming_events": shared_payload["upcoming_events"],
            "pattern_scan": pipeline_state.get("patterns"),
        }
        risk_output = None
        
        async for event in agent_llm_call_streaming("RISK", risk_input):
            if event["type"] == "token":
                yield format_sse("agent_token", {
                    "agent": "RISK",
                    "token": event["content"]
                })
            elif event["type"] == "complete":
                risk_output = event["content"]
                pipeline_state["risk"] = risk_output
                yield format_sse("agent_complete", {
                    "agent": "RISK",
                    "output": risk_output,
                    "timestamp": datetime.now().isoformat()
                })
            elif event["type"] == "error":
                yield format_sse("error", {
                    "agent": "RISK",
                    "error": event["content"]
                })
                return
        
        # ---- 7) CRITIC ----
        yield format_sse("agent_start", {
            "agent": "CRITIC",
            "timestamp": datetime.now().isoformat()
        })

        critic_input = {
            **shared_payload,
            "planner": planner_output,
            "data_context": data_outputs,
            "strategy": best_strategy,
            "risk": risk_output,
            "fundamentals": market_data_bundle[0]["fundamentals"] if market_data_bundle else None,
            "news_briefing": news_output,
            "upcoming_events": shared_payload["upcoming_events"],
            "pattern_scan": pipeline_state.get("patterns"),
        }
        critic_output = None

        async for event in agent_llm_call_streaming("CRITIC", critic_input):
            if event["type"] == "token":
                yield format_sse("agent_token", {
                    "agent": "CRITIC",
                    "token": event["content"]
                })
            elif event["type"] == "complete":
                critic_output = event["content"]
                pipeline_state["critic"] = critic_output
                yield format_sse("agent_complete", {
                    "agent": "CRITIC",
                    "output": critic_output,
                    "timestamp": datetime.now().isoformat()
                })
            elif event["type"] == "error":
                yield format_sse("error", {
                    "agent": "CRITIC",
                    "error": event["content"]
                })
                return

        # ---- 8) SUMMARY ----
        yield format_sse("agent_start", {
            "agent": "SUMMARY",
            "timestamp": datetime.now().isoformat()
        })
        
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
            "fundamentals": market_data_bundle[0]["fundamentals"] if market_data_bundle else None,
            "news_briefing": news_output,
            "market_data": [b["formatted"] for b in market_data_bundle],
            "upcoming_events": shared_payload["upcoming_events"],
            "pattern_scan": pipeline_state.get("patterns"),
        }
        summary_output = None
        
        async for event in agent_llm_call_streaming("SUMMARY", summary_input):
            if event["type"] == "token":
                yield format_sse("agent_token", {
                    "agent": "SUMMARY",
                    "token": event["content"]
                })
            elif event["type"] == "complete":
                summary_output = event["content"]
                pipeline_state["summary"] = summary_output
                yield format_sse("agent_complete", {
                    "agent": "SUMMARY",
                    "output": summary_output,
                    "timestamp": datetime.now().isoformat()
                })
            elif event["type"] == "error":
                yield format_sse("error", {
                    "agent": "SUMMARY",
                    "error": event["content"]
                })
                return
        
        # Pipeline complete
        yield format_sse("pipeline_complete", {
            "pipeline_state": pipeline_state,
            "timestamp": datetime.now().isoformat()
        })
        
    except asyncio.CancelledError:
        # Client disconnected; stop emitting to avoid socket.send errors
        return
    except Exception as e:
        yield format_sse("error", {
            "agent": "SYSTEM",
            "error": f"Pipeline error: {str(e)}"
        })
