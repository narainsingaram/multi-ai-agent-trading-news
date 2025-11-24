"""
Streaming Pipeline Endpoint - SSE Implementation

This file contains the streaming version of the pipeline that sends
Server-Sent Events (SSE) to the frontend for real-time updates.
"""

import json
from datetime import datetime


async def stream_pipeline_events(headline: str):
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
        
        # Pre-fetch market data (used by downstream agents)
        market_data_formatted = ""
        market_data_raw = {}
        fundamentals = None
        news_articles = []

        if isinstance(planner_output, dict) and planner_output.get("tickers"):
            tickers = planner_output.get("tickers", [])
            if tickers and len(tickers) > 0:
                primary_ticker = tickers[0]
                try:
                    from market_data import get_market_analysis, format_market_data_for_agent
                    market_data_raw = get_market_analysis(primary_ticker, period="3mo")
                    fundamentals = market_data_raw.get("fundamentals")
                    news_articles = market_data_raw.get("news", [])
                    market_data_formatted = format_market_data_for_agent(market_data_raw)
                except Exception as e:
                    market_data_formatted = f"Unable to fetch market data: {str(e)}"

        # ---- 2) NEWS ----
        yield format_sse("agent_start", {
            "agent": "NEWS",
            "timestamp": datetime.now().isoformat()
        })

        news_input = {
            **shared_payload,
            "planner": planner_output,
            "headline": headline,
            "articles": news_articles,
            "fundamentals": fundamentals,
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

        # ---- 3) DATA_CONTEXT ----
        yield format_sse("agent_start", {
            "agent": "DATA_CONTEXT",
            "timestamp": datetime.now().isoformat()
        })

        data_input = {
            **shared_payload,
            "planner": planner_output,
            "market_data": market_data_formatted,
            "fundamentals": fundamentals,
            "news_briefing": news_output,
        }
        data_output = None
        
        async for event in agent_llm_call_streaming("DATA_CONTEXT", data_input):
            if event["type"] == "token":
                yield format_sse("agent_token", {
                    "agent": "DATA_CONTEXT",
                    "token": event["content"]
                })
            elif event["type"] == "complete":
                data_output = event["content"]
                pipeline_state["data_context"] = data_output
                yield format_sse("agent_complete", {
                    "agent": "DATA_CONTEXT",
                    "output": data_output,
                    "timestamp": datetime.now().isoformat()
                })
            elif event["type"] == "error":
                yield format_sse("error", {
                    "agent": "DATA_CONTEXT",
                    "error": event["content"]
                })
                return
        
        # ---- 4) STRATEGY ----
        yield format_sse("agent_start", {
            "agent": "STRATEGY",
            "timestamp": datetime.now().isoformat()
        })
        
        strategy_input = {
            **shared_payload,
            "planner": planner_output,
            "data_context": data_output,
            "market_data": market_data_formatted,
            "fundamentals": fundamentals,
            "news_briefing": news_output,
        }
        strategy_output = None
        
        async for event in agent_llm_call_streaming("STRATEGY", strategy_input):
            if event["type"] == "token":
                yield format_sse("agent_token", {
                    "agent": "STRATEGY",
                    "token": event["content"]
                })
            elif event["type"] == "complete":
                strategy_output = event["content"]
                pipeline_state["strategy_v1"] = strategy_output
                yield format_sse("agent_complete", {
                    "agent": "STRATEGY",
                    "output": strategy_output,
                    "timestamp": datetime.now().isoformat()
                })
            elif event["type"] == "error":
                yield format_sse("error", {
                    "agent": "STRATEGY",
                    "error": event["content"]
                })
                return
        
        # ---- 5) RISK ----
        yield format_sse("agent_start", {
            "agent": "RISK",
            "timestamp": datetime.now().isoformat()
        })
        
        risk_input = {
            **shared_payload,
            "planner": planner_output,
            "data_context": data_output,
            "strategy": strategy_output,
            "fundamentals": fundamentals,
            "news_briefing": news_output,
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
        
        # ---- 6) CRITIC ----
        yield format_sse("agent_start", {
            "agent": "CRITIC",
            "timestamp": datetime.now().isoformat()
        })

        critic_input = {
            **shared_payload,
            "planner": planner_output,
            "data_context": data_output,
            "strategy": strategy_output,
            "risk": risk_output,
            "fundamentals": fundamentals,
            "news_briefing": news_output,
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

        # ---- 7) STRATEGY_FINAL (Optional Revision) ----
        strategy_final_output = strategy_output
        
        if isinstance(critic_output, dict) and critic_output.get("status") == "NEEDS_REVISION":
            yield format_sse("agent_start", {
                "agent": "STRATEGY_FINAL",
                "timestamp": datetime.now().isoformat()
            })
            
            revision_input = {
                **shared_payload,
                "planner": planner_output,
                "data_context": data_output,
                "previous_strategy": strategy_output,
                "critic_feedback": critic_output,
                "fundamentals": fundamentals,
                "news_briefing": news_output,
                "instruction": (
                    "Revise the previous strategy to address ALL critiques above. "
                    "Keep the same output JSON schema as STRATEGY."
                ),
            }
            
            async for event in agent_llm_call_streaming("STRATEGY", revision_input):
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
        else:
            # If no revision needed, we still emit a quick complete event for UI consistency
            # or just skip it. But since UI expects it in the list, let's emit a "skipped" or just complete immediately.
            # For simplicity, let's just mark it complete with original strategy
            pipeline_state["strategy_final"] = strategy_final_output
            yield format_sse("agent_complete", {
                "agent": "STRATEGY_FINAL",
                "output": strategy_final_output,
                "timestamp": datetime.now().isoformat()
            })

        # ---- 8) SUMMARY ----
        yield format_sse("agent_start", {
            "agent": "SUMMARY",
            "timestamp": datetime.now().isoformat()
        })
        
        summary_input = {
            **shared_payload,
            "planner": planner_output,
            "data_context": data_output,
            "strategy_final": strategy_final_output,
            "risk": risk_output,
            "critic": critic_output,
            "fundamentals": fundamentals,
            "news_briefing": news_output,
            "market_data": market_data_formatted,
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
        
    except Exception as e:
        yield format_sse("error", {
            "agent": "SYSTEM",
            "error": f"Pipeline error: {str(e)}"
        })
