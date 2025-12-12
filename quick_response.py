"""
Quick Response Handler for AI Trading System.
Fast responses for simple queries without running full pipeline.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from groq import Groq
import os
import json
from dotenv import load_dotenv
from market_data import get_market_analysis
import yfinance as yf

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


class QuickResponseOutput(BaseModel):
    """Quick response output with validation"""
    response_type: str = Field(description="Type of response: market_overview, quick_lookup, educational, follow_up")
    answer: str = Field(
        min_length=20,
        description="Direct answer to user's question"
    )
    data: Optional[Dict[str, Any]] = Field(
        None,
        description="Supporting data if applicable"
    )
    follow_up_suggestions: List[str] = Field(
        default_factory=list,
        description="Suggested follow-up questions"
    )
    needs_full_analysis: bool = Field(
        default=False,
        description="Whether user should run full pipeline for deeper analysis"
    )


class QuickResponseHandler:
    """
    Fast responses for simple queries without running full pipeline.
    Optimized for speed (< 3 seconds).
    """
    
    EDUCATIONAL_KB = {
        "rsi": {
            "name": "RSI (Relative Strength Index)",
            "definition": "RSI is a momentum oscillator that measures the speed and magnitude of price changes on a scale of 0-100.",
            "interpretation": "RSI > 70 = overbought (potential sell signal), RSI < 30 = oversold (potential buy signal), RSI 40-60 = neutral",
            "usage": "Traders use RSI to identify potential reversal points. When RSI is overbought and starts declining, it may signal a price drop. When oversold and rising, it may signal a bounce.",
            "example": "If TSLA has RSI of 28, it's oversold and may bounce soon. If RSI is 75, it's overbought and may pull back."
        },
        "macd": {
            "name": "MACD (Moving Average Convergence Divergence)",
            "definition": "MACD shows the relationship between two moving averages (12-day and 26-day EMAs) and includes a signal line (9-day EMA).",
            "interpretation": "MACD line > Signal line = bullish, MACD line < Signal line = bearish. Crossovers indicate potential trend changes.",
            "usage": "Traders look for MACD crossovers as buy/sell signals. A bullish crossover (MACD crosses above signal) suggests upward momentum.",
            "example": "If MACD line crosses above signal line, it's a bullish signal to consider buying."
        },
        "moving average": {
            "name": "Moving Average (MA / SMA)",
            "definition": "A moving average smooths price data by calculating the average price over a specific period (e.g., 50-day, 200-day).",
            "interpretation": "Price > MA = uptrend, Price < MA = downtrend. The 50-day and 200-day MAs are most commonly watched.",
            "usage": "Traders use MAs to identify trend direction and support/resistance levels. A 'golden cross' (50-day crosses above 200-day) is very bullish.",
            "example": "If AAPL is trading at $180 and the 50-day MA is $175, the stock is in an uptrend."
        },
        "support and resistance": {
            "name": "Support and Resistance Levels",
            "definition": "Support is a price level where buying pressure prevents further decline. Resistance is where selling pressure prevents further rise.",
            "interpretation": "Price bounces off support (buy opportunity), price rejected at resistance (sell opportunity).",
            "usage": "Traders buy near support and sell near resistance. A breakout above resistance or below support signals a strong move.",
            "example": "If TSLA repeatedly bounces at $240, that's a support level. If it breaks below $240, it may drop further."
        },
        "volume": {
            "name": "Trading Volume",
            "definition": "Volume is the number of shares traded during a period. High volume indicates strong interest, low volume indicates weak interest.",
            "interpretation": "Volume confirms trends. Rising prices with high volume = strong uptrend. Rising prices with low volume = weak, may reverse.",
            "usage": "Traders look for volume spikes to confirm breakouts or reversals. 'Volume precedes price' is a key principle.",
            "example": "If NVDA breaks out to new highs on 2x average volume, it's a strong bullish signal."
        },
        "bollinger bands": {
            "name": "Bollinger Bands",
            "definition": "Bollinger Bands consist of a middle band (20-day SMA) and two outer bands (2 standard deviations away).",
            "interpretation": "Price touching upper band = overbought, touching lower band = oversold. Bands squeeze before big moves.",
            "usage": "Traders use Bollinger Bands to identify volatility and potential reversals. A 'squeeze' (narrow bands) often precedes a breakout.",
            "example": "If price touches the lower Bollinger Band and RSI is oversold, it's a potential buy signal."
        },
        "atr": {
            "name": "ATR (Average True Range)",
            "definition": "ATR measures volatility by calculating the average range between high and low prices over a period (typically 14 days).",
            "interpretation": "High ATR = high volatility (larger price swings), Low ATR = low volatility (smaller price swings).",
            "usage": "Traders use ATR to set stop-loss levels. A wider ATR means you need wider stops to avoid getting stopped out prematurely.",
            "example": "If TSLA has ATR of $8, you might set a stop-loss $8-10 below your entry to account for normal volatility."
        }
    }
    
    def __init__(self):
        """Initialize quick response handler"""
        self.client = client
        self.cache = {}  # Simple cache for repeated queries
    
    def handle_market_overview(self) -> QuickResponseOutput:
        """
        Quick market overview.
        Returns: SPY/QQQ/DIA performance, VIX, sector leaders/laggards
        """
        try:
            # Get major indices
            spy = yf.Ticker("SPY")
            qqq = yf.Ticker("QQQ")
            dia = yf.Ticker("DIA")
            vix = yf.Ticker("^VIX")
            
            # Get current data
            spy_data = spy.history(period="1d")
            qqq_data = qqq.history(period="1d")
            dia_data = dia.history(period="1d")
            vix_data = vix.history(period="1d")
            
            if spy_data.empty:
                return QuickResponseOutput(
                    response_type="market_overview",
                    answer="Market data is currently unavailable (market may be closed).",
                    data={},
                    follow_up_suggestions=["Try again during market hours"]
                )
            
            # Calculate changes
            spy_change = ((spy_data['Close'].iloc[-1] - spy_data['Open'].iloc[-1]) / spy_data['Open'].iloc[-1]) * 100
            qqq_change = ((qqq_data['Close'].iloc[-1] - qqq_data['Open'].iloc[-1]) / qqq_data['Open'].iloc[-1]) * 100
            dia_change = ((dia_data['Close'].iloc[-1] - dia_data['Open'].iloc[-1]) / dia_data['Open'].iloc[-1]) * 100
            vix_level = vix_data['Close'].iloc[-1]
            
            # Determine market sentiment
            if spy_change > 0.5:
                sentiment = "bullish"
                sentiment_desc = "strong buying pressure"
            elif spy_change < -0.5:
                sentiment = "bearish"
                sentiment_desc = "selling pressure"
            else:
                sentiment = "neutral"
                sentiment_desc = "mixed trading"
            
            # Determine volatility
            if vix_level > 25:
                volatility = "high"
            elif vix_level < 15:
                volatility = "low"
            else:
                volatility = "moderate"
            
            # Build answer
            answer = f"""**Market Overview:**

📊 **Major Indices:**
- S&P 500 (SPY): {spy_change:+.2f}%
- Nasdaq (QQQ): {qqq_change:+.2f}%
- Dow Jones (DIA): {dia_change:+.2f}%

📈 **Market Sentiment:** {sentiment.upper()} ({sentiment_desc})

⚡ **Volatility (VIX):** {vix_level:.2f} ({volatility})

**Interpretation:**
{self._interpret_market(spy_change, qqq_change, vix_level)}
"""
            
            data = {
                "spy_change": float(spy_change),
                "qqq_change": float(qqq_change),
                "dia_change": float(dia_change),
                "vix": float(vix_level),
                "sentiment": sentiment,
                "volatility": volatility
            }
            
            suggestions = [
                "Find oversold stocks in this market",
                "Show me tech sector performance",
                "What stocks are breaking out today?"
            ]
            
            return QuickResponseOutput(
                response_type="market_overview",
                answer=answer,
                data=data,
                follow_up_suggestions=suggestions,
                needs_full_analysis=False
            )
            
        except Exception as e:
            return QuickResponseOutput(
                response_type="market_overview",
                answer=f"Unable to fetch market overview: {str(e)}. Market may be closed.",
                data={},
                follow_up_suggestions=["Try again during market hours"]
            )
    
    def _interpret_market(self, spy_change: float, qqq_change: float, vix: float) -> str:
        """Interpret market conditions"""
        if spy_change > 1 and qqq_change > spy_change:
            return "Tech is leading the market higher. Growth stocks are in favor."
        elif spy_change > 1 and qqq_change < spy_change:
            return "Broad market rally with defensive sectors leading. Risk-off sentiment."
        elif spy_change < -1 and vix > 25:
            return "Market selling with elevated fear (high VIX). Consider waiting for stabilization."
        elif spy_change < -1:
            return "Market weakness. Look for oversold opportunities or wait for reversal signals."
        else:
            return "Market is consolidating. Wait for clearer direction or focus on individual stock setups."
    
    def handle_quick_lookup(self, ticker: str) -> QuickResponseOutput:
        """
        Quick stock lookup.
        Returns: Current price, day change, volume, basic stats
        """
        try:
            ticker = ticker.upper()
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="1d")
            
            if hist.empty:
                return QuickResponseOutput(
                    response_type="quick_lookup",
                    answer=f"No data available for {ticker}. Ticker may be invalid or market is closed.",
                    data={},
                    follow_up_suggestions=[]
                )
            
            current_price = hist['Close'].iloc[-1]
            day_change = ((hist['Close'].iloc[-1] - hist['Open'].iloc[-1]) / hist['Open'].iloc[-1]) * 100
            volume = hist['Volume'].iloc[-1]
            
            company_name = info.get('longName', ticker)
            market_cap = info.get('marketCap', 0)
            pe_ratio = info.get('trailingPE', 'N/A')
            
            answer = f"""**{company_name} ({ticker})**

💰 **Price:** ${current_price:.2f} ({day_change:+.2f}% today)
📊 **Volume:** {volume:,.0f} shares
🏢 **Market Cap:** ${market_cap/1e9:.2f}B
📈 **P/E Ratio:** {pe_ratio if isinstance(pe_ratio, str) else f'{pe_ratio:.2f}'}

**Quick Take:**
{self._quick_stock_take(day_change, volume, info)}
"""
            
            data = {
                "ticker": ticker,
                "price": float(current_price),
                "day_change_percent": float(day_change),
                "volume": int(volume),
                "market_cap": market_cap,
                "pe_ratio": pe_ratio
            }
            
            suggestions = [
                f"Run full analysis on {ticker}",
                f"Compare {ticker} to sector peers",
                f"Show me chart patterns for {ticker}"
            ]
            
            return QuickResponseOutput(
                response_type="quick_lookup",
                answer=answer,
                data=data,
                follow_up_suggestions=suggestions,
                needs_full_analysis=True
            )
            
        except Exception as e:
            return QuickResponseOutput(
                response_type="quick_lookup",
                answer=f"Error looking up {ticker}: {str(e)}",
                data={},
                follow_up_suggestions=["Check if ticker is correct"]
            )
    
    def _quick_stock_take(self, day_change: float, volume: float, info: dict) -> str:
        """Generate quick take on stock"""
        if day_change > 3:
            return f"Strong upward momentum today (+{day_change:.1f}%). Consider running full analysis to see if trend is sustainable."
        elif day_change < -3:
            return f"Significant decline today ({day_change:.1f}%). May present buying opportunity if fundamentals are strong."
        else:
            return "Modest price movement today. Run full analysis for detailed technical and fundamental insights."
    
    def handle_educational(self, topic: str) -> QuickResponseOutput:
        """
        Educational response about trading concepts.
        Returns: Explanation of trading concept
        """
        topic_lower = topic.lower()
        
        # Find matching topic in knowledge base
        matched_topic = None
        for key in self.EDUCATIONAL_KB.keys():
            if key in topic_lower or topic_lower in key:
                matched_topic = key
                break
        
        if matched_topic:
            kb_entry = self.EDUCATIONAL_KB[matched_topic]
            
            answer = f"""**{kb_entry['name']}**

**Definition:**
{kb_entry['definition']}

**How to Interpret:**
{kb_entry['interpretation']}

**How Traders Use It:**
{kb_entry['usage']}

**Example:**
{kb_entry['example']}
"""
            
            suggestions = [
                "Show me stocks with this indicator",
                "How do I combine this with other indicators?",
                "Find me a trading opportunity using this"
            ]
            
            return QuickResponseOutput(
                response_type="educational",
                answer=answer,
                data={"topic": matched_topic, "details": kb_entry},
                follow_up_suggestions=suggestions,
                needs_full_analysis=False
            )
        else:
            # Use LLM for topics not in knowledge base
            return self._llm_educational_response(topic)
    
    def _llm_educational_response(self, topic: str) -> QuickResponseOutput:
        """Use LLM for educational topics not in knowledge base"""
        prompt = f"""Explain the following trading/investing concept in a clear, concise way:

Topic: {topic}

Provide:
1. Definition (1-2 sentences)
2. How to interpret it
3. How traders use it
4. A practical example

Keep it under 200 words and focus on practical application."""

        try:
            completion = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=512
            )
            
            answer = completion.choices[0].message.content
            
            return QuickResponseOutput(
                response_type="educational",
                answer=answer,
                data={"topic": topic},
                follow_up_suggestions=["Show me more examples", "Find stocks using this concept"],
                needs_full_analysis=False
            )
        except Exception as e:
            return QuickResponseOutput(
                response_type="educational",
                answer=f"I don't have information about '{topic}' in my knowledge base. Try asking about: RSI, MACD, Moving Averages, Support/Resistance, Volume, Bollinger Bands, or ATR.",
                data={},
                follow_up_suggestions=[]
            )
    
    def handle_follow_up(self, question: str, context: Dict) -> QuickResponseOutput:
        """
        Context-aware follow-up response.
        Uses previous query context to answer
        """
        # Use LLM with context to answer follow-up
        context_str = json.dumps(context, indent=2)
        
        prompt = f"""You are answering a follow-up question about a previous trading analysis.

Previous context:
{context_str}

User's follow-up question: {question}

Provide a concise, helpful answer based on the context. Keep it under 150 words."""

        try:
            completion = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=512
            )
            
            answer = completion.choices[0].message.content
            
            return QuickResponseOutput(
                response_type="follow_up",
                answer=answer,
                data=context,
                follow_up_suggestions=["Run new analysis", "Ask another question"],
                needs_full_analysis=False
            )
        except Exception as e:
            return QuickResponseOutput(
                response_type="follow_up",
                answer="I need more context to answer that. Could you be more specific?",
                data={},
                follow_up_suggestions=[]
            )


# Global instance
_quick_handler = None

def get_quick_handler() -> QuickResponseHandler:
    """Get global quick response handler instance"""
    global _quick_handler
    if _quick_handler is None:
        _quick_handler = QuickResponseHandler()
    return _quick_handler
