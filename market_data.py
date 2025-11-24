"""
Market Data Service
Fetches real-time stock data, historical prices, and calculates technical indicators.
Uses Finnhub API for real-time quotes and yfinance for historical data.
"""

import os
import yfinance as yf
import finnhub
import pandas as pd

# Fix for pandas_ta compatibility with newer numpy versions
import numpy as np
if not hasattr(np, 'NaN'):
    np.NaN = np.nan

import pandas_ta as ta
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from dotenv import load_dotenv

load_dotenv()

# Initialize Finnhub client
finnhub_client = finnhub.Client(api_key=os.getenv("FINNHUB_API_KEY"))

# Cache to avoid excessive API calls
_cache = {}
_cache_timeout = 300  # 5 minutes


def get_stock_quote(ticker: str) -> Optional[Dict]:
    """
    Get real-time stock quote from Finnhub.
    Returns current price, change, percent change, high, low, open, previous close.
    """
    cache_key = f"quote_{ticker}"
    if cache_key in _cache:
        cached_time, cached_data = _cache[cache_key]
        if (datetime.now() - cached_time).seconds < _cache_timeout:
            return cached_data
    
    try:
        quote = finnhub_client.quote(ticker)
        data = {
            "current_price": quote.get("c"),
            "change": quote.get("d"),
            "percent_change": quote.get("dp"),
            "high": quote.get("h"),
            "low": quote.get("l"),
            "open": quote.get("o"),
            "previous_close": quote.get("pc"),
            "timestamp": quote.get("t"),
        }
        _cache[cache_key] = (datetime.now(), data)
        return data
    except Exception as e:
        print(f"Error fetching quote for {ticker}: {e}")
        return None


def get_company_info(ticker: str) -> Optional[Dict]:
    """
    Get company profile from Finnhub.
    Returns company name, industry, market cap, etc.
    """
    cache_key = f"company_{ticker}"
    if cache_key in _cache:
        cached_time, cached_data = _cache[cache_key]
        if (datetime.now() - cached_time).seconds < 3600:  # Cache for 1 hour
            return cached_data
    
    try:
        profile = finnhub_client.company_profile2(symbol=ticker)
        data = {
            "name": profile.get("name"),
            "ticker": profile.get("ticker"),
            "industry": profile.get("finnhubIndustry"),
            "market_cap": profile.get("marketCapitalization"),
            "shares_outstanding": profile.get("shareOutstanding"),
            "country": profile.get("country"),
            "currency": profile.get("currency"),
            "exchange": profile.get("exchange"),
            "ipo_date": profile.get("ipo"),
            "logo": profile.get("logo"),
            "weburl": profile.get("weburl"),
        }
        _cache[cache_key] = (datetime.now(), data)
        return data
    except Exception as e:
        print(f"Error fetching company info for {ticker}: {e}")
        return None


def get_fundamentals(ticker: str) -> Optional[Dict]:
    """
    Fetch fundamental metrics using yfinance.
    Returns a compact snapshot useful for valuation/quality checks.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info:
            return None

        return {
            "market_cap": info.get("marketCap"),
            "forward_pe": info.get("forwardPE"),
            "trailing_pe": info.get("trailingPE"),
            "forward_eps": info.get("forwardEps"),
            "trailing_eps": info.get("trailingEps"),
            "peg_ratio": info.get("pegRatio"),
            "price_to_book": info.get("priceToBook"),
            "debt_to_equity": info.get("debtToEquity"),
            "roe": info.get("returnOnEquity"),
            "profit_margin": info.get("profitMargins"),
            "free_cashflow": info.get("freeCashflow"),
            "dividend_yield": info.get("dividendYield"),
            "earnings_date": info.get("earningsDate"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
        }
    except Exception as e:
        print(f"Error fetching fundamentals for {ticker}: {e}")
        return None


def get_latest_news(ticker: str, limit: int = 5) -> List[Dict]:
    """
    Fetch recent news for a ticker via yfinance.
    Returns a list of structured items with title, publisher, time, link, and summary if present.
    """
    articles: List[Dict] = []
    try:
        stock = yf.Ticker(ticker)
        raw_news = []
        if hasattr(stock, "news"):
            raw_news = stock.news or []
        elif hasattr(stock, "get_news"):
            raw_news = stock.get_news() or []

        for item in raw_news[:limit]:
            articles.append({
                "title": item.get("title"),
                "publisher": item.get("publisher"),
                "published": item.get("providerPublishTime"),
                "link": item.get("link"),
                "summary": item.get("summary") or item.get("content") or "",
                "tickers": item.get("relatedTickers") or [],
            })
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
    return articles


def get_live_news(ticker: str, limit: int = 5) -> List[Dict]:
    """
    Try to fetch live news via Finnhub (requires FINNHUB_API_KEY). Fallback to yfinance.
    Returns normalized list with title, source, timestamp, url, and summary if present.
    """
    articles: List[Dict] = []

    # Finnhub first
    try:
        results = finnhub_client.company_news(ticker, _from=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'), to=datetime.now().strftime('%Y-%m-%d'))
        for item in results[:limit]:
            articles.append({
                "title": item.get("headline"),
                "publisher": item.get("source"),
                "published": item.get("datetime"),
                "link": item.get("url"),
                "summary": item.get("summary") or "",
                "tickers": [ticker],
            })
    except Exception as e:
        print(f"Finnhub news failed for {ticker}: {e}")

    # Fallback if none
    if not articles:
        articles = get_latest_news(ticker, limit=limit)

    return articles


def get_historical_data(ticker: str, period: str = "3mo") -> Optional[pd.DataFrame]:
    """
    Get historical OHLCV data using yfinance.
    
    Args:
        ticker: Stock symbol
        period: Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    
    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        if df.empty:
            return None
        
        return df
    except Exception as e:
        print(f"Error fetching historical data for {ticker}: {e}")
        return None


def calculate_technical_indicators(df: pd.DataFrame) -> Dict:
    """
    Calculate comprehensive technical indicators using pandas-ta.
    
    Returns a dictionary with:
    - Trend indicators: SMA, EMA, MACD
    - Momentum indicators: RSI, Stochastic
    - Volatility indicators: Bollinger Bands, ATR
    - Volume indicators: OBV, Volume SMA
    """
    if df is None or df.empty:
        return {}
    
    try:
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Trend Indicators
        df['SMA_20'] = ta.sma(df['Close'], length=20)
        df['SMA_50'] = ta.sma(df['Close'], length=50)
        df['SMA_200'] = ta.sma(df['Close'], length=200)
        df['EMA_12'] = ta.ema(df['Close'], length=12)
        df['EMA_26'] = ta.ema(df['Close'], length=26)
        
        # MACD
        macd = ta.macd(df['Close'])
        if macd is not None:
            df = pd.concat([df, macd], axis=1)
        
        # Momentum Indicators
        df['RSI_14'] = ta.rsi(df['Close'], length=14)
        
        # Stochastic
        stoch = ta.stoch(df['High'], df['Low'], df['Close'])
        if stoch is not None:
            df = pd.concat([df, stoch], axis=1)
        
        # Volatility Indicators
        bbands = ta.bbands(df['Close'], length=20, std=2)
        if bbands is not None:
            df = pd.concat([df, bbands], axis=1)
        
        df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        
        # Volume Indicators
        df['Volume_SMA_20'] = ta.sma(df['Volume'], length=20)
        df['OBV'] = ta.obv(df['Close'], df['Volume'])
        
        # Get the latest values
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        # Determine trend based on moving averages
        price = latest['Close']
        trend = "neutral"
        if pd.notna(latest['SMA_20']) and pd.notna(latest['SMA_50']):
            if price > latest['SMA_20'] > latest['SMA_50']:
                trend = "strong_uptrend"
            elif price > latest['SMA_20']:
                trend = "uptrend"
            elif price < latest['SMA_20'] < latest['SMA_50']:
                trend = "strong_downtrend"
            elif price < latest['SMA_20']:
                trend = "downtrend"
        
        # RSI interpretation
        rsi_signal = "neutral"
        if pd.notna(latest['RSI_14']):
            if latest['RSI_14'] > 70:
                rsi_signal = "overbought"
            elif latest['RSI_14'] < 30:
                rsi_signal = "oversold"
        
        # MACD signal
        macd_signal = "neutral"
        if 'MACD_12_26_9' in latest.index and 'MACDs_12_26_9' in latest.index:
            if pd.notna(latest['MACD_12_26_9']) and pd.notna(latest['MACDs_12_26_9']):
                if latest['MACD_12_26_9'] > latest['MACDs_12_26_9']:
                    macd_signal = "bullish"
                else:
                    macd_signal = "bearish"
        
        # Bollinger Bands position
        bb_position = "middle"
        if 'BBL_20_2.0' in latest.index and 'BBU_20_2.0' in latest.index:
            if pd.notna(latest['BBL_20_2.0']) and pd.notna(latest['BBU_20_2.0']):
                bb_range = latest['BBU_20_2.0'] - latest['BBL_20_2.0']
                if bb_range > 0:
                    position = (price - latest['BBL_20_2.0']) / bb_range
                    if position > 0.8:
                        bb_position = "upper_band"
                    elif position < 0.2:
                        bb_position = "lower_band"
        
        return {
            "current_price": float(price),
            "trend": {
                "direction": trend,
                "sma_20": float(latest['SMA_20']) if pd.notna(latest['SMA_20']) else None,
                "sma_50": float(latest['SMA_50']) if pd.notna(latest['SMA_50']) else None,
                "sma_200": float(latest['SMA_200']) if pd.notna(latest['SMA_200']) else None,
                "ema_12": float(latest['EMA_12']) if pd.notna(latest['EMA_12']) else None,
                "ema_26": float(latest['EMA_26']) if pd.notna(latest['EMA_26']) else None,
            },
            "momentum": {
                "rsi_14": float(latest['RSI_14']) if pd.notna(latest['RSI_14']) else None,
                "rsi_signal": rsi_signal,
                "macd": float(latest.get('MACD_12_26_9', 0)) if pd.notna(latest.get('MACD_12_26_9')) else None,
                "macd_signal_line": float(latest.get('MACDs_12_26_9', 0)) if pd.notna(latest.get('MACDs_12_26_9')) else None,
                "macd_histogram": float(latest.get('MACDh_12_26_9', 0)) if pd.notna(latest.get('MACDh_12_26_9')) else None,
                "macd_signal": macd_signal,
            },
            "volatility": {
                "atr_14": float(latest['ATR_14']) if pd.notna(latest['ATR_14']) else None,
                "bb_upper": float(latest.get('BBU_20_2.0', 0)) if pd.notna(latest.get('BBU_20_2.0')) else None,
                "bb_middle": float(latest.get('BBM_20_2.0', 0)) if pd.notna(latest.get('BBM_20_2.0')) else None,
                "bb_lower": float(latest.get('BBL_20_2.0', 0)) if pd.notna(latest.get('BBL_20_2.0')) else None,
                "bb_position": bb_position,
            },
            "volume": {
                "current": int(latest['Volume']),
                "sma_20": float(latest['Volume_SMA_20']) if pd.notna(latest['Volume_SMA_20']) else None,
                "obv": float(latest['OBV']) if pd.notna(latest['OBV']) else None,
                "volume_trend": "above_average" if latest['Volume'] > latest['Volume_SMA_20'] else "below_average",
            },
            "support_resistance": {
                "recent_high": float(df['High'].tail(20).max()),
                "recent_low": float(df['Low'].tail(20).min()),
                "sma_50_level": float(latest['SMA_50']) if pd.notna(latest['SMA_50']) else None,
                "sma_200_level": float(latest['SMA_200']) if pd.notna(latest['SMA_200']) else None,
            }
        }
    
    except Exception as e:
        print(f"Error calculating technical indicators: {e}")
        return {}


def get_market_analysis(ticker: str, period: str = "3mo") -> Dict:
    """
    Get comprehensive market analysis for a ticker.
    Combines real-time quote, company info, historical data, and technical indicators.
    
    Args:
        ticker: Stock symbol (e.g., "AAPL", "TSLA")
        period: Historical data period (default: 3mo)
    
    Returns:
        Dictionary with all market data and analysis
    """
    ticker = ticker.upper().strip()
    
    # Get all data
    quote = get_stock_quote(ticker)
    company = get_company_info(ticker)
    fundamentals = get_fundamentals(ticker)
    news = get_live_news(ticker)
    hist_data = get_historical_data(ticker, period)
    
    if hist_data is None or hist_data.empty:
        return {
            "error": f"Unable to fetch data for ticker: {ticker}",
            "ticker": ticker,
        }
    
    technicals = calculate_technical_indicators(hist_data)
    
    # Calculate additional metrics
    returns_1d = ((hist_data['Close'].iloc[-1] / hist_data['Close'].iloc[-2]) - 1) * 100 if len(hist_data) > 1 else 0
    returns_5d = ((hist_data['Close'].iloc[-1] / hist_data['Close'].iloc[-5]) - 1) * 100 if len(hist_data) > 5 else 0
    returns_1m = ((hist_data['Close'].iloc[-1] / hist_data['Close'].iloc[-20]) - 1) * 100 if len(hist_data) > 20 else 0
    
    volatility = hist_data['Close'].pct_change().std() * (252 ** 0.5) * 100  # Annualized volatility
    
    return {
        "ticker": ticker,
        "timestamp": datetime.now().isoformat(),
        "quote": quote,
        "company": company,
        "technicals": technicals,
        "fundamentals": fundamentals,
        "news": news,
        "performance": {
            "returns_1d": round(returns_1d, 2),
            "returns_5d": round(returns_5d, 2),
            "returns_1m": round(returns_1m, 2),
            "volatility_annual": round(volatility, 2),
        },
        "data_period": period,
        "data_points": len(hist_data),
    }


def format_market_data_for_agent(market_data: Dict) -> str:
    """
    Format market data into a readable string for AI agents.
    """
    if "error" in market_data:
        return f"Error: {market_data['error']}"
    
    ticker = market_data.get("ticker", "UNKNOWN")
    company = market_data.get("company", {})
    quote = market_data.get("quote", {})
    tech = market_data.get("technicals", {})
    perf = market_data.get("performance", {})
    fundamentals = market_data.get("fundamentals", {}) or {}
    news_items = market_data.get("news", []) or []
    
    output = f"=== REAL MARKET DATA FOR {ticker} ===\n\n"
    
    # Company Info
    if company and company.get('name'):
        output += f"Company: {company.get('name', 'N/A')}\n"
        output += f"Industry: {company.get('industry', 'N/A')}\n"
        if company.get('market_cap'):
            output += f"Market Cap: ${company.get('market_cap', 0):.2f}B\n\n"
    
    # Current Quote
    if quote and quote.get('current_price'):
        output += f"Current Price: ${quote.get('current_price', 0):.2f}\n"
        output += f"Change: ${quote.get('change', 0):.2f} ({quote.get('percent_change', 0):.2f}%)\n"
        output += f"Day Range: ${quote.get('low', 0):.2f} - ${quote.get('high', 0):.2f}\n\n"
    
    # Performance
    output += f"Performance:\n"
    output += f"  1-Day: {perf.get('returns_1d', 0):.2f}%\n"
    output += f"  5-Day: {perf.get('returns_5d', 0):.2f}%\n"
    output += f"  1-Month: {perf.get('returns_1m', 0):.2f}%\n"
    output += f"  Annual Volatility: {perf.get('volatility_annual', 0):.2f}%\n\n"
    
    # Technical Indicators
    if tech:
        trend = tech.get("trend", {})
        momentum = tech.get("momentum", {})
        volatility = tech.get("volatility", {})
        volume = tech.get("volume", {})
        sr = tech.get("support_resistance", {})
        
        output += f"Technical Analysis:\n"
        output += f"  Trend: {trend.get('direction', 'N/A')}\n"
        
        sma_20 = trend.get('sma_20')
        sma_50 = trend.get('sma_50')
        sma_200 = trend.get('sma_200')
        
        output += f"  SMA(20): ${sma_20:.2f}\n" if sma_20 else "  SMA(20): N/A\n"
        output += f"  SMA(50): ${sma_50:.2f}\n" if sma_50 else "  SMA(50): N/A\n"
        output += f"  SMA(200): ${sma_200:.2f}\n\n" if sma_200 else "  SMA(200): N/A (need 1y+ data)\n\n"
        
        rsi = momentum.get('rsi_14')
        macd = momentum.get('macd')
        macd_signal = momentum.get('macd_signal_line')
        
        output += f"  RSI(14): {rsi:.2f} ({momentum.get('rsi_signal', 'N/A')})\n" if rsi else "  RSI(14): N/A\n"
        output += f"  MACD: {macd:.4f} ({momentum.get('macd_signal', 'N/A')})\n" if macd else "  MACD: N/A\n"
        output += f"  MACD Signal: {macd_signal:.4f}\n\n" if macd_signal else "  MACD Signal: N/A\n\n"
        
        bb_upper = volatility.get('bb_upper')
        bb_middle = volatility.get('bb_middle')
        bb_lower = volatility.get('bb_lower')
        atr = volatility.get('atr_14')
        
        output += f"  Bollinger Bands:\n"
        output += f"    Upper: ${bb_upper:.2f}\n" if bb_upper else "    Upper: N/A\n"
        output += f"    Middle: ${bb_middle:.2f}\n" if bb_middle else "    Middle: N/A\n"
        output += f"    Lower: ${bb_lower:.2f}\n" if bb_lower else "    Lower: N/A\n"
        output += f"    Position: {volatility.get('bb_position', 'N/A')}\n"
        output += f"  ATR(14): ${atr:.2f}\n\n" if atr else "  ATR(14): N/A\n\n"
        
        output += f"  Volume: {volume.get('current', 0):,} ({volume.get('volume_trend', 'N/A')})\n"
        vol_sma = volume.get('sma_20')
        output += f"  Volume SMA(20): {vol_sma:,.0f}\n\n" if vol_sma else "  Volume SMA(20): N/A\n\n"
        
        output += f"  Support/Resistance:\n"
        output += f"    Recent High (20d): ${sr.get('recent_high', 0):.2f}\n"
        output += f"    Recent Low (20d): ${sr.get('recent_low', 0):.2f}\n"
        
        sma_50_level = sr.get('sma_50_level')
        sma_200_level = sr.get('sma_200_level')
        
        output += f"    SMA(50) Level: ${sma_50_level:.2f}\n" if sma_50_level else "    SMA(50) Level: N/A\n"
        output += f"    SMA(200) Level: ${sma_200_level:.2f}\n" if sma_200_level else "    SMA(200) Level: N/A\n"

    # Fundamental Snapshot
    if fundamentals:
        output += "\nFundamentals:\n"
        if fundamentals.get("market_cap"):
            output += f"  Market Cap: ${fundamentals.get('market_cap'):,}\n"
        output += f"  Fwd PE: {fundamentals.get('forward_pe', 'N/A')}\n"
        output += f"  Trailing PE: {fundamentals.get('trailing_pe', 'N/A')}\n"
        output += f"  PEG: {fundamentals.get('peg_ratio', 'N/A')}\n"
        output += f"  P/B: {fundamentals.get('price_to_book', 'N/A')}\n"
        output += f"  Debt/Equity: {fundamentals.get('debt_to_equity', 'N/A')}\n"
        output += f"  ROE: {fundamentals.get('roe', 'N/A')}\n"
        output += f"  Profit Margin: {fundamentals.get('profit_margin', 'N/A')}\n"
        output += f"  Dividend Yield: {fundamentals.get('dividend_yield', 'N/A')}\n"

    # News Headlines
    if news_items:
        output += "\nRecent News Headlines (with sources):\n"
        for item in news_items[:5]:
            title = item.get('title', 'No title')
            pub = item.get('publisher', 'Unknown')
            link = item.get('link', '')
            output += f"  - {title} ({pub})\n"
            if link:
                output += f"    Source: {link}\n"

    return output


def get_chart_data(ticker: str, period: str = "3mo", limit: int = 60) -> Dict:
    """
    Get OHLCV data with technical indicators formatted for candlestick charts.
    
    Args:
        ticker: Stock symbol
        period: Historical data period (1d, 5d, 1mo, 3mo, 6mo, 1y)
        limit: Maximum number of data points to return (default: 60)
    
    Returns:
        Dictionary with:
        - ticker: Stock symbol
        - data: Array of OHLCV data with technical indicators
        - latest_values: Current values of all indicators
    """
    ticker = ticker.upper().strip()
    
    try:
        # Fetch more data than needed to ensure we have enough trading days
        # Use 6mo to get ~120 trading days, then filter and limit
        fetch_period = "6mo" if period == "3mo" else period
        hist_data = get_historical_data(ticker, fetch_period)
        
        if hist_data is None or hist_data.empty:
            return {
                "error": f"Unable to fetch data for ticker: {ticker}",
                "ticker": ticker,
            }
        
        # Filter out non-trading days (weekends, holidays with zero volume)
        # Keep only rows where volume > 0
        df = hist_data[hist_data['Volume'] > 0].copy()
        
        if df.empty:
            return {
                "error": f"No trading data available for ticker: {ticker}",
                "ticker": ticker,
            }
        
        # Calculate technical indicators on the full dataset first
        # This ensures indicators are calculated correctly before limiting
        
        # Trend Indicators
        df['SMA_20'] = ta.sma(df['Close'], length=20)
        df['SMA_50'] = ta.sma(df['Close'], length=50)
        df['SMA_200'] = ta.sma(df['Close'], length=200)
        df['EMA_12'] = ta.ema(df['Close'], length=12)
        df['EMA_26'] = ta.ema(df['Close'], length=26)
        
        # MACD
        macd = ta.macd(df['Close'])
        if macd is not None:
            df = pd.concat([df, macd], axis=1)
        
        # RSI
        df['RSI_14'] = ta.rsi(df['Close'], length=14)
        
        # Bollinger Bands
        bbands = ta.bbands(df['Close'], length=20, std=2)
        if bbands is not None:
            df = pd.concat([df, bbands], axis=1)

        # Volume SMA
        df['Volume_SMA_20'] = ta.sma(df['Volume'], length=20)

        # ATR for intraday volatility gauge
        df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        
        # Now limit to most recent trading days
        df = df.tail(limit)
        
        # Format data for charting
        chart_data = []
        for idx, row in df.iterrows():
            data_point = {
                "date": idx.strftime("%Y-%m-%d"),
                "timestamp": int(idx.timestamp() * 1000),  # milliseconds for JS
                "open": float(row['Open']) if pd.notna(row['Open']) else None,
                "high": float(row['High']) if pd.notna(row['High']) else None,
                "low": float(row['Low']) if pd.notna(row['Low']) else None,
                "close": float(row['Close']) if pd.notna(row['Close']) else None,
                "volume": int(row['Volume']) if pd.notna(row['Volume']) else None,
                "sma_20": float(row['SMA_20']) if pd.notna(row['SMA_20']) else None,
                "sma_50": float(row['SMA_50']) if pd.notna(row['SMA_50']) else None,
                "sma_200": float(row['SMA_200']) if pd.notna(row['SMA_200']) else None,
                "ema_12": float(row['EMA_12']) if pd.notna(row['EMA_12']) else None,
                "ema_26": float(row['EMA_26']) if pd.notna(row['EMA_26']) else None,
                "rsi": float(row['RSI_14']) if pd.notna(row['RSI_14']) else None,
                "bb_upper": float(row.get('BBU_20_2.0', None)) if pd.notna(row.get('BBU_20_2.0')) else None,
                "bb_middle": float(row.get('BBM_20_2.0', None)) if pd.notna(row.get('BBM_20_2.0')) else None,
                "bb_lower": float(row.get('BBL_20_2.0', None)) if pd.notna(row.get('BBL_20_2.0')) else None,
                "macd": float(row.get('MACD_12_26_9', None)) if pd.notna(row.get('MACD_12_26_9')) else None,
                "macd_signal": float(row.get('MACDs_12_26_9', None)) if pd.notna(row.get('MACDs_12_26_9')) else None,
                "macd_histogram": float(row.get('MACDh_12_26_9', None)) if pd.notna(row.get('MACDh_12_26_9')) else None,
                "volume_sma": float(row['Volume_SMA_20']) if pd.notna(row['Volume_SMA_20']) else None,
                "atr_14": float(row['ATR_14']) if pd.notna(row['ATR_14']) else None,
            }
            chart_data.append(data_point)
        
        # Get latest values
        latest = df.iloc[-1]
        latest_values = {
            "price": float(latest['Close']),
            "sma_20": float(latest['SMA_20']) if pd.notna(latest['SMA_20']) else None,
            "sma_50": float(latest['SMA_50']) if pd.notna(latest['SMA_50']) else None,
            "sma_200": float(latest['SMA_200']) if pd.notna(latest['SMA_200']) else None,
            "rsi": float(latest['RSI_14']) if pd.notna(latest['RSI_14']) else None,
            "macd": float(latest.get('MACD_12_26_9', None)) if pd.notna(latest.get('MACD_12_26_9')) else None,
            "macd_signal": float(latest.get('MACDs_12_26_9', None)) if pd.notna(latest.get('MACDs_12_26_9')) else None,
            "volume": int(latest['Volume']) if pd.notna(latest['Volume']) else None,
            "volume_sma": float(latest['Volume_SMA_20']) if pd.notna(latest['Volume_SMA_20']) else None,
            "atr_14": float(latest['ATR_14']) if pd.notna(latest['ATR_14']) else None,
            "bb_position": None,
        }
        
        # Calculate Bollinger Band position
        if pd.notna(latest.get('BBL_20_2.0')) and pd.notna(latest.get('BBU_20_2.0')):
            bb_range = latest['BBU_20_2.0'] - latest['BBL_20_2.0']
            if bb_range > 0:
                latest_values['bb_position'] = float((latest['Close'] - latest['BBL_20_2.0']) / bb_range)
        
        return {
            "ticker": ticker,
            "data": chart_data,
            "latest_values": latest_values,
            "data_points": len(chart_data),
            "period": period,
        }
        
    except Exception as e:
        print(f"Error generating chart data for {ticker}: {e}")
        return {
            "error": f"Failed to generate chart data: {str(e)}",
            "ticker": ticker,
        }
