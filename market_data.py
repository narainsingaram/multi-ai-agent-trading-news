"""
Market Data Service
Fetches real-time stock data, historical prices, and calculates technical indicators.
Uses Finnhub API for real-time quotes and yfinance for historical data.
"""

import os
import re
import yfinance as yf
import finnhub
import pandas as pd

# Fix for pandas_ta compatibility with newer numpy versions
import numpy as np
if not hasattr(np, 'NaN'):
    np.NaN = np.nan

import pandas_ta as ta
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional, List, Tuple
from dotenv import load_dotenv

load_dotenv()

# Initialize Finnhub client
finnhub_client = finnhub.Client(api_key=os.getenv("FINNHUB_API_KEY"))

# Cache to avoid excessive API calls
_cache = {}
_cache_timeout = 300  # 5 minutes

# Lightweight universe for market scans (large caps + liquid growth names)
SCAN_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "AVGO", "NFLX",
    "QCOM", "CRM", "ORCL", "ADBE", "INTC", "CSCO", "IBM", "SHOP", "UBER", "PYPL",
    "SQ", "PLTR", "SNOW", "MU", "TXN", "AMAT", "COST", "WMT", "TGT", "KO", "PEP",
    "NKE", "DIS", "JPM", "BAC", "GS", "MS", "UNH", "LLY", "PFE", "CVX", "XOM",
    "BA", "GE", "CAT", "HON", "DE", "RTX", "PANW", "NET", "ZS", "OKTA"
]


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


def get_historical_range(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """
    Get historical OHLCV data for a date range using yfinance.
    Dates are YYYY-MM-DD strings.
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start, end=end)
        if df.empty:
            return None
        return df
    except Exception as e:
        print(f"Error fetching range data for {ticker}: {e}")
        return None


def get_economic_calendar() -> List[Dict]:
    """
    Lightweight static macro calendar for demo purposes.
    In production, replace with live data source.
    """
    today = datetime.today()
    base_year = today.year
    return [
        {"event": "CPI", "date": f"{base_year}-12-12", "risk": "High volatility around release"},
        {"event": "FOMC Rate Decision", "date": f"{base_year}-12-18", "risk": "Policy shift risk"},
        {"event": "Non-Farm Payrolls", "date": f"{base_year}-12-06", "risk": "Jobs print can swing risk assets"},
        {"event": "Fed Chair Speech", "date": f"{base_year}-11-30", "risk": "Forward-guidance sensitivity"},
    ]


def get_earnings_calendar(ticker: str) -> List[Dict]:
    """
    Try to fetch upcoming earnings from yfinance calendar.
    Falls back to empty list if unavailable.
    """
    events: List[Dict] = []
    try:
        stock = yf.Ticker(ticker)
        # Helper to normalize different calendar formats
        def _normalize_event_date(value):
            if isinstance(value, (pd.Timestamp, datetime)):
                return value.strftime("%Y-%m-%d")
            if hasattr(value, "to_pydatetime"):
                return value.to_pydatetime().strftime("%Y-%m-%d")
            if isinstance(value, (list, tuple, pd.Series)):
                for item in value:
                    normalized = _normalize_event_date(item)
                    if normalized:
                        return normalized
                return None
            if value is None:
                return None
            try:
                if pd.isna(value):
                    return None
            except Exception:
                pass
            return str(value)

        def _add_event(event_name: str, raw_date):
            date_str = _normalize_event_date(raw_date)
            if date_str:
                events.append({
                    "event": event_name,
                    "date": date_str,
                    "risk": "Earnings catalyst"
                })

        # Preferred: use get_earnings_dates if available (yfinance >=0.2.40)
        get_dates = getattr(stock, "get_earnings_dates", None)
        if callable(get_dates):
            dates_df = get_dates(limit=4)
            if isinstance(dates_df, pd.DataFrame) and not dates_df.empty:
                for idx, _row in dates_df.iterrows():
                    _add_event("Earnings Date", idx)

        # Fallback: older calendar attribute (DataFrame or dict)
        if not events:
            cal = getattr(stock, "calendar", None)
            if isinstance(cal, pd.DataFrame):
                for idx, row in cal.iterrows():
                    value = row.iloc[0] if len(row) else None
                    _add_event(idx, value)
            elif isinstance(cal, dict):
                for key, value in cal.items():
                    _add_event(str(key), value)
    except Exception as e:
        print(f"Error fetching earnings calendar for {ticker}: {e}")

    # Deduplicate and keep short list
    seen = set()
    deduped = []
    for ev in events:
        key = (ev["event"], ev["date"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ev)
    return deduped[:3]


def _parse_scan_filters(query: str) -> Tuple[List[Callable[[Dict[str, Any]], Tuple[bool, str]]], List[str], bool, Tuple[str, str]]:
    """
    Parse a natural language scan query into simple predicate functions.
    
    Returns:
        filters: list of callables returning (bool passed, reason)
        labels: human-readable labels for filters
        requires_sentiment: whether we need to fetch sentiment data
        sort_hint: (field, direction) to prioritize results
    """
    text = (query or "").lower()
    filters: List[Callable[[Dict[str, Any]], Tuple[bool, str]]] = []
    labels: List[str] = []
    requires_sentiment = False
    sort_hint: Tuple[str, str] = ("score", "desc")

    for op, threshold_str in re.findall(r"rsi\s*(<=|<|>=|>)\s*(\d+)", text):
        threshold = float(threshold_str)

        def rsi_filter(data: Dict[str, Any], th=threshold, operator=op):
            rsi_val = data.get("rsi")
            if rsi_val is None:
                return False, "RSI unavailable"
            if operator in ("<", "<="):
                passed = rsi_val <= th if operator == "<=" else rsi_val < th
            else:
                passed = rsi_val >= th if operator == ">=" else rsi_val > th
            return passed, f"RSI {operator} {th}"

        filters.append(rsi_filter)
        labels.append(f"RSI {op} {threshold}")
        sort_hint = ("rsi", "asc" if op in ("<", "<=") else "desc")

    if "oversold" in text and not any("RSI" in lbl for lbl in labels):
        def oversold_filter(data: Dict[str, Any]):
            rsi_val = data.get("rsi")
            if rsi_val is None:
                return False, "RSI unavailable"
            return rsi_val <= 35, "RSI <= 35 (oversold)"
        filters.append(oversold_filter)
        labels.append("RSI <= 35 (oversold)")
        sort_hint = ("rsi", "asc")

    if "overbought" in text and not any("RSI" in lbl for lbl in labels):
        def overbought_filter(data: Dict[str, Any]):
            rsi_val = data.get("rsi")
            if rsi_val is None:
                return False, "RSI unavailable"
            return rsi_val >= 65, "RSI >= 65 (overbought)"
        filters.append(overbought_filter)
        labels.append("RSI >= 65 (overbought)")
        sort_hint = ("rsi", "desc")

    if "positive sentiment" in text or "bullish sentiment" in text or "positive social" in text:
        requires_sentiment = True

        def bullish_sentiment(data: Dict[str, Any]):
            score = data.get("sentiment_score")
            if score is None:
                return False, "Sentiment unavailable"
            return score > 0.05, f"Sentiment {score:.2f} bullish"

        filters.append(bullish_sentiment)
        labels.append("Sentiment > 0")
        sort_hint = ("sentiment_score", "desc")

    if "negative sentiment" in text or "bearish sentiment" in text:
        requires_sentiment = True

        def bearish_sentiment(data: Dict[str, Any]):
            score = data.get("sentiment_score")
            if score is None:
                return False, "Sentiment unavailable"
            return score < -0.05, f"Sentiment {score:.2f} bearish"

        filters.append(bearish_sentiment)
        labels.append("Sentiment < 0")
        sort_hint = ("sentiment_score", "asc")

    return filters, labels, requires_sentiment, sort_hint


def _build_scan_snapshot(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Lightweight snapshot used for market scans.
    Avoids heavy quote APIs to keep scans fast.
    """
    hist_data = get_historical_data(ticker, "6mo")
    if hist_data is None or hist_data.empty:
        return None

    technicals = calculate_technical_indicators(hist_data)
    if not technicals:
        return None

    latest_close = float(hist_data["Close"].iloc[-1])
    returns_1d = ((hist_data['Close'].iloc[-1] / hist_data['Close'].iloc[-2]) - 1) * 100 if len(hist_data) > 1 else 0.0
    returns_5d = ((hist_data['Close'].iloc[-1] / hist_data['Close'].iloc[-5]) - 1) * 100 if len(hist_data) > 5 else 0.0

    fundamentals = get_fundamentals(ticker) or {}
    company_name = None
    try:
        info = yf.Ticker(ticker).info or {}
        company_name = info.get("shortName") or info.get("longName")
    except Exception:
        company_name = None

    momentum = technicals.get("momentum", {})
    volume = technicals.get("volume", {})
    trend = technicals.get("trend", {})
    volatility = technicals.get("volatility", {})

    return {
        "ticker": ticker,
        "name": company_name,
        "price": round(latest_close, 2),
        "rsi": momentum.get("rsi_14"),
        "macd": momentum.get("macd"),
        "trend": trend.get("direction"),
        "volume_trend": volume.get("volume_trend"),
        "returns_1d": round(returns_1d, 2),
        "returns_5d": round(returns_5d, 2),
        "bb_position": volatility.get("bb_position"),
        "atr": volatility.get("atr_14"),
        "market_cap": fundamentals.get("market_cap"),
    }


def scan_market_for_criteria(query: str, universe: Optional[List[str]] = None, limit: int = 40) -> Dict[str, Any]:
    """
    Scan a curated ticker universe and return matches for the user's criteria.
    
    Args:
        query: Natural language criteria (e.g., "RSI < 30 and positive sentiment")
        universe: Optional list of tickers to scan
        limit: Max tickers to evaluate
    """
    tickers = list(dict.fromkeys(universe or SCAN_UNIVERSE))
    limit = max(1, min(limit, len(tickers), 75))
    tickers = tickers[:limit]

    filters, labels, requires_sentiment, sort_hint = _parse_scan_filters(query)
    results = []

    for ticker in tickers:
        try:
            snapshot = _build_scan_snapshot(ticker)
            if not snapshot:
                continue

            if requires_sentiment:
                try:
                    from sentiment_data import get_ticker_sentiment
                    sentiment = get_ticker_sentiment(ticker)
                    snapshot["sentiment_score"] = sentiment.get("sentiment_score")
                    snapshot["sentiment_label"] = sentiment.get("sentiment_label")
                except Exception:
                    snapshot["sentiment_score"] = None
                    snapshot["sentiment_label"] = None

            reasons = []
            passed = True
            for predicate in filters:
                ok, note = predicate(snapshot)
                if ok and note:
                    reasons.append(note)
                if not ok:
                    passed = False
                    break

            if not filters:
                reasons.append("Baseline technical scan (RSI + velocity)")

            if not passed:
                continue

            # Simple ranking score to surface the most interesting names first
            score = 0.0
            rsi_val = snapshot.get("rsi")
            if rsi_val is not None:
                score += max(0, (70 - min(rsi_val, 100)) / 100)
            if snapshot.get("returns_1d") is not None:
                score += max(0, -snapshot["returns_1d"]) / 200
            if snapshot.get("sentiment_score") is not None:
                score += max(0, snapshot["sentiment_score"])

            snapshot["score"] = round(score, 3)
            snapshot["match_reasons"] = reasons
            results.append(snapshot)
        except Exception as e:
            print(f"Scan failed for {ticker}: {e}")
            continue

    sort_field, sort_dir = sort_hint

    def sort_key(item: Dict[str, Any]):
        val = item.get(sort_field)
        if isinstance(val, (int, float)):
            return (0, val if sort_dir == "asc" else -val)
        return (1, 0)

    results.sort(key=sort_key)

    return {
        "summary": {
            "query": query,
            "universe_size": len(tickers),
            "filters_applied": labels or ["No explicit filters"],
            "requires_sentiment": requires_sentiment,
            "matched": len(results),
            "sort": {"field": sort_field, "direction": sort_dir},
        },
        "results": results,
    }


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
        
        # VWAP (Volume Weighted Average Price) - institutional benchmark
        df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # MFI (Money Flow Index) - volume-weighted RSI
        df['MFI_14'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14)
        
        # ADX (Average Directional Index) - trend strength
        adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
        if adx is not None:
            df = pd.concat([df, adx], axis=1)
        
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
        
        # Stochastic interpretation
        stoch_signal = "neutral"
        stoch_k = latest.get('STOCHk_14_3_3')
        stoch_d = latest.get('STOCHd_14_3_3')
        if pd.notna(stoch_k) and pd.notna(stoch_d):
            if stoch_k > 80:
                stoch_signal = "overbought"
            elif stoch_k < 20:
                stoch_signal = "oversold"
            # Check for crossovers
            prev_k = prev.get('STOCHk_14_3_3')
            prev_d = prev.get('STOCHd_14_3_3')
            if pd.notna(prev_k) and pd.notna(prev_d):
                if stoch_k > stoch_d and prev_k <= prev_d:
                    stoch_signal = "bullish_crossover"
                elif stoch_k < stoch_d and prev_k >= prev_d:
                    stoch_signal = "bearish_crossover"
        
        # OBV trend detection
        obv_trend = "neutral"
        if pd.notna(latest['OBV']) and len(df) >= 10:
            obv_sma = df['OBV'].tail(10).mean()
            if latest['OBV'] > obv_sma * 1.02:
                obv_trend = "rising"
            elif latest['OBV'] < obv_sma * 0.98:
                obv_trend = "falling"
        
        # MFI interpretation
        mfi_signal = "neutral"
        mfi = latest.get('MFI_14')
        if pd.notna(mfi):
            if mfi > 80:
                mfi_signal = "overbought"
            elif mfi < 20:
                mfi_signal = "oversold"
        
        # ADX trend strength
        adx_strength = "weak"
        adx_val = latest.get('ADX_14')
        if pd.notna(adx_val):
            if adx_val > 25:
                adx_strength = "strong"
            elif adx_val > 20:
                adx_strength = "moderate"
        
        # VWAP position
        vwap_position = "neutral"
        vwap = latest.get('VWAP')
        if pd.notna(vwap):
            if price > vwap * 1.005:
                vwap_position = "above"
            elif price < vwap * 0.995:
                vwap_position = "below"
        
        return {
            "current_price": float(price),
            "trend": {
                "direction": trend,
                "sma_20": float(latest['SMA_20']) if pd.notna(latest['SMA_20']) else None,
                "sma_50": float(latest['SMA_50']) if pd.notna(latest['SMA_50']) else None,
                "sma_200": float(latest['SMA_200']) if pd.notna(latest['SMA_200']) else None,
                "ema_12": float(latest['EMA_12']) if pd.notna(latest['EMA_12']) else None,
                "ema_26": float(latest['EMA_26']) if pd.notna(latest['EMA_26']) else None,
                "vwap": float(vwap) if pd.notna(vwap) else None,
                "vwap_position": vwap_position,
            },
            "momentum": {
                "rsi_14": float(latest['RSI_14']) if pd.notna(latest['RSI_14']) else None,
                "rsi_signal": rsi_signal,
                "macd": float(latest.get('MACD_12_26_9', 0)) if pd.notna(latest.get('MACD_12_26_9')) else None,
                "macd_signal_line": float(latest.get('MACDs_12_26_9', 0)) if pd.notna(latest.get('MACDs_12_26_9')) else None,
                "macd_histogram": float(latest.get('MACDh_12_26_9', 0)) if pd.notna(latest.get('MACDh_12_26_9')) else None,
                "macd_signal": macd_signal,
                "stochastic_k": float(stoch_k) if pd.notna(stoch_k) else None,
                "stochastic_d": float(stoch_d) if pd.notna(stoch_d) else None,
                "stoch_signal": stoch_signal,
                "mfi_14": float(mfi) if pd.notna(mfi) else None,
                "mfi_signal": mfi_signal,
                "adx_14": float(adx_val) if pd.notna(adx_val) else None,
                "adx_strength": adx_strength,
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
                "obv_trend": obv_trend,
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


def format_market_data_for_agent(market_data: Dict, institutional_levels: Optional[Dict] = None) -> str:
    """
    Format market data into a readable string for AI agents.
    Includes institutional levels (Order Blocks, Fair Value Gaps, Liquidity Pools).
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
        output += f"  SMA(200): ${sma_200:.2f}\n" if sma_200 else "  SMA(200): N/A (need 1y+ data)\n"
        
        # VWAP (institutional benchmark)
        vwap = trend.get('vwap')
        vwap_position = trend.get('vwap_position')
        if vwap:
            output += f"  VWAP: ${vwap:.2f} (Price {vwap_position} VWAP)\n"
        else:
            output += "  VWAP: N/A\n"
        
        output += "\n"
        
        rsi = momentum.get('rsi_14')
        macd = momentum.get('macd')
        macd_signal = momentum.get('macd_signal_line')
        stoch_k = momentum.get('stochastic_k')
        stoch_d = momentum.get('stochastic_d')
        stoch_signal = momentum.get('stoch_signal')
        mfi = momentum.get('mfi_14')
        mfi_signal = momentum.get('mfi_signal')
        adx = momentum.get('adx_14')
        adx_strength = momentum.get('adx_strength')
        
        output += f"  RSI(14): {rsi:.2f} ({momentum.get('rsi_signal', 'N/A')})\n" if rsi else "  RSI(14): N/A\n"
        output += f"  MACD: {macd:.4f} ({momentum.get('macd_signal', 'N/A')})\n" if macd else "  MACD: N/A\n"
        output += f"  MACD Signal: {macd_signal:.4f}\n" if macd_signal else "  MACD Signal: N/A\n"
        
        # Stochastic Oscillator
        if stoch_k and stoch_d:
            output += f"  Stochastic %K: {stoch_k:.2f}, %D: {stoch_d:.2f} ({stoch_signal})\n"
        else:
            output += "  Stochastic: N/A\n"
        
        # MFI (Money Flow Index)
        if mfi:
            output += f"  MFI(14): {mfi:.2f} ({mfi_signal})\n"
        else:
            output += "  MFI(14): N/A\n"
        
        # ADX (Trend Strength)
        if adx:
            output += f"  ADX(14): {adx:.2f} (Trend Strength: {adx_strength})\n"
        else:
            output += "  ADX(14): N/A\n"
        
        output += "\n"
        
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
        output += f"  Volume SMA(20): {vol_sma:,.0f}\n" if vol_sma else "  Volume SMA(20): N/A\n"
        
        # OBV (On-Balance Volume)
        obv = volume.get('obv')
        obv_trend = volume.get('obv_trend')
        if obv:
            output += f"  OBV: {obv:,.0f} (Trend: {obv_trend})\n"
        else:
            output += "  OBV: N/A\n"
        
        output += "\n"
        
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

    # Institutional Levels (Order Blocks, Fair Value Gaps, Liquidity Pools)
    if institutional_levels:
        output += "\nInstitutional Levels (Smart Money Zones):\n"
        
        # Order Blocks
        obs = institutional_levels.get('order_blocks', [])
        if obs:
            output += "  Order Blocks:\n"
            for ob in obs[:5]:  # Top 5 most recent
                output += f"    {ob['type'].title()}: ${ob['bottom']:.2f} - ${ob['top']:.2f} (Date: {ob['date']})\n"
        
        # Fair Value Gaps
        fvgs = institutional_levels.get('fair_value_gaps', [])
        if fvgs:
            output += "  Fair Value Gaps (FVG):\n"
            for fvg in fvgs[:5]:  # Top 5
                output += f"    {fvg['type'].title()}: ${fvg['bottom']:.2f} - ${fvg['top']:.2f}\n"
        
        # Liquidity Pools
        liq_pools = institutional_levels.get('liquidity_pools', [])
        if liq_pools:
            output += "  Liquidity Pools:\n"
            for pool in liq_pools[:5]:
                output += f"    {pool['type'].title()}: ${pool['level']:.2f} (Strength: {pool.get('strength', 'N/A')})\n"

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



def detect_institutional_levels(df: pd.DataFrame) -> Dict:
    """
    Detect institutional levels: Order Blocks, Fair Value Gaps (FVG), and Liquidity Pools.
    """
    levels = {
        "order_blocks": [],
        "fair_value_gaps": [],
        "liquidity_pools": []
    }
    
    if df is None or len(df) < 5:
        return levels

    # Helper to check for strong displacement (body > 1.5x average body)
    df['body_size'] = abs(df['Close'] - df['Open'])
    avg_body = df['body_size'].rolling(20).mean()
    
    for i in range(2, len(df) - 2):
        # --- Fair Value Gaps (FVG) ---
        # Bullish FVG: Low[i] > High[i-2]
        if df['Low'].iloc[i] > df['High'].iloc[i-2]:
            levels['fair_value_gaps'].append({
                "type": "bullish",
                "top": float(df['Low'].iloc[i]),
                "bottom": float(df['High'].iloc[i-2]),
                "start_date": df.index[i-2].strftime("%Y-%m-%d"),
                "end_date": df.index[i].strftime("%Y-%m-%d")
            })
            
        # Bearish FVG: High[i] < Low[i-2]
        if df['High'].iloc[i] < df['Low'].iloc[i-2]:
            levels['fair_value_gaps'].append({
                "type": "bearish",
                "top": float(df['Low'].iloc[i-2]),
                "bottom": float(df['High'].iloc[i]),
                "start_date": df.index[i-2].strftime("%Y-%m-%d"),
                "end_date": df.index[i].strftime("%Y-%m-%d")
            })

        # --- Order Blocks (Simplified) ---
        # Bullish OB: Last red candle before a strong green move
        current_body = df['body_size'].iloc[i]
        is_strong_move = current_body > (avg_body.iloc[i] * 1.5)
        is_green = df['Close'].iloc[i] > df['Open'].iloc[i]
        prev_is_red = df['Close'].iloc[i-1] < df['Open'].iloc[i-1]
        
        if is_strong_move and is_green and prev_is_red:
            levels['order_blocks'].append({
                "type": "bullish",
                "top": float(df['High'].iloc[i-1]),
                "bottom": float(df['Low'].iloc[i-1]),
                "date": df.index[i-1].strftime("%Y-%m-%d")
            })
            
        # Bearish OB: Last green candle before a strong red move
        is_red = df['Close'].iloc[i] < df['Open'].iloc[i]
        prev_is_green = df['Close'].iloc[i-1] > df['Open'].iloc[i-1]
        
        if is_strong_move and is_red and prev_is_green:
            levels['order_blocks'].append({
                "type": "bearish",
                "top": float(df['High'].iloc[i-1]),
                "bottom": float(df['Low'].iloc[i-1]),
                "date": df.index[i-1].strftime("%Y-%m-%d")
            })

        # --- Liquidity Pools (Swing Highs/Lows) ---
        # Swing High
        if df['High'].iloc[i] > df['High'].iloc[i-1] and df['High'].iloc[i] > df['High'].iloc[i+1]:
             levels['liquidity_pools'].append({
                "type": "buy_side", # Stops above highs
                "level": float(df['High'].iloc[i]),
                "date": df.index[i].strftime("%Y-%m-%d")
            })
            
        # Swing Low
        if df['Low'].iloc[i] < df['Low'].iloc[i-1] and df['Low'].iloc[i] < df['Low'].iloc[i+1]:
             levels['liquidity_pools'].append({
                "type": "sell_side", # Stops below lows
                "level": float(df['Low'].iloc[i]),
                "date": df.index[i].strftime("%Y-%m-%d")
            })

    # Filter to keep only recent/relevant levels (optional optimization)
    # For now, return all detected in the window
    return levels


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
        - institutional_levels: Detected OBs, FVGs, Liquidity
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
        
        # Detect Institutional Levels (on the full dataset to catch older levels)
        institutional_levels = detect_institutional_levels(df)

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
            "institutional_levels": institutional_levels,
            "data_points": len(chart_data),
            "period": period,
        }
        
    except Exception as e:
        print(f"Error generating chart data for {ticker}: {e}")
        return {
            "error": f"Failed to generate chart data: {str(e)}",
            "ticker": ticker,
        }


def run_simple_backtest(ticker: str, start: str, end: str, strategy: Optional[Dict] = None) -> Dict:
    """
    Lightweight backtest placeholder.
    - Buys at first close in range, holds until end.
    - Computes ROI, CAGR, win rate (positive days), max drawdown, Sharpe (daily, 0% rf).
    - Returns equity curve and drawdown series for charting.
    """
    df = get_historical_range(ticker, start, end)
    if df is None or df.empty:
        return {"error": f"No historical data for {ticker} in range {start} to {end}"}

    df = df.copy()
    df["pct_return"] = df["Close"].pct_change()
    df.dropna(inplace=True)
    if df.empty:
        return {"error": "Not enough data to backtest"}

    start_price = df["Close"].iloc[0]
    end_price = df["Close"].iloc[-1]
    roi = (end_price / start_price) - 1

    # CAGR approximation
    days = (df.index[-1] - df.index[0]).days or 1
    years = days / 365.25
    cagr = (end_price / start_price) ** (1 / years) - 1 if years > 0 else roi

    # Win rate: percent of positive daily returns
    win_rate = (df["pct_return"] > 0).mean()

    # Max drawdown + equity curve
    cumulative = (1 + df["pct_return"]).cumprod()
    rolling_max = cumulative.cummax()
    drawdowns = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()

    equity_curve = []
    for idx, (ts, eq, dd) in enumerate(zip(df.index, cumulative, drawdowns)):
        equity_curve.append({
            "date": ts.strftime("%Y-%m-%d"),
            "equity": float(eq),
            "drawdown": float(dd)
        })

    # Sharpe (daily)
    avg_daily = df["pct_return"].mean()
    std_daily = df["pct_return"].std()
    sharpe = (avg_daily / std_daily * (252 ** 0.5)) if std_daily and std_daily != 0 else None

    return {
        "ticker": ticker.upper(),
        "start": start,
        "end": end,
        "roi": roi,
        "cagr": cagr,
        "win_rate": win_rate,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "bars": len(df),
        "equity_curve": equity_curve,
        "strategy_echo": strategy or {},
    }


def scan_chart_patterns(ticker: str, period: str = "6mo") -> List[Dict]:
    """
    Lightweight heuristic pattern scanner. Returns list of {pattern, confidence, detail}.
    Not a production-grade scanner; meant to inform the LLM and UI.
    """
    try:
        df = get_historical_data(ticker, period)
        if df is None or df.empty:
            return []
        closes = df["Close"].tolist()
        highs = df["High"].tolist()
        lows = df["Low"].tolist()
        patterns: List[Dict] = []

        def add(pattern, conf, detail):
            patterns.append({"pattern": pattern, "confidence": conf, "detail": detail})

        n = len(closes)
        if n < 30:
            return patterns

        # Bull flag: strong advance then tight range
        rally = (closes[-10] - closes[-20]) / closes[-20] if closes[-20] else 0
        flag_range = (max(closes[-5:]) - min(closes[-5:])) / closes[-5]
        if rally > 0.07 and flag_range < 0.03:
            add("Bull flag", round(min(0.9, rally + 0.5 - flag_range), 2), "Advance over last 20 bars with tight 5-bar consolidation.")

        # Double top: two recent highs within 2% separated by 5-20 bars
        recent_highs = sorted([(h, i) for i, h in enumerate(highs[-40:])], key=lambda x: x[0], reverse=True)
        if len(recent_highs) >= 2:
            h1, i1 = recent_highs[0]
            h2, i2 = next(((h, i) for h, i in recent_highs[1:] if abs(h - h1) / h1 < 0.02 and abs(i - i1) >= 5), (None, None))
            if h2:
                add("Double top", 0.62, f"Two swing highs within 2% at bars {n-40+i1} and {n-40+i2}.")

        # Cup & handle: rounded base then small pullback
        window = closes[-60:]
        if len(window) >= 40:
            left = window[0]
            right = window[-1]
            mid = min(window)
            if abs(left - right) / left < 0.03 and (left - mid) / left > 0.08:
                handle_drop = (max(window[-5:]) - min(window[-5:])) / max(window[-5:])
                if handle_drop < 0.04:
                    add("Cup & handle", 0.58, "Rounded base with shallow handle in last 5 bars.")

        # Rising wedge: higher highs/lows with contracting range
        hh_slope = (highs[-1] - highs[-15]) / highs[-15] if highs[-15] else 0
        ll_slope = (lows[-1] - lows[-15]) / lows[-15] if lows[-15] else 0
        range_now = (highs[-1] - lows[-1]) / highs[-1]
        range_then = (highs[-15] - lows[-15]) / highs[-15]
        if hh_slope > 0 and ll_slope > 0 and hh_slope < ll_slope and range_now < range_then:
            add("Rising wedge", 0.55, "Higher highs/lows with contracting range over last 15 bars.")

        # Fair value gap (ICT): identify recent 3-candle gap
        for i in range(n - 10, n - 2):
            if lows[i] > highs[i - 2]:
                gap = (lows[i] - highs[i - 2]) / highs[i - 2]
                if gap > 0.01:
                    add("Bullish fair value gap", round(min(0.9, 0.5 + gap), 2), f"Gap of {gap*100:.1f}% between bars {i-2} and {i}.")
                    break
            if highs[i] < lows[i - 2]:
                gap = (lows[i - 2] - highs[i]) / lows[i - 2]
                if gap > 0.01:
                    add("Bearish fair value gap", round(min(0.9, 0.5 + gap), 2), f"Gap of {gap*100:.1f}% between bars {i-2} and {i}.")
                    break

        return patterns[:6]
    except Exception as e:
        print(f"Pattern scan failed for {ticker}: {e}")
        return []
