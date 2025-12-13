"""
Options data fetching and analysis module.
Fetches real options chain data from yfinance and calculates metrics.
"""

import yfinance as yf
import pandas as pd
import requests
from requests.exceptions import RequestException
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional


YAHOO_OPTIONS_URL = "https://query2.finance.yahoo.com/v7/finance/options/{ticker}"


def _safe_series_sum(df: pd.DataFrame, column: str) -> float:
    """Sum a dataframe column safely even if it's missing or contains NaNs."""
    if df is None or df.empty:
        return 0
    series = df.get(column)
    if series is None:
        return 0
    return float(pd.to_numeric(series.fillna(0), errors="coerce").sum())


def _fetch_options_via_http(ticker: str, expiration_date: Optional[str] = None):
    """Hit Yahoo's options endpoint directly to avoid yfinance curl/DNS issues."""
    params = {}
    if expiration_date:
        try:
            params["date"] = int(datetime.strptime(expiration_date, "%Y-%m-%d").timestamp())
        except ValueError:
            raise ValueError(f"Invalid expiration format: {expiration_date}. Expected YYYY-MM-DD.")

    try:
        resp = requests.get(
            YAHOO_OPTIONS_URL.format(ticker=ticker),
            params=params,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        resp.raise_for_status()
        payload = resp.json()
    except RequestException as e:
        raise RuntimeError(f"HTTP fetch failed: {e}") from e
    except ValueError as e:
        raise RuntimeError(f"Invalid response while parsing options JSON: {e}") from e

    chain_results = payload.get("optionChain", {}).get("result") or []
    if not chain_results:
        raise RuntimeError("No option chain returned from Yahoo Finance.")

    chain = chain_results[0]
    expiration_stamps = chain.get("expirationDates") or []
    available_expirations = [
        datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
        for ts in expiration_stamps
    ]

    options_list = chain.get("options") or []
    if not options_list:
        raise RuntimeError("No options data available for the requested expiration.")

    option_set = options_list[0]
    exp_timestamp = option_set.get("expirationDate")
    exp_date = (
        datetime.utcfromtimestamp(exp_timestamp).strftime("%Y-%m-%d")
        if exp_timestamp
        else (available_expirations[0] if available_expirations else None)
    )

    quote = (option_set.get("quote") or chain.get("quote")) or {}
    current_price = (
        quote.get("regularMarketPrice")
        or quote.get("postMarketPrice")
        or quote.get("preMarketPrice")
        or quote.get("lastPrice")
    )

    calls_df = pd.DataFrame(option_set.get("calls") or [])
    puts_df = pd.DataFrame(option_set.get("puts") or [])

    return calls_df, puts_df, current_price, exp_date, available_expirations


def _build_options_payload(
    ticker: str,
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    current_price: float,
    exp_date: str,
    expirations: List[str],
) -> Dict[str, Any]:
    """Convert raw calls/puts dataframes into the API response shape."""
    if current_price is None:
        return {"error": f"Could not fetch current price for {ticker}"}

    # Ensure required columns exist for aggregation
    calls = calls.copy()
    puts = puts.copy()
    for df in (calls, puts):
        if "volume" not in df:
            df["volume"] = 0
        if "openInterest" not in df:
            df["openInterest"] = 0

    calls_data = process_options_data(calls, current_price, 'call')
    puts_data = process_options_data(puts, current_price, 'put')

    total_call_volume = _safe_series_sum(calls, "volume")
    total_put_volume = _safe_series_sum(puts, "volume")
    total_call_oi = _safe_series_sum(calls, "openInterest")
    total_put_oi = _safe_series_sum(puts, "openInterest")

    pc_ratio_volume = total_put_volume / total_call_volume if total_call_volume > 0 else 0
    pc_ratio_oi = total_put_oi / total_call_oi if total_call_oi > 0 else 0

    unusual_calls = detect_unusual_activity(calls_data)
    unusual_puts = detect_unusual_activity(puts_data)
    heatmap_data = create_heatmap_data(calls_data, puts_data, current_price)

    return {
        "ticker": ticker,
        "current_price": float(current_price),
        "expiration_date": exp_date,
        "available_expirations": list(expirations or []),
        "calls": calls_data,
        "puts": puts_data,
        "put_call_ratio": {
            "volume": float(pc_ratio_volume),
            "open_interest": float(pc_ratio_oi)
        },
        "total_volume": {
            "calls": int(total_call_volume),
            "puts": int(total_put_volume)
        },
        "total_open_interest": {
            "calls": int(total_call_oi),
            "puts": int(total_put_oi)
        },
        "unusual_activity": {
            "calls": unusual_calls,
            "puts": unusual_puts
        },
        "heatmap": heatmap_data
    }


def get_options_chain(ticker: str, expiration_date: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetch real options chain data for a ticker.
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL')
        expiration_date: Specific expiration date (YYYY-MM-DD), or None for nearest
        
    Returns:
        Dictionary with options chain data, put/call ratios, and unusual activity
    """
    errors: List[str] = []

    # First try direct HTTP call to avoid yfinance DNS/consent failures (e.g., guce.yahoo.com).
    try:
        calls_df, puts_df, current_price, exp_date, expirations = _fetch_options_via_http(
            ticker, expiration_date
        )
        payload = _build_options_payload(
            ticker,
            calls_df,
            puts_df,
            current_price,
            exp_date,
            expirations,
        )
        if "error" not in payload:
            return payload
        errors.append(payload["error"])
    except Exception as e:
        errors.append(f"Yahoo HTTP fallback failed: {e}")

    try:
        stock = yf.Ticker(ticker)
        expirations = list(stock.options or [])
        if not expirations:
            errors.append(f"No options data available for {ticker}")
        else:
            exp_date = expiration_date if expiration_date in expirations else expirations[0]

            opt_chain = stock.option_chain(exp_date)
            calls = opt_chain.calls
            puts = opt_chain.puts

            current_price = stock.info.get('currentPrice') or stock.info.get('regularMarketPrice')
            if not current_price:
                hist = stock.history(period='1d')
                current_price = hist['Close'].iloc[-1] if not hist.empty else None

            payload = _build_options_payload(
                ticker,
                calls,
                puts,
                current_price,
                exp_date,
                expirations,
            )
            if "error" not in payload:
                return payload
            errors.append(payload["error"])
    except Exception as e:
        errors.append(f"yfinance fallback failed: {e}")

    detail = "; ".join(errors) if errors else "Unknown error"
    return {"error": f"Failed to fetch options data: {detail}"}


def process_options_data(df: pd.DataFrame, current_price: float, option_type: str) -> List[Dict]:
    """Process raw options dataframe into structured data."""
    if df.empty:
        return []
    
    processed = []
    for _, row in df.iterrows():
        strike = row['strike']
        
        # Determine if ITM, ATM, or OTM
        if option_type == 'call':
            moneyness = 'ITM' if strike < current_price else 'ATM' if abs(strike - current_price) < 1 else 'OTM'
        else:  # put
            moneyness = 'ITM' if strike > current_price else 'ATM' if abs(strike - current_price) < 1 else 'OTM'
        
        # Calculate volume/OI ratio (indicator of unusual activity)
        volume = row.get('volume', 0) or 0
        open_interest = row.get('openInterest', 0) or 0
        vol_oi_ratio = volume / open_interest if open_interest > 0 else 0
        
        processed.append({
            "strike": float(strike),
            "last_price": float(row.get('lastPrice', 0) or 0),
            "bid": float(row.get('bid', 0) or 0),
            "ask": float(row.get('ask', 0) or 0),
            "volume": int(volume),
            "open_interest": int(open_interest),
            "implied_volatility": float(row.get('impliedVolatility', 0) or 0),
            "delta": float(row.get('delta', 0) or 0) if 'delta' in row else None,
            "gamma": float(row.get('gamma', 0) or 0) if 'gamma' in row else None,
            "theta": float(row.get('theta', 0) or 0) if 'theta' in row else None,
            "vega": float(row.get('vega', 0) or 0) if 'vega' in row else None,
            "moneyness": moneyness,
            "vol_oi_ratio": float(vol_oi_ratio),
            "in_the_money": row.get('inTheMoney', False)
        })
    
    return processed


def detect_unusual_activity(options_data: List[Dict], threshold: float = 2.0) -> List[Dict]:
    """
    Detect unusual options activity based on volume/OI ratio.
    
    Args:
        options_data: List of processed options
        threshold: Vol/OI ratio threshold for unusual activity
        
    Returns:
        List of options with unusual activity
    """
    unusual = []
    
    for opt in options_data:
        if opt['vol_oi_ratio'] >= threshold and opt['volume'] > 100:
            unusual.append({
                "strike": opt['strike'],
                "volume": opt['volume'],
                "open_interest": opt['open_interest'],
                "vol_oi_ratio": opt['vol_oi_ratio'],
                "last_price": opt['last_price'],
                "moneyness": opt['moneyness']
            })
    
    # Sort by volume descending
    unusual.sort(key=lambda x: x['volume'], reverse=True)
    
    return unusual[:10]  # Top 10 unusual activities


def create_heatmap_data(calls: List[Dict], puts: List[Dict], current_price: float) -> List[Dict]:
    """
    Create heatmap data for strike prices.
    Combines call and put data at each strike.
    """
    # Group by strike
    strikes = {}
    
    for call in calls:
        strike = call['strike']
        if strike not in strikes:
            strikes[strike] = {
                "strike": strike,
                "call_volume": 0,
                "put_volume": 0,
                "call_oi": 0,
                "put_oi": 0,
                "call_iv": 0,
                "put_iv": 0
            }
        strikes[strike]['call_volume'] = call['volume']
        strikes[strike]['call_oi'] = call['open_interest']
        strikes[strike]['call_iv'] = call['implied_volatility']
    
    for put in puts:
        strike = put['strike']
        if strike not in strikes:
            strikes[strike] = {
                "strike": strike,
                "call_volume": 0,
                "put_volume": 0,
                "call_oi": 0,
                "put_oi": 0,
                "call_iv": 0,
                "put_iv": 0
            }
        strikes[strike]['put_volume'] = put['volume']
        strikes[strike]['put_oi'] = put['open_interest']
        strikes[strike]['put_iv'] = put['implied_volatility']
    
    # Convert to list and add metadata
    heatmap = []
    for strike, data in sorted(strikes.items()):
        total_volume = data['call_volume'] + data['put_volume']
        total_oi = data['call_oi'] + data['put_oi']
        
        # Determine if this is near ATM
        distance_from_price = abs(strike - current_price)
        is_near_money = distance_from_price / current_price < 0.05  # Within 5%
        
        heatmap.append({
            **data,
            "total_volume": total_volume,
            "total_oi": total_oi,
            "distance_from_price": float(distance_from_price),
            "is_near_money": is_near_money,
            "pc_volume_ratio": data['put_volume'] / data['call_volume'] if data['call_volume'] > 0 else 0
        })
    
    return heatmap


def get_max_pain(calls: List[Dict], puts: List[Dict]) -> Optional[float]:
    """
    Calculate max pain strike price.
    Max pain is where option holders lose the most money (option sellers profit most).
    """
    if not calls or not puts:
        return None
    
    # Get all unique strikes
    strikes = set([c['strike'] for c in calls] + [p['strike'] for p in puts])
    
    max_pain_strike = None
    min_total_value = float('inf')
    
    for strike in strikes:
        # Calculate total value of ITM options at this strike
        call_value = sum(
            max(0, strike - c['strike']) * c['open_interest']
            for c in calls if c['strike'] < strike
        )
        put_value = sum(
            max(0, p['strike'] - strike) * p['open_interest']
            for p in puts if p['strike'] > strike
        )
        
        total_value = call_value + put_value
        
        if total_value < min_total_value:
            min_total_value = total_value
            max_pain_strike = strike
    
    return float(max_pain_strike) if max_pain_strike else None
