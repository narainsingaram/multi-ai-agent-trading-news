#!/usr/bin/env python3
"""
Test script for market_data module
"""

from market_data import get_market_analysis, format_market_data_for_agent

# Test with AAPL
print("Testing market data fetching for AAPL...")
print("=" * 80)

try:
    data = get_market_analysis("AAPL", period="6mo")
    
    if "error" in data:
        print(f"ERROR: {data['error']}")
    else:
        print(f"✓ Successfully fetched data for {data['ticker']}")
        print(f"✓ Current Price: ${data['technicals']['current_price']:.2f}")
        print(f"✓ RSI(14): {data['technicals']['momentum']['rsi_14']:.2f}")
        print(f"✓ Trend: {data['technicals']['trend']['direction']}")
        
        sma_20 = data['technicals']['trend']['sma_20']
        sma_50 = data['technicals']['trend']['sma_50']
        sma_200 = data['technicals']['trend']['sma_200']
        
        print(f"✓ SMA(20): ${sma_20:.2f}" if sma_20 else "✓ SMA(20): N/A")
        print(f"✓ SMA(50): ${sma_50:.2f}" if sma_50 else "✓ SMA(50): N/A")
        print(f"✓ SMA(200): ${sma_200:.2f}" if sma_200 else "✓ SMA(200): N/A (need 1y+ data)")
        print(f"✓ MACD: {data['technicals']['momentum']['macd']:.4f}")
        print(f"✓ BB Upper: ${data['technicals']['volatility']['bb_upper']:.2f}")
        print(f"✓ BB Lower: ${data['technicals']['volatility']['bb_lower']:.2f}")
        print(f"✓ Volume: {data['technicals']['volume']['current']:,}")
        print()
        print("Formatted output for agents:")
        print("=" * 80)
        print(format_market_data_for_agent(data))
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
