
from market_data import get_chart_data
import json

print("Testing get_chart_data for AAPL...")
data = get_chart_data("AAPL", period="6mo")

if "error" in data:
    print(f"Error: {data['error']}")
else:
    levels = data.get("institutional_levels", {})
    print(f"Institutional Levels detected: {list(levels.keys())}")
    print(f"Order Blocks: {len(levels.get('order_blocks', []))}")
    print(f"FVGs: {len(levels.get('fair_value_gaps', []))}")
    print(f"Liquidity Pools: {len(levels.get('liquidity_pools', []))}")
    
    if levels.get('order_blocks'):
        print("Sample OB:", levels['order_blocks'][0])
