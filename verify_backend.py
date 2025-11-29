import sys
import os
sys.path.append(os.getcwd())

from market_data import get_chart_data, get_stock_quote

def test_data():
    print("Testing Quote...")
    quote = get_stock_quote("AAPL")
    print(f"Quote: {quote}")

    print("\nTesting Chart Data...")
    chart = get_chart_data("AAPL", period="1mo", limit=5)
    if "error" in chart:
        print(f"Chart Error: {chart['error']}")
    else:
        print(f"Chart Data Points: {len(chart.get('data', []))}")
        if chart.get('data'):
            print(f"Sample Point: {chart['data'][0]}")
        print(f"Latest Values: {chart.get('latest_values')}")

if __name__ == "__main__":
    test_data()
