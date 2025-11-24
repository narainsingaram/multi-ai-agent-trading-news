import yfinance as yf

# Test news fetching for different tickers
tickers = ["AAPL", "TSLA", "MSFT", "NVDA"]

for ticker_symbol in tickers:
    print(f"\n{'='*60}")
    print(f"Testing: {ticker_symbol}")
    print('='*60)
    
    tk = yf.Ticker(ticker_symbol)
    
    # Method 1: Direct .news property
    try:
        if hasattr(tk, 'news'):
            news = tk.news
            if news:
                print(f"✓ .news property: Found {len(news)} items")
                if len(news) > 0:
                    print(f"  First item: {news[0].get('title', 'No title')}")
            else:
                print("✗ .news property: Empty or None")
        else:
            print("✗ .news property: Not available")
    except Exception as e:
        print(f"✗ .news property failed: {e}")
    
    # Method 2: get_news()
    try:
        if hasattr(tk, 'get_news'):
            news = tk.get_news()
            if news:
                print(f"✓ .get_news(): Found {len(news)} items")
            else:
                print("✗ .get_news(): Empty or None")
        else:
            print("✗ .get_news(): Not available")
    except Exception as e:
        print(f"✗ .get_news() failed: {e}")

print("\n" + "="*60)
print("Test complete!")
