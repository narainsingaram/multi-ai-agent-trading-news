import yfinance as yf

ticker = "AAPL"
tk = yf.Ticker(ticker)

print(f"Testing {ticker}...")
print(f"Has history method: {hasattr(tk, 'history')}")

try:
    hist = tk.history(period="1mo")
    print(f"History shape: {hist.shape}")
    print(f"History columns: {list(hist.columns)}")
    print(f"First few rows:\n{hist.head()}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
