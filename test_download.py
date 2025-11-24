import yfinance as yf

ticker = "TSLA"
print(f"Testing {ticker} with yf.download...")

hist = yf.download(
    ticker, 
    period="1mo", 
    progress=False,
    show_errors=False,
    timeout=10
)

print(f"Shape: {hist.shape}")
print(f"Columns: {list(hist.columns)}")
if not hist.empty:
    print(f"✓ SUCCESS! Got {len(hist)} days of data")
    print(f"Latest close: ${hist['Close'].iloc[-1]:.2f}")
else:
    print("✗ FAILED - empty dataframe")
