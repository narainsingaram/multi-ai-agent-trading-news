"""
Ticker Extraction Service
Uses LLM to extract stock ticker symbols from natural language queries.
"""

from groq import Groq
import os
import json
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

TICKER_EXTRACTION_PROMPT = """You are a stock ticker extraction agent specialized in US stock markets.

Your job is to identify ALL publicly traded company stock ticker symbols mentioned in the user's text.

Rules:
1. Extract ticker symbols for companies mentioned by name (e.g., "Apple" -> "AAPL", "Microsoft" -> "MSFT", "Hershey's" -> "HSY")
2. Include tickers already in the text (e.g., "$TSLA" or "NVDA")
3. Only include valid US stock exchange tickers (NYSE, NASDAQ)
4. If a company has multiple share classes, default to the most common (e.g., "GOOGL" not "GOOG")
5. If no tickers are found, return an empty array
6. Do NOT include ETFs, indices, or crypto symbols unless explicitly mentioned
7. Be comprehensive - extract ALL companies mentioned, even if multiple

Common examples:
- "Apple" or "Apple Inc" -> "AAPL"
- "Tesla" -> "TSLA"
- "Nvidia" -> "NVDA"
- "Hershey's" or "Hershey" -> "HSY"
- "Coca-Cola" -> "KO"
- "McDonald's" -> "MCD"
- "Walmart" -> "WMT"
- "Target" -> "TGT"
- "Costco" -> "COST"
- "Palantir" -> "PLTR"
- "Snowflake" -> "SNOW"

Respond ONLY with valid JSON in this format:
{
  "tickers": ["AAPL", "MSFT", "TSLA"]
}
"""

def extract_tickers(text: str) -> list[str]:
    """
    Extract stock ticker symbols from natural language text using LLM.
    
    Args:
        text: User query or text containing company names
        
    Returns:
        List of ticker symbols (e.g., ["AAPL", "MSFT"])
    """
    if not text or not text.strip():
        return []
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": TICKER_EXTRACTION_PROMPT},
                {"role": "user", "content": text}
            ],
            temperature=0.1,
            max_tokens=256,
            response_format={"type": "json_object"},
        )
        
        response_content = completion.choices[0].message.content
        result = json.loads(response_content)
        
        tickers = result.get("tickers", [])
        
        # Validate and clean tickers
        valid_tickers = []
        for ticker in tickers:
            ticker = ticker.strip().upper()
            # Basic validation: 1-5 uppercase letters
            if ticker and ticker.isalpha() and 1 <= len(ticker) <= 5:
                valid_tickers.append(ticker)
        
        return valid_tickers
        
    except Exception as e:
        print(f"Error extracting tickers: {e}")
        return []


if __name__ == "__main__":
    # Test cases
    test_queries = [
        "What's happening with Apple and Microsoft today?",
        "Analyze TSLA and NVDA",
        "Should I buy Palantir stock?",
        "Compare Amazon, Google, and Meta",
        "$AAPL earnings report analysis",
        "Snowflake vs Databricks",
    ]
    
    for query in test_queries:
        tickers = extract_tickers(query)
        print(f"Query: {query}")
        print(f"Tickers: {tickers}\n")
