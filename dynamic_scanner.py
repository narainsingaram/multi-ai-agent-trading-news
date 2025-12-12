"""
Dynamic Market Scanner for AI Trading System.
Parses natural language queries into structured scans and executes them.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Literal
from groq import Groq
import os
import json
from dotenv import load_dotenv
from market_data import scan_market_for_criteria, get_market_analysis

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


class ScanCriteria(BaseModel):
    """Structured scan criteria with validation"""
    
    # Sector/Industry filters
    sectors: List[str] = Field(
        default_factory=list,
        description="Sectors to scan: Technology, Healthcare, Finance, Energy, Consumer, Industrial, Materials, Utilities, Real Estate, Communication"
    )
    
    # Market cap filters
    market_cap_min: Optional[float] = Field(
        None,
        description="Minimum market cap in dollars (e.g., 10000000000 for $10B)"
    )
    market_cap_max: Optional[float] = Field(
        None,
        description="Maximum market cap in dollars"
    )
    
    # Technical criteria
    rsi_min: Optional[float] = Field(None, ge=0, le=100, description="Minimum RSI value")
    rsi_max: Optional[float] = Field(None, ge=0, le=100, description="Maximum RSI value")
    macd_signal: Optional[Literal["bullish", "bearish", "any"]] = Field(
        None,
        description="MACD signal: bullish (line > signal), bearish (line < signal)"
    )
    price_vs_sma50: Optional[Literal["above", "below", "any"]] = Field(
        None,
        description="Price position relative to 50-day SMA"
    )
    price_vs_sma200: Optional[Literal["above", "below", "any"]] = Field(
        None,
        description="Price position relative to 200-day SMA"
    )
    volume_vs_avg: Optional[Literal["above_average", "below_average", "any"]] = Field(
        None,
        description="Volume compared to average"
    )
    patterns: List[str] = Field(
        default_factory=list,
        description="Chart patterns to look for: bull_flag, bear_flag, head_shoulders, double_top, double_bottom, cup_handle, triangle"
    )
    price_action: Optional[Literal["breakout", "pullback", "reversal", "consolidation"]] = Field(
        None,
        description="Type of price action"
    )
    
    # Fundamental criteria
    pe_ratio_min: Optional[float] = Field(None, description="Minimum P/E ratio")
    pe_ratio_max: Optional[float] = Field(None, description="Maximum P/E ratio")
    dividend_yield_min: Optional[float] = Field(None, ge=0, description="Minimum dividend yield %")
    revenue_growth_min: Optional[float] = Field(None, description="Minimum revenue growth %")
    profit_margin_min: Optional[float] = Field(None, description="Minimum profit margin %")
    debt_to_equity_max: Optional[float] = Field(None, description="Maximum debt-to-equity ratio")
    
    # Other filters
    max_results: int = Field(default=10, ge=1, le=50, description="Maximum number of results")
    sort_by: Literal["score", "volume", "price_change", "market_cap"] = Field(
        default="score",
        description="How to sort results"
    )
    
    @validator('rsi_max')
    def rsi_max_greater_than_min(cls, v, values):
        """Ensure RSI max > min if both specified"""
        if v is not None and 'rsi_min' in values and values['rsi_min'] is not None:
            if v <= values['rsi_min']:
                raise ValueError("rsi_max must be greater than rsi_min")
        return v


class ScanResult(BaseModel):
    """Single stock from scan results"""
    ticker: str = Field(description="Stock ticker symbol")
    company_name: str = Field(description="Company name")
    current_price: float = Field(description="Current stock price")
    score: float = Field(
        ge=0,
        le=100,
        description="Match quality score (0-100)"
    )
    matching_criteria: List[str] = Field(
        description="Which criteria this stock matched"
    )
    key_metrics: Dict[str, Any] = Field(
        description="Relevant metrics for this stock"
    )
    summary: str = Field(
        description="One-line summary of why this stock matches"
    )


class DynamicScanner:
    """
    Natural language market scanner.
    Parses user queries into structured criteria and executes scans.
    """
    
    SCANNER_PROMPT = """You are a market scanner that converts natural language into structured scan criteria.

Your job: Parse the user's scan request into specific, measurable criteria.

TECHNICAL CRITERIA MAPPINGS:
- "oversold" → rsi_max: 30
- "overbought" → rsi_min: 70
- "bullish MACD" → macd_signal: "bullish"
- "bearish MACD" → macd_signal: "bearish"
- "above moving average" / "above SMA50" → price_vs_sma50: "above"
- "below moving average" / "below SMA200" → price_vs_sma200: "below"
- "high volume" → volume_vs_avg: "above_average"
- "low volume" → volume_vs_avg: "below_average"
- "breakout" → price_action: "breakout"
- "pullback" → price_action: "pullback"

FUNDAMENTAL CRITERIA MAPPINGS:
- "undervalued" / "cheap" → pe_ratio_max: 15
- "expensive" / "overvalued" → pe_ratio_min: 30
- "high dividend" / "dividend stocks" → dividend_yield_min: 3.0
- "growth stock" / "high growth" → revenue_growth_min: 15.0
- "profitable" → profit_margin_min: 10.0
- "large cap" → market_cap_min: 10000000000 (10 billion)
- "mid cap" → market_cap_min: 2000000000, market_cap_max: 10000000000
- "small cap" → market_cap_max: 2000000000

SECTORS:
Technology, Healthcare, Finance, Energy, Consumer, Industrial, Materials, Utilities, Real Estate, Communication

CHART PATTERNS:
bull_flag, bear_flag, head_shoulders, inverse_head_shoulders, double_top, double_bottom, cup_handle, ascending_triangle, descending_triangle

EXAMPLES:

Query: "Find oversold tech stocks"
{
  "sectors": ["Technology"],
  "rsi_max": 30,
  "max_results": 10,
  "sort_by": "score"
}

Query: "Show me breakouts with high volume"
{
  "price_action": "breakout",
  "volume_vs_avg": "above_average",
  "max_results": 10,
  "sort_by": "volume"
}

Query: "Dividend stocks with P/E under 15"
{
  "dividend_yield_min": 2.0,
  "pe_ratio_max": 15,
  "max_results": 10,
  "sort_by": "score"
}

Query: "Undervalued large cap healthcare stocks"
{
  "sectors": ["Healthcare"],
  "market_cap_min": 10000000000,
  "pe_ratio_max": 15,
  "max_results": 10,
  "sort_by": "market_cap"
}

Query: "Tech stocks with bullish MACD above 50-day moving average"
{
  "sectors": ["Technology"],
  "macd_signal": "bullish",
  "price_vs_sma50": "above",
  "max_results": 10,
  "sort_by": "score"
}

Query: "Find stocks breaking out with RSI between 50 and 70"
{
  "price_action": "breakout",
  "rsi_min": 50,
  "rsi_max": 70,
  "max_results": 10,
  "sort_by": "volume"
}

CRITICAL: Respond ONLY with valid JSON matching the ScanCriteria schema. No additional text.
"""

    def __init__(self):
        """Initialize scanner with LLM client"""
        self.client = client
    
    def parse_scan_query(self, user_query: str) -> ScanCriteria:
        """
        Parse natural language query into structured scan criteria.
        
        Args:
            user_query: Natural language scan request
            
        Returns:
            ScanCriteria with structured filters
        """
        messages = [
            {"role": "system", "content": self.SCANNER_PROMPT},
            {"role": "user", "content": f"Query: {user_query}"}
        ]
        
        try:
            # Call LLM with JSON mode
            completion = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                temperature=0.1,
                max_tokens=1024,
                response_format={"type": "json_object"}
            )
            
            response_content = completion.choices[0].message.content
            response_json = json.loads(response_content)
            
            # Validate with Pydantic
            criteria = ScanCriteria(**response_json)
            
            print(f"📊 Scanner parsed criteria:")
            if criteria.sectors:
                print(f"   Sectors: {', '.join(criteria.sectors)}")
            if criteria.rsi_min or criteria.rsi_max:
                print(f"   RSI: {criteria.rsi_min or 0} - {criteria.rsi_max or 100}")
            if criteria.macd_signal:
                print(f"   MACD: {criteria.macd_signal}")
            if criteria.dividend_yield_min:
                print(f"   Dividend yield: >{criteria.dividend_yield_min}%")
            
            return criteria
            
        except Exception as e:
            print(f"❌ Scanner parse error: {e}")
            # Return default criteria
            return ScanCriteria()
    
    def execute_scan(self, criteria: ScanCriteria) -> List[ScanResult]:
        """
        Execute scan with given criteria.
        
        Args:
            criteria: Structured scan criteria
            
        Returns:
            List of matching stocks with scores
        """
        print(f"🔍 Executing market scan...")
        
        # Convert ScanCriteria to format expected by scan_market_for_criteria
        scan_params = {}
        
        if criteria.sectors:
            scan_params['sectors'] = criteria.sectors
        if criteria.rsi_max is not None:
            scan_params['rsi_max'] = criteria.rsi_max
        if criteria.rsi_min is not None:
            scan_params['rsi_min'] = criteria.rsi_min
        if criteria.macd_signal:
            scan_params['macd_signal'] = criteria.macd_signal
        if criteria.volume_vs_avg:
            scan_params['volume'] = criteria.volume_vs_avg
        if criteria.dividend_yield_min:
            scan_params['dividend_min'] = criteria.dividend_yield_min
        if criteria.pe_ratio_max:
            scan_params['pe_max'] = criteria.pe_ratio_max
        
        # Execute scan using existing function
        try:
            raw_results = scan_market_for_criteria(scan_params)
            
            # Convert to ScanResult objects
            results = []
            for stock in raw_results[:criteria.max_results]:
                # Calculate match score
                score = self._calculate_match_score(stock, criteria)
                
                # Identify matching criteria
                matching = self._identify_matching_criteria(stock, criteria)
                
                result = ScanResult(
                    ticker=stock.get('ticker', 'UNKNOWN'),
                    company_name=stock.get('name', 'Unknown Company'),
                    current_price=stock.get('price', 0.0),
                    score=score,
                    matching_criteria=matching,
                    key_metrics=self._extract_key_metrics(stock, criteria),
                    summary=self._generate_summary(stock, matching)
                )
                results.append(result)
            
            # Sort results
            results = self._sort_results(results, criteria.sort_by)
            
            print(f"✅ Found {len(results)} matching stocks")
            
            return results
            
        except Exception as e:
            print(f"❌ Scan execution error: {e}")
            return []
    
    def _calculate_match_score(self, stock: Dict, criteria: ScanCriteria) -> float:
        """Calculate how well stock matches criteria (0-100)"""
        score = 50.0  # Base score
        
        # Add points for each matching criterion
        if criteria.rsi_max and stock.get('rsi', 100) <= criteria.rsi_max:
            score += 10
        if criteria.rsi_min and stock.get('rsi', 0) >= criteria.rsi_min:
            score += 10
        if criteria.macd_signal == "bullish" and stock.get('macd_bullish'):
            score += 15
        if criteria.volume_vs_avg == "above_average" and stock.get('volume_high'):
            score += 10
        if criteria.dividend_yield_min and stock.get('dividend_yield', 0) >= criteria.dividend_yield_min:
            score += 10
        
        return min(100.0, score)
    
    def _identify_matching_criteria(self, stock: Dict, criteria: ScanCriteria) -> List[str]:
        """Identify which criteria the stock matches"""
        matching = []
        
        if criteria.sectors and stock.get('sector') in criteria.sectors:
            matching.append(f"Sector: {stock.get('sector')}")
        if criteria.rsi_max and stock.get('rsi', 100) <= criteria.rsi_max:
            matching.append(f"RSI {stock.get('rsi'):.1f} (oversold)")
        if criteria.macd_signal == "bullish":
            matching.append("Bullish MACD")
        if criteria.volume_vs_avg == "above_average":
            matching.append("High volume")
        if criteria.dividend_yield_min:
            matching.append(f"Dividend {stock.get('dividend_yield', 0):.1f}%")
        
        return matching
    
    def _extract_key_metrics(self, stock: Dict, criteria: ScanCriteria) -> Dict[str, Any]:
        """Extract relevant metrics based on criteria"""
        metrics = {
            'price': stock.get('price', 0),
            'volume': stock.get('volume', 0)
        }
        
        if criteria.rsi_min or criteria.rsi_max:
            metrics['rsi'] = stock.get('rsi')
        if criteria.macd_signal:
            metrics['macd'] = stock.get('macd')
        if criteria.dividend_yield_min:
            metrics['dividend_yield'] = stock.get('dividend_yield')
        if criteria.pe_ratio_max:
            metrics['pe_ratio'] = stock.get('pe_ratio')
        
        return metrics
    
    def _generate_summary(self, stock: Dict, matching_criteria: List[str]) -> str:
        """Generate one-line summary"""
        if not matching_criteria:
            return f"{stock.get('name', 'Stock')} matches scan criteria"
        
        criteria_str = ", ".join(matching_criteria[:3])
        return f"{stock.get('name', 'Stock')}: {criteria_str}"
    
    def _sort_results(self, results: List[ScanResult], sort_by: str) -> List[ScanResult]:
        """Sort results by specified criterion"""
        if sort_by == "score":
            return sorted(results, key=lambda x: x.score, reverse=True)
        elif sort_by == "volume":
            return sorted(results, key=lambda x: x.key_metrics.get('volume', 0), reverse=True)
        elif sort_by == "price_change":
            return sorted(results, key=lambda x: x.key_metrics.get('price_change', 0), reverse=True)
        else:
            return results


# Global instance
_scanner = None

def get_scanner() -> DynamicScanner:
    """Get global dynamic scanner instance"""
    global _scanner
    if _scanner is None:
        _scanner = DynamicScanner()
    return _scanner
