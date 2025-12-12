"""
Intelligent Query Router for AI Trading System.
Classifies user queries and routes them to appropriate workflows.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Literal
from enum import Enum
from groq import Groq
import os
import json
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


class QueryIntent(str, Enum):
    """Types of user queries the system can handle"""
    SINGLE_TICKER = "single_ticker"
    MARKET_SCAN = "market_scan"
    MARKET_OVERVIEW = "market_overview"
    CUSTOM_SCAN = "custom_scan"
    EDUCATIONAL = "educational"
    CONVERSATIONAL = "conversational"


class ExtractedEntity(BaseModel):
    """Entities extracted from user query"""
    tickers: List[str] = Field(
        default_factory=list,
        description="Stock tickers mentioned (e.g., TSLA, AAPL)"
    )
    sectors: List[str] = Field(
        default_factory=list,
        description="Sectors/industries mentioned (e.g., Technology, Healthcare)"
    )
    technical_criteria: Dict[str, Any] = Field(
        default_factory=dict,
        description="Technical indicators with thresholds (e.g., {'rsi': '<30', 'macd': 'bullish'})"
    )
    fundamental_criteria: Dict[str, Any] = Field(
        default_factory=dict,
        description="Fundamental metrics (e.g., {'pe_ratio': '<15', 'dividend_yield': '>3'})"
    )
    timeframe: Optional[str] = Field(
        None,
        description="Time horizon: intraday, swing, position, long_term"
    )
    sentiment: Optional[str] = Field(
        None,
        description="Bullish, bearish, or neutral bias detected in query"
    )
    price_action: Optional[str] = Field(
        None,
        description="Price action mentioned: breakout, pullback, reversal, etc."
    )


class RouterOutput(BaseModel):
    """Router classification output with validation"""
    intent: QueryIntent = Field(description="Classified intent type")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in classification (0-1)"
    )
    entities: ExtractedEntity = Field(description="Extracted entities from query")
    reasoning: str = Field(
        min_length=10,
        description="Why this intent was chosen"
    )
    suggested_workflow: str = Field(
        description="Which workflow to execute: full_pipeline, dynamic_scanner, quick_overview, educational_response"
    )
    reformulated_query: Optional[str] = Field(
        None,
        description="Clarified version of query for better processing"
    )
    
    @validator('confidence')
    def confidence_reasonable(cls, v, values):
        """Warn if confidence is too low"""
        if v < 0.5:
            print(f"⚠️  Low confidence ({v:.2f}) in intent classification")
        return v


class QueryRouter:
    """
    Intelligent router that classifies user queries and extracts entities.
    Uses LLM for intent detection and entity extraction.
    """
    
    ROUTER_PROMPT = """You are an intelligent query router for a trading AI system.

Your job: Analyze the user's query and determine:
1. INTENT: What type of query is this?
2. ENTITIES: Extract tickers, sectors, criteria, timeframes
3. WORKFLOW: Which system workflow should handle this?

INTENT TYPES:
- single_ticker: User wants analysis of a specific stock
  Examples: "Should I buy TSLA?", "NVDA earnings analysis", "Is AAPL a good buy?"
  
- market_scan: User wants to find stocks matching criteria
  Examples: "Find oversold tech stocks", "Show me breakouts", "Stocks with high volume"
  
- market_overview: User wants general market sentiment
  Examples: "How's the market today?", "What's happening in tech sector?", "Market summary"
  
- custom_scan: Complex multi-criteria scan
  Examples: "Find dividend stocks with RSI < 30 and P/E < 15", "Tech stocks breaking out with bullish MACD"
  
- educational: User asking to learn something
  Examples: "What is RSI?", "How to read MACD?", "Explain head and shoulders pattern"
  
- conversational: Follow-up or clarification
  Examples: "Tell me more", "What about the risks?", "Can you explain that?"

ENTITY EXTRACTION RULES:
- Tickers: Stock symbols in ALL CAPS (TSLA, AAPL, NVDA, etc.)
- Sectors: Technology, Healthcare, Finance, Energy, Consumer, Industrial, Materials, Utilities, Real Estate, Communication
- Technical criteria: 
  * RSI: "oversold" = <30, "overbought" = >70
  * MACD: "bullish" = line > signal, "bearish" = line < signal
  * Moving averages: "above SMA50", "below SMA200"
  * Volume: "high volume" = above average
  * Patterns: "breakout", "pullback", "reversal", "flag", "head and shoulders"
- Fundamental criteria:
  * P/E ratio: "undervalued" = <15, "expensive" = >30
  * Dividend: "high dividend" = >3%
  * Growth: "growth stock" = revenue growth >15%
  * Market cap: "large cap" = >$10B, "small cap" = <$2B
- Timeframe: intraday (hours), swing (days/weeks), position (months), long_term (years)
- Sentiment: bullish (positive), bearish (negative), neutral

EXAMPLES:

Query: "Should I buy Tesla?"
{
  "intent": "single_ticker",
  "confidence": 0.95,
  "entities": {
    "tickers": ["TSLA"],
    "sentiment": "neutral"
  },
  "reasoning": "User asking for buy/sell recommendation on specific stock (Tesla/TSLA)",
  "suggested_workflow": "full_pipeline",
  "reformulated_query": "Analyze TSLA for potential buy opportunity"
}

Query: "Find me oversold tech stocks with good fundamentals"
{
  "intent": "market_scan",
  "confidence": 0.90,
  "entities": {
    "sectors": ["Technology"],
    "technical_criteria": {"rsi": "<30"},
    "fundamental_criteria": {"quality": "high"}
  },
  "reasoning": "User wants to discover stocks matching multiple criteria (oversold + tech + fundamentals)",
  "suggested_workflow": "dynamic_scanner",
  "reformulated_query": "Scan for Technology sector stocks with RSI < 30 and strong fundamentals"
}

Query: "What's happening in the market today?"
{
  "intent": "market_overview",
  "confidence": 0.92,
  "entities": {
    "timeframe": "intraday"
  },
  "reasoning": "User wants general market sentiment for today, not specific stock analysis",
  "suggested_workflow": "quick_overview",
  "reformulated_query": "Provide market overview for today's trading session"
}

Query: "Find dividend stocks with P/E under 15 and RSI below 40"
{
  "intent": "custom_scan",
  "confidence": 0.88,
  "entities": {
    "technical_criteria": {"rsi": "<40"},
    "fundamental_criteria": {"pe_ratio": "<15", "dividend_yield": ">2"}
  },
  "reasoning": "Complex multi-criteria scan combining technical and fundamental filters",
  "suggested_workflow": "dynamic_scanner",
  "reformulated_query": "Scan for stocks with dividend yield >2%, P/E <15, and RSI <40"
}

Query: "What is the MACD indicator?"
{
  "intent": "educational",
  "confidence": 0.98,
  "entities": {},
  "reasoning": "User asking to learn about a trading concept (MACD indicator)",
  "suggested_workflow": "educational_response",
  "reformulated_query": "Explain what the MACD (Moving Average Convergence Divergence) indicator is"
}

Query: "Tell me more about that"
{
  "intent": "conversational",
  "confidence": 0.85,
  "entities": {},
  "reasoning": "Follow-up question requiring context from previous interaction",
  "suggested_workflow": "contextual_response",
  "reformulated_query": "Provide more details about the previous topic"
}

CRITICAL: Respond ONLY with valid JSON matching the RouterOutput schema. No additional text.
"""

    def __init__(self):
        """Initialize router with LLM client"""
        self.client = client
        self.context_history = []  # Track conversation context
    
    def classify_query(
        self,
        user_query: str,
        context: Optional[Dict] = None
    ) -> RouterOutput:
        """
        Classify user query and extract entities using LLM.
        
        Args:
            user_query: User's natural language query
            context: Optional context from previous queries
            
        Returns:
            RouterOutput with intent, entities, and routing info
        """
        # Build messages for LLM
        messages = [
            {"role": "system", "content": self.ROUTER_PROMPT},
            {"role": "user", "content": f"Query: {user_query}"}
        ]
        
        # Add context if available
        if context:
            context_str = f"\nPrevious context: {json.dumps(context, indent=2)}"
            messages[1]["content"] += context_str
        
        try:
            # Call LLM with JSON mode
            completion = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                temperature=0.1,  # Low temperature for consistent classification
                max_tokens=1024,
                response_format={"type": "json_object"}
            )
            
            response_content = completion.choices[0].message.content
            response_json = json.loads(response_content)
            
            # Validate with Pydantic
            router_output = RouterOutput(**response_json)
            
            # Store in context history
            self.context_history.append({
                "query": user_query,
                "intent": router_output.intent,
                "entities": router_output.entities.dict()
            })
            
            # Keep only last 5 queries in context
            if len(self.context_history) > 5:
                self.context_history.pop(0)
            
            print(f"🎯 Router: {router_output.intent.value} (confidence: {router_output.confidence:.2f})")
            print(f"   Reasoning: {router_output.reasoning}")
            
            return router_output
            
        except json.JSONDecodeError as e:
            print(f"❌ Router JSON decode error: {e}")
            # Fallback to single ticker if query contains ticker-like words
            return self._fallback_classification(user_query)
        
        except Exception as e:
            print(f"❌ Router error: {e}")
            return self._fallback_classification(user_query)
    
    def _fallback_classification(self, user_query: str) -> RouterOutput:
        """
        Fallback classification when LLM fails.
        Uses simple heuristics.
        """
        query_lower = user_query.lower()
        
        # Check for ticker patterns (3-5 uppercase letters)
        import re
        ticker_pattern = r'\b[A-Z]{2,5}\b'
        tickers = re.findall(ticker_pattern, user_query)
        
        # Heuristic rules
        if tickers:
            intent = QueryIntent.SINGLE_TICKER
            workflow = "full_pipeline"
            entities = ExtractedEntity(tickers=tickers)
        elif any(word in query_lower for word in ['find', 'scan', 'show me', 'search']):
            intent = QueryIntent.MARKET_SCAN
            workflow = "dynamic_scanner"
            entities = ExtractedEntity()
        elif any(word in query_lower for word in ['market', 'today', 'overview']):
            intent = QueryIntent.MARKET_OVERVIEW
            workflow = "quick_overview"
            entities = ExtractedEntity()
        elif any(word in query_lower for word in ['what is', 'explain', 'how to', 'what does']):
            intent = QueryIntent.EDUCATIONAL
            workflow = "educational_response"
            entities = ExtractedEntity()
        else:
            intent = QueryIntent.CONVERSATIONAL
            workflow = "contextual_response"
            entities = ExtractedEntity()
        
        return RouterOutput(
            intent=intent,
            confidence=0.6,  # Low confidence for fallback
            entities=entities,
            reasoning=f"Fallback classification based on keywords",
            suggested_workflow=workflow,
            reformulated_query=user_query
        )
    
    def get_context(self) -> List[Dict]:
        """Get conversation context history"""
        return self.context_history
    
    def clear_context(self):
        """Clear conversation context"""
        self.context_history = []


# Global instance
_router = None

def get_router() -> QueryRouter:
    """Get global query router instance"""
    global _router
    if _router is None:
        _router = QueryRouter()
    return _router
