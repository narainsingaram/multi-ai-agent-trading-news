"""
Advanced News Analysis System for AI Trading.
Pulls from multiple sources, performs deep sentiment analysis, and extracts actionable insights.
"""

import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field
import requests
import finnhub

load_dotenv()

# Initialize clients
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
finnhub_client = finnhub.Client(api_key=os.getenv("FINNHUB_API_KEY"))
newsapi_key = os.getenv("NEWSAPI_KEY", "")  # Optional: get free key from newsapi.org


class NewsArticle(BaseModel):
    """Single news article with metadata"""
    title: str = Field(description="Article headline")
    summary: str = Field(description="Article summary/snippet")
    source: str = Field(description="Publisher/source name")
    url: str = Field(description="Article URL")
    published_at: str = Field(description="Publication timestamp")
    sentiment_score: Optional[float] = Field(None, ge=-1, le=1, description="Sentiment score -1 to 1")
    relevance_score: Optional[float] = Field(None, ge=0, le=1, description="Relevance to query 0-1")


class EntityMention(BaseModel):
    """Extracted entity from news"""
    entity: str = Field(description="Entity name (company, person, product)")
    entity_type: str = Field(description="Type: company, person, product, event, metric")
    mentions: int = Field(description="Number of mentions across articles")
    sentiment: str = Field(description="Sentiment: positive, negative, neutral, mixed")


class NewsAnalysis(BaseModel):
    """Comprehensive news analysis output"""
    headline_summary: str = Field(min_length=50, description="Concise summary of main narrative")
    overall_sentiment: str = Field(description="bullish, bearish, mixed, unclear")
    sentiment_score: float = Field(ge=-1, le=1, description="Aggregate sentiment -1 to 1")
    confidence: float = Field(ge=0, le=1, description="Confidence in sentiment analysis")
    
    # Deep analysis
    key_drivers: List[str] = Field(min_items=2, description="Main drivers from news")
    catalysts: List[str] = Field(description="Upcoming catalysts mentioned")
    risks: List[str] = Field(min_items=1, description="Identified risks")
    opportunities: List[str] = Field(description="Identified opportunities")
    
    # Entity analysis
    entities_mentioned: List[EntityMention] = Field(description="Key entities extracted")
    
    # Management/Guidance analysis
    management_tone: Optional[str] = Field(None, description="confident, cautious, defensive, optimistic")
    guidance_changes: List[str] = Field(default_factory=list, description="Any guidance revisions")
    
    # Market impact
    market_moving_insights: List[str] = Field(min_items=1, description="Insights likely to move price")
    trade_implications: List[str] = Field(min_items=2, description="Trading implications")
    
    # Sources
    articles_analyzed: int = Field(description="Number of articles analyzed")
    sources: List[NewsArticle] = Field(description="Source articles with metadata")
    
    # Handoff
    handoff_note: str = Field(description="Note for next agent")


class AdvancedNewsAnalyzer:
    """
    Advanced news analysis system that pulls from multiple sources
    and performs deep sentiment and entity analysis.
    """
    
    def __init__(self):
        """Initialize news analyzer with API clients"""
        self.groq_client = groq_client
        self.finnhub_client = finnhub_client
        self.newsapi_key = newsapi_key
    
    def fetch_news_multi_source(
        self,
        ticker: str,
        query: Optional[str] = None,
        days_back: int = 7
    ) -> List[NewsArticle]:
        """
        Fetch news from multiple sources and deduplicate.
        
        Args:
            ticker: Stock ticker
            query: Optional search query
            days_back: How many days back to search
            
        Returns:
            List of NewsArticle objects
        """
        all_articles = []
        
        # Source 1: Finnhub company news
        try:
            finnhub_news = self._fetch_finnhub_news(ticker, days_back)
            all_articles.extend(finnhub_news)
            print(f"📰 Fetched {len(finnhub_news)} articles from Finnhub")
        except Exception as e:
            print(f"⚠️  Finnhub news error: {e}")
        
        # Source 2: NewsAPI (if key available)
        if self.newsapi_key:
            try:
                newsapi_articles = self._fetch_newsapi(ticker, query, days_back)
                all_articles.extend(newsapi_articles)
                print(f"📰 Fetched {len(newsapi_articles)} articles from NewsAPI")
            except Exception as e:
                print(f"⚠️  NewsAPI error: {e}")
        
        # Source 3: Finnhub general news (market-wide)
        try:
            general_news = self._fetch_finnhub_general_news(ticker)
            all_articles.extend(general_news)
            print(f"📰 Fetched {len(general_news)} general market articles")
        except Exception as e:
            print(f"⚠️  General news error: {e}")
        
        # Deduplicate by title similarity
        unique_articles = self._deduplicate_articles(all_articles)
        
        # Sort by relevance and recency
        sorted_articles = sorted(
            unique_articles,
            key=lambda x: (x.relevance_score or 0.5, x.published_at),
            reverse=True
        )
        
        print(f"✅ Total unique articles: {len(sorted_articles)}")
        
        return sorted_articles[:20]  # Return top 20
    
    def _fetch_finnhub_news(self, ticker: str, days_back: int) -> List[NewsArticle]:
        """Fetch company-specific news from Finnhub"""
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        to_date = datetime.now().strftime('%Y-%m-%d')
        
        news = self.finnhub_client.company_news(ticker, _from=from_date, to=to_date)
        
        articles = []
        for item in news:
            articles.append(NewsArticle(
                title=item.get('headline', ''),
                summary=item.get('summary', '')[:500],  # Limit summary length
                source=item.get('source', 'Finnhub'),
                url=item.get('url', ''),
                published_at=datetime.fromtimestamp(item.get('datetime', 0)).isoformat(),
                relevance_score=0.9  # Company-specific news is highly relevant
            ))
        
        return articles
    
    def _fetch_newsapi(self, ticker: str, query: Optional[str], days_back: int) -> List[NewsArticle]:
        """Fetch news from NewsAPI"""
        if not self.newsapi_key:
            return []
        
        # Build search query
        search_query = query if query else ticker
        
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': search_query,
            'from': from_date,
            'sortBy': 'relevancy',
            'language': 'en',
            'apiKey': self.newsapi_key
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        articles = []
        for item in data.get('articles', [])[:10]:  # Limit to 10
            articles.append(NewsArticle(
                title=item.get('title', ''),
                summary=item.get('description', '')[:500],
                source=item.get('source', {}).get('name', 'NewsAPI'),
                url=item.get('url', ''),
                published_at=item.get('publishedAt', ''),
                relevance_score=0.7  # Medium relevance
            ))
        
        return articles
    
    def _fetch_finnhub_general_news(self, ticker: str) -> List[NewsArticle]:
        """Fetch general market news from Finnhub"""
        news = self.finnhub_client.general_news('general', min_id=0)
        
        # Filter for relevance to ticker
        articles = []
        for item in news[:10]:  # Limit to 10
            title = item.get('headline', '').upper()
            summary = item.get('summary', '').upper()
            
            # Check if ticker mentioned
            if ticker.upper() in title or ticker.upper() in summary:
                articles.append(NewsArticle(
                    title=item.get('headline', ''),
                    summary=item.get('summary', '')[:500],
                    source=item.get('source', 'Finnhub'),
                    url=item.get('url', ''),
                    published_at=datetime.fromtimestamp(item.get('datetime', 0)).isoformat(),
                    relevance_score=0.6  # Lower relevance
                ))
        
        return articles
    
    def _deduplicate_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove duplicate articles based on title similarity"""
        unique = []
        seen_titles = set()
        
        for article in articles:
            # Simple dedup by first 50 chars of title
            title_key = article.title[:50].lower()
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique.append(article)
        
        return unique
    
    def analyze_news_deep(
        self,
        ticker: str,
        headline: str,
        articles: List[NewsArticle],
        fundamentals: Optional[Dict] = None
    ) -> NewsAnalysis:
        """
        Perform deep analysis of news articles using LLM.
        
        Args:
            ticker: Stock ticker
            headline: User's headline/query
            articles: List of news articles
            fundamentals: Optional fundamental data
            
        Returns:
            NewsAnalysis with comprehensive insights
        """
        # Build context from articles
        articles_text = self._format_articles_for_llm(articles)
        fundamentals_text = self._format_fundamentals(fundamentals) if fundamentals else "No fundamental data available"
        
        # Deep analysis prompt
        prompt = f"""You are an expert financial news analyst. Perform a DEEP, COMPREHENSIVE analysis of the following news.

TICKER: {ticker}
USER HEADLINE/QUERY: {headline}

FUNDAMENTALS SNAPSHOT:
{fundamentals_text}

NEWS ARTICLES ({len(articles)} sources):
{articles_text}

PERFORM DEEP ANALYSIS:

1. SENTIMENT ANALYSIS:
   - Overall sentiment (bullish/bearish/mixed/unclear)
   - Sentiment score (-1 to 1, where -1 = very bearish, 1 = very bullish)
   - Confidence in sentiment (0-1)

2. KEY DRIVERS:
   - What are the 3-5 main drivers from the news?
   - Be specific with numbers, dates, and facts

3. CATALYSTS & RISKS:
   - Upcoming catalysts mentioned (earnings, product launches, regulatory decisions)
   - Identified risks (competition, regulation, macro, execution)
   - Opportunities (new markets, partnerships, innovation)

4. ENTITY EXTRACTION:
   - Extract key entities mentioned: companies, people, products, events
   - For each entity: type, number of mentions, sentiment

5. MANAGEMENT TONE (if earnings/guidance mentioned):
   - Tone: confident, cautious, defensive, optimistic
   - Any guidance changes (raised, lowered, maintained)

6. MARKET-MOVING INSIGHTS:
   - What insights are most likely to move the stock price?
   - What are traders likely to focus on?

7. TRADE IMPLICATIONS:
   - How should traders interpret this news?
   - What are the actionable takeaways?

CRITICAL: Be specific, cite numbers, and provide DEEP analysis. Don't be generic.

Respond ONLY with valid JSON matching the NewsAnalysis schema."""

        try:
            # Call LLM with JSON mode
            completion = self.groq_client.chat.completions.create(
                model="llama-3.1-70b-versatile",  # Use larger model for better analysis
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,  # Low temperature for factual analysis
                max_tokens=2048,
                response_format={"type": "json_object"}
            )
            
            response_json = json.loads(completion.choices[0].message.content)
            
            # Add articles to response
            response_json['articles_analyzed'] = len(articles)
            response_json['sources'] = [a.dict() for a in articles[:10]]  # Top 10 sources
            
            # Validate with Pydantic
            analysis = NewsAnalysis(**response_json)
            
            print(f"✅ Deep news analysis complete:")
            print(f"   Sentiment: {analysis.overall_sentiment} ({analysis.sentiment_score:.2f})")
            print(f"   Confidence: {analysis.confidence:.2f}")
            print(f"   Key drivers: {len(analysis.key_drivers)}")
            print(f"   Entities: {len(analysis.entities_mentioned)}")
            
            return analysis
            
        except Exception as e:
            print(f"❌ News analysis error: {e}")
            # Return fallback analysis
            return self._fallback_analysis(ticker, headline, articles)
    
    def _format_articles_for_llm(self, articles: List[NewsArticle]) -> str:
        """Format articles for LLM consumption"""
        formatted = []
        for i, article in enumerate(articles[:15], 1):  # Top 15 articles
            formatted.append(f"""
Article {i}:
Title: {article.title}
Source: {article.source}
Published: {article.published_at}
Summary: {article.summary}
URL: {article.url}
---""")
        return "\n".join(formatted)
    
    def _format_fundamentals(self, fundamentals: Dict) -> str:
        """Format fundamentals for LLM"""
        if not fundamentals:
            return "No fundamental data"
        
        return f"""
Market Cap: ${fundamentals.get('market_cap', 0) / 1e9:.2f}B
P/E Ratio: {fundamentals.get('pe_ratio', 'N/A')}
EPS: ${fundamentals.get('eps', 'N/A')}
Revenue: ${fundamentals.get('revenue', 0) / 1e9:.2f}B
Profit Margin: {fundamentals.get('profit_margin', 0) * 100:.1f}%
Debt/Equity: {fundamentals.get('debt_to_equity', 'N/A')}
"""
    
    def _fallback_analysis(
        self,
        ticker: str,
        headline: str,
        articles: List[NewsArticle]
    ) -> NewsAnalysis:
        """Fallback analysis if LLM fails"""
        return NewsAnalysis(
            headline_summary=f"Analysis of {ticker} based on {len(articles)} news articles",
            overall_sentiment="mixed",
            sentiment_score=0.0,
            confidence=0.5,
            key_drivers=["Multiple news sources analyzed"],
            catalysts=[],
            risks=["Unable to perform deep analysis"],
            opportunities=[],
            entities_mentioned=[],
            market_moving_insights=["See individual articles for details"],
            trade_implications=["Review articles manually for trading decisions"],
            articles_analyzed=len(articles),
            sources=articles[:10],
            handoff_note="Fallback analysis - LLM analysis failed"
        )


# Global instance
_news_analyzer = None

def get_news_analyzer() -> AdvancedNewsAnalyzer:
    """Get global news analyzer instance"""
    global _news_analyzer
    if _news_analyzer is None:
        _news_analyzer = AdvancedNewsAnalyzer()
    return _news_analyzer
