"""
Social sentiment analysis module.
Fetches real-time sentiment data from Reddit and analyzes ticker mentions.
Uses Reddit's public JSON API (no auth required) and VADER sentiment analysis.
"""

import requests
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import Counter, defaultdict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time


# Initialize VADER sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# Cache for rate limiting
_cache = {}
_cache_timeout = 60  # seconds


def get_trending_stocks(limit: int = 20) -> Dict[str, Any]:
    """
    Get trending stocks from Reddit (r/wallstreetbets, r/stocks).
    Uses public JSON endpoints - no authentication required.
    
    Returns:
        Dictionary with trending tickers, sentiment scores, and discussion highlights
    """
    try:
        # Check cache
        cache_key = f"trending_{limit}"
        if cache_key in _cache:
            cached_time, cached_data = _cache[cache_key]
            if time.time() - cached_time < _cache_timeout:
                return cached_data
        
        # Fetch from multiple subreddits
        subreddits = ['wallstreetbets', 'stocks', 'investing']
        all_posts = []
        
        for subreddit in subreddits:
            posts = fetch_subreddit_posts(subreddit, limit=100)
            all_posts.extend(posts)
        
        # Extract tickers and analyze sentiment
        ticker_data = analyze_posts(all_posts)
        
        # Get trending tickers (sorted by mention count)
        trending = sorted(
            ticker_data.items(),
            key=lambda x: x[1]['mention_count'],
            reverse=True
        )[:limit]
        
        # Format response
        result = {
            "trending_tickers": [
                {
                    "ticker": ticker,
                    "mention_count": data['mention_count'],
                    "sentiment_score": data['sentiment_score'],
                    "sentiment_label": get_sentiment_label(data['sentiment_score']),
                    "bullish_ratio": data['bullish_count'] / data['mention_count'] if data['mention_count'] > 0 else 0,
                    "trend": data['trend'],  # 'hot', 'rising', 'stable'
                    "top_post": data['top_post']
                }
                for ticker, data in trending
            ],
            "discussion_highlights": get_top_discussions(all_posts, limit=10),
            "overall_sentiment": calculate_overall_sentiment(ticker_data),
            "last_updated": datetime.now().isoformat(),
            "total_posts_analyzed": len(all_posts)
        }
        
        # Cache result
        _cache[cache_key] = (time.time(), result)
        
        return result
        
    except Exception as e:
        return {"error": f"Failed to fetch trending stocks: {str(e)}"}


def get_ticker_sentiment(ticker: str) -> Dict[str, Any]:
    """
    Get detailed sentiment analysis for a specific ticker.
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL')
        
    Returns:
        Sentiment data including score, discussions, and timeline
    """
    try:
        # Check cache
        cache_key = f"ticker_{ticker}"
        if cache_key in _cache:
            cached_time, cached_data = _cache[cache_key]
            if time.time() - cached_time < _cache_timeout:
                return cached_data
        
        subreddits = ['wallstreetbets', 'stocks', 'investing']
        all_posts = []
        
        for subreddit in subreddits:
            # Search for ticker mentions
            posts = search_subreddit(subreddit, ticker, limit=50)
            all_posts.extend(posts)
        
        if not all_posts:
            return {
                "ticker": ticker,
                "sentiment_score": 0,
                "sentiment_label": "neutral",
                "mention_count": 0,
                "discussions": [],
                "error": "No recent discussions found"
            }
        
        # Analyze sentiment
        sentiments = []
        discussions = []
        
        for post in all_posts:
            text = f"{post['title']} {post.get('selftext', '')}"
            sentiment = analyze_text_sentiment(text)
            sentiments.append(sentiment['compound'])
            
            discussions.append({
                "title": post['title'],
                "author": post['author'],
                "score": post['score'],
                "num_comments": post['num_comments'],
                "created_utc": post['created_utc'],
                "url": post['url'],
                "sentiment": sentiment['compound'],
                "sentiment_label": get_sentiment_label(sentiment['compound'])
            })
        
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        
        # Sort discussions by score
        discussions.sort(key=lambda x: x['score'], reverse=True)
        
        result = {
            "ticker": ticker,
            "sentiment_score": avg_sentiment,
            "sentiment_label": get_sentiment_label(avg_sentiment),
            "mention_count": len(all_posts),
            "discussions": discussions[:20],  # Top 20
            "sentiment_distribution": {
                "bullish": sum(1 for s in sentiments if s > 0.05),
                "bearish": sum(1 for s in sentiments if s < -0.05),
                "neutral": sum(1 for s in sentiments if -0.05 <= s <= 0.05)
            },
            "last_updated": datetime.now().isoformat()
        }
        
        # Cache result
        _cache[cache_key] = (time.time(), result)
        
        return result
        
    except Exception as e:
        return {"error": f"Failed to fetch ticker sentiment: {str(e)}"}


def fetch_subreddit_posts(subreddit: str, limit: int = 100, timeframe: str = 'day') -> List[Dict]:
    """
    Fetch posts from a subreddit using public JSON API.
    No authentication required.
    """
    try:
        # Reddit's public JSON endpoint
        url = f"https://www.reddit.com/r/{subreddit}/hot.json"
        headers = {'User-Agent': 'TradingAnalysisBot/1.0'}
        params = {'limit': limit, 't': timeframe}
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        posts = []
        
        for child in data['data']['children']:
            post = child['data']
            posts.append({
                'title': post.get('title', ''),
                'selftext': post.get('selftext', ''),
                'author': post.get('author', '[deleted]'),
                'score': post.get('score', 0),
                'num_comments': post.get('num_comments', 0),
                'created_utc': post.get('created_utc', 0),
                'url': f"https://reddit.com{post.get('permalink', '')}",
                'subreddit': subreddit
            })
        
        return posts
        
    except Exception as e:
        print(f"Error fetching from r/{subreddit}: {e}")
        return []


def search_subreddit(subreddit: str, query: str, limit: int = 50) -> List[Dict]:
    """Search for posts mentioning a specific query in a subreddit."""
    try:
        url = f"https://www.reddit.com/r/{subreddit}/search.json"
        headers = {'User-Agent': 'TradingAnalysisBot/1.0'}
        params = {
            'q': query,
            'restrict_sr': 'on',
            'sort': 'relevance',
            'limit': limit,
            't': 'week'
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        posts = []
        
        for child in data['data']['children']:
            post = child['data']
            posts.append({
                'title': post.get('title', ''),
                'selftext': post.get('selftext', ''),
                'author': post.get('author', '[deleted]'),
                'score': post.get('score', 0),
                'num_comments': post.get('num_comments', 0),
                'created_utc': post.get('created_utc', 0),
                'url': f"https://reddit.com{post.get('permalink', '')}",
                'subreddit': subreddit
            })
        
        return posts
        
    except Exception as e:
        print(f"Error searching r/{subreddit}: {e}")
        return []


def analyze_posts(posts: List[Dict]) -> Dict[str, Dict]:
    """
    Analyze posts to extract ticker mentions and sentiment.
    
    Returns:
        Dictionary mapping tickers to their data (mentions, sentiment, etc.)
    """
    ticker_data = defaultdict(lambda: {
        'mention_count': 0,
        'sentiment_score': 0,
        'bullish_count': 0,
        'bearish_count': 0,
        'neutral_count': 0,
        'top_post': None,
        'trend': 'stable'
    })
    
    # Common stock tickers pattern (1-5 uppercase letters)
    ticker_pattern = r'\b[A-Z]{1,5}\b'
    
    # Exclude common words that look like tickers
    exclude_words = {
        'I', 'A', 'CEO', 'IPO', 'ETF', 'DD', 'YOLO', 'WSB', 'IMO', 'IMHO',
        'FYI', 'TL', 'DR', 'TLDR', 'USA', 'US', 'UK', 'EU', 'ATH', 'ATL',
        'PM', 'AM', 'EOD', 'AH', 'IV', 'PE', 'EPS', 'GDP', 'CPI', 'FED',
        'SEC', 'IRS', 'LLC', 'INC', 'CORP', 'LTD', 'THE', 'AND', 'FOR',
        'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE',
        'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HIS', 'HOW', 'MAN',
        'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WAY', 'WHO', 'BOY', 'DID',
        'ITS', 'LET', 'PUT', 'SAY', 'SHE', 'TOO', 'USE', 'IT', 'IS', 'OF'
    }
    
    for post in posts:
        text = f"{post['title']} {post.get('selftext', '')}"
        
        # Find potential tickers
        potential_tickers = set(re.findall(ticker_pattern, text))
        potential_tickers = potential_tickers - exclude_words
        
        # Analyze sentiment
        sentiment = analyze_text_sentiment(text)
        
        for ticker in potential_tickers:
            ticker_data[ticker]['mention_count'] += 1
            ticker_data[ticker]['sentiment_score'] += sentiment['compound']
            
            if sentiment['compound'] > 0.05:
                ticker_data[ticker]['bullish_count'] += 1
            elif sentiment['compound'] < -0.05:
                ticker_data[ticker]['bearish_count'] += 1
            else:
                ticker_data[ticker]['neutral_count'] += 1
            
            # Track top post for this ticker
            if (ticker_data[ticker]['top_post'] is None or 
                post['score'] > ticker_data[ticker]['top_post']['score']):
                ticker_data[ticker]['top_post'] = {
                    'title': post['title'],
                    'score': post['score'],
                    'url': post['url'],
                    'num_comments': post['num_comments']
                }
    
    # Calculate average sentiment
    for ticker, data in ticker_data.items():
        if data['mention_count'] > 0:
            data['sentiment_score'] /= data['mention_count']
        
        # Determine trend
        if data['mention_count'] > 50:
            data['trend'] = 'hot'
        elif data['mention_count'] > 20:
            data['trend'] = 'rising'
        else:
            data['trend'] = 'stable'
    
    return dict(ticker_data)


def analyze_text_sentiment(text: str) -> Dict[str, float]:
    """Analyze sentiment of text using VADER."""
    return sentiment_analyzer.polarity_scores(text)


def get_sentiment_label(score: float) -> str:
    """Convert sentiment score to label."""
    if score > 0.05:
        return 'bullish'
    elif score < -0.05:
        return 'bearish'
    else:
        return 'neutral'


def get_top_discussions(posts: List[Dict], limit: int = 10) -> List[Dict]:
    """Get top discussions sorted by score."""
    sorted_posts = sorted(posts, key=lambda x: x['score'], reverse=True)
    
    highlights = []
    for post in sorted_posts[:limit]:
        text = f"{post['title']} {post.get('selftext', '')}"
        sentiment = analyze_text_sentiment(text)
        
        highlights.append({
            'title': post['title'],
            'author': post['author'],
            'score': post['score'],
            'num_comments': post['num_comments'],
            'subreddit': post['subreddit'],
            'url': post['url'],
            'sentiment': sentiment['compound'],
            'sentiment_label': get_sentiment_label(sentiment['compound']),
            'created_ago': format_time_ago(post['created_utc'])
        })
    
    return highlights


def calculate_overall_sentiment(ticker_data: Dict) -> Dict[str, Any]:
    """Calculate overall market sentiment from ticker data."""
    if not ticker_data:
        return {"score": 0, "label": "neutral"}
    
    total_sentiment = sum(data['sentiment_score'] * data['mention_count'] 
                         for data in ticker_data.values())
    total_mentions = sum(data['mention_count'] for data in ticker_data.values())
    
    avg_sentiment = total_sentiment / total_mentions if total_mentions > 0 else 0
    
    return {
        "score": avg_sentiment,
        "label": get_sentiment_label(avg_sentiment),
        "total_mentions": total_mentions,
        "unique_tickers": len(ticker_data)
    }


def format_time_ago(timestamp: float) -> str:
    """Format Unix timestamp as 'X hours ago' or 'X days ago'."""
    now = datetime.now()
    post_time = datetime.fromtimestamp(timestamp)
    diff = now - post_time
    
    if diff.days > 0:
        return f"{diff.days}d ago"
    elif diff.seconds >= 3600:
        hours = diff.seconds // 3600
        return f"{hours}h ago"
    else:
        minutes = diff.seconds // 60
        return f"{minutes}m ago"
