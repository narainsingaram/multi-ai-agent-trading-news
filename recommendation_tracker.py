"""
Recommendation Tracker for AI Trading System.
Logs every recommendation and tracks outcomes for learning.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from database import get_db
from pattern_recognition import get_pattern_recognition


class RecommendationTracker:
    """
    Track all AI recommendations and their outcomes.
    Enables learning from past performance.
    """
    
    def __init__(self):
        """Initialize tracker with database connection."""
        self.db = get_db()
        self.pattern_detector = get_pattern_recognition()
    
    def log_recommendation(
        self,
        pipeline_state: Dict,
        market_data: Dict,
        ticker: str
    ) -> int:
        """
        Log a recommendation from the AI pipeline.
        
        Args:
            pipeline_state: Complete pipeline state with all agent outputs
            market_data: Market data including OHLCV and technicals
            ticker: Stock ticker
            
        Returns:
            recommendation_id: Database ID of logged recommendation
        """
        # Extract agent outputs
        consensus = pipeline_state.get('consensus', {})
        bull = pipeline_state.get('bull', {})
        bear = pipeline_state.get('bear', {})
        planner = pipeline_state.get('planner', {})
        
        # Get recommended action
        recommended_action = consensus.get('recommended_action', {})
        
        # Detect patterns in current data
        hist_data = market_data.get('historical_data')
        patterns_detected = []
        if hist_data is not None and not hist_data.empty:
            patterns_detected = self.pattern_detector.detect_all_patterns(hist_data)
        
        # Prepare recommendation data
        rec_data = {
            'timestamp': datetime.now(),
            'ticker': ticker,
            'decision': consensus.get('decision', 'NO_TRADE'),
            'entry_price': recommended_action.get('entry'),
            'target_price': recommended_action.get('target'),
            'stop_price': recommended_action.get('stop'),
            'position_size': recommended_action.get('position_size'),
            
            # Agent confidences
            'bull_confidence': bull.get('confidence'),
            'bear_confidence': bear.get('confidence'),
            'consensus_confidence': consensus.get('confidence'),
            
            # Agent theses
            'bull_thesis': json.dumps(bull.get('thesis', '')),
            'bear_thesis': json.dumps(bear.get('thesis', '')),
            'consensus_rationale': json.dumps(consensus.get('rationale', '')),
            
            # Technical snapshot
            'technical_snapshot': market_data.get('technicals', {}),
            'pattern_detected': patterns_detected,
            'market_regime': self._determine_market_regime(market_data),
        }
        
        # Log to database
        rec_id = self.db.log_recommendation(rec_data)
        
        # Log agent performance (initially as PENDING)
        self._log_agent_performance(rec_id, pipeline_state)
        
        # Add patterns to library
        for pattern in patterns_detected:
            self._add_pattern_to_library(pattern, ticker, market_data)
        
        print(f"📝 Logged recommendation #{rec_id}: {ticker} {rec_data['decision']} @ ${rec_data.get('entry_price', 0):.2f}")
        
        return rec_id
    
    def _log_agent_performance(self, rec_id: int, pipeline_state: Dict):
        """Log performance data for each agent."""
        agents = ['BULL', 'BEAR', 'CONSENSUS']
        
        for agent_name in agents:
            agent_output = pipeline_state.get(agent_name.lower(), {})
            
            if agent_output:
                perf_data = {
                    'agent_name': agent_name,
                    'recommendation_id': rec_id,
                    'predicted_direction': self._get_agent_direction(agent_name, agent_output),
                    'confidence_level': agent_output.get('confidence'),
                    'key_factors': agent_output.get('evidence', []) if agent_name == 'BULL' else agent_output.get('evidence', []),
                    'actual_outcome': None,  # Will be updated later
                    'was_correct': None,
                    'confidence_error': None,
                    'timestamp': datetime.now()
                }
                
                self.db.log_agent_performance(perf_data)
    
    def _get_agent_direction(self, agent_name: str, agent_output: Dict) -> str:
        """Determine agent's predicted direction."""
        if agent_name == 'BULL':
            return 'bullish'
        elif agent_name == 'BEAR':
            return 'bearish'
        elif agent_name == 'CONSENSUS':
            decision = agent_output.get('decision', 'NO_TRADE')
            if decision == 'LONG':
                return 'bullish'
            elif decision == 'SHORT':
                return 'bearish'
            else:
                return 'neutral'
        return 'neutral'
    
    def _determine_market_regime(self, market_data: Dict) -> str:
        """Determine current market regime from technical data."""
        technicals = market_data.get('technicals', {})
        
        # Get key indicators
        adx = technicals.get('momentum', {}).get('adx_14')
        atr = technicals.get('volatility', {}).get('atr_14')
        trend_direction = technicals.get('trend', {}).get('direction', 'neutral')
        
        # Determine regime
        if adx and adx > 25:
            if 'up' in trend_direction.lower():
                return 'trending_bullish'
            elif 'down' in trend_direction.lower():
                return 'trending_bearish'
            else:
                return 'trending'
        else:
            return 'ranging'
    
    def _add_pattern_to_library(self, pattern: Dict, ticker: str, market_data: Dict):
        """Add detected pattern to the pattern library."""
        pattern_data = {
            'pattern_type': pattern['type'],
            'pattern_subtype': pattern.get('subtype'),
            'ticker': ticker,
            'detected_date': datetime.now(),
            'pattern_data': pattern,
            'quality_score': pattern.get('quality_score', 0.5),
            'price_at_detection': market_data.get('quote', {}).get('current_price'),
            'technical_context': market_data.get('technicals', {}),
            'market_regime': self._determine_market_regime(market_data),
            'shape_signature': self._generate_shape_signature(pattern)
        }
        
        self.db.add_pattern(pattern_data)
    
    def _generate_shape_signature(self, pattern: Dict) -> List[float]:
        """Generate normalized shape signature for pattern matching."""
        # Extract key price points from pattern coordinates
        coords = pattern.get('coordinates', {})
        
        # Normalize prices to 0-1 range for comparison
        prices = []
        for key, value in coords.items():
            if isinstance(value, dict) and 'price' in value:
                prices.append(value['price'])
            elif isinstance(value, (int, float)):
                prices.append(value)
        
        if not prices:
            return []
        
        # Normalize
        min_price = min(prices)
        max_price = max(prices)
        price_range = max_price - min_price
        
        if price_range > 0:
            normalized = [(p - min_price) / price_range for p in prices]
        else:
            normalized = [0.5] * len(prices)
        
        return normalized
    
    def update_outcome(
        self,
        rec_id: int,
        actual_exit_price: float,
        exit_date: Optional[datetime] = None,
        mfe: Optional[float] = None,
        mae: Optional[float] = None
    ):
        """
        Update recommendation with actual outcome.
        
        Args:
            rec_id: Recommendation ID
            actual_exit_price: Actual exit price
            exit_date: Exit date (default: now)
            mfe: Maximum Favorable Excursion (best price reached)
            mae: Maximum Adverse Excursion (worst price reached)
        """
        # Get original recommendation
        rec = self.db.get_recommendation(rec_id)
        
        if not rec:
            print(f"❌ Recommendation #{rec_id} not found")
            return
        
        # Calculate P&L
        entry_price = rec['entry_price']
        decision = rec['decision']
        
        if decision == 'LONG':
            pnl_percent = ((actual_exit_price - entry_price) / entry_price) * 100
        elif decision == 'SHORT':
            pnl_percent = ((entry_price - actual_exit_price) / entry_price) * 100
        else:
            pnl_percent = 0
        
        # Determine outcome
        if pnl_percent > 0.5:
            outcome = 'WIN'
        elif pnl_percent < -0.5:
            outcome = 'LOSS'
        else:
            outcome = 'BREAKEVEN'
        
        # Calculate days held
        entry_date = datetime.fromisoformat(rec['timestamp']) if isinstance(rec['timestamp'], str) else rec['timestamp']
        exit_date = exit_date or datetime.now()
        days_held = (exit_date - entry_date).days
        
        # Update database
        outcome_data = {
            'outcome': outcome,
            'actual_exit_price': actual_exit_price,
            'actual_exit_date': exit_date,
            'pnl_percent': pnl_percent,
            'pnl_dollars': None,  # Would need position size
            'days_held': days_held,
            'mfe': mfe,
            'mae': mae
        }
        
        self.db.update_recommendation_outcome(rec_id, outcome_data)
        
        # Update agent performance
        self._update_agent_performance(rec_id, outcome)
        
        # Trigger calibration update if enough samples
        self._maybe_update_calibration()
        
        print(f"✅ Updated recommendation #{rec_id}: {outcome} ({pnl_percent:+.2f}%)")
    
    def _update_agent_performance(self, rec_id: int, outcome: str):
        """Update agent performance records with actual outcome."""
        # This would query agent_performance table and update was_correct field
        # Based on whether agent's prediction matched outcome
        # Implementation depends on how we define "correct" for each agent
        pass
    
    def _maybe_update_calibration(self):
        """Update calibration if we have enough new samples."""
        # Check if we should recalibrate (e.g., every 50 recommendations)
        # For now, we'll skip automatic calibration
        pass
    
    def get_performance_stats(
        self,
        ticker: Optional[str] = None,
        days: int = 90
    ) -> Dict:
        """
        Get performance statistics.
        
        Args:
            ticker: Filter by ticker (optional)
            days: Number of days to look back
            
        Returns:
            Dict with performance metrics
        """
        win_rate_data = self.db.get_win_rate(ticker, days)
        
        # Get agent-specific stats
        agent_stats = {}
        for agent in ['BULL', 'BEAR', 'CONSENSUS']:
            agent_stats[agent] = self.db.get_agent_stats(agent, days)
        
        return {
            'overall': win_rate_data,
            'agents': agent_stats,
            'period_days': days,
            'ticker': ticker or 'ALL'
        }
    
    def get_recent_recommendations(
        self,
        limit: int = 10,
        ticker: Optional[str] = None
    ) -> List[Dict]:
        """Get recent recommendations."""
        return self.db.get_recommendations_history(
            ticker=ticker,
            days=30,
            limit=limit
        )
    
    def get_pattern_success_rate(self, pattern_type: str) -> Dict:
        """Get success rate for a specific pattern type."""
        patterns = self.db.get_pattern_library(pattern_type=pattern_type, limit=1000)
        
        if not patterns:
            return {'pattern_type': pattern_type, 'total': 0, 'success_rate': 0}
        
        total = len(patterns)
        successful = sum(1 for p in patterns if p.get('outcome') in ['BULLISH_BREAKOUT', 'BEARISH_BREAKDOWN'])
        
        return {
            'pattern_type': pattern_type,
            'total': total,
            'successful': successful,
            'success_rate': (successful / total * 100) if total > 0 else 0
        }


# Global instance
_tracker = None

def get_tracker() -> RecommendationTracker:
    """Get global recommendation tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = RecommendationTracker()
    return _tracker
