"""
Confidence Calibrator for AI Trading System.
Learns optimal confidence levels from historical performance.
"""

from typing import Dict
from database import get_db


class ConfidenceCalibrator:
    """
    Calibrate agent confidence scores based on historical accuracy.
    If an agent says 70% confidence but only wins 50% of the time,
    we adjust future 70% confidence predictions down to 50%.
    """
    
    def __init__(self):
        """Initialize calibrator with database connection."""
        self.db = get_db()
        self.min_samples = 10  # Minimum samples needed for calibration
    
    def calibrate_confidence(
        self,
        agent_name: str,
        raw_confidence: int
    ) -> int:
        """
        Adjust agent confidence based on historical performance.
        
        Args:
            agent_name: Name of agent (BULL, BEAR, CONSENSUS)
            raw_confidence: Agent's raw confidence score (0-100)
            
        Returns:
            calibrated_confidence: Adjusted confidence score (0-100)
        """
        # Get calibration factor for this confidence bucket
        calibration_factor = self.db.get_calibration_factor(agent_name, raw_confidence)
        
        # Apply calibration
        calibrated = int(raw_confidence * calibration_factor)
        
        # Clamp to valid range
        calibrated = max(0, min(100, calibrated))
        
        # Log if calibration made a significant change
        if abs(calibrated - raw_confidence) > 10:
            print(f"🔧 Calibrated {agent_name} confidence: {raw_confidence}% → {calibrated}%")
        
        return calibrated
    
    def update_calibration(self, agent_name: str):
        """
        Recalculate calibration factors for an agent.
        Should be called periodically (e.g., weekly or after N recommendations).
        
        Args:
            agent_name: Name of agent to calibrate
        """
        print(f"🔄 Updating calibration for {agent_name}...")
        self.db.update_calibration(agent_name)
        print(f"✅ Calibration updated for {agent_name}")
    
    def update_all_calibrations(self):
        """Update calibration for all agents."""
        for agent in ['BULL', 'BEAR', 'CONSENSUS', 'DEVILS_ADVOCATE']:
            self.update_calibration(agent)
    
    def get_calibration_report(self, agent_name: str) -> Dict:
        """
        Get calibration report showing how well-calibrated an agent is.
        
        Args:
            agent_name: Name of agent
            
        Returns:
            Dict with calibration metrics per confidence bucket
        """
        cursor = self.db.conn.cursor()
        
        cursor.execute("""
            SELECT 
                confidence_bucket,
                total_predictions,
                correct_predictions,
                win_rate,
                expected_win_rate,
                calibration_factor,
                calibration_error,
                sample_size_sufficient
            FROM confidence_calibration
            WHERE agent_name = ?
            ORDER BY confidence_bucket
        """, (agent_name,))
        
        rows = cursor.fetchall()
        
        report = {
            'agent_name': agent_name,
            'buckets': []
        }
        
        for row in rows:
            bucket_data = dict(row)
            report['buckets'].append(bucket_data)
        
        # Calculate overall calibration error
        if report['buckets']:
            total_error = sum(b.get('calibration_error', 0) for b in report['buckets'] if b.get('sample_size_sufficient'))
            num_buckets = sum(1 for b in report['buckets'] if b.get('sample_size_sufficient'))
            report['avg_calibration_error'] = total_error / num_buckets if num_buckets > 0 else 0
        else:
            report['avg_calibration_error'] = 0
        
        return report
    
    def is_well_calibrated(self, agent_name: str, threshold: float = 0.1) -> bool:
        """
        Check if agent is well-calibrated.
        
        Args:
            agent_name: Name of agent
            threshold: Max acceptable calibration error (default 0.1 = 10%)
            
        Returns:
            True if agent is well-calibrated
        """
        report = self.get_calibration_report(agent_name)
        return report.get('avg_calibration_error', 1.0) < threshold
    
    def get_confidence_distribution(self, agent_name: str) -> Dict:
        """
        Get distribution of confidence levels used by an agent.
        Useful for understanding if agent is overconfident or underconfident.
        
        Args:
            agent_name: Name of agent
            
        Returns:
            Dict with confidence distribution
        """
        cursor = self.db.conn.cursor()
        
        cursor.execute("""
            SELECT 
                confidence_level,
                COUNT(*) as count
            FROM agent_performance
            WHERE agent_name = ?
            GROUP BY confidence_level
            ORDER BY confidence_level
        """, (agent_name,))
        
        rows = cursor.fetchall()
        
        distribution = {}
        total = 0
        
        for row in rows:
            conf = row['confidence_level']
            count = row['count']
            distribution[conf] = count
            total += count
        
        # Calculate percentages
        if total > 0:
            distribution_pct = {k: (v / total * 100) for k, v in distribution.items()}
        else:
            distribution_pct = {}
        
        return {
            'agent_name': agent_name,
            'distribution': distribution,
            'distribution_percent': distribution_pct,
            'total_predictions': total,
            'avg_confidence': sum(k * v for k, v in distribution.items()) / total if total > 0 else 0
        }


# Global instance
_calibrator = None

def get_calibrator() -> ConfidenceCalibrator:
    """Get global confidence calibrator instance."""
    global _calibrator
    if _calibrator is None:
        _calibrator = ConfidenceCalibrator()
    return _calibrator
