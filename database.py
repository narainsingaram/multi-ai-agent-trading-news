"""
Database module for AI Trading Pattern Recognition & Learning System.
Handles all database operations for recommendations, patterns, and agent performance.
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class TradingDatabase:
    """
    Main database interface for the AI trading system.
    Manages recommendations, patterns, agent performance, and confidence calibration.
    """
    
    def __init__(self, db_path: str = "trading_ai.db"):
        """Initialize database connection and create tables if needed."""
        self.db_path = db_path
        self.conn = None
        self.init_database()
    
    def init_database(self):
        """Initialize database with schema from SQL file."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        
        # Read and execute schema
        schema_path = Path(__file__).parent / "trading_ai_schema.sql"
        if schema_path.exists():
            with open(schema_path, 'r') as f:
                schema_sql = f.read()
                self.conn.executescript(schema_sql)
        else:
            print(f"Warning: Schema file not found at {schema_path}")
            # Create basic tables inline as fallback
            self._create_basic_schema()
        
        self.conn.commit()
        print(f"✅ Database initialized at {self.db_path}")
    
    def _create_basic_schema(self):
        """Create basic schema if SQL file not found."""
        cursor = self.conn.cursor()
        
        # Recommendations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                ticker TEXT NOT NULL,
                decision TEXT NOT NULL,
                entry_price REAL,
                target_price REAL,
                stop_price REAL,
                bull_confidence INTEGER,
                bear_confidence INTEGER,
                consensus_confidence INTEGER,
                technical_snapshot TEXT,
                pattern_detected TEXT,
                outcome TEXT,
                pnl_percent REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Pattern library table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pattern_library (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT NOT NULL,
                ticker TEXT NOT NULL,
                detected_date DATETIME NOT NULL,
                pattern_data TEXT NOT NULL,
                outcome TEXT,
                price_move_percent REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Agent performance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT NOT NULL,
                recommendation_id INTEGER NOT NULL,
                confidence_level INTEGER,
                was_correct BOOLEAN,
                timestamp DATETIME NOT NULL,
                FOREIGN KEY (recommendation_id) REFERENCES recommendations(id)
            )
        """)
        
        # Confidence calibration table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS confidence_calibration (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT NOT NULL,
                confidence_bucket INTEGER NOT NULL,
                total_predictions INTEGER DEFAULT 0,
                correct_predictions INTEGER DEFAULT 0,
                win_rate REAL,
                calibration_factor REAL DEFAULT 1.0,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(agent_name, confidence_bucket)
            )
        """)
        
        self.conn.commit()
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    # ========== Recommendation Methods ==========
    
    def log_recommendation(self, recommendation_data: Dict) -> int:
        """
        Log a new recommendation to the database.
        
        Args:
            recommendation_data: Dict containing all recommendation details
            
        Returns:
            recommendation_id: ID of the inserted recommendation
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO recommendations (
                timestamp, ticker, decision, entry_price, target_price, stop_price,
                position_size, bull_confidence, bear_confidence, consensus_confidence,
                bull_thesis, bear_thesis, consensus_rationale,
                technical_snapshot, pattern_detected, market_regime, outcome
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            recommendation_data.get('timestamp', datetime.now()),
            recommendation_data['ticker'],
            recommendation_data['decision'],
            recommendation_data.get('entry_price'),
            recommendation_data.get('target_price'),
            recommendation_data.get('stop_price'),
            recommendation_data.get('position_size'),
            recommendation_data.get('bull_confidence'),
            recommendation_data.get('bear_confidence'),
            recommendation_data.get('consensus_confidence'),
            recommendation_data.get('bull_thesis'),
            recommendation_data.get('bear_thesis'),
            recommendation_data.get('consensus_rationale'),
            json.dumps(recommendation_data.get('technical_snapshot', {})),
            json.dumps(recommendation_data.get('pattern_detected', [])),
            recommendation_data.get('market_regime'),
            'PENDING'
        ))
        
        self.conn.commit()
        rec_id = cursor.lastrowid
        
        print(f"✅ Logged recommendation #{rec_id} for {recommendation_data['ticker']}")
        return rec_id
    
    def update_recommendation_outcome(self, rec_id: int, outcome_data: Dict):
        """
        Update recommendation with actual outcome.
        
        Args:
            rec_id: Recommendation ID
            outcome_data: Dict with actual_exit_price, outcome, pnl_percent, etc.
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            UPDATE recommendations
            SET outcome = ?,
                actual_exit_price = ?,
                actual_exit_date = ?,
                pnl_percent = ?,
                pnl_dollars = ?,
                days_held = ?,
                max_favorable_excursion = ?,
                max_adverse_excursion = ?,
                updated_at = ?
            WHERE id = ?
        """, (
            outcome_data.get('outcome'),
            outcome_data.get('actual_exit_price'),
            outcome_data.get('actual_exit_date', datetime.now()),
            outcome_data.get('pnl_percent'),
            outcome_data.get('pnl_dollars'),
            outcome_data.get('days_held'),
            outcome_data.get('mfe'),
            outcome_data.get('mae'),
            datetime.now(),
            rec_id
        ))
        
        self.conn.commit()
        print(f"✅ Updated recommendation #{rec_id} with outcome: {outcome_data.get('outcome')}")
    
    def get_recommendation(self, rec_id: int) -> Optional[Dict]:
        """Get a single recommendation by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM recommendations WHERE id = ?", (rec_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_recommendations_history(
        self,
        ticker: Optional[str] = None,
        days: int = 30,
        outcome: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get recommendation history with optional filters.
        
        Args:
            ticker: Filter by ticker (optional)
            days: Number of days to look back
            outcome: Filter by outcome (WIN, LOSS, PENDING, etc.)
            limit: Max number of results
            
        Returns:
            List of recommendation dictionaries
        """
        cursor = self.conn.cursor()
        
        query = """
            SELECT * FROM recommendations
            WHERE timestamp >= datetime('now', '-' || ? || ' days')
        """
        params = [days]
        
        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)
        
        if outcome:
            query += " AND outcome = ?"
            params.append(outcome)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        return [dict(row) for row in rows]
    
    def get_win_rate(self, ticker: Optional[str] = None, days: int = 90) -> Dict:
        """Calculate overall win rate."""
        cursor = self.conn.cursor()
        
        query = """
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) as losses,
                AVG(pnl_percent) as avg_pnl
            FROM recommendations
            WHERE outcome IN ('WIN', 'LOSS')
            AND timestamp >= datetime('now', '-' || ? || ' days')
        """
        params = [days]
        
        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)
        
        cursor.execute(query, params)
        row = cursor.fetchone()
        
        if row and row['total'] > 0:
            return {
                'total_trades': row['total'],
                'wins': row['wins'],
                'losses': row['losses'],
                'win_rate': (row['wins'] / row['total']) * 100 if row['total'] > 0 else 0,
                'avg_pnl_percent': row['avg_pnl'] or 0
            }
        
        return {'total_trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0, 'avg_pnl_percent': 0}
    
    # ========== Pattern Library Methods ==========
    
    def add_pattern(self, pattern_data: Dict) -> int:
        """Add a detected pattern to the library."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO pattern_library (
                pattern_type, pattern_subtype, ticker, detected_date,
                pattern_data, pattern_quality_score, outcome,
                price_at_detection, technical_context, market_regime,
                shape_signature
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pattern_data['pattern_type'],
            pattern_data.get('pattern_subtype'),
            pattern_data['ticker'],
            pattern_data.get('detected_date', datetime.now()),
            json.dumps(pattern_data.get('pattern_data', {})),
            pattern_data.get('quality_score', 0.5),
            'PENDING',
            pattern_data.get('price_at_detection'),
            json.dumps(pattern_data.get('technical_context', {})),
            pattern_data.get('market_regime'),
            json.dumps(pattern_data.get('shape_signature', []))
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def get_pattern_library(
        self,
        pattern_type: Optional[str] = None,
        ticker: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Get patterns from library with optional filters."""
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM pattern_library WHERE 1=1"
        params = []
        
        if pattern_type:
            query += " AND pattern_type = ?"
            params.append(pattern_type)
        
        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)
        
        query += " ORDER BY detected_date DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        return [dict(row) for row in rows]
    
    # ========== Agent Performance Methods ==========
    
    def log_agent_performance(self, performance_data: Dict):
        """Log agent performance for a recommendation."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO agent_performance (
                agent_name, recommendation_id, predicted_direction,
                confidence_level, key_factors, actual_outcome,
                was_correct, confidence_error, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            performance_data['agent_name'],
            performance_data['recommendation_id'],
            performance_data.get('predicted_direction'),
            performance_data.get('confidence_level'),
            json.dumps(performance_data.get('key_factors', [])),
            performance_data.get('actual_outcome'),
            performance_data.get('was_correct'),
            performance_data.get('confidence_error'),
            performance_data.get('timestamp', datetime.now())
        ))
        
        self.conn.commit()
    
    def get_agent_stats(self, agent_name: str, days: int = 90) -> Dict:
        """Get performance statistics for an agent."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT * FROM v_agent_stats WHERE agent_name = ?
        """, (agent_name,))
        
        row = cursor.fetchone()
        return dict(row) if row else {}
    
    # ========== Confidence Calibration Methods ==========
    
    def get_calibration_factor(self, agent_name: str, confidence: int) -> float:
        """Get calibration factor for an agent at a confidence level."""
        bucket = (confidence // 10) * 10
        
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT calibration_factor FROM confidence_calibration
            WHERE agent_name = ? AND confidence_bucket = ?
        """, (agent_name, bucket))
        
        row = cursor.fetchone()
        return row['calibration_factor'] if row else 1.0
    
    def update_calibration(self, agent_name: str):
        """Recalculate calibration factors for an agent."""
        cursor = self.conn.cursor()
        
        for bucket in range(0, 100, 10):
            # Get performance in this confidence bucket
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN was_correct THEN 1 ELSE 0 END) as correct
                FROM agent_performance
                WHERE agent_name = ?
                AND confidence_level BETWEEN ? AND ?
            """, (agent_name, bucket, bucket + 10))
            
            row = cursor.fetchone()
            total, correct = row['total'], row['correct']
            
            if total >= 10:  # Need at least 10 samples
                win_rate = correct / total
                expected_confidence = (bucket + 5) / 100  # Midpoint
                
                # Calculate calibration factor
                if expected_confidence > 0:
                    calibration_factor = win_rate / expected_confidence
                else:
                    calibration_factor = 1.0
                
                # Update database
                cursor.execute("""
                    INSERT OR REPLACE INTO confidence_calibration
                    (agent_name, confidence_bucket, total_predictions,
                     correct_predictions, win_rate, calibration_factor,
                     expected_win_rate, calibration_error, last_updated,
                     sample_size_sufficient)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    agent_name, bucket, total, correct, win_rate,
                    calibration_factor, expected_confidence,
                    abs(win_rate - expected_confidence),
                    datetime.now(), True
                ))
        
        self.conn.commit()
        print(f"✅ Updated calibration for {agent_name}")


# Global database instance
_db_instance = None

def get_db() -> TradingDatabase:
    """Get global database instance (singleton pattern)."""
    global _db_instance
    if _db_instance is None:
        _db_instance = TradingDatabase()
    return _db_instance
