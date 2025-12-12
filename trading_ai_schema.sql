-- AI Trading Pattern Recognition & Learning Database Schema
-- SQLite database for tracking recommendations, patterns, and agent performance

-- Recommendations: Every AI recommendation with full context
CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    ticker TEXT NOT NULL,
    decision TEXT NOT NULL CHECK(decision IN ('LONG', 'SHORT', 'NO_TRADE')),
    entry_price REAL,
    target_price REAL,
    stop_price REAL,
    position_size TEXT,
    
    -- Agent outputs
    bull_confidence INTEGER CHECK(bull_confidence BETWEEN 0 AND 100),
    bear_confidence INTEGER CHECK(bear_confidence BETWEEN 0 AND 100),
    consensus_confidence INTEGER CHECK(consensus_confidence BETWEEN 0 AND 100),
    bull_thesis TEXT,
    bear_thesis TEXT,
    consensus_rationale TEXT,
    
    -- Technical snapshot at time of recommendation
    technical_snapshot TEXT,  -- JSON: RSI, MACD, OBV, VWAP, etc.
    pattern_detected TEXT,    -- JSON: detected patterns
    market_regime TEXT,       -- trending, ranging, volatile, etc.
    
    -- Outcome tracking
    outcome TEXT CHECK(outcome IN ('WIN', 'LOSS', 'PENDING', 'CANCELLED', 'BREAKEVEN')),
    actual_exit_price REAL,
    actual_exit_date DATETIME,
    pnl_percent REAL,
    pnl_dollars REAL,
    days_held INTEGER,
    max_favorable_excursion REAL,  -- MFE: best price reached
    max_adverse_excursion REAL,    -- MAE: worst price reached
    
    -- Metadata
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Pattern Library: Historical chart patterns with outcomes
CREATE TABLE IF NOT EXISTS pattern_library (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_type TEXT NOT NULL,  -- head_shoulders, cup_handle, flag, etc.
    pattern_subtype TEXT,        -- bullish, bearish
    ticker TEXT NOT NULL,
    detected_date DATETIME NOT NULL,
    
    -- Pattern measurements
    pattern_data TEXT NOT NULL,  -- JSON: coordinates, measurements, ratios
    pattern_quality_score REAL CHECK(pattern_quality_score BETWEEN 0 AND 1),
    
    -- Outcome
    outcome TEXT CHECK(outcome IN ('BULLISH_BREAKOUT', 'BEARISH_BREAKDOWN', 'FAILED', 'PENDING')),
    price_at_detection REAL,
    price_at_completion REAL,
    price_move_percent REAL,
    days_to_completion INTEGER,
    volume_confirmation BOOLEAN,
    
    -- Technical context
    technical_context TEXT,  -- JSON: RSI, MACD, trend, etc. at detection
    market_regime TEXT,
    
    -- For similarity matching
    shape_signature TEXT,    -- JSON: normalized shape for DTW matching
    similarity_hash TEXT,    -- For quick lookups
    
    -- Metadata
    chart_image_path TEXT,
    notes TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Agent Performance: Track each agent's accuracy
CREATE TABLE IF NOT EXISTS agent_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_name TEXT NOT NULL CHECK(agent_name IN ('BULL', 'BEAR', 'CONSENSUS', 'DEVILS_ADVOCATE')),
    recommendation_id INTEGER NOT NULL,
    
    -- Prediction
    predicted_direction TEXT,  -- bullish, bearish, neutral
    confidence_level INTEGER CHECK(confidence_level BETWEEN 0 AND 100),
    key_factors TEXT,  -- JSON: factors agent cited
    
    -- Actual outcome
    actual_outcome TEXT,
    was_correct BOOLEAN,
    confidence_error REAL,  -- difference between confidence and actual win rate
    
    -- Timing
    timestamp DATETIME NOT NULL,
    
    FOREIGN KEY (recommendation_id) REFERENCES recommendations(id) ON DELETE CASCADE
);

-- Confidence Calibration: Learn optimal confidence levels
CREATE TABLE IF NOT EXISTS confidence_calibration (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_name TEXT NOT NULL,
    confidence_bucket INTEGER NOT NULL,  -- 0, 10, 20, ..., 90 (bucket start)
    
    -- Performance metrics
    total_predictions INTEGER DEFAULT 0,
    correct_predictions INTEGER DEFAULT 0,
    win_rate REAL,
    
    -- Calibration
    calibration_factor REAL DEFAULT 1.0,  -- multiplier to adjust confidence
    expected_win_rate REAL,  -- what win rate should be for this confidence
    calibration_error REAL,  -- how far off we are
    
    -- Metadata
    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
    sample_size_sufficient BOOLEAN DEFAULT 0,  -- TRUE if >= 30 samples
    
    UNIQUE(agent_name, confidence_bucket)
);

-- Pattern Similarity Cache: Speed up pattern matching
CREATE TABLE IF NOT EXISTS pattern_similarity_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern1_id INTEGER NOT NULL,
    pattern2_id INTEGER NOT NULL,
    similarity_score REAL CHECK(similarity_score BETWEEN 0 AND 1),
    similarity_components TEXT,  -- JSON: shape, technical, context scores
    computed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (pattern1_id) REFERENCES pattern_library(id) ON DELETE CASCADE,
    FOREIGN KEY (pattern2_id) REFERENCES pattern_library(id) ON DELETE CASCADE,
    UNIQUE(pattern1_id, pattern2_id)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_recommendations_ticker ON recommendations(ticker);
CREATE INDEX IF NOT EXISTS idx_recommendations_timestamp ON recommendations(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_recommendations_outcome ON recommendations(outcome);
CREATE INDEX IF NOT EXISTS idx_pattern_library_type ON pattern_library(pattern_type);
CREATE INDEX IF NOT EXISTS idx_pattern_library_ticker ON pattern_library(ticker);
CREATE INDEX IF NOT EXISTS idx_pattern_library_date ON pattern_library(detected_date DESC);
CREATE INDEX IF NOT EXISTS idx_agent_performance_agent ON agent_performance(agent_name);
CREATE INDEX IF NOT EXISTS idx_agent_performance_timestamp ON agent_performance(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_confidence_calibration_agent ON confidence_calibration(agent_name, confidence_bucket);

-- Views for easy querying
CREATE VIEW IF NOT EXISTS v_agent_stats AS
SELECT 
    agent_name,
    COUNT(*) as total_predictions,
    SUM(CASE WHEN was_correct THEN 1 ELSE 0 END) as correct_predictions,
    ROUND(AVG(CASE WHEN was_correct THEN 1.0 ELSE 0.0 END) * 100, 2) as win_rate,
    ROUND(AVG(confidence_level), 2) as avg_confidence,
    ROUND(AVG(confidence_error), 2) as avg_confidence_error
FROM agent_performance
GROUP BY agent_name;

CREATE VIEW IF NOT EXISTS v_pattern_success_rates AS
SELECT 
    pattern_type,
    pattern_subtype,
    COUNT(*) as total_occurrences,
    SUM(CASE WHEN outcome LIKE '%BREAKOUT%' OR outcome LIKE '%BREAKDOWN%' THEN 1 ELSE 0 END) as successful,
    ROUND(AVG(price_move_percent), 2) as avg_price_move,
    ROUND(AVG(days_to_completion), 1) as avg_days
FROM pattern_library
WHERE outcome != 'PENDING'
GROUP BY pattern_type, pattern_subtype;

CREATE VIEW IF NOT EXISTS v_recent_recommendations AS
SELECT 
    r.id,
    r.timestamp,
    r.ticker,
    r.decision,
    r.entry_price,
    r.target_price,
    r.consensus_confidence,
    r.outcome,
    r.pnl_percent,
    r.days_held,
    json_extract(r.pattern_detected, '$[0].type') as primary_pattern
FROM recommendations r
ORDER BY r.timestamp DESC
LIMIT 100;
