"""
Enhanced Agent Prompts with Pydantic Validation.
Ensures agents produce consistent, validated output.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime


# ========== BULL AGENT ==========

class BullKeyLevels(BaseModel):
    """Price levels for bullish trade"""
    entry: float = Field(description="Entry price for long position")
    target: float = Field(description="Target price (must be > entry)")
    stop: float = Field(description="Stop loss price (must be < entry)")
    
    @validator('target')
    def target_above_entry(cls, v, values):
        if 'entry' in values and v <= values['entry']:
            raise ValueError("Target must be above entry for bullish trade")
        return v
    
    @validator('stop')
    def stop_below_entry(cls, v, values):
        if 'entry' in values and v >= values['entry']:
            raise ValueError("Stop must be below entry for bullish trade")
        return v


class BullAgentOutput(BaseModel):
    """Validated output for BULL agent"""
    thesis: str = Field(
        min_length=30,
        max_length=500,
        description="Concise bullish argument with specific reasoning"
    )
    evidence: List[str] = Field(
        min_items=3,
        max_items=7,
        description="Specific evidence with numbers and data points"
    )
    key_levels: BullKeyLevels = Field(description="Entry, target, stop prices")
    catalysts: List[str] = Field(
        min_items=1,
        max_items=5,
        description="Bullish catalysts or drivers"
    )
    confidence: int = Field(
        ge=0,
        le=100,
        description="Confidence score 0-100"
    )
    timeframe: str = Field(
        pattern=r"^\d+-\d+ (days|weeks|months)$",
        description="Expected timeframe in format: '3-5 days'"
    )
    handoff_note: str = Field(
        max_length=200,
        description="Note to BEAR agent highlighting potential weaknesses"
    )
    
    @validator('evidence')
    def evidence_must_have_numbers(cls, v):
        """Ensure evidence includes specific data points"""
        for item in v:
            if not any(char.isdigit() for char in item):
                raise ValueError(f"Evidence must include specific numbers/data: {item}")
        return v


# ========== BEAR AGENT ==========

class BearKeyLevels(BaseModel):
    """Price levels for bearish trade"""
    short_entry: float = Field(description="Entry price for short position")
    target: float = Field(description="Target price (must be < entry)")
    stop: float = Field(description="Stop loss price (must be > entry)")
    
    @validator('target')
    def target_below_entry(cls, v, values):
        if 'short_entry' in values and v >= values['short_entry']:
            raise ValueError("Target must be below entry for bearish trade")
        return v
    
    @validator('stop')
    def stop_above_entry(cls, v, values):
        if 'short_entry' in values and v <= values['short_entry']:
            raise ValueError("Stop must be above entry for bearish trade")
        return v


class BearAgentOutput(BaseModel):
    """Validated output for BEAR agent"""
    thesis: str = Field(
        min_length=30,
        max_length=500,
        description="Concise bearish argument"
    )
    evidence: List[str] = Field(
        min_items=3,
        max_items=7,
        description="Specific bearish evidence with numbers"
    )
    challenges_to_bull: List[str] = Field(
        min_items=2,
        max_items=5,
        description="Direct challenges to BULL's thesis"
    )
    key_levels: BearKeyLevels = Field(description="Short entry, target, stop")
    risks: List[str] = Field(
        min_items=1,
        max_items=5,
        description="Downside risks"
    )
    confidence: int = Field(ge=0, le=100)
    timeframe: str = Field(pattern=r"^\d+-\d+ (days|weeks|months)$")
    handoff_note: str = Field(max_length=200)
    
    @validator('evidence')
    def evidence_must_have_numbers(cls, v):
        for item in v:
            if not any(char.isdigit() for char in item):
                raise ValueError(f"Evidence must include specific numbers: {item}")
        return v


# ========== CONSENSUS AGENT ==========

class ConsensusRecommendedAction(BaseModel):
    """Recommended action from consensus"""
    entry: float = Field(description="Recommended entry price")
    entry_logic: str = Field(
        min_length=20,
        description="Why this entry price"
    )
    target: float = Field(description="Target price")
    target_logic: str = Field(min_length=20)
    stop: float = Field(description="Stop loss price")
    stop_logic: str = Field(min_length=20)
    position_size: str = Field(
        description="Position sizing guidance (e.g., '50% of normal')"
    )


class ConsensusKeyFactor(BaseModel):
    """Weighted factor in consensus decision"""
    factor: str = Field(description="The factor being weighted")
    weight: Literal["high", "medium", "low"] = Field(description="Importance weight")
    direction: Literal["bullish", "bearish", "neutral"] = Field(description="Directional bias")


class ConsensusAgentOutput(BaseModel):
    """Validated output for CONSENSUS agent"""
    decision: Literal["LONG", "SHORT", "NO_TRADE"] = Field(description="Final decision")
    confidence: int = Field(ge=0, le=100, description="Overall confidence")
    bull_weight: float = Field(ge=0, le=1, description="Weight given to bull case (0-1)")
    bear_weight: float = Field(ge=0, le=1, description="Weight given to bear case (0-1)")
    rationale: str = Field(
        min_length=50,
        max_length=800,
        description="Detailed rationale for decision"
    )
    recommended_action: ConsensusRecommendedAction = Field(description="Specific trade recommendation")
    key_factors_weighted: List[ConsensusKeyFactor] = Field(
        min_items=3,
        description="Key factors with weights"
    )
    dissenting_view: str = Field(
        min_length=20,
        description="Acknowledgment of opposing view"
    )
    invalidation_conditions: List[str] = Field(
        min_items=2,
        description="Conditions that would invalidate this thesis"
    )
    timeframe: str = Field(pattern=r"^\d+-\d+ (days|weeks|months)$")
    handoff_note: str = Field(max_length=200, description="Note to RISK agent")
    
    @validator('bear_weight')
    def weights_sum_to_one(cls, v, values):
        if 'bull_weight' in values:
            total = v + values['bull_weight']
            if abs(total - 1.0) > 0.01:  # Allow small floating point error
                raise ValueError(f"Bull weight + Bear weight must equal 1.0, got {total}")
        return v


# ========== STRATEGY AGENT ==========

class StrategyAgentOutput(BaseModel):
    """Validated output for STRATEGY agent"""
    strategy_type: Literal[
        "momentum_breakout",
        "mean_reversion",
        "trend_following",
        "swing_trade",
        "scalp",
        "position_trade"
    ] = Field(description="Type of trading strategy")
    entry_criteria: List[str] = Field(
        min_items=2,
        description="Specific entry criteria"
    )
    exit_criteria: List[str] = Field(
        min_items=2,
        description="Specific exit criteria"
    )
    risk_reward_ratio: float = Field(
        ge=1.0,
        description="Risk/reward ratio (must be >= 1.0)"
    )
    win_probability: int = Field(
        ge=0,
        le=100,
        description="Estimated win probability %"
    )
    timeframe: str = Field(description="Strategy timeframe")
    key_indicators: List[str] = Field(
        min_items=2,
        description="Key indicators for this strategy"
    )


# ========== RISK AGENT ==========

class RiskAgentOutput(BaseModel):
    """Validated output for RISK agent"""
    risk_score: int = Field(
        ge=0,
        le=100,
        description="Overall risk score (0=low, 100=extreme)"
    )
    risk_factors: List[str] = Field(
        min_items=2,
        description="Identified risk factors"
    )
    position_size_recommendation: str = Field(
        description="Position sizing recommendation"
    )
    stop_loss_validation: str = Field(
        description="Validation of stop loss placement"
    )
    risk_mitigation: List[str] = Field(
        min_items=1,
        description="Risk mitigation strategies"
    )
    max_loss_percent: float = Field(
        ge=0,
        le=100,
        description="Maximum acceptable loss %"
    )
    approval: Literal["APPROVED", "APPROVED_WITH_CAUTION", "REJECTED"] = Field(
        description="Risk approval status"
    )
    reasoning: str = Field(min_length=30, description="Risk assessment reasoning")


# ========== SUMMARY AGENT ==========

class DebateSummary(BaseModel):
    """Summary of Bull vs Bear debate"""
    bull_case: str = Field(
        min_length=50,
        max_length=300,
        description="2-3 sentence summary of Bull's strongest points"
    )
    bear_case: str = Field(
        min_length=50,
        max_length=300,
        description="2-3 sentence summary of Bear's strongest points"
    )
    devils_advocate: str = Field(
        min_length=30,
        max_length=200,
        description="1-2 sentence summary of Devil's Advocate concerns"
    )
    consensus_decision: str = Field(
        min_length=50,
        max_length=300,
        description="Final consensus decision with confidence and rationale"
    )


class SummaryAgentOutput(BaseModel):
    """Validated output for SUMMARY agent"""
    headline_analysis: str = Field(description="Analysis of the headline")
    market_context: str = Field(description="Current market context")
    final_recommendation: Literal["LONG", "SHORT", "NO_TRADE"] = Field(
        description="Final trading recommendation"
    )
    confidence_score: int = Field(ge=0, le=100, description="Overall confidence")
    entry_price: Optional[float] = Field(None, description="Entry price if trade recommended")
    target_price: Optional[float] = Field(None, description="Target price")
    stop_loss: Optional[float] = Field(None, description="Stop loss price")
    position_size: Optional[str] = Field(None, description="Position sizing")
    timeframe: str = Field(description="Expected timeframe")
    key_risks: List[str] = Field(min_items=1, description="Key risks to monitor")
    debate_summary: DebateSummary = Field(description="Summary of agent debate")
    action_items: List[str] = Field(
        min_items=1,
        description="Actionable next steps for trader"
    )


# ========== ENHANCED PROMPTS ==========

ENHANCED_BULL_PROMPT = """You are the BULL agent - the optimistic advocate in a trading debate.

CRITICAL RULES:
1. ALWAYS cite specific numbers (e.g., "RSI 45", not "RSI is low")
2. ALWAYS provide 3-7 pieces of evidence
3. ALWAYS set target > entry and stop < entry for long trades
4. ALWAYS explain WHY each piece of evidence is bullish
5. ALWAYS include institutional signals (OBV, VWAP, Order Blocks, MFI)

GOOD EXAMPLE:
{
  "thesis": "Strong institutional accumulation with confirmed uptrend momentum",
  "evidence": [
    "OBV rising 15% over 10 days = clear institutional buying",
    "ADX 32.5 = strong, sustainable uptrend (>25 threshold)",
    "Price above VWAP $244.75 = bullish institutional positioning",
    "Bullish Order Block at $242-245 provides strong support",
    "MACD bullish crossover 2 days ago confirms momentum shift"
  ],
  "key_levels": {
    "entry": 245.00,
    "target": 260.00,
    "stop": 240.00
  },
  "catalysts": [
    "Institutional accumulation evident in OBV trend",
    "Earnings beat expected based on sector strength",
    "Bullish Order Block = smart money support zone"
  ],
  "confidence": 75,
  "timeframe": "3-5 days",
  "handoff_note": "Bear should address: Stochastic overbought at 85, Fair Value Gap resistance $246-248"
}

BAD EXAMPLE (DO NOT DO THIS):
{
  "thesis": "Stock looks good",  // ❌ Too vague!
  "evidence": ["Price is up", "Volume is high", "Trend is bullish"],  // ❌ No specific numbers!
  "key_levels": {"entry": 245, "target": 240, "stop": 250},  // ❌ Illogical levels!
  "confidence": 90  // ❌ Overconfident without strong evidence!
}

RESPOND ONLY WITH VALID JSON MATCHING THE BullAgentOutput SCHEMA.
"""

ENHANCED_CONSENSUS_PROMPT = """You are the CONSENSUS agent - the final decision synthesizer after Bull vs Bear debate.

CRITICAL RULES:
1. Summarize STRONGEST points from both Bull and Bear (not everything)
2. Incorporate Devil's Advocate concerns
3. Assign weights (bull_weight + bear_weight = 1.0)
4. Make CLEAR decision: LONG, SHORT, or NO_TRADE
5. Provide PRECISE entry/target/stop with logic for each
6. Give SPECIFIC confidence score based on evidence quality
7. Acknowledge dissenting opinion
8. List invalidation conditions

GOOD EXAMPLE:
{
  "decision": "LONG",
  "confidence": 68,
  "bull_weight": 0.60,
  "bear_weight": 0.40,
  "rationale": "Bull's institutional accumulation (OBV +15%) and trend strength (ADX 32) outweigh Bear's overbought concerns. Devil's Advocate correctly identifies earnings risk as critical factor both sides underweighted. Fair Value Gap resistance at $246-248 is valid concern.",
  "recommended_action": {
    "entry": 244.75,
    "entry_logic": "Wait for pullback to VWAP support ($244.75), do NOT chase at Fair Value Gap resistance $246-248",
    "target": 258.00,
    "target_logic": "Conservative target below recent high $252, accounts for earnings volatility",
    "stop": 241.50,
    "stop_logic": "Below bullish Order Block ($242-245) invalidates Bull thesis",
    "position_size": "50% of normal (reduced for earnings risk in 3 days)"
  },
  "key_factors_weighted": [
    {"factor": "OBV rising 15% (institutional accumulation)", "weight": "high", "direction": "bullish"},
    {"factor": "ADX 32.5 (strong trend)", "weight": "high", "direction": "bullish"},
    {"factor": "Earnings in 3 days (binary risk)", "weight": "high", "direction": "neutral"},
    {"factor": "Stochastic 85 overbought", "weight": "medium", "direction": "bearish"}
  ],
  "dissenting_view": "Bear's overbought Stochastic concern valid for 1-2 day pullback",
  "invalidation_conditions": [
    "Break below $241 Order Block support",
    "Bearish earnings surprise",
    "Stochastic bearish crossover with volume spike"
  ],
  "timeframe": "3-7 days",
  "handoff_note": "RISK agent: Focus on earnings volatility, position sizing for binary event"
}

RESPOND ONLY WITH VALID JSON MATCHING THE ConsensusAgentOutput SCHEMA.
"""

# Export all models and prompts
__all__ = [
    'BullAgentOutput',
    'BearAgentOutput',
    'ConsensusAgentOutput',
    'StrategyAgentOutput',
    'RiskAgentOutput',
    'SummaryAgentOutput',
    'ENHANCED_BULL_PROMPT',
    'ENHANCED_CONSENSUS_PROMPT'
]
