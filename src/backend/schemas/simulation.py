"""
Simulation session, state snapshot, and progression schemas.
"""

from typing import Optional, List
from pydantic import BaseModel, Field

from .prediction import PredictionResponse, ProbabilityPoint


class SimulationCreateRequest(BaseModel):
    match_id: str = Field(..., description="ID of the historical match to simulate")


class BattingFigure(BaseModel):
    player_name: str = Field(..., description="Batter name")
    dismissal: Optional[str] = Field("not out", description="Dismissal method (e.g. 'not out', 'caught', 'bowled', 'lbw', 'run out')")
    runs: int = Field(..., description="Runs scored off bat")
    balls: int = Field(..., description="Balls faced (excluding wides)")
    fours: int = Field(..., description="Number of 4s scored")
    sixes: int = Field(..., description="Number of 6s scored")
    strike_rate: float = Field(..., description="Strike rate (runs / balls * 100)")
    is_striker: bool = Field(False, description="Whether this batter is currently on strike")


class BowlingFigure(BaseModel):
    player_name: str = Field(..., description="Bowler name")
    overs: str = Field(..., description="Overs bowled in cricket notation (e.g. '2.0', '2.3')")
    maidens: int = Field(0, description="Completed 6-ball maiden overs with 0 bowler runs conceded")
    runs: int = Field(..., description="Runs conceded by bowler (bat runs + wides + noballs)")
    wickets: int = Field(..., description="Bowler-credited wickets")
    economy: float = Field(..., description="Economy rate (runs / overs)")
    is_current_bowler: bool = Field(False, description="Whether this bowler is currently bowling")


class InningsScorecard(BaseModel):
    innings: int = Field(..., description="Innings number (1 or 2)")
    team: Optional[str] = Field(None, description="Batting team name")
    score: int = Field(0, description="Total runs scored")
    wickets: int = Field(0, description="Wickets lost")
    overs: str = Field("0.0", description="Cricket overs string representation e.g. '19.4'")
    legal_balls: int = Field(0, description="Legal balls bowled in innings")
    batting: List[BattingFigure] = Field(default_factory=list, description="Batting scorecard figures")
    bowling: List[BowlingFigure] = Field(default_factory=list, description="Bowling scorecard figures")


class MatchStateSnapshot(BaseModel):
    innings: int = Field(..., description="Current innings (1 or 2)")
    score: int = Field(..., description="Current cumulative runs in current innings")
    wickets: int = Field(..., description="Current wickets lost in current innings")
    legal_balls: int = Field(..., description="Total legal balls bowled in current innings")
    overs: str = Field(..., description="Overs string representation (e.g. '15.3')")
    overs_completed: float = Field(..., description="Decimal overs completed (e.g. 15.5)")
    current_run_rate: float = Field(..., description="Current run rate (runs per over)")
    target_runs: Optional[int] = Field(None, description="Target runs for Innings 2 chase (None in Innings 1)")
    runs_needed: Optional[int] = Field(None, description="Runs needed to win (None in Innings 1)")
    required_run_rate: Optional[float] = Field(None, description="Required run rate (None in Innings 1)")
    rrr_crr_differential: Optional[float] = Field(None, description="RRR minus CRR differential")
    striker: str = Field(..., description="Batter on strike")
    non_striker: str = Field(..., description="Batter at non-striker's end")
    bowler: str = Field(..., description="Active bowler")
    batting_team: Optional[str] = Field(None, description="Name of current batting team")
    bowling_team: Optional[str] = Field(None, description="Name of current bowling team")
    batting: List[BattingFigure] = Field(default_factory=list, description="List of batting figures for current innings")
    bowling: List[BowlingFigure] = Field(default_factory=list, description="List of bowling figures for current innings")
    innings_1: Optional[InningsScorecard] = Field(None, description="Innings 1 scorecard up to current replay position")
    innings_2: Optional[InningsScorecard] = Field(None, description="Innings 2 scorecard up to current replay position")


class SimulationSeekRequest(BaseModel):
    event_index: int = Field(..., ge=0, description="Target event index to navigate replay to")



class DeliverySnapshot(BaseModel):
    innings: int = Field(..., description="Innings number")
    ball: float = Field(..., description="Authoritative ball number in over format (e.g. 0.5)")
    over: Optional[int] = Field(None, description="Authoritative over number (0-19)")
    ball_in_over: Optional[int] = Field(None, description="Authoritative ball number within over (1-6)")
    display_ball: Optional[str] = Field(None, description="Authoritative cricket ball string notation (e.g. '0.5')")
    event_index: Optional[int] = Field(None, description="Replay event sequence index")
    is_legal: bool = Field(..., description="Whether delivery was legal (not wide, not noball)")
    batting_team: str = Field(..., description="Team batting")
    bowling_team: str = Field(..., description="Team bowling")
    striker: str = Field(..., description="Striker on this delivery")
    non_striker: str = Field(..., description="Non-striker on this delivery")
    bowler: str = Field(..., description="Bowler for this delivery")
    runs_off_bat: int = Field(..., description="Runs scored off the bat")
    extras: int = Field(..., description="Extra runs conceded")
    wides: int = Field(0, description="Wide runs")
    noballs: int = Field(0, description="No-ball runs")
    wicket_type: Optional[str] = Field(None, description="Dismissal type if wicket occurred")
    player_dismissed: Optional[str] = Field(None, description="Name of player dismissed if any")


class MatchIntelligenceSnapshot(BaseModel):
    current_team: Optional[str] = Field(None, description="Leading team for win probability")
    current_win_probability: Optional[float] = Field(None, description="Leading team win probability percentage")
    trend_probability: Optional[float] = Field(None, description="Trend probability over the window")
    trend_team: Optional[str] = Field(None, description="Team the trend applies to")
    volatility: Optional[float] = Field(None, description="Match volatility score 0-100")
    criticality: Optional[float] = Field(None, description="Criticality of next ball 0-100")
    turning_point_over: Optional[str] = Field(None, description="Over when biggest turning point happened")
    turning_point_event: Optional[str] = Field(None, description="Event of biggest turning point")
    turning_point_wpa: Optional[float] = Field(None, description="WPA of turning point")
    turning_point_team: Optional[str] = Field(None, description="Team that gained at turning point")
    influential_player: Optional[str] = Field(None, description="Most influential player so far")
    influential_player_wpa: Optional[float] = Field(None, description="Most influential player WPA")
    historical_similarity: Optional[str] = Field(None, description="Similar historical outcome percentage")



class SimulationCreateResponse(BaseModel):
    session_id: str = Field(..., description="Unique simulation session UUID")
    match_id: str = Field(..., description="Target match identifier")
    status: str = Field(..., description="Session state: 'ready', 'active', 'completed', 'error'")
    current_event_index: int = Field(0, description="Current event index (0 = T0 pre-match anchor)")
    total_events: int = Field(..., description="Total events in this match (including T0)")
    current_prediction: PredictionResponse = Field(..., description="Pre-match T0 win probability prediction")


class SimulationStepResponse(BaseModel):
    session_id: str = Field(..., description="Simulation session UUID")
    match_id: str = Field(..., description="Match identifier")
    event_index: int = Field(..., description="Current event index")
    total_events: int = Field(..., description="Total events in this match")
    status: str = Field(..., description="Current session state: 'ready', 'active', 'completed'")
    delivery: Optional[DeliverySnapshot] = Field(None, description="Last recorded delivery event (None at T0)")
    state: MatchStateSnapshot = Field(..., description="Instantaneous match state")
    prediction: PredictionResponse = Field(..., description="Latest model probability prediction")
    match_intelligence: Optional[MatchIntelligenceSnapshot] = Field(None, description="Match analytics context")


class SimulationRunResponse(BaseModel):
    session_id: str = Field(..., description="Simulation session UUID")
    match_id: str = Field(..., description="Match identifier")
    status: str = Field(..., description="Final session state ('completed')")
    events_processed: int = Field(..., description="Number of events processed in this run call")
    total_events: int = Field(..., description="Total events in match")
    winner: str = Field(..., description="Official match winner")
    final_state: MatchStateSnapshot = Field(..., description="Final match state at conclusion")
    final_prediction: PredictionResponse = Field(..., description="Final settled prediction")
    delivery: Optional[DeliverySnapshot] = Field(None, description="Final delivery event if any")
    match_intelligence: Optional[MatchIntelligenceSnapshot] = Field(None, description="Final match intelligence snapshot")
    history: Optional[List[ProbabilityPoint]] = Field(None, description="Complete probability trajectory if requested")
