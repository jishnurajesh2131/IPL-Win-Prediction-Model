import React from 'react';
import { Target, Clock } from 'lucide-react';
import { MatchStateSnapshot } from '../types/simulation';
import { getTeamColor } from '../utils/teamVisualConfig';

interface ScoreCardProps {
  state: MatchStateSnapshot;
  team1Name: string;
  team2Name: string;
}

export const ScoreCard: React.FC<ScoreCardProps> = ({ state, team1Name, team2Name }) => {
  const isChasing = state.innings === 2 && state.target_runs !== null;
  const battingTeam = state.batting_team || (state.innings === 1 ? team1Name : team2Name);
  const bowlingTeam = state.bowling_team || (state.innings === 1 ? team2Name : team1Name);

  const ballsRemaining = Math.max(0, 120 - state.legal_balls);

  return (
    <div className="scorecard-panel">
      <div className="scorecard-top-row">
        <div className="batting-team-info">
          <span className="innings-tag">INNINGS {state.innings}</span>
          <h2 className="current-batting-team" style={{ color: getTeamColor(battingTeam) }}>{battingTeam}</h2>
          <span className="fielding-team-sub" style={{ color: getTeamColor(bowlingTeam) }}>Bowling: {bowlingTeam}</span>
        </div>

        <div className="primary-score-display">
          <div className="score-numbers">
            <span className="score-runs">{state.score}</span>
            <span className="score-slash">/</span>
            <span className="score-wickets">{state.wickets}</span>
          </div>
          <div className="overs-badge">
            <Clock size={13} />
            <span>{state.overs} overs</span>
          </div>
        </div>
      </div>

      {/* Metrics Row: CRR, RRR, Target */}
      <div className="score-metrics-grid">
        <div className="metric-cell">
          <span className="metric-label">CURRENT RUN RATE</span>
          <span className="metric-val">{state.current_run_rate.toFixed(2)}</span>
        </div>

        {isChasing && state.target_runs && (
          <>
            <div className="metric-cell">
              <span className="metric-label">TARGET</span>
              <span className="metric-val highlight-target">{state.target_runs}</span>
            </div>

            <div className="metric-cell">
              <span className="metric-label">RUNS NEEDED</span>
              <span className="metric-val highlight-needed">
                {state.runs_needed} <span className="sub-unit">off {ballsRemaining}b</span>
              </span>
            </div>

            {state.required_run_rate !== null && (
              <div className="metric-cell">
                <span className="metric-label">REQ RUN RATE</span>
                <span className="metric-val">{state.required_run_rate.toFixed(2)}</span>
              </div>
            )}

            {state.rrr_crr_differential !== null && (
              <div className="metric-cell">
                <span className="metric-label">RRR - CRR</span>
                <span className={`metric-val ${state.rrr_crr_differential > 0 ? 'text-danger' : 'text-success'}`}>
                  {state.rrr_crr_differential > 0 ? `+${state.rrr_crr_differential.toFixed(2)}` : state.rrr_crr_differential.toFixed(2)}
                </span>
              </div>
            )}
          </>
        )}
      </div>

      {/* Chase Progress Banner during Innings 2 */}
      {isChasing && state.runs_needed !== null && (
        <div className="chase-progress-banner">
          <Target size={14} />
          <span>
            {state.runs_needed <= 0
              ? '🎯 Target Reached!'
              : `Need ${state.runs_needed} runs from ${ballsRemaining} balls to win`}
          </span>
        </div>
      )}
    </div>
  );
};
