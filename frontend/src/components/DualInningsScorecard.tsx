import React, { useState, useEffect } from 'react';
import { MatchStateSnapshot } from '../types/simulation';
import { getTeamColor } from '../utils/teamVisualConfig';

interface DualInningsScorecardProps {
  state: MatchStateSnapshot;
  team1Name: string;
  team2Name: string;
}

export const DualInningsScorecard: React.FC<DualInningsScorecardProps> = ({
  state,
  team1Name,
  team2Name,
}) => {
  // Default active tab to current innings, but allow user to switch freely between 1 and 2
  const [activeTab, setActiveTab] = useState<1 | 2>(state.innings === 2 ? 2 : 1);

  // Auto-switch tab to innings 2 when innings transitions to 2 if user hasn't manually switched
  useEffect(() => {
    if (state.innings === 2 && activeTab === 1 && (state.innings_2?.legal_balls || 0) <= 6) {
      setActiveTab(2);
    }
  }, [state.innings]);

  const inn1 = state.innings_1;
  const inn2 = state.innings_2;

  // Fallback to active state if innings_1 or innings_2 not yet populated
  const inn1Team = inn1?.team || (state.innings === 1 ? state.batting_team : state.bowling_team) || team1Name;
  const inn2Team = inn2?.team || (state.innings === 2 ? state.batting_team : state.bowling_team) || team2Name;

  const activeScorecard = activeTab === 1 ? inn1 : inn2;
  const activeTeamName = activeTab === 1 ? inn1Team : inn2Team;
  const activeColor = getTeamColor(activeTeamName);

  const battingList = activeScorecard?.batting || (activeTab === state.innings ? state.batting : []) || [];
  const bowlingList = activeScorecard?.bowling || (activeTab === state.innings ? state.bowling : []) || [];

  const displayScore = activeScorecard ? activeScorecard.score : (activeTab === state.innings ? state.score : 0);
  const displayWickets = activeScorecard ? activeScorecard.wickets : (activeTab === state.innings ? state.wickets : 0);
  const displayOvers = activeScorecard ? activeScorecard.overs : (activeTab === state.innings ? state.overs : '0.0');

  const isPreInnings = activeTab === 2 && state.innings === 1;

  return (
    <div className="analytics-card dual-scorecard-card">
      <div className="dual-scorecard-header">
        <div className="scorecard-tabs">
          <button
            type="button"
            className={`scorecard-tab ${activeTab === 1 ? 'active' : ''}`}
            onClick={() => setActiveTab(1)}
            style={activeTab === 1 ? { borderBottomColor: getTeamColor(inn1Team) } : {}}
          >
            <span className="tab-title">INNINGS 1</span>
            <span className="tab-team" style={{ color: getTeamColor(inn1Team) }}>
              {inn1Team}
            </span>
            <span className="tab-badge">
              {inn1 ? `${inn1.score}/${inn1.wickets}` : (state.innings === 1 ? `${state.score}/${state.wickets}` : 'Final')}
            </span>
          </button>

          <button
            type="button"
            className={`scorecard-tab ${activeTab === 2 ? 'active' : ''}`}
            onClick={() => setActiveTab(2)}
            style={activeTab === 2 ? { borderBottomColor: getTeamColor(inn2Team) } : {}}
          >
            <span className="tab-title">INNINGS 2</span>
            <span className="tab-team" style={{ color: getTeamColor(inn2Team) }}>
              {inn2Team}
            </span>
            <span className="tab-badge">
              {isPreInnings
                ? 'Yet to bat'
                : inn2
                ? `${inn2.score}/${inn2.wickets}`
                : (state.innings === 2 ? `${state.score}/${state.wickets}` : '0/0')}
            </span>
          </button>
        </div>

        {/* Selected Innings Score Summary */}
        <div className="scorecard-total-summary">
          <div className="inn-team-pill" style={{ borderColor: activeColor }}>
            <span className="inn-team-dot" style={{ backgroundColor: activeColor }} />
            <span className="inn-team-label">{activeTeamName}</span>
          </div>
          <div className="inn-numbers">
            <span className="inn-score">
              {displayScore}/{displayWickets}
            </span>
            <span className="inn-overs">({displayOvers} ov)</span>
          </div>
        </div>
      </div>

      {isPreInnings ? (
        <div className="pre-innings-notice">
          <p>
            <strong>{inn2Team}</strong> has not started their innings yet in this historical replay position.
          </p>
          <span className="text-muted">Innings 2 figures will unlock when replay reaches Innings 2.</span>
        </div>
      ) : (
        <div className="scorecard-tables-container">
          {/* Batting Section */}
          <div className="scorecard-subtable-section">
            <div className="subtable-header">
              <h4>BATTING</h4>
            </div>
            <div className="table-responsive">
              <table className="scorecard-data-table">
                <thead>
                  <tr>
                    <th className="th-name">Batter</th>
                    <th className="th-dismissal">Dismissal</th>
                    <th className="th-num">R</th>
                    <th className="th-num">B</th>
                    <th className="th-num">4s</th>
                    <th className="th-num">6s</th>
                    <th className="th-num">SR</th>
                  </tr>
                </thead>
                <tbody>
                  {battingList.length === 0 ? (
                    <tr>
                      <td colSpan={7} className="text-center text-muted py-3">
                        Awaiting batting figures for this innings...
                      </td>
                    </tr>
                  ) : (
                    battingList.map((b) => (
                      <tr
                        key={b.player_name}
                        className={b.is_striker && activeTab === state.innings ? 'striker-active-row' : ''}
                      >
                        <td className="td-name">
                          <span className="player-link-name">{b.player_name}</span>
                          {b.is_striker && activeTab === state.innings && <span className="on-strike-star">*</span>}
                        </td>
                        <td className="td-dismissal">
                          <span className={`dismissal-tag ${b.dismissal && b.dismissal !== 'not out' ? 'out' : 'not-out'}`}>
                            {b.dismissal || 'not out'}
                          </span>
                        </td>
                        <td className="td-num font-bold text-white">{b.runs}</td>
                        <td className="td-num">{b.balls}</td>
                        <td className="td-num text-amber">{b.fours}</td>
                        <td className="td-num text-cyan">{b.sixes}</td>
                        <td className="td-num font-mono">{b.strike_rate.toFixed(1)}</td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>

          {/* Bowling Section */}
          <div className="scorecard-subtable-section mt-4">
            <div className="subtable-header">
              <h4>BOWLING</h4>
            </div>
            <div className="table-responsive">
              <table className="scorecard-data-table">
                <thead>
                  <tr>
                    <th className="th-name">Bowler</th>
                    <th className="th-num">O</th>
                    <th className="th-num">M</th>
                    <th className="th-num">R</th>
                    <th className="th-num">W</th>
                    <th className="th-num">ECON</th>
                  </tr>
                </thead>
                <tbody>
                  {bowlingList.length === 0 ? (
                    <tr>
                      <td colSpan={6} className="text-center text-muted py-3">
                        Awaiting bowling figures for this innings...
                      </td>
                    </tr>
                  ) : (
                    bowlingList.map((bow) => (
                      <tr
                        key={bow.player_name}
                        className={bow.is_current_bowler && activeTab === state.innings ? 'bowler-active-row' : ''}
                      >
                        <td className="td-name">
                          <span className="player-link-name">{bow.player_name}</span>
                          {bow.is_current_bowler && activeTab === state.innings && <span className="bowling-icon-tag">●</span>}
                        </td>
                        <td className="td-num font-mono">{bow.overs}</td>
                        <td className="td-num">{bow.maidens}</td>
                        <td className="td-num">{bow.runs}</td>
                        <td className="td-num font-bold text-green">{bow.wickets}</td>
                        <td className="td-num font-mono">{bow.economy.toFixed(2)}</td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
