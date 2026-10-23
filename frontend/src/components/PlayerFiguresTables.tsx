import React from 'react';
import { BattingFigure, BowlingFigure } from '../types/simulation';
import { getTeamColor } from '../utils/teamVisualConfig';

interface PlayerFiguresTablesProps {
  batting?: BattingFigure[];
  bowling?: BowlingFigure[];
  battingTeam?: string;
  bowlingTeam?: string;
  currentStriker?: string;
  currentBowler?: string;
}

export const formatStrikeRate = (sr: number | undefined, balls: number | undefined): string => {
  if (!balls || balls <= 0 || sr === undefined || !Number.isFinite(sr)) {
    return '0.00';
  }
  return sr.toFixed(2);
};

export const formatEconomy = (econ: number | undefined, overs: string | undefined): string => {
  if (!overs || overs === '0.0' || overs === '0' || econ === undefined || !Number.isFinite(econ)) {
    return '0.00';
  }
  return econ.toFixed(2);
};

export const PlayerFiguresTables: React.FC<PlayerFiguresTablesProps> = ({
  batting = [],
  bowling = [],
  battingTeam,
  bowlingTeam,
  currentStriker,
  currentBowler,
}) => {
  const battingColor = getTeamColor(battingTeam);
  const bowlingColor = getTeamColor(bowlingTeam);

  return (
    <div className="figures-tables-container">
      {/* Batting Section */}
      <div
        className="figures-section batting-section"
        style={{ borderTop: `2px solid ${battingColor}` }}
      >
        <div className="figures-section-header">
          <h5>
            <span style={{ color: battingColor }}>BATTING</span>{' '}
            {battingTeam ? (
              <span className="team-sub" style={{ color: battingColor }}>
                — {battingTeam}
              </span>
            ) : (
              ''
            )}
          </h5>
        </div>

        <div className="figures-table-scroll">
          <table className="figures-table batting-table">
            <thead>
              <tr>
                <th className="th-player">BATSMAN</th>
                <th className="th-num">R</th>
                <th className="th-num">B</th>
                <th className="th-num">4s</th>
                <th className="th-num">6s</th>
                <th className="th-num">SR</th>
              </tr>
            </thead>
            <tbody>
              {batting && batting.length > 0 ? (
                batting.map((b) => {
                  const isStriker =
                    b.is_striker || (currentStriker && b.player_name === currentStriker);
                  return (
                    <tr
                      key={b.player_name}
                      className={`figure-row ${isStriker ? 'striker-highlight-row' : ''}`}
                      style={
                        isStriker
                          ? {
                              backgroundColor: `${battingColor}18`,
                              borderLeft: `3px solid ${battingColor}`,
                            }
                          : undefined
                      }
                    >
                      <td className="td-player">
                        <span className="player-name-cell" style={{ color: '#ffffff' }}>
                          {b.player_name}
                          {isStriker && (
                            <span className="striker-badge-mini" style={{ color: battingColor }}>
                              ★
                            </span>
                          )}
                        </span>
                      </td>
                      <td className="td-num runs-cell" style={{ color: '#ffffff' }}>
                        {b.runs ?? 0}
                      </td>
                      <td className="td-num" style={{ color: '#ffffff' }}>
                        {b.balls ?? 0}
                      </td>
                      <td className="td-num fours-cell">{b.fours ?? 0}</td>
                      <td className="td-num sixes-cell">{b.sixes ?? 0}</td>
                      <td className="td-num sr-cell">
                        {formatStrikeRate(b.strike_rate, b.balls)}
                      </td>
                    </tr>
                  );
                })
              ) : (
                <tr>
                  <td colSpan={6} className="empty-figures-cell">
                    Awaiting batting data...
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Bowling Section */}
      <div
        className="figures-section bowling-section"
        style={{ borderTop: `2px solid ${bowlingColor}` }}
      >
        <div className="figures-section-header">
          <h5>
            <span style={{ color: bowlingColor }}>BOWLING</span>{' '}
            {bowlingTeam ? (
              <span className="team-sub" style={{ color: bowlingColor }}>
                — {bowlingTeam}
              </span>
            ) : (
              ''
            )}
          </h5>
        </div>

        <div className="figures-table-scroll">
          <table className="figures-table bowling-table">
            <thead>
              <tr>
                <th className="th-player">BOWLER</th>
                <th className="th-num">O</th>
                <th className="th-num">M</th>
                <th className="th-num">R</th>
                <th className="th-num">W</th>
                <th className="th-num">ECON</th>
              </tr>
            </thead>
            <tbody>
              {bowling && bowling.length > 0 ? (
                bowling.map((bow) => {
                  const isBowler =
                    bow.is_current_bowler || (currentBowler && bow.player_name === currentBowler);
                  return (
                    <tr
                      key={bow.player_name}
                      className={`figure-row ${isBowler ? 'bowler-highlight-row' : ''}`}
                      style={
                        isBowler
                          ? {
                              backgroundColor: `${bowlingColor}18`,
                              borderLeft: `3px solid ${bowlingColor}`,
                            }
                          : undefined
                      }
                    >
                      <td className="td-player">
                        <span className="player-name-cell" style={{ color: '#ffffff' }}>
                          {bow.player_name}
                          {isBowler && (
                            <span className="bowler-badge-mini" style={{ color: bowlingColor }}>
                              🛡
                            </span>
                          )}
                        </span>
                      </td>
                      <td className="td-num overs-cell" style={{ color: '#ffffff' }}>
                        {bow.overs || '0.0'}
                      </td>
                      <td className="td-num maidens-cell" style={{ color: '#ffffff' }}>
                        {bow.maidens ?? 0}
                      </td>
                      <td className="td-num" style={{ color: '#ffffff' }}>
                        {bow.runs ?? 0}
                      </td>
                      <td className="td-num wickets-cell" style={{ color: '#ffffff' }}>
                        {bow.wickets ?? 0}
                      </td>
                      <td className="td-num econ-cell">
                        {formatEconomy(bow.economy, bow.overs)}
                      </td>
                    </tr>
                  );
                })
              ) : (
                <tr>
                  <td colSpan={6} className="empty-figures-cell">
                    Awaiting bowling data...
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
