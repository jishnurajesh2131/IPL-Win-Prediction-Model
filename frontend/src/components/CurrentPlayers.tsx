import React from 'react';
import { User, Shield } from 'lucide-react';
import { MatchStateSnapshot } from '../types/simulation';
import { PlayerFiguresTables } from './PlayerFiguresTables';
import { getTeamColor } from '../utils/teamVisualConfig';

interface CurrentPlayersProps {
  state: MatchStateSnapshot;
  team1Name?: string;
  team2Name?: string;
}

export const CurrentPlayers: React.FC<CurrentPlayersProps> = ({ state, team1Name, team2Name }) => {
  const battingTeam = state.batting_team || (state.innings === 2 ? team2Name : team1Name) || '';
  const bowlingTeam = state.bowling_team || (state.innings === 2 ? team1Name : team2Name) || '';
  const battingColor = getTeamColor(battingTeam);
  const bowlingColor = getTeamColor(bowlingTeam);

  return (
    <div className="players-card">
      <div className="players-header">
        <h4>ACTIVE AT CREASE</h4>
      </div>

      <div className="players-grid">
        {/* Batters */}
        <div className="player-column">
          {/* Striker: team-colored border, team-colored icon/accent, subtle team-colored background */}
          <div
            className="player-slot active-striker"
            style={{
              borderColor: battingColor,
              backgroundColor: `${battingColor}15`,
              boxShadow: `0 0 12px ${battingColor}25`,
            }}
          >
            <div
              className="player-avatar striker-icon"
              style={{
                backgroundColor: `${battingColor}28`,
                color: battingColor,
              }}
            >
              <User size={15} />
            </div>
            <div className="player-details">
              <span className="player-role-tag striker-tag" style={{ color: battingColor }}>
                ★ STRIKER
              </span>
              <span className="player-name-text" style={{ color: '#ffffff' }}>
                {state.striker}
              </span>
            </div>
          </div>

          {/* Non-Striker: subtle batting-team color */}
          <div
            className="player-slot non-striker"
            style={{
              borderLeft: `3px solid ${battingColor}80`,
              backgroundColor: `${battingColor}08`,
            }}
          >
            <div
              className="player-avatar"
              style={{
                backgroundColor: `${battingColor}18`,
                color: battingColor,
              }}
            >
              <User size={15} />
            </div>
            <div className="player-details">
              <span className="player-role-tag" style={{ color: `${battingColor}aa` }}>
                NON-STRIKER
              </span>
              <span className="player-name-text" style={{ color: '#ffffff' }}>
                {state.non_striker}
              </span>
            </div>
          </div>
        </div>

        {/* Bowler: bowling-team-colored border, bowling-team-colored icon/accent, subtle bowling-team-colored background */}
        <div className="player-column bowler-column">
          <div
            className="player-slot active-bowler"
            style={{
              borderColor: bowlingColor,
              backgroundColor: `${bowlingColor}15`,
              boxShadow: `0 0 12px ${bowlingColor}25`,
            }}
          >
            <div
              className="player-avatar bowler-icon"
              style={{
                backgroundColor: `${bowlingColor}28`,
                color: bowlingColor,
              }}
            >
              <Shield size={15} />
            </div>
            <div className="player-details">
              <span className="player-role-tag bowler-tag" style={{ color: bowlingColor }}>
                BOWLER
              </span>
              <span className="player-name-text" style={{ color: '#ffffff' }}>
                {state.bowler}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Batting & Bowling Figures Tables */}
      <PlayerFiguresTables
        batting={state.batting}
        bowling={state.bowling}
        battingTeam={battingTeam}
        bowlingTeam={bowlingTeam}
        currentStriker={state.striker}
        currentBowler={state.bowler}
      />
    </div>
  );
};
