import React from 'react';
import { Calendar, MapPin, Award, CheckCircle2 } from 'lucide-react';
import { MatchDetails } from '../types/match';
import { getTeamColor } from '../utils/teamVisualConfig';

interface MatchHeaderProps {
  match: MatchDetails;
  winner: string | null;
  status: string;
}

export const MatchHeader: React.FC<MatchHeaderProps> = ({ match, winner, status }) => {
  const isCompleted = status === 'completed' || !!winner;

  return (
    <div className="match-header-panel">
      <div className="match-title-row">
        <div className="match-title">
          <h2>
            <span className="team-highlight team-1" style={{ color: getTeamColor(match.team_1) }}>{match.team_1}</span>
            <span className="vs-badge">VS</span>
            <span className="team-highlight team-2" style={{ color: getTeamColor(match.team_2) }}>{match.team_2}</span>
          </h2>
        </div>

        <div className="match-status-badge-container">
          {isCompleted ? (
            <span className="badge badge-success">
              <CheckCircle2 size={12} /> MATCH COMPLETED
            </span>
          ) : status === 'active' ? (
            <span className="badge badge-live">
              <span className="live-dot"></span> LIVE IN-PLAY
            </span>
          ) : (
            <span className="badge badge-ready">T0 PRE-MATCH</span>
          )}
        </div>
      </div>

      <div className="match-meta-grid">
        <div className="meta-pill">
          <Calendar size={13} />
          <span>{match.date} ({match.season})</span>
        </div>

        <div className="meta-pill" title={match.venue}>
          <MapPin size={13} />
          <span>{match.venue}, {match.city}</span>
        </div>

        {match.toss_winner && (
          <div className="meta-pill toss-pill">
            <Award size={13} />
            <span>
              Toss: <strong>{match.toss_winner}</strong> chose to{' '}
              <strong>{match.toss_decision}</strong>
            </span>
          </div>
        )}

        {isCompleted && winner && (
          <div className="meta-pill winner-pill">
            <span>
              🏆 <strong>Result:</strong> {winner} won
            </span>
          </div>
        )}
      </div>
    </div>
  );
};
