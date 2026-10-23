import React from 'react';
import { X, Search, Calendar, MapPin, Trophy, Play } from 'lucide-react';
import { MatchSummary } from '../types/match';

interface MatchSelectorProps {
  isOpen: boolean;
  onClose: () => void;
  matches: MatchSummary[];
  total: number;
  loading: boolean;
  selectedSeason: string;
  onSelectSeason: (season: string) => void;
  selectedTeam: string;
  onSelectTeam: (team: string) => void;
  onChooseMatch: (matchId: string) => void;
  availableSeasons?: string[];
  availableTeams?: string[];
  mode?: 'historical' | 'evaluation';
}

const DEFAULT_SEASONS = [
  '2024', '2023', '2022', '2021', '2020/21', '2019', '2018',
  '2017', '2016', '2015', '2014', '2013', '2012', '2011',
  '2009/10', '2009', '2007/08'
];

export const MatchSelector: React.FC<MatchSelectorProps> = ({
  isOpen,
  onClose,
  matches,
  total,
  loading,
  selectedSeason,
  onSelectSeason,
  selectedTeam,
  onSelectTeam,
  onChooseMatch,
  availableSeasons = DEFAULT_SEASONS,
  availableTeams = [],
  mode = 'historical',
}) => {
  if (!isOpen) return null;

  let seasonsList = availableSeasons && availableSeasons.length > 0 ? availableSeasons : DEFAULT_SEASONS;
  
  if (mode === 'evaluation') {
    seasonsList = seasonsList.filter(s => {
      const year = parseInt(s.substring(0, 4), 10);
      return year >= 2024;
    });
  } else if (mode === 'historical') {
    seasonsList = seasonsList.filter(s => {
      const year = parseInt(s.substring(0, 4), 10);
      return year <= 2023;
    });
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div>
            <h2>Select Historical IPL Match</h2>
            <p className="text-muted">Choose from 1,243 clean Cricsheet matches to simulate live</p>
          </div>
          <button className="btn-icon" onClick={onClose}>
            <X size={20} />
          </button>
        </div>

        {/* Filters Bar */}
        <div className="filters-bar">
          <div className="filter-group">
            <label htmlFor="season-select">Season</label>
            <select
              id="season-select"
              value={selectedSeason}
              onChange={(e) => onSelectSeason(e.target.value)}
              className="select-input"
            >
              {seasonsList.map((s) => (
                <option key={s} value={s}>
                  {s} Season
                </option>
              ))}
            </select>
          </div>

          <div className="filter-group">
            <label htmlFor="team-select">Team Filter</label>
            <select
              id="team-select"
              value={selectedTeam}
              onChange={(e) => onSelectTeam(e.target.value)}
              className="select-input"
            >
              <option value="">All Teams</option>
              {availableTeams.filter((t) => t !== '').map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
          </div>

          <div className="total-badge">
            <span>{total} matches found</span>
          </div>
        </div>

        {/* Matches List */}
        <div className="matches-grid-container">
          {loading ? (
            <div className="loading-state-box">
              <div className="spinner"></div>
              <span>Loading matches...</span>
            </div>
          ) : matches.length === 0 ? (
            <div className="empty-state-box">
              <Search size={32} className="text-muted" />
              <p>No matches found matching the selected filters.</p>
            </div>
          ) : (
            <div className="matches-grid">
              {matches.map((m) => (
                <div key={m.match_id} className="match-card">
                  <div className="match-card-header">
                    <span className="match-season-badge">{m.season}</span>
                    <span className="match-date">
                      <Calendar size={12} /> {m.date}
                    </span>
                  </div>

                  <div className="match-card-teams">
                    <div className="team-row">
                      <span className="team-name team-1">{m.team_1}</span>
                    </div>
                    <div className="vs-divider">vs</div>
                    <div className="team-row">
                      <span className="team-name team-2">{m.team_2}</span>
                    </div>
                  </div>

                  <div className="match-card-meta">
                    <div className="meta-item">
                      <MapPin size={12} />
                      <span title={`${m.venue}, ${m.city}`}>{m.city || m.venue}</span>
                    </div>
                    {m.winner && (
                      <div className="meta-item winner-item">
                        <Trophy size={12} />
                        <span>Winner: {m.winner}</span>
                      </div>
                    )}
                  </div>

                  <button
                    className="btn btn-primary btn-block"
                    onClick={() => {
                      onChooseMatch(m.match_id);
                      onClose();
                    }}
                  >
                    <Play size={14} /> Simulate Match
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
