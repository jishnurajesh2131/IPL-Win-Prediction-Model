import React, { useMemo } from 'react';
import { useMatches } from '../hooks/useMatches';
import { Calendar, Play, Trophy, Activity } from 'lucide-react';
import { getTeamColor } from '../utils/teamVisualConfig';

interface FinalTestDashboardProps {
  onSimulateMatch: (matchId: string) => void;
}

export const FinalTestDashboard: React.FC<FinalTestDashboardProps> = ({ onSimulateMatch }) => {
  const { matches, availableSeasons, loading, selectedSeason, setSelectedSeason } = useMatches('2024');

  // Only consider 2024-2026 test matches
  const testMatches = useMemo(() => {
    return matches.filter(m => {
      const year = parseInt(m.season.substring(0, 4), 10);
      return year >= 2024 && year <= 2026;
    });
  }, [matches]);

  const filteredMatches = testMatches; // Already filtered by useMatches backend fetch

  const availableTestSeasons = useMemo(() => {
    const s = availableSeasons.filter(s => {
      const year = parseInt(s.substring(0, 4), 10);
      return year >= 2024 && year <= 2026;
    });
    return s.sort((a, b) => b.localeCompare(a));
  }, [availableSeasons]);

  return (
    <div className="dashboard-grid" style={{ maxWidth: '1200px', margin: '0 auto', gap: '2rem' }}>
      
      {/* HEADER SECTION */}
      <div className="grid-full-width" style={{ textAlign: 'center', padding: '2rem 1rem', background: '#1e293b', borderRadius: '12px' }}>
        <h1 style={{ margin: '0 0 0.5rem 0', color: '#f8fafc' }}>Final Test Evaluation</h1>
        <h3 style={{ margin: '0 0 1rem 0', color: '#94a3b8', fontWeight: 'normal' }}>2024–2026 Held-Out Test Set</h3>
        <p style={{ margin: 0, color: '#cbd5e1' }}>
          Final evaluation of the frozen live win-probability model on chronologically later matches. <br/>
          <span style={{ color: '#f59e0b', fontSize: '0.9em', marginTop: '0.5rem', display: 'inline-block' }}>
            These matches were not used for model training or model selection.
          </span>
        </p>
      </div>

      {/* SEASON FILTER */}
      <div className="grid-full-width" style={{ display: 'flex', gap: '1rem', justifyContent: 'center' }}>
        <button 
          className={`btn ${selectedSeason === '' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => setSelectedSeason('')}
        >
          All
        </button>
        {availableTestSeasons.map(s => (
          <button 
            key={s}
            className={`btn ${selectedSeason === s ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => setSelectedSeason(s)}
          >
            {s}
          </button>
        ))}
      </div>

      {/* TEST MATCH LIST */}
      <div className="grid-full-width">
        <h2 style={{ marginBottom: '1rem', color: '#f8fafc' }}>{selectedSeason === '' ? 'All Test Matches' : `${selectedSeason} Test Matches`} ({filteredMatches.length})</h2>
        
        {loading ? (
          <div style={{ textAlign: 'center', padding: '2rem' }}><Activity className="spin" size={32} /></div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: '1rem' }}>
            {filteredMatches.map(m => {
              // We do not have direct predictions in MatchSummary right now, but we can display the actual result
              return (
                <div key={m.match_id} style={{ background: '#1e293b', borderRadius: '12px', padding: '1rem', border: '1px solid #334155' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', fontSize: '0.85rem', color: '#94a3b8' }}>
                    <span><Calendar size={12} style={{ display: 'inline', marginRight: '4px' }}/> {m.date} ({m.season})</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
                    <span style={{ fontWeight: 'bold', color: getTeamColor(m.team_1) }}>{m.team_1}</span>
                    <span style={{ fontSize: '0.8rem', color: '#64748b' }}>vs</span>
                    <span style={{ fontWeight: 'bold', color: getTeamColor(m.team_2) }}>{m.team_2}</span>
                  </div>
                  
                  <div style={{ background: '#0f172a', padding: '0.75rem', borderRadius: '8px', marginBottom: '1rem' }}>
                    <div style={{ fontSize: '0.85rem', color: '#94a3b8', marginBottom: '0.25rem' }}>ACTUAL RESULT</div>
                    <div style={{ fontWeight: 'bold', color: '#f8fafc', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <Trophy size={14} color="#f59e0b" />
                      {m.winner || 'No Result'}
                    </div>
                  </div>

                  <button 
                    className="btn btn-primary" 
                    style={{ width: '100%', display: 'flex', justifyContent: 'center', gap: '0.5rem' }}
                    onClick={() => onSimulateMatch(m.match_id)}
                  >
                    <Play size={16} /> Evaluate Replay
                  </button>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};
