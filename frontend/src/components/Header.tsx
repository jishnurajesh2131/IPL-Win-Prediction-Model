import React, { useEffect, useState } from 'react';
import { Activity, RefreshCw } from 'lucide-react';
import { checkBackendHealth } from '../api/simulations';
import { HealthResponse } from '../types/simulation';

interface HeaderProps {
  onOpenMatchSelector: () => void;
  selectedMatchId: string | null;
  mode: 'historical' | 'evaluation';
  onToggleMode: (mode: 'historical' | 'evaluation') => void;
}

export const Header: React.FC<HeaderProps> = ({
  onOpenMatchSelector,
  selectedMatchId,
  mode,
  onToggleMode,
}) => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [checking, setChecking] = useState<boolean>(false);

  const checkHealth = async () => {
    setChecking(true);
    try {
      const h = await checkBackendHealth();
      setHealth(h);
    } catch {
      setHealth(null);
    } finally {
      setChecking(false);
    }
  };

  useEffect(() => {
    checkHealth();
  }, []);

  const isOnline = health?.status === 'ok';

  return (
    <header className="app-header">
      <div className="header-brand">
        <div className="brand-icon">🏏</div>
        <div className="brand-text">
          <h1>IPL WIN PROBABILITY & LIVE ANALYTICS</h1>
          <span className="brand-subtitle">Two-Model XGBoost Live Match Simulator</span>
        </div>
      </div>

      {/* Mode Switcher Tabs */}
      <div className="header-mode-nav">
        <button
          className={`nav-tab ${mode === 'historical' ? 'active' : ''}`}
          onClick={() => onToggleMode('historical')}
        >
          Historical Replay
        </button>
        <button
          className={`nav-tab ${mode === 'evaluation' ? 'active' : ''}`}
          onClick={() => onToggleMode('evaluation')}
        >
          Final Test Evaluation
        </button>
      </div>

      <div className="header-actions">
        <div className={`status-pill ${isOnline ? 'online' : 'offline'}`} title="Backend ML API Status">
          <Activity size={14} className={checking ? 'spin' : ''} />
          <span>{isOnline ? 'API Online' : 'API Offline'}</span>
          {isOnline && health?.matches_available && (
            <span className="match-count-tag">({health.matches_available} matches)</span>
          )}
        </div>

        {mode === 'historical' && (
          <button className="btn btn-secondary" onClick={onOpenMatchSelector}>
            <RefreshCw size={14} />
            <span>{selectedMatchId ? 'Change Match' : 'Select Match'}</span>
          </button>
        )}
      </div>
    </header>
  );
};

