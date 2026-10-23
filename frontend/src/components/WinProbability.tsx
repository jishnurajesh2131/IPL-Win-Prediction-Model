import React from 'react';
import { Sparkles, TrendingUp } from 'lucide-react';
import { PredictionResponse } from '../types/prediction';
import { getTeamColor } from '../utils/teamVisualConfig';

interface WinProbabilityProps {
  prediction: PredictionResponse;
}

export const WinProbability: React.FC<WinProbabilityProps> = ({ prediction }) => {
  const p1 = Math.max(0, Math.min(100, prediction.team_1_probability * 100));
  const p2 = Math.max(0, Math.min(100, prediction.team_2_probability * 100));

  const p1Formatted = p1.toFixed(1);
  const p2Formatted = p2.toFixed(1);

  // Format human-readable prediction engine type
  let engineBadge = 'Live XGBoost Model';
  if (prediction.prediction_type === 'pre_match') {
    engineBadge = 'Walk-Forward Pre-Match Anchor (T0)';
  } else if (prediction.prediction_type.startsWith('terminal')) {
    engineBadge = 'Terminal Match Settlement';
  }

  return (
    <div className="win-probability-hero">
      <div className="hero-header">
        <div className="hero-title-group">
          <Sparkles size={16} className="text-accent" />
          <span className="hero-title">MODEL WIN PROBABILITY</span>
        </div>
        <div className="engine-badge">
          <TrendingUp size={12} />
          <span>{engineBadge}</span>
        </div>
      </div>

      {/* Main Dual Probability Display */}
      <div className="prob-display-grid">
        <div className="team-prob-block team-1-block">
          <span className="team-prob-name" style={{ color: getTeamColor(prediction.team_1) }}>
            {prediction.team_1}
          </span>
          <div className="prob-value-row">
            <span className="prob-percentage" style={{ color: '#ffffff' }}>{p1Formatted}%</span>
          </div>
        </div>

        <div className="prob-divider">
          <span className="prob-vs-badge">VS</span>
        </div>

        <div className="team-prob-block team-2-block">
          <span className="team-prob-name" style={{ color: getTeamColor(prediction.team_2) }}>
            {prediction.team_2}
          </span>
          <div className="prob-value-row">
            <span className="prob-percentage" style={{ color: '#ffffff' }}>{p2Formatted}%</span>
          </div>
        </div>
      </div>

      {/* Dual Color Gauge Progress Bar */}
      <div className="prob-gauge-container" title={`${prediction.team_1}: ${p1Formatted}% | ${prediction.team_2}: ${p2Formatted}%`}>
        <div
          className="prob-gauge-bar"
          style={{ width: `${p1}%`, backgroundColor: getTeamColor(prediction.team_1) }}
        ></div>
        <div
          className="prob-gauge-bar"
          style={{ width: `${p2}%`, backgroundColor: getTeamColor(prediction.team_2) }}
        ></div>
      </div>

      <div className="gauge-markers">
        <span>0%</span>
        <span className="gauge-midpoint">50% Parity</span>
        <span>100%</span>
      </div>
    </div>
  );
};
