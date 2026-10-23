import React, { useMemo } from 'react';
import { Activity } from 'lucide-react';
import { ProbabilityPoint } from '../types/prediction';

interface MatchVolatilityMeterProps {
  probabilityHistory: ProbabilityPoint[];
  eventIndex: number;
}

/**
 * DETERMINISTIC VOLATILITY FORMULA:
 * 1. Window: up to last 12 recorded delivery transitions [eventIndex - 12, eventIndex].
 * 2. Absolute step changes: delta_i = |P_t(Team A) - P_(t-1)(Team A)| for each delivery in window.
 * 3. Mean absolute step: mean_delta = sum(delta_i) / N.
 * 4. Normalized Score (0-100):
 *    A mean step of 8% (0.08) represents standard maximum baseline volatility.
 *    volatility_score = Math.min(100, Math.round((mean_delta / 0.08) * 100)).
 * 5. Deterministic Category Thresholds:
 *    - 0 to 24: LOW
 *    - 25 to 54: MODERATE
 *    - 55 to 74: HIGH
 *    - 75 to 100: EXTREME
 */
export const MatchVolatilityMeter: React.FC<MatchVolatilityMeterProps> = ({
  probabilityHistory,
  eventIndex,
}) => {
  const { score, level, meanDeltaPct, sampleSize } = useMemo(() => {
    if (eventIndex <= 1 || probabilityHistory.length <= 1) {
      return { score: 10, level: 'LOW', meanDeltaPct: 0.5, sampleSize: 0 };
    }

    const startIdx = Math.max(1, eventIndex - 11);
    let totalAbsDelta = 0;
    let count = 0;

    for (let i = startIdx; i <= eventIndex && i < probabilityHistory.length; i++) {
      const curr = probabilityHistory[i];
      const prev = probabilityHistory[i - 1];
      if (curr && prev) {
        const delta = Math.abs(curr.team_1_probability - prev.team_1_probability);
        totalAbsDelta += delta;
        count++;
      }
    }

    if (count === 0) {
      return { score: 10, level: 'LOW', meanDeltaPct: 0.5, sampleSize: 0 };
    }

    const meanDelta = totalAbsDelta / count;
    // Scale against reference high volatility benchmark of 0.08 (8% average ball swing)
    const rawScore = Math.min(100, Math.max(0, Math.round((meanDelta / 0.08) * 100)));

    let category: 'LOW' | 'MODERATE' | 'HIGH' | 'EXTREME' = 'LOW';
    if (rawScore >= 75) {
      category = 'EXTREME';
    } else if (rawScore >= 55) {
      category = 'HIGH';
    } else if (rawScore >= 25) {
      category = 'MODERATE';
    } else {
      category = 'LOW';
    }

    return {
      score: rawScore,
      level: category,
      meanDeltaPct: Math.round(meanDelta * 1000) / 10,
      sampleSize: count,
    };
  }, [probabilityHistory, eventIndex]);

  const levelColor = useMemo(() => {
    switch (level) {
      case 'EXTREME':
        return '#ef4444'; // Red
      case 'HIGH':
        return '#f97316'; // Orange
      case 'MODERATE':
        return '#eab308'; // Amber/Yellow
      case 'LOW':
      default:
        return '#22c55e'; // Green
    }
  }, [level]);

  return (
    <div className="analytics-card volatility-card">
      <div className="analytics-card-header">
        <div className="analytics-title-group">
          <Activity size={18} className="analytics-icon text-cyan" />
          <h3 className="analytics-title">MATCH VOLATILITY</h3>
        </div>
        <span
          className="volatility-level-badge"
          style={{ backgroundColor: `${levelColor}22`, color: levelColor, borderColor: levelColor }}
        >
          {level} VOLATILITY
        </span>
      </div>

      <div className="volatility-meter-body">
        <div className="volatility-score-display">
          <span className="volatility-number" style={{ color: levelColor }}>
            {score}
          </span>
          <span className="volatility-scale">/ 100</span>
        </div>

        {/* Visual Gauge Bar: LOW ────────●──────── HIGH */}
        <div className="volatility-gauge-container">
          <div className="gauge-axis-labels">
            <span className="gauge-label-low">LOW</span>
            <span className="gauge-label-mod">MODERATE</span>
            <span className="gauge-label-high">HIGH</span>
            <span className="gauge-label-extreme">EXTREME</span>
          </div>

          <div className="gauge-track">
            <div
              className="gauge-progress"
              style={{
                width: `${score}%`,
                background: `linear-gradient(90deg, #22c55e 0%, #eab308 50%, ${levelColor} 100%)`,
              }}
            />
            <div
              className="gauge-thumb"
              style={{ left: `${Math.min(97, Math.max(3, score))}%`, backgroundColor: levelColor }}
            />
          </div>
        </div>

        <div className="volatility-formula-caption">
          <span>Rolling 12-ball probability swing avg: </span>
          <strong>{meanDeltaPct}% / ball</strong>
          <span className="sample-badge">({sampleSize} balls sampled)</span>
        </div>
      </div>
    </div>
  );
};
