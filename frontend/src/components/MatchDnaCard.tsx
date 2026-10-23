import React, { useMemo } from 'react';
import { Dna } from 'lucide-react';
import { MatchStateSnapshot } from '../types/simulation';
import { ProbabilityPoint } from '../types/prediction';

interface MatchDnaCardProps {
  currentState: MatchStateSnapshot;
  probabilityHistory: ProbabilityPoint[];
  eventIndex: number;
}

interface DnaMetric {
  name: string;
  value: number;
  label: string;
  color: string;
  description: string;
}

/**
 * DETERMINISTIC MATCH DNA FORMULAS:
 * 1. Pressure (0-100):
 *    - Innings 1: 30 + (wickets * 6) + Math.max(0, (8.0 - CRR) * 4)
 *    - Innings 2: (RRR / 14.0) * 60 + (wickets * 4)
 * 2. Volatility (0-100):
 *    - Rolling 12-ball mean absolute swing normalized to 8% par.
 * 3. Batting Edge (0-100):
 *    - 50 + (CRR - 8.0) * 5 + ((10 - wickets) - 5) * 4
 * 4. Bowling Edge (0-100):
 *    - 100 - Batting Edge
 * 5. Chase Difficulty (0-100):
 *    - Innings 1: 0 (Targets not established yet)
 *    - Innings 2: (RRR / 15.0) * 100
 * 6. Momentum (0-100):
 *    - 50 + (P_t - P_(t-6)) / 0.20 * 50
 */
export const MatchDnaCard: React.FC<MatchDnaCardProps> = ({
  currentState,
  probabilityHistory,
  eventIndex,
}) => {
  const metrics = useMemo<DnaMetric[]>(() => {
    const inn = currentState.innings;
    const crr = currentState.current_run_rate || 0;
    const rrr = currentState.required_run_rate;
    const wkts = currentState.wickets || 0;

    // 1. Pressure
    let pressureRaw = 40;
    if (inn === 2 && rrr !== null) {
      pressureRaw = (rrr / 14.0) * 60 + wkts * 4;
    } else {
      pressureRaw = 30 + wkts * 6 + Math.max(0, (8.0 - crr) * 4);
    }
    const pressure = Math.min(100, Math.max(0, Math.round(pressureRaw)));

    // 2. Volatility
    let volRaw = 20;
    if (eventIndex > 1 && probabilityHistory.length > 1) {
      const startIdx = Math.max(1, eventIndex - 11);
      let sum = 0;
      let cnt = 0;
      for (let i = startIdx; i <= eventIndex && i < probabilityHistory.length; i++) {
        sum += Math.abs(probabilityHistory[i].team_1_probability - probabilityHistory[i - 1].team_1_probability);
        cnt++;
      }
      if (cnt > 0) {
        volRaw = (sum / cnt / 0.08) * 100;
      }
    }
    const volatility = Math.min(100, Math.max(0, Math.round(volRaw)));

    // 3. Batting Edge
    const batEdgeRaw = 50 + (crr - 8.0) * 5 + (5 - wkts) * 4;
    const battingEdge = Math.min(100, Math.max(0, Math.round(batEdgeRaw)));

    // 4. Bowling Edge
    const bowlingEdge = 100 - battingEdge;

    // 5. Chase Difficulty
    let chaseDiff = 0;
    if (inn === 2 && rrr !== null) {
      chaseDiff = Math.min(100, Math.max(0, Math.round((rrr / 15.0) * 100)));
    }

    // 6. Momentum
    let momRaw = 50;
    if (eventIndex > 0 && probabilityHistory.length > 1) {
      const curr = probabilityHistory[Math.min(eventIndex, probabilityHistory.length - 1)];
      const prevIdx = Math.max(0, eventIndex - 6);
      const prev = probabilityHistory[prevIdx];
      if (curr && prev) {
        const delta = curr.team_1_probability - prev.team_1_probability;
        momRaw = 50 + (delta / 0.20) * 50;
      }
    }
    const momentum = Math.min(100, Math.max(0, Math.round(momRaw)));

    return [
      {
        name: 'Pressure',
        value: pressure,
        label: `${pressure}`,
        color: pressure > 70 ? '#ef4444' : pressure > 45 ? '#f59e0b' : '#3b82f6',
        description: inn === 2 ? 'Calculated from required run rate vs wickets remaining' : 'Calculated from current score velocity and wickets down',
      },
      {
        name: 'Volatility',
        value: volatility,
        label: `${volatility}`,
        color: volatility > 70 ? '#ef4444' : volatility > 40 ? '#f59e0b' : '#10b981',
        description: '12-ball rolling win probability movement',
      },
      {
        name: 'Batting Edge',
        value: battingEdge,
        label: `${battingEdge}`,
        color: battingEdge > 55 ? '#10b981' : '#64748b',
        description: 'Run rate premium over par and wickets in hand',
      },
      {
        name: 'Bowling Edge',
        value: bowlingEdge,
        label: `${bowlingEdge}`,
        color: bowlingEdge > 55 ? '#3b82f6' : '#64748b',
        description: 'Defensive containment and breakthrough pressure',
      },
      {
        name: 'Chase Difficulty',
        value: chaseDiff,
        label: inn === 2 ? `${chaseDiff}` : '1st Inn',
        color: chaseDiff > 70 ? '#ef4444' : chaseDiff > 40 ? '#f59e0b' : '#10b981',
        description: inn === 2 ? 'Required run rate scaled against benchmark target' : 'Target not yet set in 1st Innings',
      },
      {
        name: 'Momentum',
        value: momentum,
        label: `${momentum}`,
        color: momentum > 60 ? '#10b981' : momentum < 40 ? '#ef4444' : '#64748b',
        description: '6-ball probability shift directional momentum',
      },
    ];
  }, [currentState, probabilityHistory, eventIndex]);

  return (
    <div className="analytics-card match-dna-card">
      <div className="analytics-card-header">
        <div className="analytics-title-group">
          <Dna size={18} className="analytics-icon text-teal" />
          <h3 className="analytics-title">MATCH DNA</h3>
        </div>
        <span className="dna-badge">State Fingerprint</span>
      </div>

      <div className="match-dna-body">
        <div className="dna-meters-list">
          {metrics.map((m) => (
            <div key={m.name} className="dna-meter-row" title={m.description}>
              <div className="dna-label-group">
                <span className="dna-metric-name">{m.name}</span>
                <span className="dna-metric-val" style={{ color: m.color }}>
                  {m.label}
                </span>
              </div>
              <div className="dna-track">
                <div
                  className="dna-fill"
                  style={{
                    width: `${m.value}%`,
                    backgroundColor: m.color,
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
