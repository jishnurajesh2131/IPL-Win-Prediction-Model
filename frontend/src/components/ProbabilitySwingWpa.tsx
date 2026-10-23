import React, { useMemo } from 'react';
import { TrendingUp, TrendingDown, Minus, Zap } from 'lucide-react';
import { ProbabilityPoint, PredictionResponse } from '../types/prediction';
import { DeliverySnapshot, MatchStateSnapshot } from '../types/simulation';
import { getTeamColor } from '../utils/teamVisualConfig';
import { formatCricketBall } from '../utils/deliveryFormat';

interface ProbabilitySwingWpaProps {
  currentPrediction: PredictionResponse;
  currentDelivery: DeliverySnapshot | null;
  currentState: MatchStateSnapshot;
  eventIndex: number;
  probabilityHistory: ProbabilityPoint[];
  onSeekToBall: (eventIndex: number) => void;
}

export const ProbabilitySwingWpa: React.FC<ProbabilitySwingWpaProps> = ({
  currentPrediction,
  currentDelivery,
  currentState,
  eventIndex,
  probabilityHistory,
  onSeekToBall,
}) => {
  const team1Name = currentPrediction.team_1;
  const team2Name = currentPrediction.team_2;
  const team1Color = getTeamColor(team1Name);
  const team2Color = getTeamColor(team2Name);

  // Compute current swing WPA: P_after - P_before for Team A and Team B
  const { prevPoint, currentPoint, swingA, swingB, lastEventLabel } = useMemo(() => {
    if (eventIndex <= 0 || probabilityHistory.length <= 1) {
      return {
        prevPoint: null,
        currentPoint: probabilityHistory[0] || null,
        swingA: 0,
        swingB: 0,
        lastEventLabel: 'T0 PRE-MATCH ANCHOR',
      };
    }

    const curr = probabilityHistory[eventIndex] || probabilityHistory[probabilityHistory.length - 1];
    const prev = probabilityHistory[eventIndex - 1] || probabilityHistory[0];

    const sA = curr && prev ? (curr.team_1_probability - prev.team_1_probability) * 100 : 0;
    const sB = curr && prev ? (curr.team_2_probability - prev.team_2_probability) * 100 : 0;

    let eventLabel = 'DELIVERY';
    if (currentDelivery) {
      if (currentDelivery.wicket_type) {
        eventLabel = `WICKET (${currentDelivery.wicket_type.toUpperCase()})`;
      } else if (currentDelivery.runs_off_bat === 6) {
        eventLabel = 'SIX';
      } else if (currentDelivery.runs_off_bat === 4) {
        eventLabel = 'FOUR';
      } else if (currentDelivery.wides > 0) {
        eventLabel = `WIDE (+${currentDelivery.extras})`;
      } else if (currentDelivery.noballs > 0) {
        eventLabel = `NO BALL (+${currentDelivery.extras})`;
      } else if (currentDelivery.runs_off_bat === 0 && currentDelivery.extras === 0) {
        eventLabel = 'DOT BALL';
      } else {
        const total = currentDelivery.runs_off_bat + currentDelivery.extras;
        eventLabel = `${total} RUN${total > 1 ? 'S' : ''}`;
      }
    }

    return {
      prevPoint: prev,
      currentPoint: curr,
      swingA: Math.round(sA * 10) / 10,
      swingB: Math.round(sB * 10) / 10,
      lastEventLabel: eventLabel,
    };
  }, [eventIndex, probabilityHistory, currentDelivery]);

  // Compute recent probability swings from actual history (most recent 6-8 deliveries)
  const recentSwings = useMemo(() => {
    if (probabilityHistory.length <= 1) return [];

    const list: Array<{
      eventIndex: number;
      ballLabel: string;
      eventDesc: string;
      swing: number;
    }> = [];

    // Traverse backward from current eventIndex
    const startIdx = Math.min(eventIndex, probabilityHistory.length - 1);
    for (let i = startIdx; i >= 1 && list.length < 6; i--) {
      const pCurr = probabilityHistory[i];
      const pPrev = probabilityHistory[i - 1];
      if (!pCurr || !pPrev) continue;

      const diff = (pCurr.team_1_probability - pPrev.team_1_probability) * 100;
      const roundedDiff = Math.round(diff * 10) / 10;

      // Ball label
      const ballNum = formatCricketBall(pCurr.ball, pCurr.overs_completed);

      let desc = 'DELIVERY';
      const scoreDiff = pCurr.score - pPrev.score;
      const wktDiff = pCurr.wickets - pPrev.wickets;

      if (wktDiff > 0) {
        desc = 'WICKET';
      } else if (scoreDiff === 6) {
        desc = 'SIX';
      } else if (scoreDiff === 4) {
        desc = 'FOUR';
      } else if (scoreDiff === 0) {
        desc = 'DOT';
      } else {
        desc = `${scoreDiff} RUN${scoreDiff > 1 ? 'S' : ''}`;
      }

      list.push({
        eventIndex: pCurr.event_index,
        ballLabel: `Inn ${pCurr.innings} · ${ballNum}`,
        eventDesc: desc,
        swing: roundedDiff,
      });
    }

    return list;
  }, [eventIndex, probabilityHistory]);

  const pBeforeA = prevPoint ? (prevPoint.team_1_probability * 100).toFixed(1) : (currentPrediction.team_1_probability * 100).toFixed(1);
  const pAfterA = currentPoint ? (currentPoint.team_1_probability * 100).toFixed(1) : (currentPrediction.team_1_probability * 100).toFixed(1);

  const pBeforeB = prevPoint ? (prevPoint.team_2_probability * 100).toFixed(1) : (currentPrediction.team_2_probability * 100).toFixed(1);
  const pAfterB = currentPoint ? (currentPoint.team_2_probability * 100).toFixed(1) : (currentPrediction.team_2_probability * 100).toFixed(1);

  return (
    <div className="analytics-card swing-wpa-card">
      <div className="analytics-card-header">
        <div className="analytics-title-group">
          <Zap size={18} className="analytics-icon text-amber" />
          <h3 className="analytics-title">PROBABILITY SWING</h3>
        </div>
        <span className="wpa-badge">WPA / Win Probability Added</span>
      </div>

      <div className="swing-hero-section">
        <div className="last-event-strip">
          <span className="last-event-label">LAST EVENT</span>
          <span className="last-event-value">{lastEventLabel}</span>
          {currentDelivery && (
            <span className="last-event-context">
              Over {formatCricketBall(currentDelivery.ball)} ({currentState.batting_team || 'Batting'})
            </span>
          )}
        </div>

        {/* Dual Team WPA Breakdown */}
        <div className="dual-team-swings-grid">
          {/* Team A */}
          <div className="team-swing-box" style={{ borderLeftColor: team1Color }}>
            <div className="team-swing-header">
              <span className="team-name" style={{ color: team1Color }}>{team1Name}</span>
              <span className={`swing-pill ${swingA > 0 ? 'positive' : swingA < 0 ? 'negative' : 'neutral'}`}>
                {swingA > 0 ? <TrendingUp size={14} /> : swingA < 0 ? <TrendingDown size={14} /> : <Minus size={14} />}
                {swingA > 0 ? `+${swingA.toFixed(1)}%` : `${swingA.toFixed(1)}%`}
              </span>
            </div>
            <div className="prob-transition-row">
              <div className="prob-val-group">
                <span className="prob-sublabel">Before</span>
                <span className="prob-val">{pBeforeA}%</span>
              </div>
              <span className="transition-arrow">→</span>
              <div className="prob-val-group">
                <span className="prob-sublabel">After</span>
                <span className="prob-val highlight">{pAfterA}%</span>
              </div>
            </div>
          </div>

          {/* Team B */}
          <div className="team-swing-box" style={{ borderLeftColor: team2Color }}>
            <div className="team-swing-header">
              <span className="team-name" style={{ color: team2Color }}>{team2Name}</span>
              <span className={`swing-pill ${swingB > 0 ? 'positive' : swingB < 0 ? 'negative' : 'neutral'}`}>
                {swingB > 0 ? <TrendingUp size={14} /> : swingB < 0 ? <TrendingDown size={14} /> : <Minus size={14} />}
                {swingB > 0 ? `+${swingB.toFixed(1)}%` : `${swingB.toFixed(1)}%`}
              </span>
            </div>
            <div className="prob-transition-row">
              <div className="prob-val-group">
                <span className="prob-sublabel">Before</span>
                <span className="prob-val">{pBeforeB}%</span>
              </div>
              <span className="transition-arrow">→</span>
              <div className="prob-val-group">
                <span className="prob-sublabel">After</span>
                <span className="prob-val highlight">{pAfterB}%</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* RECENT PROBABILITY SWINGS TABLE */}
      <div className="recent-swings-container">
        <div className="recent-swings-header">
          <span>RECENT PROBABILITY SWINGS</span>
          <span className="click-hint">Click row to navigate</span>
        </div>
        {recentSwings.length === 0 ? (
          <div className="empty-recent-swings">Awaiting recorded deliveries...</div>
        ) : (
          <div className="recent-swings-table-wrapper">
            <table className="recent-swings-table">
              <thead>
                <tr>
                  <th>Delivery</th>
                  <th>Event</th>
                  <th>Swing ({team1Name})</th>
                </tr>
              </thead>
              <tbody>
                {recentSwings.map((row) => (
                  <tr
                    key={row.eventIndex}
                    onClick={() => onSeekToBall(row.eventIndex)}
                    className={`swing-row ${row.eventIndex === eventIndex ? 'active-ball-row' : ''}`}
                    title={`Click to jump replay to ${row.ballLabel}`}
                  >
                    <td className="ball-cell">{row.ballLabel}</td>
                    <td className="event-desc-cell">
                      <span className={`event-tag tag-${row.eventDesc.toLowerCase().replace(/\s+/g, '-')}`}>
                        {row.eventDesc}
                      </span>
                    </td>
                    <td className={`swing-val-cell ${row.swing > 0 ? 'text-green' : row.swing < 0 ? 'text-red' : ''}`}>
                      {row.swing > 0 ? `+${row.swing.toFixed(1)}%` : `${row.swing.toFixed(1)}%`}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
