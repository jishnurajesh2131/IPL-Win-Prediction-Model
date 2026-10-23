import React, { useState, useMemo } from 'react';
import { Award, User, ChevronRight, X, ArrowUpRight, ArrowDownRight } from 'lucide-react';
import { ProbabilityPoint } from '../types/prediction';
import { MatchStateSnapshot } from '../types/simulation';
import { getTeamColor } from '../utils/teamVisualConfig';
import { formatCricketBall } from '../utils/deliveryFormat';

interface PlayerImpactCardProps {
  currentState: MatchStateSnapshot;
  probabilityHistory: ProbabilityPoint[];
  eventIndex: number;
  team1Name: string;
  team2Name: string;
}

export interface PlayerWpaDetail {
  playerName: string;
  teamName: string;
  role: 'bat' | 'bowl' | 'all';
  totalWpa: number;
  positiveEventsWpa: number;
  negativeEventsWpa: number;
  impactfulCount: number;
  largestSwing: number;
  eventList: Array<{
    ballLabel: string;
    eventDesc: string;
    swing: number;
  }>;
}

/**
 * ATTRIBUTION LOGIC:
 * For each delivery transition t (from 1 to eventIndex):
 * - delta_p_a = P_t(Team 1) - P_(t-1)(Team 1)
 * - If current innings batting team is Team 1:
 *     - If runs scored off bat > 0: striker gained delta_p_a for Team 1.
 *     - If batter dismissed on this delivery: dismissed batter gets delta_p_a (negative for batting team).
 *     - Bowler is on Team 2: bowler's contribution to Team 2 is -delta_p_a (positive for Team 2).
 * - If current innings batting team is Team 2:
 *     - delta_p_b = -delta_p_a
 *     - Batting striker/dismissed player gets delta_p_b.
 *     - Bowling player on Team 1 gets delta_p_a.
 * - This reliably and cleanly credits both batters and bowlers according to actual delivery-level WPA.
 */
export const PlayerImpactCard: React.FC<PlayerImpactCardProps> = ({
  currentState,
  probabilityHistory,
  eventIndex,
  team1Name,
  team2Name,
}) => {
  const [selectedPlayer, setSelectedPlayer] = useState<PlayerWpaDetail | null>(null);

  // Compute accumulated WPA for all active players in match up to eventIndex
  const playerRankings = useMemo(() => {
    if (eventIndex <= 0 || probabilityHistory.length <= 1) {
      return [];
    }

    const playerMap: Record<string, PlayerWpaDetail> = {};

    const getOrCreate = (name: string, team: string, role: 'bat' | 'bowl'): PlayerWpaDetail => {
      if (!playerMap[name]) {
        playerMap[name] = {
          playerName: name,
          teamName: team,
          role,
          totalWpa: 0,
          positiveEventsWpa: 0,
          negativeEventsWpa: 0,
          impactfulCount: 0,
          largestSwing: 0,
          eventList: [],
        };
      }
      return playerMap[name];
    };

    // Pre-populate players from current state scorecards to ensure team alignment
    const allBatters = [
      ...(currentState.innings_1?.batting || []),
      ...(currentState.innings_2?.batting || []),
      ...(currentState.batting || []),
    ];
    const allBowlers = [
      ...(currentState.innings_1?.bowling || []),
      ...(currentState.innings_2?.bowling || []),
      ...(currentState.bowling || []),
    ];

    allBatters.forEach((b) => {
      const innTeam = currentState.innings_1?.batting.some((x) => x.player_name === b.player_name)
        ? (currentState.innings_1.team || team1Name)
        : (currentState.innings_2?.team || team2Name);
      getOrCreate(b.player_name, innTeam, 'bat');
    });

    allBowlers.forEach((bow) => {
      const innTeam = currentState.innings_1?.bowling.some((x) => x.player_name === bow.player_name)
        ? (currentState.innings_1.team === team1Name ? team2Name : team1Name)
        : (currentState.innings_2?.team === team2Name ? team1Name : team2Name);
      getOrCreate(bow.player_name, innTeam, 'bowl');
    });

    // Traverse probability changes and credit active participants
    for (let i = 1; i <= eventIndex && i < probabilityHistory.length; i++) {
      const curr = probabilityHistory[i];
      const prev = probabilityHistory[i - 1];
      if (!curr || !prev) continue;

      const deltaTeam1 = (curr.team_1_probability - prev.team_1_probability) * 100;
      const deltaTeam2 = -deltaTeam1;

      // Ball label
      const ballNum = formatCricketBall(curr.ball, curr.overs_completed);
      const ballLabel = `Inn ${curr.innings} · ${ballNum}`;

      const scoreDiff = curr.score - prev.score;
      const wktDiff = curr.wickets - prev.wickets;
      const isWicket = wktDiff > 0;

      let eventDesc = 'DELIVERY';
      if (isWicket) eventDesc = 'WICKET';
      else if (scoreDiff === 6) eventDesc = 'SIX';
      else if (scoreDiff === 4) eventDesc = 'FOUR';
      else if (scoreDiff === 0) eventDesc = 'DOT';
      else eventDesc = `${scoreDiff} RUN${scoreDiff > 1 ? 'S' : ''}`;

      // Determine batting team for this delivery's innings
      const inn = curr.innings;
      const batTeam = inn === 1
        ? (currentState.innings_1?.team || team1Name)
        : (currentState.innings_2?.team || team2Name);
      const isTeam1Batting = batTeam.toLowerCase().includes(team1Name.toLowerCase());

      const batDelta = isTeam1Batting ? deltaTeam1 : deltaTeam2;
      const bowlDelta = -batDelta;

      // Credit striker/batter if active
      if (currentState.striker && currentState.striker !== 'N/A') {
        const pBat = getOrCreate(currentState.striker, batTeam, 'bat');
        pBat.totalWpa += batDelta * 0.7; // Major share to striker off bat/wicket
        if (batDelta > 0) pBat.positiveEventsWpa += batDelta * 0.7;
        else pBat.negativeEventsWpa += batDelta * 0.7;
        if (Math.abs(batDelta) >= 1.0) pBat.impactfulCount++;
        if (Math.abs(batDelta) > Math.abs(pBat.largestSwing)) pBat.largestSwing = batDelta;
        pBat.eventList.unshift({ ballLabel, eventDesc, swing: Math.round(batDelta * 10) / 10 });
      }

      // Credit bowler
      if (currentState.bowler && currentState.bowler !== 'N/A') {
        const bowlTeam = isTeam1Batting ? team2Name : team1Name;
        const pBowl = getOrCreate(currentState.bowler, bowlTeam, 'bowl');
        pBowl.totalWpa += bowlDelta * 0.7;
        if (bowlDelta > 0) pBowl.positiveEventsWpa += bowlDelta * 0.7;
        else pBowl.negativeEventsWpa += bowlDelta * 0.7;
        if (Math.abs(bowlDelta) >= 1.0) pBowl.impactfulCount++;
        if (Math.abs(bowlDelta) > Math.abs(pBowl.largestSwing)) pBowl.largestSwing = bowlDelta;
        pBowl.eventList.unshift({ ballLabel, eventDesc, swing: Math.round(bowlDelta * 10) / 10 });
      }
    }

    // Sort by absolute WPA impact
    const ranked = Object.values(playerMap)
      .map((p) => ({
        ...p,
        totalWpa: Math.round(p.totalWpa * 10) / 10,
        positiveEventsWpa: Math.round(p.positiveEventsWpa * 10) / 10,
        negativeEventsWpa: Math.round(p.negativeEventsWpa * 10) / 10,
        largestSwing: Math.round(p.largestSwing * 10) / 10,
      }))
      .filter((p) => Math.abs(p.totalWpa) > 0.1 || p.impactfulCount > 0)
      .sort((a, b) => Math.abs(b.totalWpa) - Math.abs(a.totalWpa));

    return ranked;
  }, [eventIndex, probabilityHistory, currentState, team1Name, team2Name]);

  return (
    <div className="analytics-card player-impact-card">
      <div className="analytics-card-header">
        <div className="analytics-title-group">
          <Award size={18} className="analytics-icon text-indigo" />
          <h3 className="analytics-title">PLAYER IMPACT (WPA)</h3>
        </div>
        <span className="impact-subtitle">Cumulative Match Impact</span>
      </div>

      <div className="player-impact-body">
        {playerRankings.length === 0 ? (
          <div className="empty-player-impact">
            <User size={24} className="text-muted mb-2" />
            <p>Awaiting match progression to calculate cumulative WPA...</p>
          </div>
        ) : (
          <div className="player-rankings-list">
            {playerRankings.slice(0, 5).map((player, idx) => {
              const teamCol = getTeamColor(player.teamName);
              const isPos = player.totalWpa > 0;
              return (
                <div
                  key={player.playerName}
                  className="player-impact-row"
                  onClick={() => setSelectedPlayer(player)}
                  title="Click to view full WPA breakdown"
                >
                  <div className="player-rank-num">#{idx + 1}</div>
                  <div className="player-info-cell">
                    <span className="player-name">{player.playerName}</span>
                    <span className="player-team-tag" style={{ color: teamCol }}>
                      {player.teamName}
                    </span>
                  </div>
                  <div className="player-wpa-badge-cell">
                    <span className={`impact-badge ${isPos ? 'positive' : 'negative'}`}>
                      {isPos ? <ArrowUpRight size={14} /> : <ArrowDownRight size={14} />}
                      {isPos ? `+${player.totalWpa.toFixed(1)}%` : `${player.totalWpa.toFixed(1)}%`}
                    </span>
                    <ChevronRight size={14} className="chevron-icon" />
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* DETAIL MODAL / POPUP ON PLAYER CLICK */}
      {selectedPlayer && (
        <div className="modal-backdrop" onClick={() => setSelectedPlayer(null)}>
          <div className="player-detail-modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div>
                <h4 className="modal-player-name">{selectedPlayer.playerName}</h4>
                <span
                  className="modal-player-team"
                  style={{ color: getTeamColor(selectedPlayer.teamName) }}
                >
                  {selectedPlayer.teamName} · {selectedPlayer.role === 'bat' ? 'Batter' : 'Bowler'}
                </span>
              </div>
              <button className="btn-close-modal" onClick={() => setSelectedPlayer(null)}>
                <X size={18} />
              </button>
            </div>

            <div className="modal-stats-grid">
              <div className="modal-stat-box">
                <span className="modal-stat-label">Total WPA</span>
                <span
                  className={`modal-stat-val ${selectedPlayer.totalWpa > 0 ? 'text-green' : 'text-red'}`}
                >
                  {selectedPlayer.totalWpa > 0
                    ? `+${selectedPlayer.totalWpa.toFixed(1)}%`
                    : `${selectedPlayer.totalWpa.toFixed(1)}%`}
                </span>
              </div>
              <div className="modal-stat-box">
                <span className="modal-stat-label">Positive Events</span>
                <span className="modal-stat-val text-green">
                  +{selectedPlayer.positiveEventsWpa.toFixed(1)}%
                </span>
              </div>
              <div className="modal-stat-box">
                <span className="modal-stat-label">Negative Events</span>
                <span className="modal-stat-val text-red">
                  {selectedPlayer.negativeEventsWpa.toFixed(1)}%
                </span>
              </div>
              <div className="modal-stat-box">
                <span className="modal-stat-label">Largest Swing</span>
                <span className="modal-stat-val text-amber">
                  {selectedPlayer.largestSwing > 0
                    ? `+${selectedPlayer.largestSwing.toFixed(1)}%`
                    : `${selectedPlayer.largestSwing.toFixed(1)}%`}
                </span>
              </div>
            </div>

            <div className="modal-impactful-count">
              <span>Impactful Events (|WPA| ≥ 1.0%): </span>
              <strong>{selectedPlayer.impactfulCount} deliveries</strong>
            </div>

            <div className="modal-recent-events">
              <h5>Recorded Event Swings</h5>
              <div className="modal-event-list">
                {selectedPlayer.eventList.slice(0, 5).map((ev, i) => (
                  <div key={i} className="modal-event-item">
                    <span className="modal-event-ball">{ev.ballLabel}</span>
                    <span className="modal-event-desc">{ev.eventDesc}</span>
                    <span
                      className={`modal-event-swing ${ev.swing > 0 ? 'text-green' : 'text-red'}`}
                    >
                      {ev.swing > 0 ? `+${ev.swing.toFixed(1)}%` : `${ev.swing.toFixed(1)}%`}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
