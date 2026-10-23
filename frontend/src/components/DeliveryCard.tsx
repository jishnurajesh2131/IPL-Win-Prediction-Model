import React from 'react';
import { CircleDot, AlertTriangle, Disc } from 'lucide-react';
import { DeliverySnapshot } from '../types/simulation';
import { formatCricketBall } from '../utils/deliveryFormat';

interface DeliveryCardProps {
  delivery: DeliverySnapshot | null;
}

export const DeliveryCard: React.FC<DeliveryCardProps> = ({ delivery }) => {
  if (!delivery) {
    return (
      <div className="delivery-card empty">
        <Disc size={20} className="text-muted" />
        <div className="delivery-empty-text">
          <strong>Pre-Match (T0)</strong>
          <span>Awaiting first ball of the match</span>
        </div>
      </div>
    );
  }

  // Determine outcome badge
  let badgeType = 'dot';
  let badgeText = `${delivery.runs_off_bat + delivery.extras}`;

  if (delivery.wicket_type) {
    badgeType = 'wicket';
    badgeText = 'WICKET';
  } else if (delivery.runs_off_bat === 6) {
    badgeType = 'six';
    badgeText = 'SIX';
  } else if (delivery.runs_off_bat === 4) {
    badgeType = 'four';
    badgeText = 'FOUR';
  } else if (delivery.wides > 0) {
    badgeType = 'extra';
    badgeText = `WIDE (${delivery.extras})`;
  } else if (delivery.noballs > 0) {
    badgeType = 'extra';
    badgeText = `NO BALL (+${delivery.extras})`;
  } else if (delivery.runs_off_bat === 0 && delivery.extras === 0) {
    badgeType = 'dot';
    badgeText = 'DOT';
  } else {
    badgeType = 'runs';
    badgeText = `${delivery.runs_off_bat} RUN${delivery.runs_off_bat > 1 ? 'S' : ''}`;
  }

  const ballDisplay = delivery.display_ball || formatCricketBall(delivery.ball);

  return (
    <div className={`delivery-card ${badgeType}`}>
      <div className="delivery-top">
        <div className="delivery-ball-indicator">
          <CircleDot size={14} />
          <span>Ball {ballDisplay}</span>
        </div>
        <span className={`delivery-outcome-badge ${badgeType}`}>{badgeText}</span>
      </div>

      <div className="delivery-actors">
        <div className="actor-item">
          <span className="actor-role">BOWLER</span>
          <span className="actor-name">{delivery.bowler}</span>
        </div>
        <div className="actor-arrow">→</div>
        <div className="actor-item">
          <span className="actor-role">STRIKER</span>
          <span className="actor-name">{delivery.striker}</span>
        </div>
      </div>

      {delivery.wicket_type && (
        <div className="dismissal-callout">
          <AlertTriangle size={14} />
          <span>
            OUT! <strong>{delivery.player_dismissed || delivery.striker}</strong> ({delivery.wicket_type})
          </span>
        </div>
      )}
    </div>
  );
};
