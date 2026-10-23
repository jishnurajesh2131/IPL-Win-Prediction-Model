/**
 * Authoritative global cricket delivery formatting utilities.
 * Single source of truth for cricket ball numbering and display labels across:
 * - DeliveryCard
 * - Go To Ball & Simulation Controls
 * - Recent Events / WPA
 * - Probability Chart
 * - Player Impact
 */

export interface FormattableDelivery {
  ball?: number | null;
  display_ball?: string | null;
  overs_completed?: number | null;
  innings?: number | null;
  event_index?: number | null;
}

/**
 * Format a delivery ball number to standard cricket notation (e.g. "0.5", "17.3").
 * Guarantees that ball-in-over strictly conforms to cricket notation (1..6).
 * Never emits raw attempt indices such as .7, .8, or .9.
 */
export function formatCricketBall(
  ball: number | undefined | null,
  oversCompleted?: number | null
): string {
  if (ball !== undefined && ball !== null && !isNaN(ball)) {
    const b = Number(ball);
    const over = Math.floor(b);
    const ballInOver = Math.round((b - over) * 10);

    // If ball in over is valid cricket notation 1..6, return standard over.ball
    if (ballInOver >= 1 && ballInOver <= 6) {
      return `${over}.${ballInOver}`;
    }

    // If an invalid attempt index was passed (> 6 or 0), fallback to oversCompleted if available
    if (oversCompleted !== undefined && oversCompleted !== null && !isNaN(oversCompleted)) {
      const ocOver = Math.floor(oversCompleted);
      const ocBall = Math.round((oversCompleted % 1) * 6);
      return `${ocOver}.${ocBall}`;
    }

    // Otherwise clamp ballInOver to legal 1..6 bounds
    const clampedBall = Math.min(6, Math.max(1, ballInOver));
    return `${over}.${clampedBall}`;
  }

  if (oversCompleted !== undefined && oversCompleted !== null && !isNaN(oversCompleted)) {
    const over = Math.floor(oversCompleted);
    const ballInOver = Math.round((oversCompleted % 1) * 6);
    return `${over}.${ballInOver}`;
  }

  return '0.0';
}

/**
 * Returns the standardized display label for a replay event or delivery.
 */
export function getDisplayDeliveryLabel(
  item: FormattableDelivery | null | undefined,
  options?: {
    includeInnings?: boolean;
    prefixBall?: boolean;
    t0Label?: string;
  }
): string {
  if (!item) return options?.t0Label || 'T0';
  if (item.event_index === 0) return options?.t0Label || 'T0';

  const ballStr = item.display_ball || formatCricketBall(item.ball, item.overs_completed);
  const prefix = options?.prefixBall ? 'Ball ' : '';

  if (options?.includeInnings && item.innings) {
    return `Inn ${item.innings} · ${prefix}${ballStr}`;
  }
  return `${prefix}${ballStr}`;
}
