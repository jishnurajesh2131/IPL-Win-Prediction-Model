"""
Cricket delivery formatting utilities.
Single authoritative source of truth for cricket ball numbering and display labels.
"""

from typing import Dict, Any, Optional


def get_authoritative_delivery_ball(
    delivery: Dict[str, Any],
    legal_balls_bowled: int = 0,
    is_legal: bool = True,
) -> float:
    """
    Computes the authoritative cricket delivery number (e.g. 0.5 for over 0, ball 5)
    for a delivery row, avoiding raw Cricsheet row-within-over counters (such as 0.7 or 0.8).

    Args:
        delivery: Dictionary representing the delivery record.
        legal_balls_bowled: Number of cumulative legal balls bowled in the innings up to this delivery.
        is_legal: Whether this delivery was legal (not wide, not no-ball).

    Returns:
        float representation in standard cricket notation: over.ball (e.g. 0.5, 17.3).
    """
    # 1. Prefer actual_delivery from clean deliveries dataset if present and valid
    if "actual_delivery" in delivery:
        val = delivery.get("actual_delivery")
        if val is not None and str(val).strip() not in ("", "nan", "None"):
            try:
                ad = float(val)
                # Verify sanity: ball digit should strictly be 1-6 (valid cricket over balls)
                dec = round(ad * 10) % 10
                if 1 <= dec <= 6:
                    return round(ad, 1)
            except (ValueError, TypeError):
                pass

    # 2. Generic rule-based calculation from legal balls bowled in this innings
    if is_legal and legal_balls_bowled > 0:
        over = (legal_balls_bowled - 1) // 6
        ball_in_over = ((legal_balls_bowled - 1) % 6) + 1
    else:
        over = legal_balls_bowled // 6
        ball_in_over = (legal_balls_bowled % 6) + 1

    # Guarantee ball_in_over is strictly within 1..6
    ball_in_over = min(6, max(1, ball_in_over))
    return round(over + (ball_in_over / 10.0), 1)


def format_cricket_ball(ball: Optional[float]) -> str:
    """
    Format a decimal cricket ball (e.g. 0.5) to standard cricket string '0.5'.
    """
    if ball is None:
        return "0.0"
    b = round(float(ball), 1)
    over = int(b)
    ball_in_over = int(round((b - over) * 10))
    if 1 <= ball_in_over <= 6:
        return f"{over}.{ball_in_over}"
    return f"{over}.{min(6, max(1, ball_in_over))}"


def format_authoritative_cricket_ball(ball: Optional[float]) -> str:
    """
    Authoritative cricket ball string formatter.
    """
    return format_cricket_ball(ball)

