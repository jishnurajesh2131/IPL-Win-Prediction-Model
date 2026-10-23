"""
Simulation Output Formatters, CLI Display, and Plotting.

Provides:
1. JSON exporter saving structured events to data/processed/simulations/.
2. Clean, readable CLI match progression printer.
3. Matplotlib win-probability trajectory visualizer.
"""

import os
import json
from typing import Optional
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend safe for headless/CLI environments
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

from .simulator import SimulationResult


def save_simulation_json(
    sim_result: SimulationResult,
    output_path: Optional[str] = None,
) -> str:
    """
    Saves simulation result to a structured JSON file.
    """
    if output_path is None:
        os.makedirs("data/processed/simulations", exist_ok=True)
        output_path = f"data/processed/simulations/simulation_{sim_result.match_id}.json"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sim_result.to_dict(), f, indent=2)

    return output_path


def print_cli_summary(sim_result: SimulationResult, step_interval: int = 6):
    """
    Prints a human-readable CLI summary of the simulated match.
    Displays every `step_interval` deliveries (default: end of each over) plus wickets and terminal states.
    """
    print("\n" + "=" * 65)
    print("                 IPL LIVE MATCH SIMULATION")
    print("=" * 65)
    print(f"Match:    {sim_result.team_a} vs {sim_result.team_b}")
    print(f"Match ID: {sim_result.match_id} | Season: {sim_result.season} | Date: {sim_result.date}")
    print(f"Venue:    {sim_result.venue}, {sim_result.city}")
    print(f"Toss:     {sim_result.toss_winner} won and chose to {sim_result.toss_decision}")
    print("-" * 65)

    # T0 event
    t0 = sim_result.events[0]
    print(f"T0 (Pre-Match Anchor):")
    print(f"  {sim_result.team_a} Win Probability: {t0.team_a_win_probability * 100:.1f}%")
    print(f"  {sim_result.team_b} Win Probability: {t0.team_b_win_probability * 100:.1f}%")
    print("-" * 65)

    current_inn = 1
    print(f"\n--- INNINGS 1 ---")
    prev_wickets = 0

    inn1_final_event = None
    for e in sim_result.events[1:]:
        if e.innings == 1:
            inn1_final_event = e
            # Print at over intervals or upon a wicket falling
            is_over_end = (e.legal_balls_bowled > 0 and e.legal_balls_bowled % 6 == 0 and e.is_legal)
            is_wicket_fall = (e.wickets > prev_wickets)
            prev_wickets = e.wickets

            if is_over_end or is_wicket_fall or e.event_number == 1:
                ball_str = f"{int(e.overs_completed)}.{e.legal_balls_bowled % 6}"
                print(
                    f"Over {ball_str:>4} | {e.score:>3}/{e.wickets:<2} (CRR {e.current_run_rate:4.1f}) | "
                    f"{sim_result.team_a}: {e.team_a_win_probability * 100:5.1f}% | "
                    f"{sim_result.team_b}: {e.team_b_win_probability * 100:5.1f}% | "
                    f"Striker: {e.striker} vs {e.bowler}"
                )

        elif e.innings == 2:
            if current_inn == 1:
                print("-" * 65)
                print(f"\n--- INNINGS BREAK ---")
                if inn1_final_event:
                    print(f"Innings 1 Total: {inn1_final_event.score}/{inn1_final_event.wickets}")
                    print(f"Target for Innings 2: {inn1_final_event.score + 1}")
                print("-" * 65)
                print(f"\n--- INNINGS 2 ---")
                current_inn = 2
                prev_wickets = 0

            is_over_end = (e.legal_balls_bowled > 0 and e.legal_balls_bowled % 6 == 0 and e.is_legal)
            is_wicket_fall = (e.wickets > prev_wickets)
            is_terminal = "terminal" in e.prediction_type
            prev_wickets = e.wickets

            if is_over_end or is_wicket_fall or e.event_number == sim_result.events[-1].event_number or is_terminal:
                ball_str = f"{int(e.overs_completed)}.{e.legal_balls_bowled % 6}"
                print(
                    f"Over {ball_str:>4} | {e.score:>3}/{e.wickets:<2} (Need {e.runs_needed:>3} off {max(0, 120 - e.legal_balls_bowled):>3}, RRR {e.required_run_rate:4.1f}) | "
                    f"{sim_result.team_a}: {e.team_a_win_probability * 100:5.1f}% | "
                    f"{sim_result.team_b}: {e.team_b_win_probability * 100:5.1f}% | "
                    f"{e.prediction_type}"
                )

    print("\n" + "=" * 65)
    print("                        FINAL RESULT")
    print("=" * 65)
    print(f"Official Winner: {sim_result.winner}")
    last_event = sim_result.events[-1]
    print(f"Final Probability Output:")
    print(f"  {sim_result.team_a}: {last_event.team_a_win_probability * 100:.1f}%")
    print(f"  {sim_result.team_b}: {last_event.team_b_win_probability * 100:.1f}%")
    print(f"Prediction State: {last_event.prediction_type}")
    print("=" * 65 + "\n")


def plot_simulation_trajectory(
    sim_result: SimulationResult,
    output_png_path: Optional[str] = None,
) -> str:
    """
    Plots the Team A win probability trajectory across the match progression.
    """
    if not HAS_MATPLOTLIB:
        raise RuntimeError("matplotlib is required for trajectory plotting. Install it with: pip install matplotlib")

    if output_png_path is None:
        os.makedirs("data/processed/simulations", exist_ok=True)
        output_png_path = f"data/processed/simulations/trajectory_{sim_result.match_id}.png"

    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)

    event_indices = [e.event_number for e in sim_result.events]
    team_a_probs = [e.team_a_win_probability for e in sim_result.events]
    team_b_probs = [e.team_b_win_probability for e in sim_result.events]

    # Find innings break index
    inn_break_idx = None
    for e in sim_result.events:
        if e.innings == 2:
            inn_break_idx = e.event_number
            break

    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(12, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    # Top Plot: Win Probabilities
    ax1.plot(
        event_indices,
        team_a_probs,
        label=f"{sim_result.team_a} Win Prob",
        color="#1f77b4",
        linewidth=2.2,
    )
    ax1.plot(
        event_indices,
        team_b_probs,
        label=f"{sim_result.team_b} Win Prob",
        color="#ff7f0e",
        linewidth=1.8,
        linestyle="--",
    )
    ax1.axhline(0.50, color="gray", linestyle=":", alpha=0.7, label="Even Odds (50%)")

    if inn_break_idx:
        ax1.axvline(
            inn_break_idx,
            color="red",
            linestyle="-.",
            alpha=0.8,
            label="Innings Break",
        )

    ax1.set_ylim(-0.05, 1.05)
    ax1.set_ylabel("Win Probability", fontsize=12)
    ax1.set_title(
        f"Live Win Probability Trajectory: {sim_result.team_a} vs {sim_result.team_b} (Match {sim_result.match_id})\n"
        f"Winner: {sim_result.winner} | Venue: {sim_result.venue}",
        fontsize=13,
        fontweight="bold",
    )
    ax1.legend(loc="best", framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # Bottom Plot: Runs and Wickets Progression
    scores = [e.score for e in sim_result.events]
    wickets = [e.wickets for e in sim_result.events]

    ax2.plot(event_indices, scores, color="green", linewidth=1.5, label="Cumulative Score")
    ax2_w = ax2.twinx()
    ax2_w.plot(event_indices, wickets, color="crimson", linewidth=1.2, linestyle=":", label="Wickets Fallen")
    ax2_w.set_ylabel("Wickets", color="crimson")
    ax2_w.set_ylim(-0.5, 10.5)

    if inn_break_idx:
        ax2.axvline(inn_break_idx, color="red", linestyle="-.", alpha=0.8)

    ax2.set_xlabel("Delivery / Event Sequence", fontsize=12)
    ax2.set_ylabel("Innings Runs", color="green")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_png_path, dpi=200)
    plt.close()

    return output_png_path
