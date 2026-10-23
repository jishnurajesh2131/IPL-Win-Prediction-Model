"""
Command-Line Interface for Live Match Simulation.

Usage:
    python -m src.live_simulator.cli --match-id <MATCH_ID> [--plot] [--output-json <PATH>]
or:
    python -m src.live_simulator.simulate_match --match-id <MATCH_ID> [--plot]
"""

import argparse
import sys
from .simulator import MatchSimulator
from .output import save_simulation_json, print_cli_summary, plot_simulation_trajectory


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay historical IPL match delivery-by-delivery with live ML win probability predictions."
    )
    parser.add_argument(
        "--match-id",
        type=str,
        required=True,
        help="Historical match ID to simulate (e.g. 1426274, 501228, 419155)",
    )
    parser.add_argument(
        "--matches-csv",
        type=str,
        default="data/processed/matches_clean.csv",
        help="Path to clean matches CSV",
    )
    parser.add_argument(
        "--deliveries-csv",
        type=str,
        default="data/processed/deliveries_clean.csv",
        help="Path to clean deliveries CSV",
    )
    parser.add_argument(
        "--walk-forward-csv",
        type=str,
        default="data/processed/walk_forward_pre_match_probs.csv",
        help="Path to walk-forward pre-match probabilities CSV",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save simulation JSON (default: data/processed/simulations/simulation_<match_id>.json)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate matplotlib win-probability trajectory plot",
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default=None,
        help="Custom path to save trajectory plot PNG",
    )
    parser.add_argument(
        "--no-print",
        action="store_true",
        help="Suppress terminal CLI match progression output",
    )
    return parser.parse_args()


def run_cli():
    args = parse_args()
    print(f"\nInitializing Live Match Simulator for Match ID: {args.match_id}...")

    simulator = MatchSimulator(
        matches_csv=args.matches_csv,
        deliveries_csv=args.deliveries_csv,
        walk_forward_csv=args.walk_forward_csv,
    )

    try:
        sim_result = simulator.simulate(match_id=args.match_id)
    except Exception as e:
        print(f"Error during simulation: {e}", file=sys.stderr)
        sys.exit(1)

    # CLI Output
    if not args.no_print:
        print_cli_summary(sim_result)

    # Save JSON
    json_path = save_simulation_json(sim_result, output_path=args.output_json)
    print(f"Simulation JSON saved to: {json_path}")

    # Plot
    if args.plot or args.plot_path:
        plot_path = plot_simulation_trajectory(sim_result, output_png_path=args.plot_path)
        print(f"Win probability trajectory plot saved to: {plot_path}")


if __name__ == "__main__":
    run_cli()
