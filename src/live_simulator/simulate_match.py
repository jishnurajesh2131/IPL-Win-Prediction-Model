"""
Convenience entry point for live match simulation.

Usage:
    python -m src.live_simulator.simulate_match --match-id <MATCH_ID> [--plot]
"""

from .cli import run_cli

if __name__ == "__main__":
    run_cli()
