"""
Validation Script: End-to-End Replay Suite for Phase 5 Step 5.

Replays 11 diverse historical matches across all IPL eras:
- 335982 (2008, KKR vs RCB, Match 1, McCullum 158*, cold-start 0.50 anchor, 140-run win)
- 419155 (2010, CSK vs DC, 121-ball anomaly test)
- 501228 (2011, PWI vs CSK, high-scoring defense, 205 vs 147)
- 598027 (2013, SRH vs RCB, regulation tie: 130/7 vs 130/8)
- 1082591 (2017, RPS vs MI, 2017 Final, 1-run thriller: 129 vs 128)
- 1136620 (2018, SRH vs CSK, low-scoring defense thriller: 118 vs 87)
- 1359507 (2023, GT vs KKR, Rinku Singh 5 sixes miracle chase: 204 vs 207)
- 1359544 (2023, CSK vs GT, 2023 Qualifier 1: 172 vs 157)
- 1422131 (2024, KKR vs SRH, held-out test period, 200+ thriller: 208 vs 204)
- 1426274 (2024, KKR vs RCB, held-out test period, 222 vs 221 all out)
- 1426298 (2024, KKR vs MI, held-out test period, moderate defense: 157 vs 139)

Computes probability validity, terminal states, checkpoint summaries, and runtimes.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import json
import pandas as pd
import numpy as np

from src.live_simulator.simulator import MatchSimulator
from src.live_simulator.output import save_simulation_json, plot_simulation_trajectory


def run_replay_validation():
    print("=" * 70)
    print("PHASE 5 STEP 5: END-TO-END HISTORICAL REPLAY VALIDATION (11 MATCHES)")
    print("=" * 70)

    sim = MatchSimulator()
    matches = [
        ("335982", "2007/08 (Train)", "First-ever IPL match, McCullum 158*, cold start"),
        ("419155", "2009/10 (Train)", "121-ball anomaly test (Over 19 had 7 balls)"),
        ("501228", "2011 (Train)", "2011 Final, dominant high-scoring defense"),
        ("598027", "2013 (Train)", "Regulation tie (130/7 vs 130/8)"),
        ("1082591", "2017 (Train)", "2017 Final, 1-run low-scoring thriller (129 vs 128)"),
        ("1136620", "2018 (Train)", "Low-scoring defense thriller (118 vs 87)"),
        ("1359507", "2023 (Val)", "Rinku Singh 5 sixes miracle chase (204 vs 207)"),
        ("1359544", "2023 (Val)", "2023 Qualifier 1, defense (172 vs 157)"),
        ("1422131", "2024 (Test)", "Held-out test period, 200+ thriller (208 vs 204)"),
        ("1426274", "2024 (Test)", "Held-out test period, 1-run thriller (222 vs 221)"),
        ("1426298", "2024 (Test)", "Held-out test period, moderate defense (157 vs 139)"),
    ]

    total_events_all = 0
    invalid_prob_events = 0
    min_p_a_all = 1.0
    max_p_a_all = 0.0
    min_p_b_all = 1.0
    max_p_b_all = 0.0
    max_prob_sum_error = 0.0

    active_events_sub_1pct = 0
    active_events_super_99pct = 0
    min_active_p_a = 1.0
    max_active_p_a = 0.0

    runtimes = []
    checkpoint_rows = []
    summary_rows = []

    for mid, era_label, note in matches:
        t0 = time.time()
        res = sim.simulate(mid)
        elapsed = time.time() - t0
        runtimes.append(elapsed)

        # Save JSON & PNG
        json_path = save_simulation_json(res)
        plot_path = plot_simulation_trajectory(res)

        n_events = len(res.events)
        total_events_all += n_events

        # Checkpoints extraction
        t0_p_a = res.events[0].team_a_win_probability
        pp_p_a = None
        inn_break_p_a = None
        over15_p_a = None
        pre_terminal_p_a = None
        terminal_p_a = res.events[-1].team_a_win_probability
        terminal_pred_type = res.events[-1].prediction_type

        # Delivery probability audits
        for idx, e in enumerate(res.events):
            p_a = e.team_a_win_probability
            p_b = e.team_b_win_probability

            # Range & sum validity
            if p_a < 0.0 or p_a > 1.0 or p_b < 0.0 or p_b > 1.0:
                invalid_prob_events += 1
            sum_err = abs((p_a + p_b) - 1.0)
            if sum_err > max_prob_sum_error:
                max_prob_sum_error = sum_err

            min_p_a_all = min(min_p_a_all, p_a)
            max_p_a_all = max(max_p_a_all, p_a)
            min_p_b_all = min(min_p_b_all, p_b)
            max_p_b_all = max(max_p_b_all, p_b)

            # Active (non-terminal, non-T0) probabilities
            if e.prediction_type not in ("pre_match", "terminal_team_a_win", "terminal_team_b_win", "terminal_tie"):
                min_active_p_a = min(min_active_p_a, p_a)
                max_active_p_a = max(max_active_p_a, p_a)
                if p_a < 0.01:
                    active_events_sub_1pct += 1
                elif p_a > 0.99:
                    active_events_super_99pct += 1

            # Track checkpoints
            if e.innings == 1 and int(e.overs_completed) == 6 and pp_p_a is None:
                pp_p_a = p_a

            if e.innings == 2 and inn_break_p_a is None:
                # Event before innings 2 is final innings 1 event
                inn_break_p_a = res.events[idx - 1].team_a_win_probability

            if e.innings == 2 and int(e.overs_completed) == 15 and over15_p_a is None:
                over15_p_a = p_a

        # Pre-terminal event is the second to last event
        if len(res.events) >= 2:
            pre_terminal_p_a = res.events[-2].team_a_win_probability

        checkpoint_rows.append({
            "Match ID": mid,
            "Season / Era": era_label,
            "Teams": f"{res.team_a} vs {res.team_b}",
            "Winner": res.winner,
            "T0 P(A)": f"{t0_p_a:.4f}",
            "Powerplay P(A)": f"{pp_p_a:.4f}" if pp_p_a is not None else "N/A",
            "Inn Break P(A)": f"{inn_break_p_a:.4f}" if inn_break_p_a is not None else "N/A",
            "15-Over P(A)": f"{over15_p_a:.4f}" if over15_p_a is not None else "N/A",
            "Pre-Term P(A)": f"{pre_terminal_p_a:.4f}" if pre_terminal_p_a is not None else "N/A",
            "Term P(A)": f"{terminal_p_a:.4f}",
            "Terminal State": terminal_pred_type,
        })

        summary_rows.append({
            "match_id": mid,
            "era": era_label,
            "teams": f"{res.team_a} vs {res.team_b}",
            "events": n_events,
            "runtime_s": round(elapsed, 2),
            "winner": res.winner,
            "terminal_state": terminal_pred_type,
        })

        print(f"Replayed {mid} ({era_label}) in {elapsed:.2f}s | Events: {n_events} | Winner: {res.winner} | Term: {terminal_pred_type}")

    # Output Summary Table
    df_checkpoints = pd.DataFrame(checkpoint_rows)
    print("\n" + "=" * 70)
    print("HISTORICAL REPLAY CHECKPOINT SUMMARY TABLE")
    print("=" * 70)
    print(df_checkpoints.to_string(index=False))

    df_summary = pd.DataFrame(summary_rows)
    print("\n" + "=" * 70)
    print("RUNTIME BENCHMARK SUMMARY")
    print("=" * 70)
    print(df_summary.to_string(index=False))
    print("-" * 70)
    print(f"Total Matches Replayed: {len(matches)}")
    print(f"Total Delivery Events Evaluated: {total_events_all:,}")
    print(f"Average Runtime: {np.mean(runtimes):.2f}s | Median: {np.median(runtimes):.2f}s | Min: {np.min(runtimes):.2f}s | Max: {np.max(runtimes):.2f}s")
    print(f"Invalid Probability Events: {invalid_prob_events} (Expected: 0)")
    print(f"Max Probability Sum Error: {max_prob_sum_error:.6f} (Expected: < 1e-4)")
    print(f"Overall Prob Bounds: P(A) in [{min_p_a_all:.4f}, {max_p_a_all:.4f}] | P(B) in [{min_p_b_all:.4f}, {max_p_b_all:.4f}]")
    print(f"Active Non-Terminal P(A) Bounds: [{min_active_p_a:.4f}, {max_active_p_a:.4f}]")
    print(f"Active Events < 1%: {active_events_sub_1pct} | Active Events > 99%: {active_events_super_99pct}")

    # Save metrics to scratch for report inclusion
    metrics_out = {
        "total_matches_replayed": len(matches),
        "total_events_all": total_events_all,
        "invalid_prob_events": invalid_prob_events,
        "max_prob_sum_error": max_prob_sum_error,
        "min_p_a_all": min_p_a_all,
        "max_p_a_all": max_p_a_all,
        "min_p_b_all": min_p_b_all,
        "max_p_b_all": max_p_b_all,
        "min_active_p_a": min_active_p_a,
        "max_active_p_a": max_active_p_a,
        "active_events_sub_1pct": active_events_sub_1pct,
        "active_events_super_99pct": active_events_super_99pct,
        "runtime_mean": round(float(np.mean(runtimes)), 2),
        "runtime_median": round(float(np.median(runtimes)), 2),
        "runtime_min": round(float(np.min(runtimes)), 2),
        "runtime_max": round(float(np.max(runtimes)), 2),
        "checkpoints": checkpoint_rows,
        "summary": summary_rows,
    }
    with open("scratch/replay_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)
    print("\nSaved replay_metrics.json successfully.")


if __name__ == "__main__":
    run_replay_validation()
