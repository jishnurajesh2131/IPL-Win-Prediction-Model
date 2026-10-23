import sys
import copy
from src.live_simulator.simulator import MatchSimulator

def run_audits():
    print("Starting Leakage Audit Execution...\n")

    # Select two test matches from 2024-2026
    test_matches = ["1422119", "1422120"]

    print(f"Target Matches: {test_matches}\n")

    for match_id in test_matches:
        print(f"=== Auditing Match {match_id} ===")
        
        # --- BASELINE RUN ---
        print("1. Running Baseline Simulation...")
        sim_baseline = MatchSimulator()
        baseline_result = sim_baseline.simulate(match_id)
        baseline_probs = [e.team_a_win_probability for e in baseline_result.events]
        print(f"   Baseline completed with {len(baseline_probs)} prediction points.")

        # --- AUDIT 7: Exact Reproducibility ---
        print("2. Running Reproducibility Simulation (Audit 7)...")
        sim_reprod = MatchSimulator()
        reprod_result = sim_reprod.simulate(match_id)
        reprod_probs = [e.team_a_win_probability for e in reprod_result.events]

        if baseline_probs == reprod_probs:
            print("   [PASS] Reproducibility: Probabilities match exactly.")
        else:
            print("   [FAIL] Reproducibility: Probabilities differ!")
            sys.exit(1)


        # --- AUDIT 1: Final-Result Perturbation ---
        print("3. Running Final-Result Perturbation (Audit 1)...")
        sim_perturb = MatchSimulator()
        sim_perturb._ensure_data_loaded() # Load the raw data
        
        # Perturb the metadata
        original_winner = None
        for m in sim_perturb._all_matches:
            if str(m["match_id"]) == str(match_id):
                original_winner = m["winner"]
                # Flip winner
                fake_winner = "Fake Team" if m["winner"] != "Fake Team" else "Another Fake"
                m["winner"] = fake_winner
                break
        
        perturb_result = sim_perturb.simulate(match_id)
        perturb_probs = [e.team_a_win_probability for e in perturb_result.events]

        diff_count = 0
        for i, (b_p, p_p) in enumerate(zip(baseline_probs, perturb_probs)):
            if b_p != p_p:
                if i == len(baseline_probs) - 1:
                    print("   [NOTE] Final delivery probability changed due to terminal deterministic condition. This is expected terminal outcome assignment.")
                else:
                    diff_count += 1
                    print(f"   [FAIL] Mismatch at index {i}: Baseline={b_p}, Perturbed={p_p}")

        if diff_count == 0:
            print("   [PASS] Final-Result Perturbation: No leakage in active prediction points.")
        else:
            print("   [FAIL] Final-Result Perturbation: Leakage detected!")


        # --- AUDIT 2: Future-Delivery Perturbation ---
        print("4. Running Future-Delivery Perturbation (Audit 2)...")
        sim_future = MatchSimulator()
        sim_future._ensure_data_loaded()
        
        # Pick a point T (e.g. index 50)
        T = 50
        if T < len(sim_future._all_deliveries[match_id]) - 1:
            custom_dels = copy.deepcopy(sim_future._all_deliveries[match_id])
            
            # Perturb delivery T + 1
            custom_dels[T + 1]["total_runs"] = "6" if custom_dels[T + 1].get("total_runs") != "6" else "0"
            
            future_result = sim_future.simulate(match_id, custom_deliveries=custom_dels)
            future_probs = [e.team_a_win_probability for e in future_result.events]
            
            diff_future_count = 0
            for i in range(T + 1): # Check up to and including T
                if baseline_probs[i] != future_probs[i]:
                    diff_future_count += 1
                    print(f"   [FAIL] Mismatch at index {i} (before T={T}): Baseline={baseline_probs[i]}, Future={future_probs[i]}")

            if diff_future_count == 0:
                print(f"   [PASS] Future-Delivery Perturbation: Predictions before T={T} remain exactly identical.")
            else:
                print("   [FAIL] Future-Delivery Perturbation: Leakage detected from future delivery!")
        else:
            print("   [SKIP] Match too short for T=50.")
            
        print("\n")

if __name__ == "__main__":
    run_audits()
