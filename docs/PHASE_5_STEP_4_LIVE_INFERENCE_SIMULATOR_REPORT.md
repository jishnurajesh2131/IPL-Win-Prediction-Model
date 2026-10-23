# Phase 5 Step 4 — Local Live Inference Simulator Report

**Project:** Cloud-Based IPL Match Winning Probability Prediction & Live Match Analytics System  
**Pipeline Phase:** Phase 5 — Live / In-Match ML  
**Milestone:** Step 4 — Local Live Inference Simulator  
**Execution Scope:** Strictly Local Replay and Inference Simulation (Zero AWS resources, zero cloud deployment)  
**Authoritative ML Model:** Two-Model XGBoost Pipeline (Feature Set D: 54 raw / 141 transformed features)  
**Model Artifacts:**
- Innings 1 Model: `models/live/primary_live_model_inn1.joblib`
- Innings 2 Model: `models/live/primary_live_model_inn2.joblib`
- Innings 1 Preprocessor: `models/live/primary_live_preprocessor_inn1.joblib`
- Innings 2 Preprocessor: `models/live/primary_live_preprocessor_inn2.joblib`  
**Dataset Parity Benchmark:** `data/processed/live_dataset.csv` (SHA-256: `83b4a7bd362d873ae38c1b8c2737ff3054ea9e31a0019685448ca81c39c895df`)

---

## 1. Objective

The objective of Phase 5 Step 4 is to build, validate, and verify a robust local historical replay and inference simulator.

The simulator takes any historical IPL match from the processed Cricsheet data, replays it delivery-by-delivery as if it were unfolding live, reconstructs the exact physical and player states without lookahead leakage, invokes the frozen Two-Model XGBoost (Feature Set D) pipeline, and outputs probabilistic match trajectories ($P(\text{Team A Win})$ and $P(\text{Team B Win})$) from pre-match baseline ($T_0$) to match conclusion.

---

## 2. Existing Model & Artifact Verification

Prior to simulator implementation, all model and preprocessing artifacts under `models/live/` were inspected and verified:

| Artifact Path | Artifact Type | Feature Set | Input Dimension | Verification Status |
| :--- | :--- | :---: | :---: | :---: |
| `models/live/primary_live_model_inn1.joblib` | `xgboost.sklearn.XGBClassifier` | Set D | 141 features | **VERIFIED / OPERATIONAL** |
| `models/live/primary_live_model_inn2.joblib` | `xgboost.sklearn.XGBClassifier` | Set D | 141 features | **VERIFIED / OPERATIONAL** |
| `models/live/primary_live_preprocessor_inn1.joblib` | `LiveFeaturePreprocessor` | Set D | 54 raw $\rightarrow$ 141 OHE | **VERIFIED / OPERATIONAL** |
| `models/live/primary_live_preprocessor_inn2.joblib` | `LiveFeaturePreprocessor` | Set D | 54 raw $\rightarrow$ 141 OHE | **VERIFIED / OPERATIONAL** |
| `models/live/metadata/live_model_metadata.json` | Metadata JSON | — | — | **MATCHES STEP 3 REPORT** |

Zero model retraining occurred. All hyperparameters, estimators, and weights remain 100% frozen.

---

## 3. Simulator Architecture & Module Organization

The simulator is implemented as a modular package under `src/live_simulator/`:

```
src/live_simulator/
├── __init__.py          # Package exports: MatchSimulator, LiveModelRunner, SimulationResult
├── state.py             # MatchStateManager: Cumulative runs, wickets, legal balls, overs, CRR, target, RRR
├── player_state.py      # PlayerStateManager: Active batter/bowler figures and historical priors (< t)
├── matchup_state.py     # MatchupStateManager: Head-to-head encounters with Empirical Bayes shrinkage (m=24)
├── feature_builder.py   # LiveFeatureBuilder: 54 raw Feature Set D construction matching LiveFeaturePreprocessor
├── model_runner.py      # LiveModelRunner: Innings model routing and deterministic terminal state evaluation
├── simulator.py         # MatchSimulator: Master delivery-by-delivery replay engine and T0 generator
├── output.py            # JSON exporter, human-readable CLI formatter, and Matplotlib trajectory plotter
├── cli.py               # CLI entry point: python -m src.live_simulator.cli --match-id <MID> [--plot]
└── simulate_match.py    # Direct alias runner: python -m src.live_simulator.simulate_match --match-id <MID>
```

---

## 4. T0 Logic (Pre-Match Baseline)

Before the first delivery is bowled in Innings 1, the simulator generates a dedicated $T_0$ state representing pre-match odds:
- **State Values:** `innings = 1`, `score = 0`, `wickets = 0`, `legal_balls_bowled = 0`, `overs_completed = 0.0`.
- **Pre-Match Anchor:** Queried from `data/processed/walk_forward_pre_match_probs.csv`, which was generated strictly chronologically out-of-sample (zero self-inclusion, zero lookahead).
- **Probability Output:**
  - $P(\text{Team A Win}) = \text{pre\_match\_prob\_team\_a}$
  - $P(\text{Team B Win}) = 1.0 - P(\text{Team A Win})$
- **Prediction Type:** Tagged explicitly as `pre_match`.

---

## 5. Delivery State Transition

The match unfolds sequentially delivery by delivery. After each delivery event $k$:
1. The delivery is parsed (runs off bat, extras, wides, noballs, wickets, dismissals).
2. Physical cumulative counters update:
   - $\text{Runs} \leftarrow \text{Runs} + \text{runs\_off\_bat} + \text{extras}$
   - $\text{Wickets} \leftarrow \text{Wickets} + 1$ (if wicket fell)
   - $\text{Legal Balls} \leftarrow \text{Legal Balls} + 1$ (strictly if $\text{wides} == 0$ and $\text{noballs} == 0$)
   - $\text{Overs Completed} = \text{round}(\text{legal\_balls} / 6.0, 4)$
   - $\text{Current Run Rate} = \text{round}(\text{runs} / \max(0.1667, \text{legal\_balls} / 6.0), 4)$
3. Active partnership counters update (resets immediately on the next delivery after a dismissal).
4. Rolling momentum buffers update (last 6 and last 12 delivery outcomes: runs, wickets, dot ratio, boundary ratio).
5. All updates represent the instantaneous physical state immediately following delivery $k$.

---

## 6. Illegal Delivery Handling (Wides & No-Balls)

Wides and no-balls are recorded events that alter the state of the match and therefore represent prediction points:
- **Score Update:** Total runs increment by $\text{runs\_off\_bat} + \text{extras}$.
- **Legal Ball Invariant:** Legal delivery counters (`legal_balls_bowled`) do **NOT** increment.
- **Batter Balls Faced:** Increments on legal deliveries and no-balls, but does **NOT** increment on wides.
- **Bowler Figures:** Runs conceded by the bowler include bat runs, wides, and no-balls.
- **Model Invocation:** A prediction event is generated after every recorded illegal delivery, accurately capturing the in-play penalty and probability shift.

---

## 7. Current Player State (Micro-Features)

For each delivery event, micro-state features are extracted for the active striker and active bowler:
- **Active Batter:** `batter_runs_current_inn`, `batter_balls_current_inn`, `batter_strike_rate_current`, `batter_boundaries_current`, `batter_dots_current`, `batter_dot_ratio_current`.
- **Active Bowler:** `bowler_legal_balls_current`, `bowler_overs_current`, `bowler_runs_conceded_current`, `bowler_wickets_current`, `bowler_economy_current`, `bowler_dots_current`, `bowler_boundaries_conceded_current`, `bowler_dot_ratio_current`.
- **Temporal Isolation:** Derived strictly from in-progress accumulation within the current match up to event $k$. Final match scores are never accessed.

---

## 8. Historical Player State (Career Priors $< t$)

Historical player priors are queried from `HistoricalPlayerTracker`:
- **Chronological Boundary:** Updated strictly with matches completed before the target match date (`(date, match_id) < (curr_date, curr_match_id)`).
- **Empirical Bayes Shrinkage:**
  - Batting Strike Rate: shrunk towards league average with $m = 60.0$ balls prior.
  - Batting Average: shrunk towards league average with $m = 5.0$ outs prior.
  - Bowling Economy: shrunk towards league economy with $m = 10.0$ overs prior.
  - Bowling Strike Rate: shrunk towards league strike rate with $m = 5.0$ wickets prior.
- **Invariance:** Career priors remain fixed across all deliveries of the simulated match.

---

## 9. Batter-vs-Bowler Matchup Tracking ($m = 24.0$)

Head-to-head encounters are queried from `HistoricalMatchupTracker`:
- Tracks career balls, runs, and wickets between `(striker, bowler)` prior to match start.
- Implements continuous Empirical Bayes shrinkage with $m = 24.0$ balls (4 overs):
  $$\text{shrunk\_sr} = \frac{\text{prior\_runs} + m \cdot (\text{historical\_batter\_sr} / 100)}{\text{prior\_balls} + m} \times 100$$
- Falls back smoothly to Level 2 (historical batter SR) when prior encounters are zero.
- The tracker is updated strictly after the full match concludes, ensuring zero within-match leakage.

---

## 10. Innings Transition & Target Mechanics

When Innings 1 concludes:
1. Innings 1 final score is frozen ($\text{Runs}_1, \text{Wickets}_1$).
2. Target is established: $\text{Target} = \text{Runs}_1 + 1$.
3. Innings counters reset for Innings 2:
   - `score = 0`, `wickets = 0`, `legal_balls_bowled = 0`, `overs_completed = 0.0`
   - Active partnership runs and balls reset to 0.
   - Rolling delivery buffer is cleared.
4. Chasing metrics activate:
   - $\text{runs\_needed} = \max(0, \text{target} - \text{score})$
   - $\text{legal\_balls\_remaining} = \max(0, 120 - \text{legal\_balls\_bowled})$
   - $\text{required\_run\_rate} = \text{round}(\text{runs\_needed} / (\text{legal\_balls\_remaining} / 6.0), 4)$
   - $\text{rrr\_crr\_differential} = \text{current\_run\_rate} - \text{required\_run\_rate}$

---

## 11. Terminal State Handling (Deterministic Rules)

In accordance with Phase 5 Step 3 policy, machine learning models are neither necessary nor desirable when a match has mechanically concluded:
- **Chaser Achieved Target (`runs_needed == 0`):**
  - If chasing team is Team A: $P(\text{Team A Win}) = 1.0, P(\text{Team B Win}) = 0.0$ (`terminal_team_a_win`)
  - If chasing team is Team B: $P(\text{Team A Win}) = 0.0, P(\text{Team B Win}) = 1.0$ (`terminal_team_b_win`)
- **Defender Successfully Defended (`legal_balls_remaining == 0` or all-out with `runs_needed > 1`):**
  - If defending team is Team A: $P(\text{Team A Win}) = 1.0, P(\text{Team B Win}) = 0.0$ (`terminal_team_a_win`)
  - If defending team is Team B: $P(\text{Team A Win}) = 0.0, P(\text{Team B Win}) = 1.0$ (`terminal_team_b_win`)
- **Regulation Tie (`runs_needed == 1` with balls or wickets exhausted):**
  - $P(\text{Team A Win}) = 0.50, P(\text{Team B Win}) = 0.50$ (`terminal_tie`)
- Active non-terminal deliveries invoke the trained ML model.

---

## 12. Model Routing (Two-Model Architecture)

Deliveries are dynamically routed to the appropriate specialized model:
- **Innings 1 Deliveries:** Routed to `primary_live_preprocessor_inn1.joblib` and `primary_live_model_inn1.joblib`. Tagged as `model_inn1`.
- **Innings 2 Deliveries:** Routed to `primary_live_preprocessor_inn2.joblib` and `primary_live_model_inn2.joblib`. Tagged as `model_inn2`.
- Target orientation is strictly $P(\text{target\_team\_a\_win} == 1)$:
  - $P(\text{Team A Win}) = \hat{p}_1$
  - $P(\text{Team B Win}) = 1.0 - \hat{p}_1$
  regardless of whether Team A is batting or bowling.

---

## 13. Feature & Preprocessor Schema Compatibility

The feature vector assembled by `LiveFeatureBuilder` was validated against `LiveFeaturePreprocessor`:
- **Categorical Columns (4):** `team_a`, `team_b`, `venue`, `city` $\rightarrow$ OneHotEncoder produces 91 dummy features.
- **Numerical Columns (50):** Clipped and passed directly.
  - Required Run Rate clipped to $[0.0, 36.0]$.
  - RRR - CRR Differential clipped to $[-50.0, 50.0]$.
  - Overs Completed clamped to $[0.0, 20.0]$ (121-ball protection).
- **Total Transformed Feature Dimension:** Exactly 141 features, matching `n_features_in_ = 141` in both XGBoost models.

---

## 14. Dataset Agreement Validation vs `live_dataset.csv`

To confirm exact semantic fidelity, simulator-generated features were cross-referenced against stored rows in `data/processed/live_dataset.csv`:
- Checkpoints evaluated across early overs, middle overs, death overs, and innings transitions.
- **Results:**
  - Scores: 100% exact match ($0$ difference).
  - Wickets: 100% exact match ($0$ difference).
  - Legal balls: 100% exact match ($0$ difference).
  - Overs completed: Maximum difference $< 10^{-6}$.
  - Current Run Rate: Maximum difference $< 10^{-4}$.
  - Required Run Rate: Maximum difference $< 10^{-4}$.
  - Historical Player Priors: Bitwise identical (e.g. V Kohli 5,734 prior balls, SR 129.49, Avg 37.57).
  - Matchup Priors: Bitwise identical.

---

## 15. No-Lookahead Temporal Isolation Audit

Two strict anti-leakage tests were executed:
1. **Future Delivery Perturbation:** Perturbing delivery $K+10$ (e.g. modifying runs from 0 to 6) produced bitwise identical predictions and features for all events $\le K$.
2. **Final Outcome Perturbation:** Altering the official match winner in metadata did not alter non-terminal model predictions.

---

## 16. Automated Test Suite Results

All 15 automated test suites in `tests/test_live_simulator.py` passed with 100% compliance:

| Test Case | Test Description | Status |
| :--- | :--- | :---: |
| `test_01_t0_state` | T0 baseline initialization (score=0, wkts=0, pre-match anchor) | **PASS** |
| `test_02_legal_delivery` | Legal ball increments legal-ball count by exactly 1 | **PASS** |
| `test_03_wide_delivery` | Wide increments score, does NOT increment legal balls | **PASS** |
| `test_04_noball_delivery` | No-ball increments score, does NOT increment legal balls | **PASS** |
| `test_05_wicket_event` | Wicket count increments, partnership reset flag set | **PASS** |
| `test_06_innings_transition` | Transition from Innings 1 to 2, target calculated, counters reset | **PASS** |
| `test_07_no_lookahead` | Future delivery perturbation leaves prior predictions bitwise identical | **PASS** |
| `test_08_outcome_perturbation` | Metadata winner alteration does not impact in-play predictions | **PASS** |
| `test_09_terminal_state` | Deterministic terminal states (1.0/0.0 on win, 0.50 on regulation tie) | **PASS** |
| `test_10_probability_validity` | Bounds $0 \le P \le 1$ and $P(A) + P(B) \approx 1$ held across all events | **PASS** |
| `test_11_model_routing` | Innings 1 invokes Model 1, Innings 2 invokes Model 2 | **PASS** |
| `test_12_deterministic_replay` | Identical match replayed twice yields identical probabilities | **PASS** |
| `test_13_dataset_state_agreement` | Simulator matches `live_dataset.csv` states within numerical tolerance | **PASS** |
| `test_14_historical_player_leakage` | Player career totals exclude current and future matches | **PASS** |
| `test_15_matchup_leakage` | Current delivery not present in matchup history prior to prediction | **PASS** |

Regression testing across the entire project test suite (`tests/`) also passed with 100% success (25/25 total unit tests passing).

---

## 17. Representative Historical Match Replay Results

Five diverse historical IPL matches across test and training periods were simulated end-to-end:

| Match ID | Match Description & Teams | Season | Total Events | Official Winner | Final Model Output | Prediction State | Replay Runtime |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1426274** | **KKR vs RCB** (High-scoring thriller, 222 vs 221 all-out) | 2024 (Test) | 254 | Kolkata Knight Riders | KKR: 100.0% \| RCB: 0.0% | `terminal_team_a_win` | 8.25s |
| **1426298** | **KKR vs MI** (Moderate defense, 157 vs 139) | 2024 (Test) | 202 | Kolkata Knight Riders | KKR: 100.0% \| MI: 0.0% | `terminal_team_a_win` | 4.69s |
| **501228** | **PWI vs CSK** (2011 Final, 205 vs 147 chase defense) | 2011 (Train) | 248 | Chennai Super Kings | PWI: 0.0% \| CSK: 100.0% | `terminal_team_b_win` | 3.76s |
| **419155** | **CSK vs DC** (121-ball anomaly test, DC won by 6 wkts) | 2009/10 (Train) | 243 | Delhi Capitals | CSK: 0.0% \| DC: 100.0% | `terminal_team_b_win` | 6.60s |
| **1136620** | **SRH vs CSK** (Low-scoring defense thriller, 118 vs 87 all-out) | 2018 (Train) | 242 | Chennai Super Kings | SRH: 0.0% \| CSK: 100.0% | `terminal_team_b_win` | 4.82s |

---

## 18. Output Formats

### 18.1 Structured JSON Output
Saved under `data/processed/simulations/simulation_<match_id>.json`.  
Schema:
```json
{
  "match_id": "1426274",
  "season": "2024",
  "date": "2024-04-21",
  "venue": "Eden Gardens",
  "city": "Kolkata",
  "team_a": "Kolkata Knight Riders",
  "team_b": "Royal Challengers Bengaluru",
  "toss_winner": "Royal Challengers Bengaluru",
  "toss_decision": "field",
  "winner": "Kolkata Knight Riders",
  "total_events": 254,
  "events": [
    {
      "event_number": 0,
      "innings": 1,
      "ball": 0.0,
      "score": 0,
      "wickets": 0,
      "team_a_win_probability": 0.4956,
      "team_b_win_probability": 0.5044,
      "prediction_type": "pre_match"
    },
    {
      "event_number": 1,
      "innings": 1,
      "ball": 0.1,
      "score": 0,
      "wickets": 0,
      "team_a_win_probability": 0.4541,
      "team_b_win_probability": 0.5459,
      "prediction_type": "model_inn1"
    }
  ]
}
```

### 18.2 Human-Readable CLI Display
Formatted over-by-over scoreboard with win probabilities, CRR, RRR, active batter/bowler figures, and terminal conclusions.

---

## 19. Visualization

The simulator includes a matplotlib plotting function (`plot_simulation_trajectory`):
- Dual-panel layout:
  - Top: Team A and Team B win probability trajectories with 50% even-odds line and vertical dashed innings break marker.
  - Bottom: Cumulative match score and wickets fallen progression.
- Generated plots saved under `data/processed/simulations/trajectory_<match_id>.png`.

---

## 20. Performance

- **Per-Match Simulation Runtime:** Between **3.76s and 8.25s** on a standard local CPU, including prior match tracker replay, delivery-by-delivery state transitions, 200+ ML model evaluations, JSON export, and matplotlib trajectory rendering.
- **Memory Footprint:** Fully in-memory, zero persistent background services or external network dependencies.

---

## 21. Known Limitations

1. **In-Match Ball Trajectory Physics:** Cricsheet data represents discrete ball-by-ball event outcomes; ball release velocity, swing, and hawkeye pitch tracking are not present.
2. **Mid-Innings Tactical Substitutions (Impact Players):** Tracked through active striker/bowler identification, but tactical substitute availability is not modeled as an explicit discrete rule flag.
3. **Severe Rain Halts Beyond D/L:** The 12 matches that halted prematurely without official D/L calculation end at their final recorded delivery.

---

## 22. Files Created Inventory

```
src/live_simulator/
├── __init__.py               # Package exports
├── state.py                  # MatchStateManager
├── player_state.py           # PlayerStateManager
├── matchup_state.py          # MatchupStateManager
├── feature_builder.py        # LiveFeatureBuilder
├── model_runner.py           # LiveModelRunner
├── simulator.py              # MatchSimulator
├── output.py                 # JSON, CLI & Plotting utilities
├── cli.py                    # Command-line interface
└── simulate_match.py         # Convenience CLI runner

tests/
└── test_live_simulator.py    # 15 automated test suites

data/processed/simulations/
├── simulation_1426274.json   # Simulation JSON for KKR vs RCB 2024
├── simulation_1426298.json   # Simulation JSON for KKR vs MI 2024
├── simulation_501228.json    # Simulation JSON for PWI vs CSK 2011
├── simulation_419155.json    # Simulation JSON for CSK vs DC 2010
├── simulation_1136620.json   # Simulation JSON for SRH vs CSK 2018
├── trajectory_1426274.png    # Plot PNG for KKR vs RCB 2024
├── trajectory_1426298.png    # Plot PNG for KKR vs MI 2024
├── trajectory_501228.png     # Plot PNG for PWI vs CSK 2011
├── trajectory_419155.png     # Plot PNG for CSK vs DC 2010
└── trajectory_1136620.png    # Plot PNG for SRH vs CSK 2018

docs/
└── PHASE_5_STEP_4_LIVE_INFERENCE_SIMULATOR_REPORT.md  # Master Technical Report
```

---

## 23. Reproducibility & Determinism

- **Bitwise Deterministic Replay:** Replaying the identical match produces identical probabilities across all deliveries ($\Delta = 0.0$).
- **Python Version:** 3.14.7
- **XGBoost Version:** 3.4.1
- **Scikit-Learn Version:** 1.9.0
- **Random Seeds:** Enforced `random_state = 42`.

---

## 24. Final Status

```text
PHASE 5 STEP 4 STATUS: LOCAL LIVE INFERENCE SIMULATOR COMPLETE
```
