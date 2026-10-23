# Phase 5 Step 3 — Live ML Model Training & Evaluation Report

**Project:** Cloud-Based IPL Match Winning Probability Prediction & Live Match Analytics System  
**Pipeline Phase:** Phase 5 — Live / In-Match ML  
**Milestone:** Live Model Training, Ablation, Architecture Comparison & Selection  
**Execution Scope:** Strictly Local Machine Learning (Zero AWS resources, zero cloud deployment)  
**Authoritative Dataset:** `data/processed/live_dataset.csv` (SHA-256: `83b4a7bd362d873ae38c1b8c2737ff3054ea9e31a0019685448ca81c39c895df`)  
**Frozen Upstream Artifacts:** Phase 3 pre-match models (`models/random_forest.joblib`, `models/preprocessor.joblib`) remain 100% frozen.

---

## 1. Objective

The objective of Phase 5 Step 3 is to train, evaluate, ablate, and select the primary local machine learning model for **Live / In-Match Win Probability Prediction**.

At any arbitrary legal delivery $N$ during a live match, the model consumes the current physical match state, player micro-states, historical player priors ($< t$), batter-vs-bowler matchups ($< t$), and the leak-free pre-match walk-forward anchor to output:

$$P(\text{Team A Win} \mid \mathbf{x}_{\text{live}, N})$$

The model candidate is selected strictly through out-of-time probability quality evaluated on historical out-of-sample seasons, following a rigorous metric hierarchy:

$$\text{Log Loss} \longrightarrow \text{Brier Score} \longrightarrow \text{Expected Calibration Error (ECE)} \longrightarrow \text{Phase Consistency} \longrightarrow \text{Accuracy} \longrightarrow \text{ROC-AUC}$$

---

## 2. Dataset Verification

The live dataset generated and verified in Phase 5 Step 2 was re-audited before model training:

| Dimension | Expected / Audited Specification | Actual Dataset Value | Verification Status |
| :--- | :--- | :--- | :---: |
| **Dataset Path** | `data/processed/live_dataset.csv` | `data/processed/live_dataset.csv` | **MATCH** |
| **Total Rows** | 290,598 delivery states | 290,598 | **EXACT MATCH** |
| **Unique Matches** | 1,211 eligible matches | 1,211 | **EXACT MATCH** |
| **Total Columns** | 66 columns | 66 | **EXACT MATCH** |
| **Target Column** | `target_team_a_win` | `target_team_a_win` | **EXACT MATCH** |
| **Target Distribution** | 50.24% Class 1, 49.76% Class 0 | 145,987 Class 1 (50.24%), 144,611 Class 0 (49.76%) | **EXACT MATCH** |
| **Innings Split** | ~51.6% Innings 1, ~48.4% Innings 2 | 149,928 Innings 1 (51.59%), 140,670 Innings 2 (48.41%) | **EXACT MATCH** |
| **Dataset Hash** | Bitwise reproducible SHA-256 | `83b4a7bd362d873ae38c1b8c2737ff3054ea9e31a0019685448ca81c39c895df` | **VERIFIED** |

---

## 3. Terminal-State Audit

In `data/processed/live_dataset.csv`, every row represents the state of a match **after** a delivery has occurred. Consequently, the final recorded delivery of Innings 2 produces a state where the match has conclusively ended.

### Audit Findings:
1. **Mechanical Ending States in Innings 2:**
   - `runs_needed == 0` (chasing team achieved target): **650 rows across 650 matches**
   - `wickets_remaining == 0` (chasing team all out): **140 rows across 140 matches**
   - `legal_balls_remaining == 0` (20 overs completed): **463 rows across 463 matches**
   - **Union of Terminal States:** Exactly **1,199 rows across 1,199 matches**.
2. **Delivery Position Audit:**
   - Exactly **0 non-last deliveries** have terminal states. All 1,199 rows occur strictly on the final recorded delivery of the match.
   - The remaining 12 matches out of 1,211 concluded on their final delivery due to rain / revised target stoppages before standard regulation limits were reached.
3. **Core ML Treatment & Live Inference Decision Rule:**
   - In actual live inference, an ML model is neither necessary nor desirable once a match is concluded.
   - For the core ML experiment, these **1,199 deterministic post-match states were isolated** from the model fitting and evaluation matrices, leaving **289,399 active in-match predictive deliveries**.
   - **Planned Live Inference Decision Logic:**
     - Match conclusively won (`runs_needed == 0` or all-out / overs expired) $\longrightarrow$ output `1.0` or `0.0` deterministically.
     - Regulation tie pending Super Over $\longrightarrow$ output `0.50` / `0.50` deterministically.
     - Active in-progress delivery $\longrightarrow$ invoke trained ML model.

---

## 4. Chronological Split

The dataset was partitioned using strict **match-level chronological separation** to guarantee zero intra-match delivery leakage:

| Split | Season Window | Eligible Matches | Active Delivery States | Delivery Team A Win % | Match Team A Win % | Match Overlap |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **TRAIN** | 2008–2021 (14 seasons) | **853** | **203,458** (70.30%) | 50.66% | 50.64% | **0 matches** |
| **VALIDATION** | 2022–2023 (2 seasons) | **145** | **35,074** (12.12%) | 53.45% | 52.41% | **0 matches** |
| **TEST** | 2024–2026 (3 seasons) | **213** | **50,867** (17.58%) | 46.34% | 45.07% | **0 matches** |
| **TOTAL** | **2008–2026 (19 seasons)** | **1,211** | **289,399** (100.0%) | **50.24%** | **49.88%** | **0 overlap** |

No match in the Validation or Test sets was ever seen by the training pipeline or preprocessor.

---

## 5. Feature Role Audit

The 66 columns in `live_dataset.csv` were systematically audited and classified into four distinct roles:

1. **Target (1 column):** `target_team_a_win` (isolated strictly as label $y$).
2. **Metadata / Traceability (10 columns):** `match_id`, `season`, `date`, `delivery_index`, `ball`, `batting_team`, `bowling_team`, `striker`, `non_striker`, `bowler`.
3. **Redundant / Excluded (1 column):** `innings` (excluded because `current_innings` represents the identical information).
4. **Raw Predictor Features (54 columns):**
   - **Categorical (4):** `team_a`, `team_b`, `venue`, `city`
   - **Binary Match Context (3):** `toss_winner_is_team_a`, `toss_decision_is_bat`, `team_a_bats_first`
   - **Macro Match State (23):** `current_innings`, `is_team_a_batting`, `current_innings_runs`, `current_innings_wickets`, `wickets_remaining`, `legal_balls_bowled`, `legal_balls_remaining`, `overs_completed`, `current_run_rate`, `target_runs`, `runs_needed`, `required_run_rate`, `rrr_crr_differential`, `runs_last_6_balls`, `wickets_last_6_balls`, `runs_last_12_balls`, `wickets_last_12_balls`, `dot_ball_ratio_last_12`, `boundary_ratio_last_12`, `current_partnership_runs`, `current_partnership_balls`, `current_partnership_run_rate`, `pre_match_prob_team_a`
   - **Current Player Micro-State (14):** `batter_runs_current_inn`, `batter_balls_current_inn`, `batter_strike_rate_current`, `batter_boundaries_current`, `batter_dots_current`, `batter_dot_ratio_current`, `bowler_legal_balls_current`, `bowler_overs_current`, `bowler_runs_conceded_current`, `bowler_wickets_current`, `bowler_economy_current`, `bowler_dots_current`, `bowler_boundaries_conceded_current`, `bowler_dot_ratio_current`
   - **Historical Player Priors (6):** `historical_batter_strike_rate`, `historical_batter_average`, `historical_batter_balls_faced`, `historical_bowler_economy`, `historical_bowler_strike_rate`, `historical_bowler_balls_bowled`
   - **Batter-vs-Bowler Matchups (4):** `matchup_balls_prior`, `matchup_runs_prior`, `matchup_wickets_prior`, `matchup_strike_rate_shrunk`

**Transformed Dimension:** Fitting `OneHotEncoder(handle_unknown='ignore')` on the 4 categorical columns across the Training split produces **91 encoded dummy features**. Combined with the 50 numerical features, the full design matrix for Model D contains **141 transformed features**.

---

## 6. Preprocessing

The preprocessing pipeline was encapsulated inside `LiveFeaturePreprocessor` (`src/live_prediction/live_model_preprocessor.py`):

1. **Categorical Encoding:** `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` was fitted **strictly on the Training split**. Unseen categories in Validation or Test are represented as all-zero vectors.
2. **Linear Preprocessing (Logistic Regression):** Numerical features are clipped and transformed using `StandardScaler` fitted strictly on the Training split.
3. **Tree Preprocessing (Random Forest & XGBoost):** Numerical features are clipped and passed directly without scaling, preserving physical tree splitting thresholds.

---

## 7. RRR / Extreme-Value Handling & 121-Ball Anomaly

1. **Required Run Rate (RRR):**
   - **Raw Dataset Range:** Min = `0.0`, Max = `792.0` (in mathematically lost chases needing 132 off 1 ball).
   - **Model-Facing Pipeline:** Clipped strictly to $[0.0, 36.0]$.
2. **RRR / CRR Differential:**
   - **Raw Dataset Range:** Min = `-785.34`, Max = `50.36`.
   - **Model-Facing Pipeline:** Clipped strictly to $[-50.0, 50.0]$.
3. **Overs Completed (121-Ball Anomaly in Match 419155):**
   - **Raw Dataset Range:** Min = `0.0`, Max = `20.1667` (CSK vs DC, 2010/04/15, 7 legal deliveries bowled in Over 19).
   - **Model-Facing Pipeline:** Clamped strictly to $[0.0, 20.0]$.
   - Raw event logs remain 100% unaltered.

---

## 8. Model A — Match State Baseline

- **Features (30 raw / 117 transformed):** Match state, toss, venue, polarity, score, wickets, overs, CRR, target, RRR, momentum, and pre-match probability anchor.
- **Results:**
  - Unified LR: Val LogLoss = `0.7184`, Test LogLoss = `0.6914`
  - Unified RF: Val LogLoss = `0.6619`, Test LogLoss = `0.6392`
  - Unified XGB: Val LogLoss = `0.6475`, Test LogLoss = `0.5889`
  - Two-Model XGB: Val LogLoss = `0.6204`, Test LogLoss = `0.5747`

---

## 9. Model B — Current Player Performance

- **Features (44 raw / 131 transformed):** Model A + 14 in-match active striker and bowler micro-features.
- **Results:**
  - Unified LR: Val LogLoss = `0.7191`, Test LogLoss = `0.6906`
  - Unified RF: Val LogLoss = `0.6627`, Test LogLoss = `0.6405`
  - Unified XGB: Val LogLoss = `0.6543`, Test LogLoss = `0.5942`
- **Finding:** In a single unified model, adding current player figures without innings separation introduces slight noise (+0.0068 Val LogLoss for XGBoost).

---

## 10. Model C — Historical Player Priors

- **Features (50 raw / 137 transformed):** Model B + 6 career player priors ($< t$) shrunk towards league baselines.
- **Results:**
  - Unified LR: Val LogLoss = `0.7213`, Test LogLoss = `0.6906`
  - Unified RF: Val LogLoss = `0.6622`, Test LogLoss = `0.6389`
  - Unified XGB: Val LogLoss = `0.6617`, Test LogLoss = `0.5903`

---

## 11. Model D — Batter-vs-Bowler Matchups

- **Features (54 raw / 141 transformed):** Model C + 4 Empirical Bayes shrunk head-to-head interaction features ($m = 24.0$ balls).
- **Results:**
  - Unified LR: Val LogLoss = `0.7220`, Test LogLoss = `0.6912`
  - Unified RF: Val LogLoss = `0.6572`, Test LogLoss = `0.6371`
  - Unified XGB: Val LogLoss = `0.6554`, Test LogLoss = `0.5906`
  - Two-Model RF: Val LogLoss = `0.6293`, Test LogLoss = `0.6089`
  - **Two-Model XGB: Val LogLoss = `0.6197`, Test LogLoss = `0.5735` (WINNER — Best Evaluated Two-Model Configuration)**
- **Note on Two-Model Evaluations:** Two-Model experiments were evaluated for Feature Set A and Feature Set D; Two-Model B and Two-Model C were not included in the reported Two-Model comparison.

---

## 12. Logistic Regression Evaluation

- Baseline linear model with `StandardScaler`, `C=1.0`, `max_iter=1000`.
- Validation Log Loss hovered around `0.7015` – `0.7220`; Test Log Loss around `0.6714` – `0.6914`.
- **Diagnosis:** Severely underperforms tree models because live cricket probabilities require capturing non-linear interactions (e.g. 10 runs needed off 6 balls with 8 wickets in hand vs 1 wicket in hand). A linear hyperplane cannot effectively separate high-dimensional boundary states.

---

## 13. Random Forest Evaluation

- Baseline tree ensemble (`n_estimators=100`, `max_depth=6`, `min_samples_leaf=5`, `random_state=42`).
- Robust and stable across all feature sets.
- Unified Set D: Val LogLoss = `0.6572`, Test LogLoss = `0.6371`.
- Two-Model Set D: Val LogLoss = `0.6293`, Test LogLoss = `0.6089`, Test Brier = `0.2094`, ROC-AUC = `0.7804`.

---

## 14. XGBoost Evaluation

- Gradient boosted decision trees (`n_estimators=100`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_alpha=0.1`, `reg_lambda=1.0`, `random_state=42`).
- Fast training (2.5 – 5.5s per run) and superior probability estimation.
- Unified Set A: Val LogLoss = `0.6475`, Test LogLoss = `0.5889`.
- **Two-Model Set D:** Val LogLoss = `0.6197`, Test LogLoss = `0.5735`, Test Brier = `0.1967`, Test Accuracy = `68.29%`, Test ROC-AUC = `0.7742`.

---

## 15. Unified Architecture (Architecture A)

- A single model fitted on both Innings 1 and Innings 2 deliveries ($N = 203,458$).
- Must learn that chase features (`target_runs`, `runs_needed`, `required_run_rate`, `rrr_crr_differential`) are identically 0.0 during Innings 1 and physically activate only in Innings 2.
- Best Unified run: **XGBoost Set A** (Val LogLoss = `0.6475`, Test LogLoss = `0.5889`).

---

## 16. Innings-Specific Architecture (Architecture B)

- Splits the problem into two specialized models, both predicting `target_team_a_win`:
  - **Model 1 (Innings 1):** Trained on 105,274 Innings 1 deliveries, learning projected total dynamics and first-innings wicket rates.
  - **Model 2 (Innings 2):** Trained on 98,184 Innings 2 deliveries, learning chase physics, target proximity, and death-over pressure.
- **Result:** Architecture B clearly outperformed Unified Architecture A across every algorithm in the reported experiments. For XGBoost Set D, Val Log Loss dropped from `0.6554` to `0.6197` (-0.0357), and Test Log Loss dropped from `0.5906` to `0.5735` (-0.0171).

---

## 17. Master Probability Metrics Table

The complete evaluation across all 18 experimental configurations is detailed below:

| Architecture | Feature Set | Model Algorithm | Train Time (s) | Val LogLoss | Val Brier | Val ECE | Val Acc | Test LogLoss | Test Brier | Test ECE | Test Acc | Test AUC |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Unified | A | Logistic Regression | 13.44 | 0.7184 | 0.2597 | 0.0747 | 0.5567 | 0.6914 | 0.2485 | 0.1046 | 0.5495 | 0.5910 |
| Unified | A | Random Forest | 17.15 | 0.6619 | 0.2348 | 0.0469 | 0.5829 | 0.6392 | 0.2237 | 0.0951 | 0.6555 | 0.7142 |
| Unified | A | XGBoost | 2.51 | 0.6475 | 0.2298 | 0.0649 | 0.5855 | 0.5889 | 0.2028 | 0.0564 | 0.6702 | 0.7444 |
| Unified | B | Logistic Regression | 14.52 | 0.7191 | 0.2599 | 0.0709 | 0.5628 | 0.6906 | 0.2480 | 0.1030 | 0.5528 | 0.5933 |
| Unified | B | Random Forest | 18.91 | 0.6627 | 0.2352 | 0.0433 | 0.5876 | 0.6405 | 0.2244 | 0.0980 | 0.6536 | 0.7130 |
| Unified | B | XGBoost | 4.23 | 0.6543 | 0.2329 | 0.0768 | 0.5665 | 0.5942 | 0.2052 | 0.0563 | 0.6583 | 0.7348 |
| Unified | C | Logistic Regression | 19.21 | 0.7213 | 0.2607 | 0.0764 | 0.5531 | 0.6906 | 0.2478 | 0.0959 | 0.5601 | 0.5938 |
| Unified | C | Random Forest | 18.57 | 0.6622 | 0.2349 | 0.0449 | 0.5894 | 0.6389 | 0.2235 | 0.1012 | 0.6525 | 0.7213 |
| Unified | C | XGBoost | 3.68 | 0.6617 | 0.2362 | 0.0772 | 0.5643 | 0.5903 | 0.2034 | 0.0554 | 0.6613 | 0.7401 |
| Unified | D | Logistic Regression | 18.40 | 0.7220 | 0.2610 | 0.0771 | 0.5517 | 0.6912 | 0.2481 | 0.0964 | 0.5597 | 0.5930 |
| Unified | D | Random Forest | 17.28 | 0.6572 | 0.2326 | 0.0541 | 0.5948 | 0.6371 | 0.2227 | 0.1000 | 0.6513 | 0.7183 |
| Unified | D | XGBoost | 5.12 | 0.6554 | 0.2337 | 0.0841 | 0.5594 | 0.5906 | 0.2037 | 0.0608 | 0.6688 | 0.7390 |
| **Two-Model** | **A** | **Random Forest** | 14.94 | 0.6384 | 0.2236 | 0.0680 | 0.6469 | 0.6144 | 0.2119 | 0.1219 | 0.6990 | 0.7771 |
| **Two-Model** | **A** | **XGBoost** | 5.49 | 0.6204 | 0.2184 | 0.0365 | 0.6264 | 0.5747 | 0.1975 | 0.0626 | 0.6781 | 0.7704 |
| **Two-Model** | **A** | **Logistic Regression**| 12.62 | 0.7015 | 0.2516 | 0.0535 | 0.5773 | 0.6726 | 0.2398 | 0.0782 | 0.5715 | 0.6311 |
| **Two-Model** | **D** | **Random Forest** | 19.10 | 0.6293 | 0.2193 | 0.0717 | 0.6556 | 0.6089 | 0.2094 | 0.1185 | 0.7039 | 0.7804 |
| **Two-Model** | **D** | **XGBoost (WINNER)** | **4.39** | **0.6197** | **0.2180** | **0.0307** | **0.6283** | **0.5735** | **0.1967** | **0.0649** | **0.6829** | **0.7742** |
| **Two-Model** | **D** | **Logistic Regression**| 16.15 | 0.7061 | 0.2531 | 0.0575 | 0.5754 | 0.6714 | 0.2389 | 0.0737 | 0.5876 | 0.6335 |

---

## 18. Calibration & Reliability Analysis

The winning model (**Two-Model Architecture, XGBoost, Set D**) was evaluated for calibration across 10 uniform probability bins on the Validation and Test sets:

### Validation Set Calibration Table ($N = 35,074$, $\text{ECE} = \mathbf{0.0307}$):

| Bin Index | Probability Range | Observation Count | Fraction of Total | Mean Predicted Prob | Observed Win Frequency | Calibration Error |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `[0.0, 0.1]` | 919 | 2.62% | 0.0780 | 0.0109 | 0.0671 |
| 2 | `[0.1, 0.2]` | 1,396 | 3.98% | 0.1446 | 0.1340 | 0.0107 |
| 3 | `[0.2, 0.3]` | 1,490 | 4.25% | 0.2567 | 0.3993 | 0.1426 |
| 4 | `[0.3, 0.4]` | 5,058 | 14.42% | 0.3572 | 0.4618 | 0.1046 |
| 5 | `[0.4, 0.5]` | 8,591 | 24.49% | 0.4478 | 0.4604 | 0.0125 |
| 6 | `[0.5, 0.6]` | 7,438 | 21.21% | 0.5495 | 0.5506 | 0.0010 |
| 7 | `[0.6, 0.7]` | 4,100 | 11.69% | 0.6422 | 0.6349 | 0.0074 |
| 8 | `[0.7, 0.8]` | 2,450 | 6.99% | 0.7493 | 0.7245 | 0.0248 |
| 9 | `[0.8, 0.9]` | 3,065 | 8.74% | 0.8552 | 0.8594 | 0.0042 |
| 10 | `[0.9, 1.0]` | 567 | 1.62% | 0.9148 | 0.9841 | 0.0693 |

### Test Set Calibration Table ($N = 50,867$, $\text{ECE} = \mathbf{0.0649}$):

| Bin Index | Probability Range | Observation Count | Fraction of Total | Mean Predicted Prob | Observed Win Frequency | Calibration Error |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `[0.0, 0.1]` | 1,944 | 3.82% | 0.0819 | 0.0237 | 0.0582 |
| 2 | `[0.1, 0.2]` | 3,024 | 5.94% | 0.1396 | 0.1038 | 0.0358 |
| 3 | `[0.2, 0.3]` | 1,994 | 3.92% | 0.2546 | 0.1515 | 0.1031 |
| 4 | `[0.3, 0.4]` | 5,278 | 10.38% | 0.3574 | 0.2327 | 0.1247 |
| 5 | `[0.4, 0.5]` | 9,867 | 19.40% | 0.4511 | 0.3630 | 0.0881 |
| 6 | `[0.5, 0.6]` | 11,271 | 22.16% | 0.5516 | 0.4813 | 0.0702 |
| 7 | `[0.6, 0.7]` | 8,517 | 16.74% | 0.6425 | 0.6047 | 0.0378 |
| 8 | `[0.7, 0.8]` | 3,882 | 7.63% | 0.7468 | 0.7494 | 0.0025 |
| 9 | `[0.8, 0.9]` | 4,109 | 8.08% | 0.8514 | 0.9027 | 0.0512 |
| 10 | `[0.9, 1.0]` | 981 | 1.93% | 0.9121 | 0.9246 | 0.0125 |

**Verdict:** The winning model demonstrates good overall calibration according to aggregate ECE, particularly on the Validation set (0.0307). However, calibration is not uniform across all probability regions, with larger deviations visible in several lower-probability bins and selected extreme bins. For example, in the Validation set, bin `[0.2, 0.3]` exhibits a calibration error of `0.1426`, bin `[0.3, 0.4]` has an error of `0.1046`, and bin `[0.9, 1.0]` has an error of `0.0693`. The Test set also contains notable errors, including bin `[0.2, 0.3]` (`0.1031`), bin `[0.3, 0.4]` (`0.1247`), and bin `[0.4, 0.5]` (`0.0881`). The Test ECE of 0.0649 indicates that some out-of-time calibration drift remains. Therefore, the model should be described as reasonably well calibrated overall rather than perfectly calibrated.

---

## 19. Phase-Based Evaluation

Probability metrics were evaluated across match phases on the **frozen Test set** (2024–2026):

### Standard 3-Phase Breakdown:
| Phase | Completed Overs | Delivery Rows | Match Count | Log Loss | Brier Score | ECE | Accuracy | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Powerplay** | 0–6 overs | 15,591 | 213 | 0.6366 | 0.2230 | 0.0636 | 62.27% | 0.6929 |
| **Middle** | 6–15 overs | 23,601 | 213 | 0.5614 | 0.1912 | 0.0647 | 69.81% | 0.7879 |
| **Death** | 15–20 overs | 11,675 | 210 | 0.5137 | 0.1728 | 0.0841 | 73.23% | 0.8385 |

### Finer 6-Phase Progression:
| Progression Window | Delivery Rows | Match Count | Log Loss | Brier Score | ECE | Accuracy | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0–2 overs** | 4,976 | 213 | 0.6680 | 0.2368 | 0.0706 | 58.24% | 0.6418 |
| **2–6 overs** | 10,615 | 213 | 0.6219 | 0.2165 | 0.0652 | 64.15% | 0.7164 |
| **6–10 overs** | 10,523 | 213 | 0.5840 | 0.2004 | 0.0649 | 67.35% | 0.7630 |
| **10–15 overs** | 13,078 | 213 | 0.5432 | 0.1838 | 0.0646 | 71.80% | 0.8061 |
| **15–18 overs** | 7,396 | 210 | 0.5065 | 0.1695 | 0.0826 | 73.92% | 0.8431 |
| **18–20 overs** | 4,279 | 205 | 0.5263 | 0.1784 | 0.0890 | 72.05% | 0.8292 |

**Physical Insight:** Performance generally improves as the match progresses, with Log Loss and Brier Score declining substantially from the early overs into the death phase. However, the final 18–20-over window shows a small deterioration relative to 15–18 overs, with Brier increasing from 0.1695 to 0.1784 and ROC-AUC decreasing from 0.8431 to 0.8292. This late-stage variation is reported as an observed characteristic of the test sample rather than treated as a failure condition.

---

## 20. Innings-Based Evaluation

The model was evaluated independently on Innings 1 vs Innings 2 deliveries on the Test set:

| Innings | Delivery Rows | Match Count | Log Loss | Brier Score | ECE | Accuracy | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Innings 1 (First Innings Total)** | 26,558 | 213 | 0.6702 | 0.2384 | 0.0736 | 58.92% | 0.6418 |
| **Innings 2 (Chase Dynamics)** | 24,309 | 213 | **0.4678** | **0.1511** | **0.0671** | **78.51%** | **0.8713** |

**Interpretation:** Innings 1 is inherently higher-entropy because the second innings has not commenced. In Innings 2, where target boundaries, balls remaining, and required run rate constrain the state space, the model achieves an extraordinary **0.4678 Log Loss and 0.8713 ROC-AUC**.

---

## 21. Delivery vs Match-Level Evaluation

Because deliveries within the same match are correlated, metrics were aggregated by match:

| Aggregation Level | Evaluation Sample | Mean Log Loss | Median Log Loss | Std Log Loss | Mean Brier | Median Brier | Std Brier |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Delivery-Level** | 50,867 deliveries | 0.5735 | — | — | 0.1967 | — | — |
| **Match-Level** | 213 test matches | **0.5731** | **0.5405** | 0.2283 | **0.1964** | **0.1820** | 0.0982 |

The match-level mean Log Loss (`0.5731`) closely tracks the delivery-level score (`0.5735`), proving that long, high-ball-count matches do not distort performance.

---

## 22. Ablation Results & Feature Interpretation

| Feature Set | Features | Val LogLoss (Unified) | Val Brier (Unified) | Val LogLoss (Two-Model) | Val Brier (Two-Model) | Test LogLoss (Two-Model) | Test Brier (Two-Model) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A** | 30 | 0.6475 | 0.2298 | 0.6204 | 0.2184 | 0.5747 | 0.1975 |
| **Model B** | 44 | 0.6543 | 0.2329 | — | — | — | — |
| **Model C** | 50 | 0.6617 | 0.2362 | — | — | — | — |
| **Model D** | 54 | 0.6554 | 0.2337 | **0.6197** | **0.2180** | **0.5735** | **0.1967** |

### Key Findings:
1. **Current Player Features (Model B):** In a unified model, active batter/bowler figures slightly degraded validation log loss (+0.0068), acting as noisy micro-signals.
2. **Historical Player Priors (Model C):** In a unified model, adding career priors without innings partitioning caused minor overfitting (+0.0074).
3. **Feature Set D Evaluated Scope:** Feature Set D achieved the best performance among the Two-Model feature configurations actually evaluated in this experiment (Validation Log Loss = `0.6197`, Test Log Loss = `0.5735` vs Two-Model Set A's `0.6204` Validation and `0.5747` Test). Two-Model B and Two-Model C were not included in the reported Two-Model comparison, so superiority over those configurations cannot be claimed from this experiment.
4. **Dominant Architecture vs. Matchup Incremental Impact:** The dominant improvement in the experiment came from separating Innings 1 and Innings 2 into specialized models. Adding the matchup features from Set A to Set D produced a comparatively small incremental improvement in the evaluated Two-Model XGBoost configuration (Validation improvement = `0.0007`, Test improvement = `0.0012`). The continuous Empirical Bayes formulation ($m=24$) provided regularized head-to-head interaction signals without threshold discontinuities, but its impact is modest relative to the architectural separation.

---

## 23. Unified vs Two-Model Architecture Comparison

| Architecture | Model Algorithm | Feature Set | Val Log Loss | Val Brier | Val ECE | Test Log Loss | Test Brier | Test ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unified (Arch A)** | Random Forest | Set D | 0.6572 | 0.2326 | 0.0541 | 0.6371 | 0.2227 | 0.7183 |
| **Two-Model (Arch B)** | Random Forest | Set D | 0.6293 | 0.2193 | 0.0717 | 0.6089 | 0.2094 | 0.7804 |
| **Unified (Arch A)** | XGBoost | Set D | 0.6554 | 0.2337 | 0.0841 | 0.5906 | 0.2037 | 0.7390 |
| **Two-Model (Arch B)** | **XGBoost (WINNER)** | **Set D** | **0.6197** | **0.2180** | **0.0307** | **0.5735** | **0.1967** | **0.7742** |

**Architecture Decision:** The Two-Model architecture clearly outperformed the Unified architecture across the reported experiments. Isolating the distinct physical states of Innings 1 (scoring velocity, projected totals) from Innings 2 (chase pressure, legal balls remaining, required run rate) eliminated model interference and drove the dominant performance gains (improving XGBoost Set D Validation Log Loss by 0.0357 points from 0.6554 to 0.6197, and Test Log Loss by 0.0171 points from 0.5906 to 0.5735).

In comparison, adding matchup features (moving from Two-Model Set A to Two-Model Set D) produced a comparatively small incremental improvement (0.0007 in Validation Log Loss, 0.0012 in Test Log Loss). Feature Set D achieved the best performance among the Two-Model feature configurations actually evaluated in this experiment; Two-Model B and Two-Model C were not included in the reported Two-Model comparison, so superiority over those configurations cannot be claimed.

---

## 24. Feature Importance

### Native Tree Importance — Model 1 (Innings 1):
1. `city_Chennai`: `0.0426`
2. `venue_Punjab Cricket Association IS Bindra Stadium`: `0.0263`
3. `current_run_rate`: `0.0258`
4. `wickets_remaining`: `0.0246`
5. `city_Chandigarh`: `0.0217`
6. `city_Bangalore`: `0.0190`
7. `venue_MA Chidambaram Stadium`: `0.0169`
8. `current_innings_wickets`: `0.0161`
9. `team_a_bats_first`: `0.0154`
10. `runs_last_12_balls`: `0.0153`

### Native Tree Importance — Model 2 (Innings 2 Chase Model):
1. `team_a_bats_first`: `0.1156` (Establishes whether Team A is defending or chasing)
2. `is_team_a_batting`: `0.0710` (Batting polarity)
3. `required_run_rate`: `0.0389` (Chase pressure)
4. `rrr_crr_differential`: `0.0342` (Momentum differential)
5. `venue_Eden Gardens`: `0.0204`
6. `venue_MA Chidambaram Stadium`: `0.0183`
7. `toss_winner_is_team_a`: `0.0164`
8. `runs_needed`: `0.0148`
9. `target_runs`: `0.0139`
10. `toss_decision_is_bat`: `0.0134`

### Validation Permutation Importance (Increase in Log Loss when shuffled):
- **Innings 1:** Ground/scoring environment indicators (`venue_Wankhede Stadium`, `current_run_rate`, `team_b` strength).
- **Innings 2:** `toss_decision_is_bat` (+0.0432), `toss_winner_is_team_a` (+0.0420), `city_Mumbai` (+0.0031), `required_run_rate`, `runs_needed`.

---

## 25. Qualitative Match Sanity Checks

Trajectory samples across representative historical match archetypes confirmed physical realism:

### Example 1: Kolkata Knight Riders vs Royal Challengers Bengaluru (Match 1426274, Season 2024)
- **Actual Winner:** Team A (KKR Won) | **Target:** 223
- **Trajectory Progression:**
  - Ball 0.1 (Inn 1, Score 0/0, Balls Left 119): Pre-match prob = `0.4956`, Model prob = `0.4541`
  - Ball 7.1 (Inn 1, Score 84/3, Balls Left 77): Model prob = `0.6087` (Accelerating run rate increases win prob)
  - Ball 13.6 (Inn 1, Score 141/5, Balls Left 37): Model prob = `0.6267`
  - Ball 19.5 (Inn 1, Score 217/6, Balls Left 2): Model prob = `0.6528` (Massive first-innings total posted)
  - Ball 6.3 (Inn 2, Score 76/2, Needed 147 off 81, RRR 10.89): Model prob = `0.5386` (RCB chasing strongly, prob drops to even)
  - Ball 13.1 (Inn 2, Score 156/6, Needed 67 off 41, RRR 9.80): Model prob = `0.7503` (Wickets tumble, KKR secures dominant position)

### Example 2: Kolkata Knight Riders vs Mumbai Indians (Match 1426298, Season 2024)
- **Actual Winner:** Team A (KKR Won) | **Target:** 158
- **Trajectory Progression:**
  - Ball 0.1 (Inn 1, Score 6/0): Pre-match prob = `0.5035`, Model prob = `0.5243`
  - Ball 5.3 (Inn 1, Score 48/3, Balls Left 87): Model prob = `0.4732` (Early top-order collapse)
  - Ball 10.5 (Inn 1, Score 106/4, Balls Left 55): Model prob = `0.5173` (Recovery)
  - Ball 5.1 (Inn 2, Score 60/0, Needed 98 off 89, RRR 6.61): Model prob = `0.1049` (MI cruising at 60/0, KKR win prob collapses to 10.5%)
  - Ball 14.2 (Inn 2, Score 120/7, Needed 38 off 34, RRR 6.71): Model prob = `0.7840` (Sensational collapse from 60/0 to 120/7, KKR surges to 78.4% and wins).

---

## 26. Final Model Selection

Based on the frozen evaluation protocol:
- **Winning Architecture:** **Architecture B — Two-Model Pipeline** (clearly outperformed Unified architecture across reported experiments)
- **Winning Algorithm:** **XGBoost** (`n_estimators=100`, `max_depth=4`, `learning_rate=0.05`)
- **Selected Feature Set:** **Feature Set D (Best evaluated Two-Model configuration, including Matchup Bayes $m=24$)**  
  *(Note: Evaluated Two-Model configurations were Set A and Set D. Two-Model B and Two-Model C were not included in the reported Two-Model comparison, so superiority over those configurations cannot be claimed from this experiment).*
- **Primary Model Artifacts:**
  - Innings 1 Model: `models/live/primary_live_model_inn1.joblib`
  - Innings 2 Model: `models/live/primary_live_model_inn2.joblib`
  - Innings 1 Preprocessor: `models/live/primary_live_preprocessor_inn1.joblib`
  - Innings 2 Preprocessor: `models/live/primary_live_preprocessor_inn2.joblib`
- **Validation & Readiness Status:** Validated as the primary local model for integration into the planned live inference pipeline. Local model training and evaluation are complete; live inference simulator integration and AWS deployment are not part of Step 3 and remain separate subsequent milestones.

---

## 27. Saved Artifacts Inventory

All models and evaluation metadata were saved under `models/live/`:

```
models/live/
├── primary_live_model_inn1.joblib
├── primary_live_model_inn2.joblib
├── primary_live_preprocessor_inn1.joblib
├── primary_live_preprocessor_inn2.joblib
├── preprocessing/
│   ├── preprocessor_set_A.joblib / preprocessor_set_A_scaled.joblib
│   ├── preprocessor_set_B.joblib / preprocessor_set_B_scaled.joblib
│   ├── preprocessor_set_C.joblib / preprocessor_set_C_scaled.joblib
│   ├── preprocessor_set_D.joblib / preprocessor_set_D_scaled.joblib
│   ├── prep_set_A_inn1.joblib / prep_set_A_inn2.joblib
│   └── prep_set_D_inn1.joblib / prep_set_D_inn2.joblib
├── unified/
│   ├── logistic_regression_set_[A-D].joblib
│   ├── random_forest_set_[A-D].joblib
│   └── xgboost_set_[A-D].joblib
├── innings_1/
│   ├── random_forest_set_A_inn1.joblib / random_forest_set_D_inn1.joblib
│   ├── xgboost_set_A_inn1.joblib / xgboost_set_D_inn1.joblib
│   └── logistic_regression_set_A_inn1.joblib / logistic_regression_set_D_inn1.joblib
├── innings_2/
│   ├── random_forest_set_A_inn2.joblib / random_forest_set_D_inn2.joblib
│   ├── xgboost_set_A_inn2.joblib / xgboost_set_D_inn2.joblib
│   └── logistic_regression_set_A_inn2.joblib / logistic_regression_set_D_inn2.joblib
├── evaluation/
│   ├── all_model_runs.json
│   └── winning_run_metrics.json
└── metadata/
    └── live_model_metadata.json
```

---

## 28. Reproducibility & Determinism

- **Python Version:** `3.14.7`
- **XGBoost Version:** `3.4.1`
- **Scikit-Learn Version:** `1.6.1`
- **Seeds Enforced:** `random_state = 42` across all models and splits.
- **Dataset Hash:** `83b4a7bd362d873ae38c1b8c2737ff3054ea9e31a0019685448ca81c39c895df`
- **Determinism Check:** The winning model pipeline was trained twice independently. Maximum absolute prediction difference across all validation rows = `0.0` (**100% bitwise identical determinism**).

---

## 29. Limitations

1. **Ball Trajectory Physics:** Cricsheet data contains discrete event outcomes (runs, wickets, extras) but lacks continuous ball tracking (hawkeye trajectory, release speed, turn/swing angle).
2. **Mid-Innings Substitutions / Tactical Impact Players:** Introduced in season 2023; captured through active striker/bowler identifiers but not explicitly modeled as a rule-shift parameter.
3. **Severe Rain Interruption Beyond D/L:** The 12 rain-halted matches that did not employ official D/L methods end prematurely before regulation overs expire.

---

## 30. Conclusion

Phase 5 Step 3 has completed the local training, ablation, and evaluation of candidate live models. Based strictly on the empirical evidence from the experiments actually conducted:

1. **Architecture Performance:** The Two-Model architecture clearly outperformed the Unified architecture across the reported experiments, providing the dominant performance gains by separating first-innings total projection from second-innings chase dynamics.
2. **Best Evaluated Configuration:** XGBoost combined with Feature Set D achieved the best performance among the Two-Model feature configurations actually evaluated in this experiment (Validation Log Loss = `0.6197`, Test Log Loss = `0.5735`, Test Brier = `0.1967`). Two-Model B and Two-Model C were not included in the reported Two-Model comparison, so superiority over those configurations cannot be claimed from this experiment.
3. **Out-of-Time Generalization:** The model demonstrates strong out-of-time probability performance across the 2024–2026 Test seasons (Test Accuracy = `68.29%`, Test ROC-AUC = `0.7742`).
4. **Calibration Quality:** The winning model demonstrates good overall calibration according to aggregate ECE (Validation ECE = `0.0307`, Test ECE = `0.0649`). However, calibration is not uniform across all probability regions, with larger deviations visible in several lower-probability bins and selected extreme bins, and some out-of-time calibration drift remains on the Test set; the model should be described as reasonably well calibrated overall rather than perfectly calibrated.
5. **Phase-Based Progression:** Performance generally improves as the match progresses, with Log Loss and Brier Score declining substantially from the early overs into the death phase. However, the final 18–20-over window shows a small deterioration relative to 15–18 overs, with Brier increasing from `0.1695` to `0.1784` and ROC-AUC decreasing from `0.8431` to `0.8292`. This late-stage variation is reported as an observed characteristic of the test sample rather than treated as a failure condition.
6. **Contribution of Matchup Features:** The dominant improvement in the experiment came from separating Innings 1 and Innings 2 into specialized models. Adding the matchup features from Set A to Set D produced a comparatively small incremental improvement in the evaluated Two-Model XGBoost configuration (Validation Log Loss improved by `0.0007`, Test Log Loss improved by `0.0012`).
7. **Production & Deployment Readiness:** The model pipeline is 100% local, fully deterministic, and validated as the primary local model for integration into the planned live inference pipeline. Local model training and offline evaluation are complete; the live inference simulator (Step 4) and cloud deployment (AWS) have intentionally not started and remain separate subsequent milestones.

---

PHASE 5 STEP 3 STATUS: REPORT CORRECTED — MODEL SELECTION UNCHANGED
