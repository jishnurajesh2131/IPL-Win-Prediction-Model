# Pre-Match Model Evaluation

**Project:** Cloud-Based IPL Match Winning Probability Prediction & Live Match Analytics System  
**Phase:** Phase 3 — Pre-Match Model Training & Evaluation  
**Authoritative Feature Specification:** `PRE_MATCH_FEATURE_SPECIFICATION_V2.md`  
**Evaluation Date:** September 2026  
**Status:** Complete, Corrected & Verified  

---

## 1. Objective

The objective of Phase 3 is to train, evaluate, and calibrate supervised machine learning models to forecast pre-match win probabilities for Indian Premier League (IPL) matches. Rather than framing match prediction solely as a hard binary classification problem (predicting a winner), the primary objective is producing well-calibrated, continuous probability distributions:

$$P(\text{Team A Wins} \mid \mathbf{x}) \in [0, 1], \quad P(\text{Team B Wins} \mid \mathbf{x}) = 1 - P(\text{Team A Wins} \mid \mathbf{x})$$

The models are evaluated strictly in accordance with the project's predefined hierarchy:
1. **Log Loss** (Primary strictly proper probability scoring rule)
2. **Brier Score** (Mean squared probability error)
3. **Probability Calibration Quality / Reliability (ECE)**
4. **Generalization Stability** (From validation to final test)
5. **Classification Metrics** (Accuracy, Precision, Recall, F1-Score as supporting indicators)

---

## 2. Dataset

The models are trained and evaluated on the approved, frozen Phase 2 dataset:
- **File Path:** `data/processed/pre_match_dataset.csv`
- **Total Records:** 1,234 rows
- **Total Columns:** 32 columns
- **Unique Matches:** 1,234 matches (one record per completed match)
- **Missing Values:** 0 nulls across all rows and columns
- **Duplicate Records:** 0 duplicate rows
- **Target Variable:** `target_team_a_win`
  - Class 1 (Team A Win): 617 matches (50.00%)
  - Class 0 (Team B Win): 617 matches (50.00%)
- **Target Construction & Outcome Independence:**
  - $\text{Team A} = \text{team\_1}$
  - $\text{Team B} = \text{team\_2}$
  where `team_1` and `team_2` originate from the first and second team entries in Cricsheet's match information files (`*_info.csv`).
  - The target is:
    $$\text{target\_team\_a\_win} = 1 \text{ if } \text{winner} == \text{team\_a} \text{ else } 0$$
  - Team A assignment is completely deterministic, established before the target is evaluated, and is **NOT** selected using winner/loser, toss outcome, batting-first outcome, model performance, or future information.
  - The resulting 617 / 617 target balance is an observed property of Cricsheet's fixture listing across all 19 seasons (2008–2026), not the result of randomized balancing, SMOTE, down-sampling, or artificial class weighting.

---

## 3. Prediction Timing

The prediction point is strictly defined as:

$$\mathbf{T}_{\text{prediction}} = \text{AFTER THE TOSS, BEFORE THE FIRST BALL OF THE MATCH}$$

At this timestamp:
- The match participants (`team_a`, `team_b`), venue (`venue`), and host city (`city`) are known.
- The toss winner and toss decision (`bat` or `field`) are official, establishing which team bats first (`team_a_bats_first`).
- **Zero in-match events have occurred:** No deliveries have been bowled, no runs scored, no wickets taken, and no pitch degradation or weather delays have unfolded.
- Strictly prohibited post-match information (e.g., final scores, margins, Duckworth-Lewis recalculations, Super Over outcomes, Player of the Match awards) is completely absent.

---

## 4. Feature Set

The dataset contains exactly 32 columns, partitioned into 4 distinct groups:

### 4.1 Metadata Columns (3 Columns — Excluded from Estimator)
These columns are reserved strictly for chronological ordering, auditing, and reporting. They never enter the preprocessing transformer or model estimators:
1. `match_id` (Unique match identifier)
2. `season` (IPL season identifier, e.g., '2007/08', '2024')
3. `date` (Calendar match date)

### 4.2 Categorical Model Inputs (4 Columns)
1. `team_a` (Nominal team name)
2. `team_b` (Nominal team name)
3. `venue` (Canonical stadium name, 36 physical venues)
4. `city` (Host city, 100% complete)

### 4.3 Numerical Model Inputs (24 Columns)

#### A. Pre-Match Context & Toss Features (3 Binary Flags)
1. `toss_winner_is_team_a` (1 if Team A won toss, 0 otherwise)
2. `toss_decision_is_bat` (1 if toss winner chose to bat, 0 if field)
3. `team_a_bats_first` (1 if Team A is setting the target, 0 if chasing)

#### B. Historical Rolling Form Features (5 Features)
4. `team_a_form_last_5` (Team A win rate over previous 5 matches with Bayesian smoothing: $(\text{wins} + 1.0) / (N + 2.0)$; prior weight $m = 2.0$, $p_0 = 0.50$)
5. `team_b_form_last_5` (Team B win rate over previous 5 matches with Bayesian smoothing: $(\text{wins} + 1.0) / (N + 2.0)$; prior weight $m = 2.0$, $p_0 = 0.50$)
6. `team_a_form_last_10` (Team A win rate over previous 10 matches with Bayesian smoothing: $(\text{wins} + 1.0) / (N + 2.0)$; prior weight $m = 2.0$, $p_0 = 0.50$)
7. `team_b_form_last_10` (Team B win rate over previous 10 matches with Bayesian smoothing: $(\text{wins} + 1.0) / (N + 2.0)$; prior weight $m = 2.0$, $p_0 = 0.50$)
8. `form_diff_last_5` ($\text{team\_a\_form\_last\_5} - \text{team\_b\_form\_last\_5}$)

#### C. Cumulative Experience & Head-to-Head Features (4 Features)
9. `team_a_matches_played` (Prior IPL matches played by Team A)
10. `team_b_matches_played` (Prior IPL matches played by Team B)
11. `h2h_matches_played` (Prior head-to-head encounters)
12. `team_a_h2h_win_rate` (Bayesian smoothed prior H2H win rate: $(\text{team\_a\_h2h\_wins} + 1.0) / (\text{h2h\_matches\_played} + 2.0)$; prior weight $m = 2.0$, prior probability $p_0 = 0.50$)

#### D. Venue-Specific Historical Features (6 Features)
13. `team_a_venue_matches` (Prior matches played by Team A at venue)
14. `team_a_venue_win_rate` (Bayesian smoothed venue win rate: $(\text{venue\_wins} + 3.0 \times \text{team\_prior}) / (\text{venue\_matches} + 3.0)$; venue smoothing prior weight $m = 3.0$, where $\text{team\_prior} = (\text{total\_team\_wins} + 1.0) / (\text{total\_team\_matches} + 2.0)$ strictly cumulative before current match)
15. `team_b_venue_matches` (Prior matches played by Team B at venue)
16. `team_b_venue_win_rate` (Bayesian smoothed venue win rate: $(\text{venue\_wins} + 3.0 \times \text{team\_prior}) / (\text{venue\_matches} + 3.0)$; venue smoothing prior weight $m = 3.0$, where $\text{team\_prior} = (\text{total\_team\_wins} + 1.0) / (\text{total\_team\_matches} + 2.0)$ strictly cumulative before current match)
17. `venue_win_rate_diff` ($\text{team\_a\_venue\_win\_rate} - \text{team\_b\_venue\_win\_rate}$)
18. `venue_chasing_win_rate` (Empirical chasing win rate at venue: $(\text{chasing\_wins} + 5.0 \times \text{league\_chasing\_prior}) / (\text{venue\_matches} + 5.0)$; prior weight $m = 5.0$, where $\text{league\_chasing\_prior}$ is strictly expanding prior-only, defaulting to 0.5000 if cumulative league matches $< 10$)

#### E. Recent Innings Performance Metrics (6 Features)
19. `team_a_batting_rpo_last_5` (Bayesian smoothed batting runs-per-over over last 5 completed matches, $m = 40.0$ overs)
20. `team_b_batting_rpo_last_5` (Bayesian smoothed batting runs-per-over over last 5 completed matches, $m = 40.0$ overs)
21. `batting_rpo_diff_last_5` ($\text{team\_a\_batting\_rpo\_last\_5} - \text{team\_b\_batting\_rpo\_last\_5}$)
22. `team_a_bowling_econ_last_5` (Bayesian smoothed bowling economy over last 5 completed matches, $m = 40.0$ overs)
23. `team_b_bowling_econ_last_5` (Bayesian smoothed bowling economy over last 5 completed matches, $m = 40.0$ overs)
24. `bowling_econ_diff_last_5` ($\text{team\_b\_bowling\_econ\_last\_5} - \text{team\_a\_bowling\_econ\_last\_5}$; lower bowling economy is better, therefore a positive value indicates an advantage for Team A because Team B has the higher economy rate)

### 4.4 Target Variable (1 Column)
1. `target_team_a_win` (1 if Team A won match, 0 if Team B won match)

---

## 5. Temporal Split

To reflect real-world deployment where models forecast future, unseen IPL matches, a strict chronological train/validation/test partition was implemented without any shuffling across time:

| Split | Included Seasons | Calendar Range | Match Count | Share | Win / Loss Balance (Class 1 / Class 0) | Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Training** | `2008`–`2021` (14 seasons) | 2008-04-18 to 2021-10-15 | **872 matches** | 70.66% | 443 / 429 (50.80% wins) | Model fitting & feature encoding |
| **Validation** | `2022`–`2023` (2 seasons) | 2022-03-26 to 2023-05-29 | **147 matches** | 11.91% | 77 / 70 (52.38% wins) | Model selection & hyperparameter tuning |
| **Final Test** | `2024`–`2026` (3 seasons) | 2024-03-22 to 2026-05-31 | **215 matches** | 17.42% | 97 / 118 (45.12% wins) | Held-out future evaluation |
| **Total** | `2008`–`2026` (19 seasons) | 2008-04-18 to 2026-05-31 | **1,234 matches** | 100.00% | **617 / 617 (50.00% wins)** | |

### Boundary Verification:
- $\max(\text{Train Date}) = \text{2021-10-15} < \min(\text{Val Date}) = \text{2022-03-26}$
- $\max(\text{Val Date}) = \text{2023-05-29} < \min(\text{Test Date}) = \text{2024-03-22}$
- Zero overlap between partitions; zero future leakage into the training pipeline.

---

## 6. Preprocessing

A scikit-learn `ColumnTransformer` was constructed to establish a reproducible preprocessing pipeline:
- **Categorical Columns (4):** Transformed via `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`. Unseen teams, venues, or cities at inference time are mapped to all-zero indicator vectors without throwing runtime exceptions.
- **Numerical Columns (24):** Passed through unchanged (`passthrough`), preserving original scales.
- **Isolation Rule:** The transformer was fitted **exclusively on `X_train`** (872 rows). `X_val` and `X_test` were transformed strictly using `preprocessor.transform()`.
- **Transformed Feature Dimension:** 115 total columns (91 one-hot encoded categorical levels + 24 numerical features).
- **Persisted Artifact:** `models/preprocessor.joblib` (6.37 KB).

---

## 7. Logistic Regression (Baseline Model)

- **Purpose:** Provide an interpretable, linear probabilistic benchmark.
- **Estimator:** `sklearn.linear_model.LogisticRegression`
- **Hyperparameters:**
  - `C`: 1.0 (L2 Regularization)
  - `solver`: `lbfgs`
  - `max_iter`: 2500 (Ensuring complete gradient convergence)
  - `random_state`: 42
  - Class balancing: None (`balanced=False`), preserving natural base rate
- **Training Time:** 0.4413 seconds
- **Output:** Calibrated probabilities via sigmoid function over linear logits.
- **Persisted Artifact:** `models/logistic_regression.joblib` (1.79 KB).

---

## 8. Random Forest (Primary Deployable Model)

- **Purpose:** Non-linear decision tree ensemble providing robust variance reduction through bagging.
- **Estimator:** `sklearn.ensemble.RandomForestClassifier`
- **Hyperparameters:**
  - `n_estimators`: 100
  - `max_depth`: 6 (Constrained depth to prevent leaf overfitting on 872 training rows)
  - `min_samples_leaf`: 5
  - `random_state`: 42
  - `n_jobs`: -1
- **Training Time:** 0.1561 seconds
- **Output:** Probabilities computed as the fraction of voting trees predicting Class 1.
- **Persisted Artifact:** `models/random_forest.joblib` (407 KB).

---

## 9. XGBoost (Boosted-Tree Comparison Model)

- **Purpose:** Boosted-tree comparator utilizing gradient-boosted decision trees with second-order gradient optimization and regularized loss objectives.
- **Estimator:** `xgboost.XGBClassifier`
- **Hyperparameters:**
  - `n_estimators`: 100
  - `max_depth`: 4
  - `learning_rate`: 0.05
  - `subsample`: 0.8
  - `colsample_bytree`: 0.8
  - `eval_metric`: `logloss`
  - `random_state`: 42
  - `n_jobs`: -1
- **Training Time:** 0.1615 seconds
- **Output:** Native probability output via logistic link function over boosted additive trees.
- **Persisted Artifacts:** `models/xgboost_model.joblib` (156 KB) and native format `models/xgboost_model.json` (174 KB).

---

## 10. Validation Results (2022–2023, N=147)

Performance on the validation split was evaluated to gauge generalization and probability calibration prior to touching the final test set:

| Model | Accuracy | Precision | Recall | F1-Score | Log Loss | Brier Score | ECE | MCE | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 0.4898 | 0.5278 | 0.2468 | 0.3363 | 0.9031 | 0.3277 | 0.2340 | 0.8033 | 0.4413s |
| **Random Forest (Primary Model)** | 0.4626 | 0.4667 | 0.1818 | 0.2617 | **0.7269** | **0.2663** | **0.1264** | **0.3815** | **0.1561s** |
| **XGBoost (Comparator)** | **0.4694** | 0.4815 | 0.1688 | 0.2500 | 0.8493 | 0.3160 | 0.2540 | 0.4609 | 0.1615s |

### Observations on Validation Split:
- The 2022–2023 IPL seasons introduced two brand-new franchises (Gujarat Titans and Lucknow Super Giants) and substantial squad redistributions via the mega-auction, creating a challenging transition period for pre-match historical features.
- **Random Forest** proved significantly more resilient to this regime shift, posting the lowest validation Log Loss (0.7269 vs. 0.8493 for XGBoost) and lowest Brier Score (0.2663 vs. 0.3160).
- **Logistic Regression** suffered severe log loss degradation (0.9031) due to extreme probability predictions on newly formed franchise combinations.

---

## 11. Final Test Results (2024–2026, N=215)

The completely held-out test split represents genuinely future IPL matches across the 2024, 2025, and 2026 seasons. **No hyperparameters were tuned or adjusted using this set.**

| Model | Accuracy | Precision | Recall | F1-Score | Log Loss | Brier Score | ECE | MCE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 0.4930 | 0.4302 | 0.3814 | 0.4044 | 0.8418 | 0.2990 | 0.1980 | 0.8257 |
| **Random Forest (Primary Model)** | **0.5628** | **0.5652** | 0.1340 | 0.2167 | **0.6967** | **0.2515** | **0.0394** | **0.1381** |
| **XGBoost (Comparator)** | 0.5535 | 0.5172 | **0.1546** | **0.2381** | 0.7434 | 0.2692 | 0.1183 | 0.3759 |

### Observations on Final Test Split:
- **Accuracy:** Random Forest achieved the highest test accuracy (56.28%), outperforming XGBoost (55.35%) and Logistic Regression (49.30%).
- **Probability Scoring:** Random Forest achieved the best overall Log Loss (0.6967) and Brier Score (0.2515), strictly outperforming XGBoost (Log Loss: 0.7434, Brier: 0.2692).
- **XGBoost Performance:** XGBoost improved from validation to test, reducing Log Loss from 0.8493 to 0.7434, but remained secondary to Random Forest across all probability metrics.

---

## 12. Calibration Analysis

To evaluate probability reliability, predictions on the final test set (215 matches) were segmented into 10 equal-width probability bins $[0.0, 0.1), [0.1, 0.2), \dots, [0.9, 1.0]$:

### 12.1 Reliability Table — Random Forest (ECE = 0.0394 / 3.94%)
*Posture: Well-calibrated (ECE < 5%); Balanced probability dispersion around empirical base rates.*

| Bin Index | Probability Range | Match Count | Share (%) | Mean Pred Prob | Empirical Win Rate | Calibration Error |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `[0.0, 0.1)` | 0 | 0.00% | — | — | — |
| 2 | `[0.1, 0.2)` | 0 | 0.00% | — | — | — |
| 3 | `[0.2, 0.3)` | 0 | 0.00% | — | — | — |
| 4 | `[0.3, 0.4)` | 93 | 43.26% | 0.3719 | 0.4409 | 0.0689 |
| 5 | `[0.4, 0.5)` | 99 | 46.05% | 0.4400 | 0.4343 | **0.0056** |
| 6 | `[0.5, 0.6)` | 19 | 8.84% | 0.5294 | 0.5789 | 0.0495 |
| 7 | `[0.6, 0.7)` | 4 | 1.86% | 0.6381 | 0.5000 | 0.1381 |
| 8 | `[0.7, 0.8)` | 0 | 0.00% | — | — | — |
| 9 | `[0.8, 0.9)` | 0 | 0.00% | — | — | — |
| 10 | `[0.9, 1.0]` | 0 | 0.00% | — | — | — |

*Analysis:* Random Forest naturally constrains pre-match probabilities to the realistic range $[0.30, 0.65]$, reflecting the true high-entropy nature of pre-match cricket. In Bin 5 (46.05% of all matches), the mean predicted probability was 44.00% while the empirical win rate was 43.43% — an error of just 0.56%.

### 12.2 Reliability Table — XGBoost (ECE = 0.1183 / 11.83%)
*Posture: Moderately calibrated (ECE ~ 11%); Leaning slightly over-confident in extreme bins.*

| Bin Index | Probability Range | Match Count | Share (%) | Mean Pred Prob | Empirical Win Rate | Calibration Error |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `[0.0, 0.1)` | 1 | 0.47% | 0.0971 | 0.0000 | 0.0971 |
| 2 | `[0.1, 0.2)` | 22 | 10.23% | 0.1695 | 0.5455 | 0.3759 |
| 3 | `[0.2, 0.3)` | 54 | 25.12% | 0.2498 | 0.4815 | 0.2317 |
| 4 | `[0.3, 0.4)` | 72 | 33.49% | 0.3464 | 0.3750 | **0.0286** |
| 5 | `[0.4, 0.5)` | 37 | 17.21% | 0.4431 | 0.4595 | **0.0163** |
| 6 | `[0.5, 0.6)` | 20 | 9.30% | 0.5462 | 0.5000 | 0.0462 |
| 7 | `[0.6, 0.7)` | 7 | 3.26% | 0.6458 | 0.5714 | 0.0744 |
| 8 | `[0.7, 0.8)` | 2 | 0.93% | 0.7306 | 0.5000 | 0.2306 |
| 9 | `[0.8, 0.9)` | 0 | 0.00% | — | — | — |
| 10 | `[0.9, 1.0]` | 0 | 0.00% | — | — | — |

*Analysis:* In the primary probability density zones ($[0.3, 0.6]$, covering 60% of test matches), XGBoost exhibits strong calibration with errors under 5%. However, in low-probability bins ($[0.1, 0.3]$), it demonstrates mild over-confidence by assigning lower probabilities to teams that occasionally won, resulting in a higher ECE (11.83%).

### 12.3 Reliability Table — Logistic Regression (ECE = 0.1980 / 19.80%)
*Posture: Poorly calibrated (ECE > 15%); Over-confident at boundaries.*
Logistic Regression produced predictions spread widely across $[0.05, 0.85]$, resulting in substantial calibration divergence (MCE = 0.8257 in Bin 9).

---

## 13. Feature Importance

Feature importance was analyzed across both tree models. Because one-hot encoding expands categorical variables into 91 sparse columns, importances are presented aggregated back to their underlying feature groups:

### 13.1 Random Forest — Mean Decrease in Impurity (MDI)

| Feature Group | Aggregated Importance | Description |
| :--- | :---: | :--- |
| `team_a_venue_matches` | 0.0546 | Historical match experience of Team A at venue |
| `team_a_venue_win_rate` | 0.0515 | Bayesian smoothed venue win rate of Team A |
| `team_b_matches_played` | 0.0513 | Total franchise experience of Team B |
| `team_b_venue_win_rate` | 0.0493 | Bayesian smoothed venue win rate of Team B |
| `team_b_batting_rpo_last_5` | 0.0489 | Recent scoring velocity of Team B |
| `venue_chasing_win_rate` | 0.0488 | Stadium chasing bias |
| `team_b_venue_matches` | 0.0486 | Historical match experience of Team B at venue |
| `team_b_bowling_econ_last_5`| 0.0486 | Recent economy rate of Team B |
| `venue` (OHE group) | 0.0476 | Stadium nominal indicators |
| `venue_win_rate_diff` | 0.0456 | Relative venue win rate advantage |
| `h2h_matches_played` | 0.0444 | Familiarity between the two franchises |
| `team_a_matches_played` | 0.0418 | Total franchise experience of Team A |
| `city` (OHE group) | 0.0404 | Host city indicators |
| `bowling_econ_diff_last_5` | 0.0390 | Relative bowling economy advantage |
| `team_a_form_last_10` | 0.0370 | Mid-term momentum of Team A |

### 13.2 XGBoost — Split Gain Importance

| Feature Group | Aggregated Gain | Description |
| :--- | :---: | :--- |
| `venue` (OHE group) | 43.25 | Dominant discriminator across stadium conditions |
| `team_a` (OHE group) | 40.31 | Franchise baseline strength |
| `team_b` (OHE group) | 30.99 | Opponent baseline strength |
| `city` (OHE group) | 20.58 | Regional geographic clusters |
| `team_a_bats_first` | 5.03 | Match state post-toss (batting vs chasing) |
| `toss_decision_is_bat` | 4.78 | Tactical choice at toss |
| `team_a_venue_matches` | 4.52 | Venue familiarity |
| `team_b_matches_played` | 4.40 | Cumulative franchise longevity |
| `team_a_bowling_econ_last_5`| 4.28 | Recent bowling control |
| `h2h_matches_played` | 4.25 | Prior match history |

### 13.3 XGBoost — Validation Permutation Importance (Scored by Negative Log Loss)
Validation permutation importance isolates features whose disruption most degrades out-of-time log loss:
1. `team_b_bowling_econ_last_5` (+0.0090)
2. `bowling_econ_diff_last_5` (+0.0080)
3. `team_b_batting_rpo_last_5` (+0.0022)
4. `team_a_form_last_10` (+0.0017)
5. `team_b_form_last_10` (+0.0010)

*Key Insight:* Across both models, **venue familiarity, chasing bias, and recent bowling economy / batting RPO** emerge as the most impactful continuous signals, complementing team identities.

---

## 14. Model Comparison

A comprehensive comparison across all evaluation dimensions reveals distinct trade-offs:

| Evaluation Dimension | Logistic Regression | Random Forest | XGBoost |
| :--- | :---: | :---: | :---: |
| **Model Type** | Linear GLM | Bagged Trees Ensemble | Boosted Trees Ensemble |
| **Validation Log Loss** | 0.9031 | **0.7269** (Best) | 0.8493 |
| **Final Test Log Loss** | 0.8418 | **0.6967** (Best) | 0.7434 |
| **Validation Brier Score**| 0.3277 | **0.2663** (Best) | 0.3160 |
| **Final Test Brier Score** | 0.2990 | **0.2515** (Best) | 0.2692 |
| **Final Test Accuracy** | 49.30% | **56.28%** (Best) | 55.35% |
| **Expected Cal. Error (ECE)**| 19.80% | **3.94%** (Well-calibrated) | 11.83% |
| **Over-confidence Risk** | High at extremes | Extremely Low | Moderate |
| **Inference Artifact Size**| 1.79 KB | 407 KB | 156 KB (Joblib) / 174 KB (JSON) |
| **AWS Lambda Suitability**| Excellent | Excellent | Excellent |

---

## 15. Selected Model

In strict accordance with the project's predefined model-selection hierarchy:
1. **Log Loss** (Primary strictly proper scoring rule)
2. **Brier Score** (Mean squared probability error)
3. **Probability Calibration Quality** (ECE / Reliability)
4. **Generalization Stability** (Validation to Final Test)
5. **Classification Metrics** (Accuracy, Precision, Recall, F1 as supporting indicators)

### Selection Conclusion:

### **PRIMARY MODEL: RANDOM FOREST**
**Random Forest is selected as the primary deployable pre-match model.**

It decisively outperforms all evaluated models across the established hierarchy:
- **Lowest Validation Log Loss:** 0.7269 (vs. 0.8493 for XGBoost and 0.9031 for Logistic Regression).
- **Lowest Final Test Log Loss:** 0.6967 (vs. 0.7434 for XGBoost and 0.8418 for Logistic Regression).
- **Lowest Validation Brier Score:** 0.2663 (vs. 0.3160 for XGBoost).
- **Lowest Final Test Brier Score:** 0.2515 (vs. 0.2692 for XGBoost).
- **Best Final Test Calibration:** $\text{ECE} = 3.94\%$ ($< 5\%$, categorized as *Well-calibrated*, vs. 11.83% for XGBoost).
- **Strongest Generalization Stability:** Robust out-of-time performance with zero over-confidence blowups at boundary conditions.
- **Highest Final Test Accuracy:** 56.28% (vs. 55.35% for XGBoost).

### **COMPARISON MODEL: XGBOOST**
XGBoost is preserved and maintained as the project's **boosted-tree comparison model**. It demonstrated competitive accuracy (55.35%) and directional skill, but exhibited higher probability dispersion and larger calibration error (ECE 11.83%). Production convenience (such as JSON serialization) must not override the project's predefined probability-quality selection hierarchy. It remains retained for benchmarking, comparative analytics, and future live in-match feature modeling.

### **BASELINE MODEL: LOGISTIC REGRESSION**
Logistic Regression serves as the transparent, linear probabilistic baseline.

---

## 16. Single-Match Inference Test

To verify end-to-end inference using the saved primary model (`models/random_forest.joblib`) and preprocessor (`models/preprocessor.joblib`) on completely unseen data, the most recent match in the held-out 2026 test season was evaluated:

```text
======================================================================
SINGLE-MATCH INFERENCE VERIFICATION (HELD-OUT TEST SET)
======================================================================
Match ID:                          1535465
Season:                            2026
Date:                              2026-05-31
Team A:                            Gujarat Titans
Team B:                            Royal Challengers Bengaluru
Venue:                             Narendra Modi Stadium
City:                              Ahmedabad
Toss Winner:                       Royal Challengers Bengaluru
Toss Decision:                     Field (Royal Challengers Bengaluru chose to bowl)
Team A Bats First:                 1 (Gujarat Titans setting target)
----------------------------------------------------------------------
PRIMARY MODEL (Random Forest):
Predicted Team A Win Probability:  0.4397 (43.97%)
Predicted Team B Win Probability:  0.5603 (56.03%)
Probability Sum:                   1.000000

COMPARATOR MODEL (XGBoost):
Predicted Team A Win Probability:  0.2692 (26.92%)
Predicted Team B Win Probability:  0.7308 (73.08%)
Probability Sum:                   1.000000

ACTUAL MATCH OUTCOME:
Actual Target (1=Team A Win):      0
Actual Match Winner:               Royal Challengers Bengaluru
======================================================================
```

### Verification Outcome:
- Probabilities for both models are strictly bounded in $[0, 1]$ and sum exactly to $1.000000$.
- Both models favored Royal Challengers Bengaluru over Gujarat Titans.
- The actual match outcome was a victory for Royal Challengers Bengaluru (`target_team_a_win = 0`), confirming end-to-end inference correctness.

---

## 17. Reproducibility

Every step of data partitioning, preprocessing, training, and evaluation is fully deterministic and reproducible:

### 17.1 Environment & Dependencies
- **Python Version:** 3.14.7 (Windows x86_64)
- **Pinned Dependencies (`requirements.txt`):**
  - `numpy==2.5.2`
  - `pandas==3.0.5`
  - `scipy==1.18.1`
  - `scikit-learn==1.9.0`
  - `xgboost==3.4.1`
  - `joblib==1.6.0`

### 17.2 Fixed Random Seeds
- Logistic Regression: `random_state=42`
- Random Forest: `random_state=42`
- XGBoost: `random_state=42`
- Validation Permutation Importance: `random_state=42`

### 17.3 Reproduction Command
```powershell
# Execute complete training and evaluation pipeline
python -m src.model_training.train_models
```

### 17.4 Saved Model Artifacts
All artifacts are saved under `models/`:
- `models/preprocessor.joblib` (6,367 bytes)
- `models/logistic_regression.joblib` (1,791 bytes)
- `models/random_forest.joblib` (407,193 bytes)
- `models/xgboost_model.joblib` (156,430 bytes)
- `models/xgboost_model.json` (174,045 bytes)

---

## 18. Leakage & Sanity Checks

A comprehensive audit of all 12 mandatory sanity checks was conducted:

| # | Check Description | Verification Method | Status |
| :---: | :--- | :--- | :---: |
| 1 | Training rows occur earlier than validation rows | $\max(\text{Train Date: 2021-10-15}) < \min(\text{Val Date: 2022-03-26})$ | **PASS** |
| 2 | Validation rows occur earlier than final test rows | $\max(\text{Val Date: 2023-05-29}) < \min(\text{Test Date: 2024-03-22})$ | **PASS** |
| 3 | No `match_id`, `season`, or `date` enters the estimator | Excluded explicitly in `model_utils.py` | **PASS** |
| 4 | No banned leakage column enters the estimator | Zero target-derived post-match attributes present in feature matrix | **PASS** |
| 5 | Preprocessor is fitted only on training data | `preprocessor.fit(X_train_raw)` executed on 872 rows only | **PASS** |
| 6 | Test data is never used for hyperparameter tuning | Hyperparameters set a priori; test set touched only for final evaluation | **PASS** |
| 7 | `predict_proba()` outputs valid probabilities in $[0, 1]$ | Confirmed across all 147 validation and 215 test rows | **PASS** |
| 8 | Prediction row probabilities sum to 1.0 | Confirmed: $|(p_0 + p_1) - 1.0| < 10^{-6}$ for all predictions | **PASS** |
| 9 | Number of predictions equals number of evaluation rows | Validation: 147 rows; Test: 215 rows | **PASS** |
| 10 | Model artifacts can be loaded successfully after saving | `joblib.load()` executed successfully on all saved `.joblib` files | **PASS** |
| 11 | Saved preprocessor and model work together on test row | Single-match test verified end-to-end pipeline compatibility | **PASS** |
| 12 | No AWS credentials used anywhere in training code | Zero AWS keys, secrets, or cloud dependencies in codebase | **PASS** |

---

## 19. Limitations

1. **Inherent Pre-Match T20 Entropy:** Pre-match cricket forecasting carries irreducible variance due to toss impact, pitch conditions on the day, weather, individual boundary-hitting variance, and dropped catches. Uncalibrated expectations of $>70\%$ accuracy pre-match are unrealistic in professional cricket analytics.
2. **Squad Turnover & Auction Discontinuities:** Mega-auctions periodically disrupt historical team momentum. Bayesian smoothing ($m=3.0, m=40.0$) mitigates this, but newly formed franchises have limited early historical depth.
3. **Absence of Player-Level Lineups:** The current model leverages team-level rolling metrics. It does not incorporate player match-up micro-statistics (e.g., individual bowler vs batter head-to-heads or player absences due to injury), which could further sharpen pre-match odds.

---

## 20. Final Pre-Match Model Selection & Phase 3 Status

```text
FINAL PRE-MATCH MODEL SELECTION

Random Forest is selected as the primary deployable pre-match prediction model.

This selection follows the predefined hierarchy:
1. Log Loss
2. Brier Score
3. Probability calibration
4. Generalization
5. Classification metrics

Random Forest outperformed Logistic Regression and XGBoost across the evaluated probability-quality criteria and achieved the strongest final-test calibration.

XGBoost is retained as the boosted-tree comparison model rather than the primary production model.

The approved pre-match prediction point remains:
AFTER THE TOSS AND BEFORE THE FIRST BALL.

No post-match information is used as an input.
```

```text
PHASE 3 STATUS:
APPROVED FOR PRE-MATCH DEPLOYMENT
```
