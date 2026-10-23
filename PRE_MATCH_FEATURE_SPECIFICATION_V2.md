# Pre-Match Feature Specification V2

**Project:** Cloud-Based IPL Match Winning Probability Prediction & Live Match Analytics System  
**Pipeline Phase:** Phase 2 — Pre-Match Feature Engineering (V2 Mathematical Specification)  
**Target Dataset:** `data/processed/pre_match_dataset.csv` (1,234 rows $\times$ 32 columns)  
**Prediction Timing:** Evaluated strictly **AFTER THE COIN TOSS** and **BEFORE THE FIRST BALL OF THE MATCH IS BOWLED**.  

---

## 1. Universal Eligible Historical Match Definition

A match in `data/processed/matches_clean.csv` is defined as an **Eligible Historical Match** if and only if:
$$\text{match\_result\_type} \neq \text{'no\_result'}$$

### Universal Classification Rules:
1. **Normal Completed Matches (`normal`):** **Eligible.** Full 20-over matches with a decisive winner.
2. **Super Over Completed Matches (`super_over`):** **Eligible.** Tied matches decided by a Super Over. The effective winner is the `winner` already resolved in `matches_clean.csv` (from the raw eliminator).
3. **Duckworth-Lewis / Stern Matches (`method == 'D/L'`):** **Eligible.** Weather-shortened matches with an official result. The official winner counts toward historical win/loss records.
4. **Abandoned / No-Result Matches (`no_result`):** **EXCLUDED.** The 9 genuine abandoned matches without a result never generate a supervised training example and **never** contribute to any team's win/loss, batting, or bowling statistics.

---

## 2. Recent Form Specification

Calculates rolling competitive momentum for both Team A and Team B over the previous 5 and 10 completed matches.

- **Feature Names:**
  - `team_a_form_last_5`, `team_b_form_last_5`
  - `team_a_form_last_10`, `team_b_form_last_10`
  - `form_diff_last_5` (`team_a_form_last_5 - team_b_form_last_5`)
- **Historical Window:** Strictly prior eligible matches. For last-5: $N \le 5$. For last-10: $N \le 10$.
- **Mathematical Smoothing Formula (Empirical Bayes / m-estimate):**
  $$\text{smoothed\_win\_rate} = \frac{\text{wins} + m \times \text{prior}}{N + m}$$
  where:
  - $m = 2.0$ (pseudo-match prior weight)
  - $\text{prior} = 0.50$ (neutral expectation)
  - $N =$ number of prior eligible matches actually available within the window ($0 \le N \le 5$ or $0 \le N \le 10$)
  - $\text{wins} =$ number of wins in those $N$ matches
- **Boundary Conditions:**
  - If $N = 0$ (team's first ever IPL match): $\frac{0 + 2(0.50)}{0 + 2} = \mathbf{0.5000}$ exactly.
  - If $N = 1$ (1 win): $\frac{1 + 1.0}{1 + 2.0} = \frac{2}{3} \approx \mathbf{0.6667}$.
  - Current match $t$ **never** contributes to its own form.
  - Future matches **never** contribute to earlier rows.

---

## 3. Head-to-Head (H2H) Specification

Measures historical matchup advantage between Team A and Team B prior to match $t$.

- **Feature Names:**
  - `h2h_matches_played`: Total prior eligible encounters between Team A and Team B.
  - `team_a_h2h_win_rate`: Smoothed historical win rate of Team A against Team B.
- **Mathematical Formula:**
  $$\text{smoothed\_h2h\_rate} = \frac{\text{team\_a\_h2h\_wins} + 2 \times 0.50}{\text{h2h\_matches\_played} + 2} = \frac{\text{team\_a\_h2h\_wins} + 1.0}{\text{h2h\_matches\_played} + 2.0}$$
- **Boundary Conditions:**
  - If $\text{h2h\_matches\_played} = 0$: Win rate $= \mathbf{0.5000}$ exactly.
- **Franchise Identity Continuity:**
  - Standardized canonical franchise identities preserve complete historical continuity across rebrandings:
    - `Delhi Daredevils` $\rightarrow$ `Delhi Capitals`
    - `Kings XI Punjab` $\rightarrow$ `Punjab Kings`
    - `Royal Challengers Bangalore` $\rightarrow$ `Royal Challengers Bengaluru`
    - `Rising Pune Supergiants` $\rightarrow$ `Rising Pune Supergiant`
  - Historically distinct franchises remain strictly isolated:
    - `Deccan Chargers` $\neq$ `Sunrisers Hyderabad`
    - `Pune Warriors`, `Gujarat Lions`, and `Kochi Tuskers Kerala` are distinct historical entities.

---

## 4. Venue Win Rate Specification

Measures ground familiarity and performance across the **36 canonical venues**.

- **Feature Names:**
  - `team_a_venue_matches`, `team_b_venue_matches`: Count of prior eligible matches played by the team at the venue.
  - `team_a_venue_win_rate`, `team_b_venue_win_rate`: Smoothed venue win rate.
  - `venue_win_rate_diff`: `team_a_venue_win_rate - team_b_venue_win_rate`.
- **Mathematical Formula:**
  $$\text{smoothed\_venue\_rate} = \frac{\text{venue\_wins} + m \times \text{team\_prior}}{N + m}$$
  where:
  - $m = 3.0$ (prior weight)
  - $N =$ prior eligible matches played by that team at that venue
  - $\text{venue\_wins} =$ prior wins at that venue
  - $\text{team\_prior} =$ the team's overall eligible historical win rate strictly **BEFORE** the current match:
    $$\text{team\_prior} = \frac{\text{total\_team\_wins} + 1.0}{\text{total\_team\_matches} + 2.0}$$
    (If the team has zero prior matches in IPL history: $\text{team\_prior} = \mathbf{0.5000}$).
- **Anti-Leakage Rule:** $\text{team\_prior}$ is strictly cumulative up to match $t-1$; it never uses full-dataset or future statistics.

---

## 5. Venue Chasing Win Rate Specification

Measures stadium-level second-innings bias (e.g. dew factor, pitch deterioration, boundary chasing dynamics).

- **Feature Name:** `venue_chasing_win_rate`
- **Mathematical Formula:**
  $$\text{smoothed\_chasing\_rate} = \frac{\text{chasing\_wins} + m \times \text{league\_chasing\_prior}}{N + m}$$
  where:
  - $m = 5.0$ (prior weight)
  - $N =$ prior eligible matches played at this canonical venue across all teams
  - $\text{chasing\_wins} =$ prior matches at this venue where the team batting second (chasing) won
  - $\text{league\_chasing\_prior} =$ expanding historical league-wide chasing win rate available strictly **BEFORE** the current match:
    $$\text{league\_chasing\_prior} = \frac{\text{cumulative\_league\_chasing\_wins}_{<t}}{\text{cumulative\_league\_matches}_{<t}}$$
    (If $\text{cumulative\_league\_matches} < 10$: defaults to $\mathbf{0.5000}$).
- **Anti-Leakage Rule:** $\text{league\_chasing\_prior}$ is computed chronologically; it never uses post-match or full-dataset totals.

---

## 6. Batting Runs Per Over (RPO) Specification

Measures recent offensive scoring velocity over the team's previous 5 completed innings.

- **Feature Names:**
  - `team_a_batting_rpo_last_5`, `team_b_batting_rpo_last_5`
  - `batting_rpo_diff_last_5` (`team_a_batting_rpo_last_5 - team_b_batting_rpo_last_5`)
- **Inclusions:** Normal completed innings (Innings 1 and 2) from normal and completed D/L matches.
- **Exclusions:** `no_result` matches, Super Over deliveries (Innings 3–6).
- **Deliveries Definition:**
  - $\text{Runs Scored} = \text{runs\_off\_bat} + \text{extras}$.
  - $\text{Legal Balls} =$ deliveries where $\text{wides} == 0$ and $\text{noballs} == 0$.
  - $\text{Legal Overs} = \frac{\text{Legal Balls}}{6.0}$.
- **Mathematical Smoothing Formula (m = 40 overs):**
  $$\text{batting\_rpo} = \frac{\sum_{i=1}^{k} \text{runs}_i + 40.0 \times \text{league\_rpo\_prior}}{\sum_{i=1}^{k} \text{overs}_i + 40.0}$$
  where:
  - $k = \min(5, \text{available\_prior\_innings})$
  - $40.0$ overs represents a prior weight of two standard 20-over T20 innings.
  - $\text{league\_rpo\_prior} =$ expanding historical league run rate strictly before match $t$:
    $$\text{league\_rpo\_prior} = \frac{\text{cumulative\_league\_runs}_{<t}}{\text{cumulative\_league\_legal\_overs}_{<t}}$$
    (If $<120$ legal balls bowled in tournament history: defaults to baseline $8.2000$).

---

## 7. Bowling Economy Specification

Measures defensive run containment rate over the team's previous 5 completed bowling innings.

- **Feature Names:**
  - `team_a_bowling_econ_last_5`, `team_b_bowling_econ_last_5`
  - `bowling_econ_diff_last_5` (`team_b_bowling_econ_last_5 - team_a_bowling_econ_last_5`)  
    *(Note: Positive value indicates Team A advantage, as Team B concedes more runs per over).*
- **Inclusions & Extras Treatment:**
  - $\text{Runs Conceded} = \text{runs\_off\_bat} + \text{extras}$ (all conceded extras—wides, no-balls, byes, leg-byes, penalties—count as runs conceded by the fielding side).
  - $\text{Legal Overs Bowled} = \frac{\text{legal\_balls}}{6.0}$ (re-bowled wides and no-balls do not advance the over count).
- **Exclusions:** `no_result` matches, Super Over deliveries (Innings 3–6).
- **Mathematical Smoothing Formula (m = 40 overs):**
  $$\text{bowling\_economy} = \frac{\sum_{i=1}^{k} \text{runs\_conceded}_i + 40.0 \times \text{league\_econ\_prior}}{\sum_{i=1}^{k} \text{overs\_bowled}_i + 40.0}$$
  where $\text{league\_econ\_prior} = \text{league\_rpo\_prior}$ strictly before match $t$.

---

## 8. Duckworth-Lewis / Stern (D/L) Policy

1. Completed D/L matches (23 matches) **remain supervised training examples** with their official winner.
2. Their official result contributes normally to team form, H2H, and venue win rates.
3. Their completed innings statistics contribute to batting RPO and bowling economy. Because RPO and Economy divide runs by actual legal overs bowled, rain-shortened 12-over or 15-over matches are **automatically rate-normalized**.
4. Super Over deliveries are strictly excluded.
5. Post-match D/L target revisions or rain calculations are **strictly banned** from pre-match features.

---

## 9. Super Over Policy

1. Completed Super Over matches (16 matches) **remain supervised training examples**.
2. The match outcome uses the effective eliminator winner from `matches_clean.csv`.
3. The match contributes to team form, H2H records, and venue win rates as a normal win/loss.
4. All Super Over deliveries (Innings 3, 4, 5, 6) are **strictly excluded** from batting RPO and bowling economy calculations.

---

## 10. Team Experience

- **Feature Names:** `team_a_matches_played`, `team_b_matches_played`.
- **Definition:** Total cumulative eligible completed matches played by the franchise prior to match $t$.
- **Anti-Leakage Rule:** Strictly cumulative up to match $t-1$. Match $t$ is never counted.

---

## 11. Categorical Encoding Strategy for Modeling & AWS Lambda Inference

To guarantee complete consistency between offline model development and cloud serverless inference, the project adheres to the approved, budget-conscious serverless architecture:

```text
Local / Google Colab Training
        ↓
Saved ML Model + Preprocessor
        ↓
   Amazon S3
        ↓
   AWS Lambda
        ↓
API Gateway / Frontend
```

### 1. Model Preprocessing Strategy:
- **Logistic Regression / Ridge / ElasticNet:**
  - Use `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` or another explicitly documented anti-symmetric differential encoding if deliberately selected.
  - Do not use arbitrary ordinal integer encoding.
- **Random Forest / Extra Trees:**
  - Use a fitted `OneHotEncoder(handle_unknown='ignore')` pipeline.
- **XGBoost / LightGBM:**
  - Use the same fitted scikit-learn `Pipeline` / `ColumnTransformer` approach unless a different encoding is explicitly justified and tested.
  - **XGBoost remains the project's primary model.**

### 2. Artifact Persistence:
- Save the fitted preprocessing pipeline as `preprocessor.joblib`.
- Save the trained model artifact separately (e.g. `model.joblib` or `model.json`).
- Store both deployment artifacts in Amazon S3.
- AWS Lambda loads the model and preprocessing artifact from S3 during container / execution context initialization for inference.

### 3. Inference Consistency & Safety:
- Incoming pre-match JSON payloads are transformed using the exact fitted preprocessing pipeline generated during training.
- `handle_unknown='ignore'` prevents inference failure when an unseen or unexpected categorical value appears in production, mapping unseen levels to all-zero vectors.
- Do **not** fit or refit the encoder during Lambda inference; it operates strictly in transform-only mode.

### 4. Architecture & Budget Constraints:
- **Zero SageMaker footprint:** Do NOT introduce Amazon SageMaker, SageMaker Endpoints, or SageMaker Training Jobs.
- Model training remains strictly on local compute or Google Colab (free-tier / local resources).
- Serverless inference deployment uses Amazon S3 + AWS Lambda.
- Amazon API Gateway exposes the HTTPS endpoint triggering the Lambda function for external or frontend access.

### 5. Security & Credentials:
- No AWS access keys or secret credentials may be hard-coded into application code or deployment packages.
- AWS Lambda accesses Amazon S3 securely via its assigned IAM execution role (`iam:GetRole` / S3 Read permissions).

---

## 12. Date and Season Policy

- **Retained Columns:** `match_id`, `season`, `date`.
- **Policy:** Strictly non-predictive metadata.
- **Usage:**
  - `date`: Chronological sorting and leak auditing.
  - `season`: Temporal train/test splitting (e.g., Train: 2008–2023, Validation: 2024, Test: 2025–2026).
  - Raw date timestamps are **never** passed into estimator feature vectors.
