# Phase 5 Specification: Live / In-Match Win Probability Prediction System & Player Data Audit

**Project:** Cloud-Based IPL Match Winning Probability Prediction & Live Match Analytics System  
**Pipeline Phase:** Phase 5 — Live / In-Match Data Pipeline & Machine Learning Design  
**Authoritative Scope:** Strictly Local ML Pipeline (Zero AWS resources, zero cloud deployments)  
**Prediction Point:** Ball-by-ball during the live match (post-toss through match conclusion)  
**Status:** DESIGN + PLAYER DATA AUDIT COMPLETE — WAITING FOR REVIEW  

---

## Executive Summary

Phase 1 (Data Preprocessing), Phase 2 (Pre-Match Feature Engineering), and Phase 3 (Pre-Match Model Training & Evaluation) established an approved, leak-free pre-match machine learning system forecasting match outcomes strictly at timestamp $T_0$ (**after the toss, before the first delivery**).

Phase 5 extends the system to **live, in-match win probability forecasting**. As a match unfolds, every delivery changes the score, wickets in hand, overs remaining, and required run rate. The Phase 5 system defines the data structures, state machine, feature engineering logic, player data audit, temporal split, and evaluation criteria required to compute and update the win probability:

$$P(\text{Team A Wins} \mid \mathbf{S}_{m, i, k}) \in [0, 1], \quad P(\text{Team B Wins} \mid \mathbf{S}_{m, i, k}) = 1 - P(\text{Team A Wins} \mid \mathbf{S}_{m, i, k})$$

where $\mathbf{S}_{m, i, k}$ represents the cumulative state of match $m$, innings $i$, immediately after delivery $k$.

---

## 1. Live Prediction Point

### 1.1 Trigger Definition
A live prediction state is officially recorded **immediately after each completed delivery event**:
- The delivery's physical and scoring outcome is finalized (runs off bat, extras, wicket event, player dismissed).
- The cumulative match score, balls completed, and wickets lost are updated.
- The model evaluates the updated state $\mathbf{S}_{m, i, k}$ and forecasts the win probability for the remainder of the match.

### 1.2 Legal Delivery vs. Non-Legal Delivery Treatment
In cricket, illegal deliveries (wides and no-balls) score penalty runs and/or extras, but do not increment the over/legal ball counter:
- **State Update on Extras:** When a wide or no-ball occurs, runs are added to the cumulative total and wickets (e.g. run-out, stumping off wide) are recorded immediately.
- **Granularity Decision:** Every recorded delivery in `data/processed/deliveries_clean.csv` triggers a state transition. A wide increases the score by the conceded extras, leaving `legal_balls_bowled` unchanged, which immediately increases the Current Run Rate (CRR) and reduces the runs required in a chase.

### 1.3 Match Lifecycle Transitions
1. **Match Start ($T_0$ — Pre-First Ball):**
   - 0 legal balls bowled in Innings 1.
   - The pre-match model (`models/random_forest.joblib`) provides the initial baseline probability $P_0$.
2. **Innings 1 Active Play ($T_{1.1} \dots T_{1.K_1}$):**
   - Deliveries 1 through $K_1$ of Innings 1.
   - Features represent offensive score accumulation against overs elapsed and wickets lost.
3. **Innings Break ($T_{\text{break}}$):**
   - Innings 1 is completed. The first-innings final total $R_1$ is locked.
   - The target score $T = R_1 + 1$ is officially established.
   - 0 balls bowled in Innings 2. Wickets in hand $= 10$, balls remaining $= 120$.
4. **Innings 2 Active Play ($T_{2.1} \dots T_{2.K_2}$):**
   - Deliveries 1 through $K_2$ of Innings 2.
   - Features represent chase dynamics: runs required, balls remaining, wickets in hand, and required run rate (RRR).
5. **Match Conclusion ($T_{\text{end}}$):**
   - The target is reached, the chasing team is bowled out, or 20 overs expire.
   - Once decisive victory is achieved, the outcome is deterministic ($P = 1.0$ or $P = 0.0$). No further predictions are generated.

---

## 2. Target Definition

### 2.1 Formal Formulation
The supervised learning target for every ball-by-ball state row is strictly:

$$\text{target\_team\_a\_win} = \begin{cases} 1 & \text{if Team A eventually won the match} \\ 0 & \text{if Team B eventually won the match} \end{cases}$$

### 2.2 Team A / Team B Identity Invariance
Consistent with Phase 2 and Phase 3:
- $\text{Team A} = \text{team\_1}$ from `matches_clean.csv`.
- $\text{Team B} = \text{team\_2}$ from `matches_clean.csv`.
- Target polarity is static across the entire match: whether Team A is batting first, fielding first, setting a target, or chasing, `target_team_a_win == 1` always means Team A won the match.

### 2.3 Strict Anti-Leakage Guard
- The final match outcome is used **exclusively as the supervised training label $y$**.
- It is **strictly prohibited** from entering the feature matrix $\mathbf{X}$ as an input.
- Target derivation never accesses delivery-level match ending flags or post-match summaries.

### 2.4 Abandoned & No-Result Match Treatment
- In `data/processed/matches_clean.csv`, exactly 9 matches are classified as `match_result_type == 'no_result'` (828 deliveries in `deliveries_clean.csv`).
- **Rule:** All 9 abandoned/no-result matches are **strictly excluded from live model training**. Without an official winner, no valid binary supervised ground truth exists.

---

## 3. Match Type Handling

| Match Type | Count in IPL History | Deliveries | Proposed Phase 5 Live Treatment | Approval Status |
| :--- | :---: | :---: | :--- | :---: |
| **Normal Completed Matches** | 1,218 matches | 290,598 deliveries | **Fully Included.** Standard 20-over setting and chasing dynamics. | Baseline Approved |
| **Super Over Tied Matches** | 16 matches | 3,956 reg. deliveries | **Regular Deliveries Included; Super Over Deliveries Excluded.** Deliveries for Innings 1 and 2 are trained with target set to the eventual eliminator winner. Super Over deliveries (Innings 3–6, 175 balls) are excluded from the main live model. | **Requires Sign-Off** |
| **Duckworth-Lewis (D/L) Matches** | 23 matches | 4,131 deliveries | **Recommendation: Exclude from Live Training Dataset.** Mid-match rain interruptions cause sudden retrospective over reductions and revised targets that distort ball-by-ball RRR calculations unless raw D/L sheet metadata is parsed. | **Requires Sign-Off** |
| **Abandoned / No-Result Matches** | 9 matches | 828 deliveries | **Strictly Excluded.** No definitive target label exists. | Baseline Approved |
| **Premature All-Out / Early Chase Finish** | ~450 matches | N/A | **Fully Included.** Natural termination of cricket innings. Wickets remaining $= 0$ or runs needed $\le 0$. | Baseline Approved |

### Detailed Rationale for Proposed Exclusions:
1. **Super Over Deliveries (175 deliveries across 16 matches):**
   - A Super Over is a 1-over tiebreaker with fundamentally distinct physical constraints: only 2 wickets allowed, 6 legal deliveries maximum.
   - Applying a 20-over required run rate model to a 6-ball shootout causes extreme feature distortion.
   - *Proposal:* Main live model governs overs 1–20. If a match ties at ball 20.0 of Innings 2, win probability is exactly $50.00\%$ pending the Super Over.
2. **Duckworth-Lewis Matches (23 matches):**
   - In matches with weather interruptions, the overs allocated to Innings 2 are truncated retroactively (e.g. target 53 runs in 8 overs).
   - In raw Cricsheet data, delivery timestamps do not record rain delay announcements. A 20-over feature pipeline would assume 120 balls remaining when in reality only 48 balls were allotted, generating severely corrupted required run rates.
   - *Proposal:* Exclude the 23 D/L matches from the live training set to preserve mathematical purity.

---

## 4. Live Team & Match-State Features

Every live feature must be strictly computable using information available **at or before delivery $k$ of innings $i$**.

```
Live Match Feature Vector (At Delivery k)
├── 1. Static Match & Team Context (Known Pre-Match)
│   ├── team_a, team_b, venue, city
│   ├── toss_winner_is_team_a, toss_decision_is_bat, team_a_bats_first
│   └── pre_match_win_prob_team_a (From Phase 3 Model)
├── 2. Core Innings State (Cumulative)
│   ├── current_innings (1 or 2)
│   ├── is_team_a_batting (1 or 0)
│   ├── current_innings_runs (Total runs scored so far)
│   ├── current_innings_wickets (Wickets lost: 0 to 10)
│   ├── wickets_remaining (10 - wickets lost)
│   ├── legal_balls_bowled (Completed legal deliveries: 0 to 120)
│   ├── legal_balls_remaining (120 - legal_balls_bowled)
│   └── overs_completed_fractional (legal_balls_bowled / 6.0)
├── 3. Dynamic Rate & Chase Metrics
│   ├── current_run_rate (CRR = runs / overs_completed)
│   ├── target_runs (Innings 2 only: Innings 1 total + 1; 0 in Innings 1)
│   ├── runs_needed (Innings 2 only: max(0, target - current_runs))
│   ├── required_run_rate (RRR = runs_needed / (balls_remaining / 6.0))
│   └── run_rate_differential (CRR - RRR in Innings 2; CRR - baseline in Innings 1)
├── 4. Rolling Micro-Momentum (Last 6, 12, 18 Balls)
│   ├── runs_scored_last_6_balls
│   ├── wickets_lost_last_6_balls
│   ├── runs_scored_last_12_balls
│   ├── wickets_lost_last_12_balls
│   ├── dot_ball_ratio_last_12 (Percentage of zero-run balls)
│   └── boundary_ratio_last_12 (Percentage of 4s and 6s)
└── 5. Active Partnership Dynamics
    ├── current_partnership_runs
    ├── current_partnership_balls
    └── current_partnership_run_rate
```

### 4.1 Exhaustive Match-State Feature Dictionary

| Feature Name | Type | Physical Meaning | Calculation / Formula | Availability | Leakage Risk & Prevention |
| :--- | :---: | :--- | :--- | :---: | :--- |
| `current_innings` | Categorical / Int | Active innings number | 1 for 1st innings, 2 for 2nd innings | Immediate | **Zero.** Known from match state. |
| `is_team_a_batting` | Binary | Whether Team A is currently batting | $1 \text{ if batting\_team} == \text{team\_a else } 0$ | Immediate | **Zero.** Fixed during innings. |
| `current_innings_runs`| Integer | Total runs scored in current innings | $\sum_{d=1}^k (\text{runs\_off\_bat}_d + \text{extras}_d)$ | Immediate | **Zero.** Strictly cumulative up to ball $k$. |
| `current_innings_wickets`| Integer | Wickets lost in current innings | $\sum_{d=1}^k \mathbb{I}(\text{player\_dismissed}_d \neq \emptyset)$ | Immediate | **Zero.** Strictly cumulative up to ball $k$. |
| `wickets_remaining` | Integer | Wickets in hand for batting side | $10 - \text{current\_innings\_wickets}$ | Immediate | **Zero.** Bounded in $[0, 10]$. |
| `legal_balls_bowled` | Integer | Completed legal deliveries | $\sum_{d=1}^k \mathbb{I}(\text{wides}_d == 0 \land \text{noballs}_d == 0)$ | Immediate | **Zero.** Ignores wides/noballs for over count. |
| `legal_balls_remaining`| Integer | Legal balls left in 20-over innings | $\max(0, 120 - \text{legal\_balls\_bowled})$ | Immediate | **Zero.** Monotonically decreases to 0. |
| `overs_completed` | Float | Fractional completed overs | $\text{legal\_balls\_bowled} / 6.0$ | Immediate | **Zero.** Standard cricket convention. |
| `current_run_rate` | Float | Runs scored per over bowled | $\frac{\text{current\_innings\_runs}}{\max(0.1667, \text{overs\_completed})}$ | Immediate | **Zero.** Guarded against division by zero at start. |
| `target_runs` | Integer | Target runs to win match | Innings 1: $0$; Innings 2: $\text{Innings 1 Total} + 1$ | After Innings 1 | **Safe in Innings 2.** Innings 1 target is 0. |
| `runs_needed` | Integer | Runs remaining to achieve target | Innings 1: $0$; Innings 2: $\max(0, \text{target\_runs} - \text{runs})$ | After Innings 1 | **Zero.** Bounded at 0 once target reached. |
| `required_run_rate` | Float | Runs required per over remaining | Innings 1: $0.0$; Innings 2: $\frac{\text{runs\_needed}}{\max(0.1667, \text{balls\_remaining}/6.0)}$ | After Innings 1 | **Zero.** Guarded when balls remaining $< 1$. |
| `rrr_crr_differential`| Float | Run rate pressure index | Innings 1: $0.0$; Innings 2: $\text{current\_run\_rate} - \text{required\_run\_rate}$ | After Innings 1 | **Zero.** Positive = batting side ahead of chase. |
| `runs_last_6_balls` | Integer | Runs scored in previous 6 deliveries | $\sum_{d=k-5}^k \text{runs}_d$ | Immediate | **Zero.** Strictly backward-looking window. |
| `wickets_last_6_balls`| Integer | Wickets lost in previous 6 deliveries | $\sum_{d=k-5}^k \text{wickets}_d$ | Immediate | **Zero.** Backward-looking window. |
| `runs_last_12_balls`| Integer | Runs scored in previous 12 deliveries | $\sum_{d=k-11}^k \text{runs}_d$ | Immediate | **Zero.** Backward-looking window. |
| `wickets_last_12_balls`| Integer | Wickets lost in previous 12 deliveries | $\sum_{d=k-11}^k \text{wickets}_d$ | Immediate | **Zero.** Backward-looking window. |
| `dot_ball_ratio_last_12`| Float | Fraction of dots in last 12 balls | $\frac{\text{count}(\text{runs} == 0 \text{ in last 12})}{\min(12, k)}$ | Immediate | **Zero.** Captures bowling pressure. |
| `boundary_ratio_last_12`| Float | Fraction of 4s/6s in last 12 balls | $\frac{\text{count}(\text{boundary in last 12})}{\min(12, k)}$ | Immediate | **Zero.** Captures batting acceleration. |
| `current_partnership_runs`| Integer | Runs scored by current pair | Cumulative runs since last wicket fell | Immediate | **Zero.** Resets to 0 upon wicket. |
| `current_partnership_balls`| Integer | Legal balls faced by current pair | Cumulative legal balls since last wicket fell | Immediate | **Zero.** Resets to 0 upon wicket. |
| `pre_match_prob_team_a`| Float | Baseline pre-match win probability for Team A | Chronologically generated out-of-sample probability using ONLY information available before match $M$ | At $T_0$ | **Zero.** Expanding match-level walk-forward simulation with strict chronological cutoff. |

### 4.2 Leakage-Safe Chronological Walk-Forward Pre-Match Anchor (Non-Negotiable)

To guarantee zero temporal leakage, `pre_match_prob_team_a` is defined strictly as:
> **`pre_match_prob_team_a` = chronologically generated out-of-sample pre-match probability using ONLY information available before the current match.**

#### 1. Chronological Ordering
- All eligible matches are sorted strictly chronologically by:
  $$\text{date ASC, then match\_id ASC}$$
- For match $M_i$, the historical training dataset consists strictly of eligible matches $M_k$ where:
  $$(\text{date}_k, \text{match\_id}_k) < (\text{date}_i, \text{match\_id}_i)$$
- Match $M_i$ itself, its final outcome, and all future matches are strictly excluded.
- Same-date player overlap was audited as zero across IPL history, making `(date, match_id)` ordering completely leakage-safe.

#### 2. Match-Level Expanding Walk-Forward Procedure
For every match $i \in [0, N-1]$:
1. Identify all eligible historical matches strictly prior to match $i$ ($k < i$).
2. Extract $X_{\text{train}}$ and $y_{\text{train}}$ using Phase 2 pre-match feature definitions.
3. Fit learned transformations (`OneHotEncoder` via `ColumnTransformer`) **ONLY** on this historical training subset.
4. Fit the pre-match Random Forest **ONLY** on this historical training subset.
5. Predict out-of-sample win probability for match $i$ using its pre-match features:
   $$\text{pre\_match\_prob\_team\_a} = P(\text{Team A Win} \mid \mathbf{x}_{i, \text{pre-match}})$$
6. Only after match $i$'s probability is computed and recorded does match $i$ become available for future matches ($j > i$).

#### 3. Cold-Start Handling
- For the earliest matches ($i < 40$, the first 40 matches of the inaugural 2008 season), there is insufficient historical data to train a stable Random Forest.
- These 40 matches are assigned the neutral, unbiased prior:
  $$\text{pre\_match\_prob\_team\_a} = 0.50 \quad (\text{cold\_start\_flag} = 1)$$
- Zero future data or season-end statistics are used in the cold start.
- From match index 40 onwards, the expanding walk-forward Random Forest generates all probabilities ($N=1,194$ matches).

#### 4. Model Configuration
- Matches the frozen Phase 3 pre-match Random Forest specification:
  - `n_estimators = 100`
  - `max_depth = 6`
  - `min_samples_leaf = 5`
  - `random_state = 42`
  - `n_jobs = 1`
- Same target definition (`target_team_a_win`) and Phase 2 feature matrix (21 continuous, 3 binary, 4 categorical columns).

#### 5. Anti-Leakage Verification Tests
The walk-forward generator is verified by 4 automated unit tests:
1. `test_walk_forward_has_no_self_inclusion`: Confirms match $M$'s ID is never present in its training set and $\text{training\_count} == i$.
2. `test_walk_forward_future_match_perturbation`: Modifying matches after $M$ has 0.0000 effect on $M$'s pre-match probability.
3. `test_walk_forward_current_match_outcome_perturbation`: Inverting match $M$'s own target outcome has 0.0000 effect on its pre-match probability.
4. `test_walk_forward_training_data_boundary`: Formally proves that for every checkpoint, $\max(\text{train\_date}, \text{train\_match\_id}) < (\text{curr\_date}, \text{curr\_match\_id})$.

---

## 5. Current Player Performance (In-Match State)

A key objective of Phase 5 is auditing and specifying **current-match player micro-state**. At any delivery $N$, the physical state of the active striker, non-striker, and active bowler can be tracked strictly up to delivery $N$.

### 5.1 Current Batter State Features

| Feature Name | Exact Calculation | Physical Meaning | Availability | Leakage Risk & Recommendation |
| :--- | :--- | :--- | :---: | :--- |
| `batter_runs_current_inn` | $\sum_{d \in \text{batter}} \text{runs\_off\_bat}_d$ | Runs accumulated by active striker | Immediate | **Safe.** Use in Model B/C/D. |
| `batter_balls_current_inn` | $\sum_{d \in \text{batter}} \mathbb{I}(\text{wides} == 0)$ | Legal balls faced by active striker | Immediate | **Safe.** Use in Model B/C/D. |
| `batter_strike_rate_current` | $\frac{\text{batter\_runs}}{\max(1, \text{batter\_balls})} \times 100$ | Striker's scoring velocity in innings | Immediate | **Safe.** Bounded with shrinkage for balls $< 5$. |
| `batter_boundaries_current` | $\sum_{d \in \text{batter}} \mathbb{I}(\text{runs\_off\_bat} \in \{4, 6\})$ | Boundaries struck by batter | Immediate | **Safe.** Measures batter settledness. |
| `batter_dots_current` | $\sum_{d \in \text{batter}} \mathbb{I}(\text{runs\_off\_bat} == 0 \land \text{extras} == 0)$ | Dot balls absorbed by batter | Immediate | **Safe.** Measures batter pressure. |

### 5.2 Current Bowler State Features

| Feature Name | Exact Calculation | Physical Meaning | Availability | Leakage Risk & Recommendation |
| :--- | :--- | :--- | :---: | :--- |
| `bowler_legal_balls_current` | $\sum_{d \in \text{bowler}} \mathbb{I}(\text{wides} == 0 \land \text{noballs} == 0)$ | Legal deliveries bowled in match | Immediate | **Safe.** Cannot exceed 24 (4 overs). |
| `bowler_overs_current` | $\text{bowler\_legal\_balls\_current} / 6.0$ | Fractional overs bowled so far | Immediate | **Safe.** |
| `bowler_runs_conceded_current` | $\sum_{d \in \text{bowler}} (\text{runs\_off\_bat} + \text{extras})$ | Runs conceded in match so far | Immediate | **Safe.** Excludes byes/leg-byes per standard cricket scoring. |
| `bowler_wickets_current` | $\sum_{d \in \text{bowler}} \mathbb{I}(\text{wicket\_type} \in \text{BOWLER\_WICKET\_TYPES})$ | Wickets credited to bowler in match | Immediate | **Safe.** Run-outs excluded. |
| `bowler_economy_current` | $\frac{\text{bowler\_runs\_conceded}}{\max(0.1667, \text{bowler\_overs})}$ | Bowler's in-match economy rate | Immediate | **Safe.** Guarded for early overs. |
| `bowler_dots_current` | $\sum_{d \in \text{bowler}} \mathbb{I}(\text{total\_runs} == 0)$ | Dot balls bowled in match | Immediate | **Safe.** Measures bowling control. |

### 5.3 Partnership & Micro-Interaction Features
- `striker_partnership_share`: $\frac{\text{batter\_runs\_current\_inn}}{\max(1, \text{current\_partnership\_runs})}$ (Captures who is dominating the crease).
- `bowler_spells_in_innings`: Number of distinct over spells the bowler has bowled in this match.

---

## 6. Historical Player Performance Audit & Specifications

### 6.1 Historical Player Data Audit (Cricsheet Data)
An exhaustive inspection of `data/processed/deliveries_clean.csv` (295,732 deliveries, 1,243 matches) revealed:
- **Total Unique Players in IPL History:** **808 players**
  - Unique Strikers: 739
  - Unique Non-Strikers: 728
  - Unique Bowlers: 578
  - Unique Dismissed Players: 687
- **Null Fields:** Exactly 0 null strikers, 0 null non-strikers, 0 null bowlers across all 295,732 rows.
- **Name Formatting:** Clean, consistent initials + surname convention (e.g. `V Kohli`, `MS Dhoni`, `JJ Bumrah`).
- **Disambiguated Player Suffixes:** Exactly two players share identical initials/surnames and are cleanly disambiguated with `(2)` by Cricsheet: `Arshad Khan (2)` and `Harmeet Singh (2)`.
- **Missing External Metadata:** Cricsheet raw CSVs **do not contain** batting handedness (LH/RH) or bowling style (off-spin, leg-spin, fast-medium). Any role categorization must be inferred strictly from bowling/batting frequencies.

### 6.2 The Sample Size / Sparsity Trap
The career ball distribution reveals extreme skewness:

```
Batter Balls Faced Distribution (N = 739 Batters):
├── Mean:   400.2 balls
├── Median:  74.0 balls
├── 25th %:  17.0 balls
├── 75th %: 311.0 balls
├── < 10 balls:  123 batters (16.6%)
├── < 50 balls:  319 batters (43.2%)
├── < 100 balls: 412 batters (55.8%)
└── >= 500 balls: 142 batters (19.2%)

Bowler Balls Bowled Distribution (N = 578 Bowlers):
├── Mean:   511.6 balls
├── Median: 197.0 balls
├── < 30 balls (< 5 overs): 96 bowlers (16.6%)
├── < 60 balls (< 10 overs): 168 bowlers (29.1%)
└── >= 500 balls: 160 bowlers (27.7%)
```

### 6.3 Shrinkage & Cold-Start Policy (Non-Negotiable)
Because **55.8% of batters faced fewer than 100 career balls** and **29.1% of bowlers bowled fewer than 10 overs**, raw historical strike rates or economy rates are mathematically toxic (e.g. a tail-ender with 1 ball and 6 runs has raw $\text{SR} = 600.0$).
- **Rule:** All historical player features **must use Empirical Bayesian Shrinkage** towards expanding league priors:
  $$\text{smoothed\_historical\_sr} = \frac{\sum \text{career\_runs}_{<t} + m \times \text{league\_sr}_{<t}}{\sum \text{career\_balls}_{<t} + m}, \quad m = 60.0 \text{ balls}$$
  $$\text{smoothed\_historical\_econ} = \frac{\sum \text{career\_runs\_conceded}_{<t} + m \times \text{league\_econ}_{<t}}{\sum \text{career\_overs\_bowled}_{<t} + m}, \quad m = 10.0 \text{ overs (60 balls)}$$
- **Anti-Leakage Rule:** Historical statistics must be strictly cumulative up to match $t-1$. Match $t$'s deliveries are strictly excluded from the historical prior.

---

## 7. Batter-vs-Bowler Historical Matchup Audit

### 7.1 Matchup Sparsity Audit
An empirical analysis of all 295,732 delivery interactions in IPL history evaluated the viability of head-to-head batter-vs-bowler features:
- **Total Unique (Striker, Bowler) Pairs:** **31,381 unique matchups**
- **Mean Balls per Matchup:** 9.42 deliveries
- **Median Balls per Matchup:** **6.0 deliveries (exactly 1 over)**
- **Matchups with exactly 1 ball:** 2,895 pairs (9.2%)
- **Matchups with $< 6$ balls ($< 1$ over):** 15,354 pairs (**48.9%**)
- **Matchups with $< 12$ balls ($< 2$ overs):** 23,834 pairs (**76.0%**)
- **Matchups with $\ge 30$ balls (5 overs):** 1,726 pairs (5.5%)
- **Matchups with $\ge 60$ balls (10 overs):** 291 pairs (0.9%)

### 7.2 The Zero-Prior Matchup Reality
A chronological traversal of all deliveries revealed that across the entire dataset:

$$\mathbf{141,622 \text{ deliveries (47.89\%)} \text{ occur with ZERO prior historical matchup balls between that striker and bowler!}}$$

### 7.3 Fallback Hierarchy & Empirical Bayes Shrinkage Specification
Using raw batter-vs-bowler head-to-heads as direct model features is **statistically invalid and dangerous** because 47.89% of deliveries have 0 prior encounters, and 76.0% have fewer than 12 balls.

To eliminate step-function discontinuities in feature space while adhering strictly to the fallback hierarchy, the system implements continuous **Empirical Bayesian Shrinkage** with prior strength $m = 24.0$ balls (equivalent to 4 completed overs):

$$\text{matchup\_strike\_rate\_shrunk} = \frac{\text{prior\_runs} + m \times \left(\frac{\text{historical\_batter\_strike\_rate}}{100}\right)}{\text{prior\_balls} + m} \times 100, \quad m = 24.0 \text{ balls}$$

#### Mathematical Behavior & Fallback Verification:
1. **Level 2 Seamless Fallback ($\text{prior\_balls} = 0$):**
   When a batter faces a bowler for the first time in IPL history (47.89% of all deliveries), $\text{prior\_balls} = 0$ and $\text{prior\_runs} = 0$:
   $$\text{matchup\_strike\_rate\_shrunk} = \frac{0 + 24.0 \times \left(\frac{\text{historical\_batter\_strike\_rate}}{100}\right)}{0 + 24.0} \times 100 = \text{historical\_batter\_strike\_rate}$$
   The feature seamlessly evaluates to the batter's shrunk historical strike rate with zero discontinuity or missing values.
2. **Intermediate Encounters ($0 < \text{prior\_balls} < 24$):**
   Smoothly transitions from the batter's baseline prior to direct head-to-head performance without artificial threshold steps.
3. **Established Encounters ($\text{prior\_balls} \ge 24$):**
   At exactly 24 balls, weight is split evenly (50% direct matchup, 50% historical prior). As $\text{prior\_balls} \to \infty$, weight approaches 100% direct matchup.
- *Recommendation for Ablation:* Isolate batter-vs-bowler features (`matchup_balls_prior`, `matchup_runs_prior`, `matchup_wickets_prior`, `matchup_strike_rate_shrunk`) into **Model D** of the ablation plan to verify whether they improve or degrade Log Loss.

---

## 8. Player Identity & Data Quality Audit

| Audit Category | Check Performed | Finding | Impact / Action |
| :--- | :--- | :--- | :--- |
| **Unique Identifiers** | Player name consistency in Cricsheet | 808 unique names across striker, non-striker, bowler, player_dismissed | Clean data, no numeric IDs needed. |
| **Missing Values** | Null or empty strings in player columns | 0 nulls across all 295,732 delivery rows | 100% complete player fields. |
| **Whitespace & Formatting** | Leading/trailing spaces or double spaces | 0 occurrences across all 808 player names | Perfectly trimmed by Phase 1 preprocessing. |
| **Name Disambiguation** | Duplicate initials and surnames | Exactly 2 disambiguated pairs: `Arshad Khan (2)` and `Harmeet Singh (2)` | Suffixes preserved as distinct historical players. |
| **Player Rebrandings/Spelling** | Aliases / alternate spellings | Cricsheet standardizes player names across seasons | No manual alias dictionary required. |
| **External Role Metadata** | Bowling style & batting handedness | **Absent from Cricsheet raw CSVs** | Exclude handedness/style from Phase 5 unless externally augmented. |

---

## 9. First-Innings State Handling

In Innings 1:
- The target does **not** exist yet. There is no "runs needed" or "required run rate".
- **First-Innings Feature Set:**
  1. `current_innings_runs`: Total runs accumulated so far.
  2. `current_run_rate`: Scoring pace relative to overs bowled.
  3. `wickets_remaining`: Resources remaining ($10 - W$).
  4. `legal_balls_remaining`: Finite horizon ($120 - B$).
  5. `projected_score`: $\text{current\_runs} + (\text{balls\_remaining} / 6.0) \times \text{venue\_death\_rpo\_prior}$.
  6. `venue_par_differential`: $\text{projected\_score} - \text{venue\_historical\_first\_innings\_average}$.
  7. `is_team_a_batting`: Polarity indicator (if Team A batting, higher projected score increases $P(\text{Team A})$; if Team B batting, it decreases $P(\text{Team A})$).

---

## 10. Second-Innings / Chase State Handling

In Innings 2, match dynamics are governed by a finite-resource chase boundary:
1. **Target Score $T$:** $\text{Innings 1 Total Score} + 1$.
2. **Runs Required $R_{\text{req}}$:** $\max(0, T - \text{current\_innings\_2\_runs})$.
3. **Balls Remaining $B_{\text{rem}}$:** $\max(0, 120 - \text{legal\_balls\_bowled})$.
4. **Wickets Remaining $W_{\text{rem}}$:** $10 - \text{wickets\_lost}$.
5. **Required Run Rate (RRR):**
   $$\text{RRR} = \begin{cases} 0.0 & \text{if } R_{\text{req}} == 0 \\ \frac{R_{\text{req}}}{B_{\text{rem}} / 6.0} & \text{if } B_{\text{rem}} > 0 \\ 99.0 & \text{if } B_{\text{rem}} == 0 \text{ and } R_{\text{req}} > 0 \end{cases}$$
6. **Pressure Index:** $\text{run\_rate\_differential} = \text{current\_run\_rate} - \text{required\_run\_rate}$.

---

## 11. Strict Ball-by-Ball Anti-Leakage Rules

### 11.1 The Golden Invariant of Live Modeling

$$\mathbf{S}_{m, i, k} \text{ depends ONLY on deliveries } d \in \{1, 2, \dots, k\} \text{ of match } m$$

### 11.2 Correct vs. Incorrect Feature Construction Examples

| Feature Concept | INCORRECT (LEAKAGE) | CORRECT (LEAK-FREE) |
| :--- | :--- | :--- |
| **Innings 1 Target** | Using final Innings 1 score to compute target in Innings 1 | Target is set to `0` during Innings 1; only exists in Innings 2 |
| **Innings Length** | Dividing runs by total balls actually bowled in the innings | Dividing runs by completed fractional overs up to ball $k$ |
| **Player Performance** | Using batter's final match score at ball 10.2 | Using batter's runs accumulated strictly up to ball 10.2 |
| **Bowler Economy** | Using bowler's 4-over completed match economy | Using bowler's runs conceded divided by overs bowled up to ball $k$ |
| **Player Matchup** | Using matchup statistics from the entire match | Using matchup statistics from matches *prior* to match $m$ plus balls in match $m$ up to $k-1$ |
| **Target Variable** | Using `target_team_a_win` inside the feature vector | Using `target_team_a_win` strictly as training label $y$ |

---

## 12. Dataset Granularity

### 12.1 Granularity Options Analysis
- **Option A (Every Legal Delivery — Recommended):**
  - Dataset Size: ~283,000 rows across 1,211 matches.
  - Strengths: Full real-time resolution; captures momentum shifts on wickets and sixes; matches broadcast tracking.
  - Feasibility: 283,000 rows train in $< 10$ seconds with XGBoost on local CPU.
- **Option B (Over-by-Over Checkpoints):**
  - Dataset Size: ~45,000 rows.
  - Weaknesses: Cannot evaluate live probability mid-over when a critical wicket falls on ball 2.
- **Recommendation:** **Adopt Option A (Every Legal Delivery)**. Non-legal deliveries (wides, noballs) update state totals and do not create separate duplicate legal rows.

---

## 13. Chronological Train / Validation / Test Partition

### 13.1 Match-Level Isolation Rule
**CRITICAL RULE:** Delivery rows from the same match must **NEVER** be randomly partitioned across train, validation, and test splits.
- Splitting deliveries randomly across splits means training on ball 10.2 and testing on ball 10.3 of the same match. This constitutes catastrophic data leakage.
- **All deliveries for a match belong entirely to that match's split.**

### 13.2 Partition Boundaries (Season-Based)

| Split | Seasons Included | Completed Matches | Estimated Delivery Rows | Target Balance |
| :--- | :--- | :---: | :---: | :---: |
| **Training** | `2008`–`2021` (14 seasons) | 872 matches | ~208,000 deliveries | ~50.8% Class 1 |
| **Validation** | `2022`–`2023` (2 seasons) | 147 matches | ~35,000 deliveries | ~52.4% Class 1 |
| **Final Test** | `2024`–`2026` (3 seasons) | 215 matches | ~51,000 deliveries | ~45.1% Class 1 |
| **Total** | `2008`–`2026` (19 seasons) | 1,234 matches | **~294,000 deliveries** | **50.00% Class 1** |

---

## 14. Super Over Handling

1. **Regular Deliveries (Innings 1 and 2):** Fully **retained** in the live training dataset. The target $y$ is set to the eventual eliminator winner.
2. **Super Over Deliveries (Innings 3+):** Strictly **excluded** from the main 20-over live dataset.
   - A 6-ball innings with 2 wickets follows fundamentally different risk dynamics than 20-over cricket.
   - If a live match reaches ball 20.0 of Innings 2 with scores level, the system outputs $P = 50.00\%$ win probability.

---

## 15. Feature Encoding & Normalization

1. **Categorical Variables:** Nominal features (`team_a`, `team_b`, `venue`, `city`) are transformed using `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` fitted **strictly on the training split**.
2. **Player Identifiers:** **Do NOT one-hot encode player names.** One-hot encoding 808 player names would create 1,600+ sparse binary columns, causing catastrophic overfitting on rare players. Player information must enter the model **strictly as continuous, Bayesian-smoothed performance indices** (e.g. `batter_shrunk_sr`, `bowler_shrunk_econ`).
3. **Numerical Variables:** Rate and count features (`current_run_rate`, `required_run_rate`, `balls_remaining`, `wickets_remaining`) are left unscaled for tree-based models, and scaled with `StandardScaler` inside a Pipeline for Logistic Regression.

---

## 16. Class Balance in Live States

- Because $\text{Team A} = \text{team\_1}$ and $\text{Team B} = \text{team\_2}$ naturally resulted in exactly 617 wins for Team A and 617 wins for Team B, the delivery-level dataset naturally hovers near **50.2% Class 1**.
- **Policy:** **Zero artificial balancing.** No SMOTE, no random under-sampling, no class-weight manipulation. The empirical match distribution is preserved.

---

## 17. Candidate Models for Live Prediction

| Model Candidate | Role | Strengths | Weaknesses |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | Linear Baseline | Ultra-fast inference ($< 1\text{ ms}$), smooth sigmoidal output, monotonic curves. | Incapable of capturing sharp non-linear interactions ($\text{RRR} \times \text{wickets\_remaining}$). |
| **Random Forest** | Bagged Trees Baseline | Highly resilient to overfitting, variance reduction, reliable calibration. | Large memory footprint on 280k rows, slower tree ensemble inference. |
| **XGBoost** | **Primary Live Candidate** | Outstanding ability to capture sharp non-linear boundaries (e.g. steep probability drop when 8th wicket falls in a chase), handles tabular interactions natively, ultra-fast C++ inference. | Requires careful calibration to prevent over-confident probabilities near 0.0 or 1.0. |

---

## 18. Probability Evaluation Framework

Live models predict continuous win probability. Evaluation relies on strictly proper scoring rules:

1. **Log Loss (Binary Cross-Entropy):** Strongly penalizes over-confident incorrect probabilities.
2. **Brier Score:** Mean squared probability error ($B = \frac{1}{N}\sum (\hat{p}_i - y_i)^2$).
3. **Expected Calibration Error (ECE):** 10-bin reliability curves.
4. **Phase-Segmented Brier Score:** Evaluates uncertainty resolution across Powerplay (overs 1–6), Middle (overs 7–15), and Death (overs 16–20).
   - *Physical Sanity Check:* Brier score must monotonically decrease as the match progresses and uncertainty resolves towards $T_{\text{end}}$.

---

## 19. Live Match Simulator Architecture

```
Deliveries Clean CSV (Match ID M)
                ↓
    ┌───────────────────────┐
    │ LiveMatchSimulator    │
    │ 1. Reads delivery d   │
    │ 2. Updates State      │
    │ 3. Generates Features │
    │ 4. Invokes Live Model │
    │ 5. Emits Live JSON    │
    └───────────────────────┘
                ↓
Live Match Output Event (JSON):
{
  "match_id": 1082591,
  "innings": 2,
  "over": 14.2,
  "legal_ball": 86,
  "score": "124/4",
  "target": 173,
  "runs_needed": 49,
  "balls_remaining": 34,
  "crr": 8.65,
  "rrr": 8.65,
  "team_a_win_probability": 0.524,
  "team_b_win_probability": 0.476
}
```

---

## 20. Player Feature Ablation Plan

To determine whether player-level micro-features genuinely improve probability quality or simply overfit, a rigorous 4-model ablation experiment is designed:

```
Ablation Hierarchy:
├── Model A (Baseline): Match-State / Team Features Only
│   └── Inputs: Core innings state, CRR, RRR, wickets in hand, pre-match win prob.
├── Model B: Match-State + Current-Match Player State
│   └── Inputs: Model A + active batter runs/balls/SR in current match + active bowler figures in current match.
├── Model C: Match-State + Current Player + Historical Player Priors
│   └── Inputs: Model B + Bayesian-smoothed historical batter SR/avg + historical bowler economy.
└── Model D: Full System (+ Batter-vs-Bowler Matchups)
    └── Inputs: Model C + Bayesian-shrunk head-to-head batter-vs-bowler interaction features.
```

### Ablation Evaluation Criteria:
1. **Log Loss & Brier Score Comparison:** Does Model B/C/D lower out-of-time Log Loss on the 2024–2026 Test Set compared to Model A?
2. **Calibration (ECE):** Does micro-player information increase over-confidence (higher ECE)?
3. **Generalization:** If Model D performs well on validation but degrades on the final test set, batter-vs-bowler matchups will be rejected as overfitted noise.

---

## 21. Proposed Source Code Structure

All live pipeline code will reside under `src/live/`:

```text
src/
├── data_processing/            # Phase 1 code (frozen, untouched)
├── features/                   # Phase 2 pre-match code (frozen, untouched)
├── model_training/             # Phase 3 pre-match code (frozen, untouched)
└── live/                       # Phase 5 Live System (New)
    ├── __init__.py
    ├── live_state_tracker.py   # Ball-by-ball cumulative state machine
    ├── live_features.py        # Macro match-state feature extractor
    ├── player_features.py      # Current & historical player feature extractor
    ├── matchup_features.py     # Batter-vs-bowler head-to-head extractor with fallback
    ├── generate_live_data.py   # Dataset builder emitting data/processed/live_deliveries_dataset.csv
    ├── train_live_models.py    # Training & persistence of live models
    ├── evaluate_live.py        # Metric computation, calibration, and ablation runner
    └── live_simulator.py       # Ball-by-ball historical match replay engine
tests/
└── test_live_pipeline.py       # Unit tests for state transitions and anti-leakage invariants
docs/
└── PHASE_5_LIVE_ML_SPEC.md     # Authoritative specification (This document)
```

---

## 22. Verification Checklist

Before live dataset generation or live model training is permitted, the pipeline must pass all 20 verification checks:

### Data & Identity Integrity:
- [ ] 1. All 808 unique player names mapped consistently with zero nulls.
- [ ] 2. Match IDs match `matches_clean.csv`.
- [ ] 3. Deliveries sorted strictly chronologically `(date ASC, match_id ASC, innings ASC, ball ASC)`.
- [ ] 4. Innings ordering strictly $1 \rightarrow 2$.

### Match State Integrity:
- [ ] 5. Cumulative score equals sum of completed deliveries.
- [ ] 6. Wickets lost equals sum of bowler and run-out dismissals up to ball $k$.
- [ ] 7. Wides and no-balls update score/wickets but do not advance `legal_balls_bowled`.
- [ ] 8. Innings 2 target is strictly $\text{Innings 1 Total Score} + 1$.
- [ ] 9. Runs needed and RRR handle boundary conditions ($B_{\text{rem}} = 0$, $R_{\text{needed}} \le 0$) safely.
- [ ] 10. Innings 1 chase features are strictly set to 0.0 with zero lookahead to Innings 2.

### Player & Matchup Integrity:
- [ ] 11. Current batter runs/balls strictly reflect deliveries faced in the active innings up to ball $k$.
- [ ] 12. Current bowler figures strictly reflect deliveries bowled in the active match up to ball $k$.
- [ ] 13. Historical player statistics use only matches strictly *before* the current match ($< t$).
- [ ] 14. Batter-vs-bowler matchups use only matches strictly *before* the current match ($< t$).
- [ ] 15. The 47.89% zero-prior matchups fall back cleanly to independent player priors.

### Leakage & Partitioning:
- [ ] 16. State at delivery $k$ strictly uses deliveries $1 \dots k$. Delivery $k+1$ is completely unknown.
- [ ] 17. Final match outcome is strictly the target label $y$ and never an input feature.
- [ ] 18. All deliveries from a match belong entirely to that match's split (zero intra-match splitting).
- [ ] 19. All 9 abandoned/no-result matches are 100% excluded.
- [ ] 20. Super Over deliveries (Innings 3+) are isolated from the 20-over live training set.

---

## 23. Final Design Summary

### A. Recommended Live Prediction Definition
Prediction generated immediately after each completed delivery event, evaluating updated cumulative match state. Pre-match model anchors probability at $T_0$.

### B. Recommended Target
$\text{target\_team\_a\_win} = 1$ if Team A wins, $0$ if Team B wins. Team A = `team_1`, Team B = `team_2`. Strictly outcome-independent.

### C. Recommended Match-State Features
Current score, wickets lost, wickets remaining, legal balls bowled, legal balls remaining, CRR, target, runs needed, RRR, $\text{CRR} - \text{RRR}$ diff, rolling momentum (last 6/12 balls), partnership runs/balls, pre-match win prob anchor.

### D. Recommended Current-Player Features
Active batter runs, balls, strike rate, boundaries, dots in current innings; active bowler overs, runs conceded, wickets, economy, dots in current match.

### E. Recommended Historical-Player Features
Empirical Bayesian-shrunk historical batter strike rate ($m=60$ balls) and bowler economy ($m=10$ overs) prior to current match.

### F. Recommended Batter-vs-Bowler Features
Empirical Bayesian-shrunk head-to-head metrics with strict fallback hierarchy for the 47.89% zero-prior encounters.

### G. Features Explicitly Rejected
1. **Raw One-Hot Player IDs:** Rejected due to extreme high cardinality (808 players) and overfitting.
2. **Raw Unsmoothed Historical Averages:** Rejected because 55.8% of batters have $<100$ balls, producing extreme mathematical noise.
3. **Batting Handedness & Bowling Style:** Rejected because Cricsheet raw data lacks these attributes.
4. **Final Innings Score Lookahead:** Strictly banned as catastrophic leakage.

### H. Recommended Dataset Granularity
Ball-by-ball granularity (every completed delivery event ~283,000 rows across 1,211 matches).

### I. Recommended Chronological Split
Match-based chronological partition: Train (2008–2021: 872 matches), Val (2022–2023: 147 matches), Test (2024–2026: 215 matches).

### J. Recommended Model Candidates
XGBoost as primary live candidate; Random Forest and Logistic Regression as comparative baselines.

### K. Player Ablation Experiment Plan
Evaluate Model A (Macro) vs Model B (Current Player) vs Model C (Historical Player) vs Model D (Matchup) to verify out-of-time Log Loss improvement.

### L. Major Unresolved Decisions Requiring Human Approval
1. **Exclusion of 23 D/L Matches:** Recommended to exclude to avoid corrupted required run rates from unrecorded rain delays.
2. **Super Over Delivery Exclusion:** Recommended to exclude 175 Super Over deliveries from 20-over training.
3. **Architecture Choice:** Unified Single Model vs. Dedicated Two-Model Pipeline (Innings 1 Model + Innings 2 Chase Model).

---

## 24. Implementation Restriction & Status

```text
PHASE 5 STEP 1 STATUS:
DESIGN + PLAYER DATA AUDIT COMPLETE — WAITING FOR REVIEW
```
