# Pre-Match Feature Definitions & Data Dictionary

**Project:** Cloud-Based IPL Match Winning Probability Prediction & Live Match Analytics System  
**Pipeline Phase:** Phase 2 — Pre-Match Feature Engineering  
**Dataset Artifact:** `data/processed/pre_match_dataset.csv`  
**Dataset Size:** 1,234 rows $\times$ 32 columns  
**Prediction Timing:** Evaluated strictly **AFTER THE TOSS** and **BEFORE THE FIRST BALL OF THE MATCH IS BOWLED**.  

---

## 1. Metadata and Key Attributes (Non-Feature Columns)

| Column Name | Data Type | Description & Purpose | Leakage Safety Policy |
| :--- | :--- | :--- | :--- |
| `match_id` | String / Int | Unique identifier linking to `matches_clean.csv` and `deliveries_clean.csv`. | Non-predictive primary key. |
| `season` | String | IPL tournament year label (e.g., `2008`, `2024`). | Non-predictive metadata; used for out-of-time chronological train/test splits. |
| `date` | Date (`YYYY/MM/DD`) | Official match date. | Non-predictive metadata; used strictly for chronological auditing and sorting. |

---

## 2. Supervised Target Attribute

| Column Name | Data Type | Range | Definition & Mathematical Formulation |
| :--- | :--- | :--- | :--- |
| `target_team_a_win` | Binary Integer | `{0, 1}` | **Target Variable:** `1` if Team A (`team_1`) won the match, `0` if Team B (`team_2`) won. Exactly 617 ones (50.00%) and 617 zeros (50.00%) across all 1,234 decisive matches. |

---

## 3. Team & Venue Categorical Context Features

| Column Name | Data Type | Cardinality | Definition & Encoding Advice |
| :--- | :--- | :--- | :--- |
| `team_a` | Categorical | 15 teams | Canonical name of Team A (`team_1`). Declared as native category for tree models or differential encoding for linear models. |
| `team_b` | Categorical | 15 teams | Canonical name of Team B (`team_2`). Declared as native category for tree models or differential encoding for linear models. |
| `venue` | Categorical | 36 venues | Standardized canonical stadium hosting the fixture. Consolidates 60 raw alias variations. |
| `city` | Categorical | 31 cities | Host city of the venue (includes verified imputations for Dubai and Sharjah). |

---

## 4. Toss & Inning Role Features

| Column Name | Data Type | Range | Formula / Definition | Physical Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| `toss_winner_is_team_a` | Binary | `{0, 1}` | `1 if toss_winner == team_a else 0` | Indicates if Team A won the toss. |
| `toss_decision_is_bat` | Binary | `{0, 1}` | `1 if toss_decision == 'bat' else 0` | Indicates whether the toss-winning captain chose to bat first ($1$) or field first ($0$). |
| `team_a_bats_first` | Binary | `{0, 1}` | `(toss_a & bat) \| (!toss_a & field)` | **Crucial role indicator:** $1$ if Team A sets the target in 1st innings; $0$ if Team A chases in 2nd innings. |

---

## 5. Team Form & Momentum Features

| Column Name | Data Type | Range | Formulation & Window | Cold-Start Fallback ($N < 5$ or $N < 10$) |
| :--- | :--- | :--- | :--- | :--- |
| `team_a_form_last_5` | Float | `[0.0, 1.0]` | Win rate of Team A across its previous 5 completed matches strictly prior to match $t$. | Bayesian smoothing: $\frac{\text{wins} + 1.0}{N + 2.0}$ ($0.50$ when $N=0$). |
| `team_b_form_last_5` | Float | `[0.0, 1.0]` | Win rate of Team B across its previous 5 completed matches strictly prior to match $t$. | Bayesian smoothing: $\frac{\text{wins} + 1.0}{N + 2.0}$ ($0.50$ when $N=0$). |
| `team_a_form_last_10` | Float | `[0.0, 1.0]` | Win rate of Team A across its previous 10 completed matches strictly prior to match $t$. | Bayesian smoothing: $\frac{\text{wins} + 1.0}{N + 2.0}$ ($0.50$ when $N=0$). |
| `team_b_form_last_10` | Float | `[0.0, 1.0]` | Win rate of Team B across its previous 10 completed matches strictly prior to match $t$. | Bayesian smoothing: $\frac{\text{wins} + 1.0}{N + 2.0}$ ($0.50$ when $N=0$). |
| `form_diff_last_5` | Float | `[-1.0, 1.0]` | `team_a_form_last_5 - team_b_form_last_5` | Positive indicates Team A has higher recent momentum. |
| `team_a_matches_played`| Integer | $\ge 0$ | Total cumulative completed matches played by Team A prior to match $t$. | Starts at $0$ for inaugural matches. |
| `team_b_matches_played`| Integer | $\ge 0$ | Total cumulative completed matches played by Team B prior to match $t$. | Starts at $0$ for inaugural matches. |

---

## 6. Head-to-Head (H2H) Features

| Column Name | Data Type | Range | Formulation & Window | Cold-Start Fallback |
| :--- | :--- | :--- | :--- | :--- |
| `h2h_matches_played` | Integer | $\ge 0$ | Cumulative prior completed encounters between Team A and Team B. | $0$ on first encounter. |
| `team_a_h2h_win_rate` | Float | `[0.0, 1.0]` | Historical win rate of Team A against Team B prior to match $t$. | Bayesian m-estimate ($m=2, p_0=0.50$): $\frac{\text{wins} + 1.0}{N + 2.0}$ ($0.50$ when $N=0$). |

---

## 7. Venue & Ground Bias Features

| Column Name | Data Type | Range | Formulation & Window | Cold-Start Fallback |
| :--- | :--- | :--- | :--- | :--- |
| `team_a_venue_matches` | Integer | $\ge 0$ | Cumulative matches played by Team A at this canonical venue prior to match $t$. | $0$ if never played at ground. |
| `team_a_venue_win_rate`| Float | `[0.0, 1.0]` | Team A win rate at this venue prior to match $t$. | Smoothed towards Team A's overall prior win rate ($m=3$). |
| `team_b_venue_matches` | Integer | $\ge 0$ | Cumulative matches played by Team B at this canonical venue prior to match $t$. | $0$ if never played at ground. |
| `team_b_venue_win_rate`| Float | `[0.0, 1.0]` | Team B win rate at this venue prior to match $t$. | Smoothed towards Team B's overall prior win rate ($m=3$). |
| `venue_win_rate_diff` | Float | `[-1.0, 1.0]` | `team_a_venue_win_rate - team_b_venue_win_rate` | Positive indicates Team A venue familiarity/success advantage. |
| `venue_chasing_win_rate`| Float | `[0.0, 1.0]` | Historical win rate of teams batting second (chasing) at this ground. | Smoothed towards expanding league chasing average ($m=5$). |

---

## 8. Batting & Bowling Strength Features

| Column Name | Data Type | Formulation & Window | Physical Significance | Cold-Start Fallback |
| :--- | :--- | :--- | :--- | :--- |
| `team_a_batting_rpo_last_5` | Float | Team A runs scored per legal over in its last 5 completed innings. | Measures recent scoring velocity, normalized across full and rain-shortened games. | Smoothed towards expanding league average RPO ($m=40$ overs). |
| `team_b_batting_rpo_last_5` | Float | Team B runs scored per legal over in its last 5 completed innings. | Measures recent scoring velocity, normalized across full and rain-shortened games. | Smoothed towards expanding league average RPO ($m=40$ overs). |
| `batting_rpo_diff_last_5` | Float | `team_a_batting_rpo_last_5 - team_b_batting_rpo_last_5` | Positive indicates Team A scores at a faster run rate. | $0.00$ prior. |
| `team_a_bowling_econ_last_5` | Float | Team A runs conceded per legal over in its last 5 completed bowling innings. | Measures defensive run containment rate (lower is better). | Smoothed towards expanding league average economy ($m=40$ overs). |
| `team_b_bowling_econ_last_5` | Float | Team B runs conceded per legal over in its last 5 completed bowling innings. | Measures defensive run containment rate (lower is better). | Smoothed towards expanding league average economy ($m=40$ overs). |
| `bowling_econ_diff_last_5` | Float | `team_b_bowling_econ_last_5 - team_a_bowling_econ_last_5` | **Positive indicates Team A advantage** (Team B concedes more runs per over than Team A). | $0.00$ prior. |

---

## 9. Banned Feature Policy (Zero Look-Ahead Guarantee)

The following attributes are explicitly barred from the feature set:
- **Direct Target Leakage:** `winner`, `eliminator`, `player_of_match`, `margin`.
- **In-Match State Leakage:** Total match runs, wickets fallen, balls bowled in match $t$, innings scores.
- **Mid-Game Weather Leakage:** `method` (Duckworth-Lewis rain revisions occurring during play).
- **Future Information Leakage:** Tournament end-of-season standings, net run rates calculated post-season, or subsequent fixtures.
