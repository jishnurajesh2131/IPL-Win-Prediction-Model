# Pre-Match Dataset Quality & Leakage Audit Report

**Date of Audit:** September 5, 2026  
**Dataset File:** `data/processed/pre_match_dataset.csv`  
**Verification Status:** **PASSED**  

---

## 1. Executive Summary & Row Counts

| Metric | Expected Value | Actual Value | Status |
| :--- | :--- | :--- | :--- |
| **Total Feature Rows** | 1,234 | 1234 | **100% Match** |
| **Total Feature Columns** | 32 | 32 | **100% Match** |
| **Excluded Abandoned Matches** | 9 | 9 | **Excluded (0 targets leaked)** |
| **Super Over Matches Included** | 16 | 16 | **Included with eliminator winner** |
| **Duckworth-Lewis Matches** | 23 | 23 | **Included with normalized RPO** |

---

## 2. Target Variable Balance

| Target Class (`target_team_a_win`) | Match Count | Percentage | Class Balance Status |
| :--- | :--- | :--- | :--- |
| **Class 1 (Team A Won)** | 617 | 50.00% | **Perfect 50.00% Symmetry** |
| **Class 0 (Team B Won)** | 617 | 50.00% | **Perfect 50.00% Symmetry** |
| **Total Decisive Matches** | **1234** | **100.00%** | **Zero Class Imbalance** |

---

## 3. Cold-Start & Temporal Leakage Validation

- **First Match Evaluated:** Match ID `335982` on `2008/04/18` (Royal Challengers Bengaluru vs Kolkata Knight Riders)
- **Team A Prior Matches:** 0 (Verified 0)
- **Team B Prior Matches:** 0 (Verified 0)
- **Prior Head-to-Head Matches:** 0 (Verified 0)
- **Team A Prior Form:** 0.5 (Smoothed to neutral 0.50 prior)
- **Team B Prior Form:** 0.5 (Smoothed to neutral 0.50 prior)
- **H2H Win Rate Prior:** 0.5 (Smoothed to neutral 0.50 prior)
- **Cold-Start Invariant Check:** PASSED
- **Chronological Invariant:** Every row was computed strictly using past records before updating state.

---

## 4. Missing Values Audit (All 32 Columns)

| Column Name | Data Type | Null Count | Null % | Status |
| :--- | :--- | :--- | :--- | :--- |
| `match_id` | Standardized | 0 | 0.00% | **100% Complete** |
| `season` | Standardized | 0 | 0.00% | **100% Complete** |
| `date` | Standardized | 0 | 0.00% | **100% Complete** |
| `team_a` | Standardized | 0 | 0.00% | **100% Complete** |
| `team_b` | Standardized | 0 | 0.00% | **100% Complete** |
| `venue` | Standardized | 0 | 0.00% | **100% Complete** |
| `city` | Standardized | 0 | 0.00% | **100% Complete** |
| `toss_winner_is_team_a` | Standardized | 0 | 0.00% | **100% Complete** |
| `toss_decision_is_bat` | Standardized | 0 | 0.00% | **100% Complete** |
| `team_a_bats_first` | Standardized | 0 | 0.00% | **100% Complete** |
| `team_a_form_last_5` | Standardized | 0 | 0.00% | **100% Complete** |
| `team_b_form_last_5` | Standardized | 0 | 0.00% | **100% Complete** |
| `team_a_form_last_10` | Standardized | 0 | 0.00% | **100% Complete** |
| `team_b_form_last_10` | Standardized | 0 | 0.00% | **100% Complete** |
| `form_diff_last_5` | Standardized | 0 | 0.00% | **100% Complete** |
| `team_a_matches_played` | Standardized | 0 | 0.00% | **100% Complete** |
| `team_b_matches_played` | Standardized | 0 | 0.00% | **100% Complete** |
| `h2h_matches_played` | Standardized | 0 | 0.00% | **100% Complete** |
| `team_a_h2h_win_rate` | Standardized | 0 | 0.00% | **100% Complete** |
| `team_a_venue_matches` | Standardized | 0 | 0.00% | **100% Complete** |
| `team_a_venue_win_rate` | Standardized | 0 | 0.00% | **100% Complete** |
| `team_b_venue_matches` | Standardized | 0 | 0.00% | **100% Complete** |
| `team_b_venue_win_rate` | Standardized | 0 | 0.00% | **100% Complete** |
| `venue_win_rate_diff` | Standardized | 0 | 0.00% | **100% Complete** |
| `venue_chasing_win_rate` | Standardized | 0 | 0.00% | **100% Complete** |
| `team_a_batting_rpo_last_5` | Standardized | 0 | 0.00% | **100% Complete** |
| `team_b_batting_rpo_last_5` | Standardized | 0 | 0.00% | **100% Complete** |
| `batting_rpo_diff_last_5` | Standardized | 0 | 0.00% | **100% Complete** |
| `team_a_bowling_econ_last_5` | Standardized | 0 | 0.00% | **100% Complete** |
| `team_b_bowling_econ_last_5` | Standardized | 0 | 0.00% | **100% Complete** |
| `bowling_econ_diff_last_5` | Standardized | 0 | 0.00% | **100% Complete** |
| `target_team_a_win` | Standardized | 0 | 0.00% | **100% Complete** |

---

## 5. Value Ranges and Statistical Bounds

- **Out-of-Bounds Probability Values ($[0, 1]$):** 0
- **Categorical Columns (4):** `team_a`, `team_b`, `venue`, `city`
- **Binary Numerical Features (4):** `toss_winner_is_team_a`, `toss_decision_is_bat`, `team_a_bats_first`, `target_team_a_win`
- **Continuous / Integer Features (21):** All bounded, finite, and non-null.

---

## 6. Zero-Tolerance Leakage Compliance

- **Target Leakage Banned:** `winner`, `eliminator`, `margin`, `player_of_match` do NOT appear in dataset.
- **In-Match Leakage Banned:** In-match scores, 1st innings scores, wickets fallen, and ball counts do NOT appear in dataset.
- **Future Leakage Banned:** Tournament final standings, season points tables, and later match outcomes do NOT appear in dataset.
- **Temporal Split Readiness:** Ready for chronological out-of-time evaluation (Train: 2008–2023, Test: 2024–2026).