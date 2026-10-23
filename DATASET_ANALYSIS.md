# IPL Dataset Analysis Report (Cricsheet Raw Data)

**Date of Analysis:** September 5, 2026  
**Dataset Source:** Cricsheet Men's IPL CSV Dataset (`data/raw/cricsheet/`)  
**Project:** Cloud-Based IPL Match Winning Probability Prediction & Live Match Analytics System  
**Status:** Preprocessing On Hold (Awaiting User Approval)

---

## Executive Summary

A comprehensive inspection of the raw IPL dataset provided in `data/raw/cricsheet/` was conducted. The dataset represents one of the most complete and granular public archives of Indian Premier League cricket, spanning from the inaugural 2008 match through season 2026. 

The dataset contains **1,243 matches** with **295,732 legal and extra deliveries**, with **0 duplicate delivery rows** and **0 duplicate match records**. The data is **highly sufficient** for both **Pre-Match / Post-Toss Prediction** and **Live / In-Match Ball-by-Ball Win Probability Prediction**, subject to standardizing team renames and consolidating venue aliases.

---

## 1. Number of Files

The directory `data/raw/cricsheet/` contains **2,488 files** in total:

| File Type | Pattern / Name | Count | Description |
| :--- | :--- | :--- | :--- |
| **Match Info CSVs** | `*_info.csv` | **1,243** | Match-level metadata, toss, venue, results, and full playing XI player lists |
| **Ball-by-Ball Individual CSVs** | `[0-9]*.csv` | **1,243** | Granular delivery-by-delivery logs for each match |
| **Aggregated Master CSV** | `all_matches.csv` | **1** | Combined master CSV aggregating all 1,243 matches into a single file (45.5 MB) |
| **Documentation** | `README.txt` | **1** | Cricsheet CSV format specifications and match list |
| **Total** | | **2,488** | |

### File Integrity & Parity Check
- Number of unique match IDs in info files: **1,243**
- Number of unique match IDs in individual CSVs: **1,243**
- Number of unique match IDs in `all_matches.csv`: **1,243**
- Total rows across 1,243 individual CSVs: **295,732**
- Total rows in `all_matches.csv`: **295,732**
- **Parity Status:** **100% Exact Match** (No missing files or delivery discrepancies).

---

## 2. Number of Matches

- **Total Recorded Matches:** **1,243 matches**
- **Matches with a Direct Decisive Winner:** **1,218 matches** (98.0%)
- **Matches Tied and Decided via Super Over / Eliminator:** **16 matches** (1.3%)
- **Matches Abandoned / No Result (Rain/Weather):** **9 matches** (0.7%)
- **Matches Usable for Supervised Outcome Modeling:** **1,234 matches** (99.3%)

---

## 3. Seasons Available

The dataset spans **19 distinct IPL seasons** from 2008 to 2026:

| Season Label in Data | Calendar Year(s) | Match Count | Date Range (YYYY/MM/DD) | Key Tournament Notes |
| :--- | :--- | :--- | :--- | :--- |
| **2007/08** | 2008 | 58 | 2008/04/18 – 2008/06/01 | Inaugural IPL season |
| **2009** | 2009 | 57 | 2009/04/18 – 2009/05/24 | Hosted in South Africa |
| **2009/10** | 2010 | 60 | 2010/03/12 – 2010/04/25 | Season 3 |
| **2011** | 2011 | 73 | 2011/04/08 – 2011/05/28 | 10 teams (KTK & PWI added) |
| **2012** | 2012 | 74 | 2012/04/04 – 2012/05/27 | 9 teams |
| **2013** | 2013 | 76 | 2013/04/03 – 2013/05/26 | 9 teams (SRH replaced DC) |
| **2014** | 2014 | 60 | 2014/04/16 – 2014/06/01 | Initial leg in UAE |
| **2015** | 2015 | 59 | 2015/04/08 – 2015/05/24 | 8 teams |
| **2016** | 2016 | 60 | 2016/04/09 – 2016/05/29 | GL & RPS added (CSK & RR banned) |
| **2017** | 2017 | 59 | 2017/04/05 – 2017/05/21 | GL & RPS |
| **2018** | 2018 | 60 | 2018/04/07 – 2018/05/27 | CSK & RR returned |
| **2019** | 2019 | 60 | 2019/03/23 – 2019/05/12 | Delhi Daredevils renamed DC |
| **2020/21** | 2020 | 60 | 2020/09/19 – 2020/11/10 | Hosted in UAE (COVID-19) |
| **2021** | 2021 | 60 | 2021/04/09 – 2021/10/15 | Split season (India + UAE) |
| **2022** | 2022 | 74 | 2022/03/26 – 2022/05/29 | 10 teams (GT & LSG added) |
| **2023** | 2023 | 74 | 2023/03/31 – 2023/05/29 | Impact Player rule introduced |
| **2024** | 2024 | 71 | 2024/03/22 – 2024/05/26 | RCB renamed RCBengaluru |
| **2025** | 2025 | 74 | 2025/03/22 – 2025/06/03 | 10 teams |
| **2026** | 2026 | 74 | 2026/03/28 – 2026/05/31 | 10 teams |
| **Total** | | **1,243** | **2008/04/18 – 2026/05/31** | **19 Seasons** |

---

## 4. Structure of Match-Info CSV Files (`*_info.csv`)

Each match has a corresponding `{match_id}_info.csv`. Unlike typical tabular CSVs, this file is structured in **key-value / entity-attribute format**:
```csv
version,2.3.0
info,balls_per_over,6
info,team,Sunrisers Hyderabad
info,team,Royal Challengers Bangalore
info,team_type,club
info,gender,male
info,season,2017
info,date,2017/04/05
info,venue,"Rajiv Gandhi International Stadium, Uppal"
info,city,Hyderabad
info,toss_winner,Royal Challengers Bangalore
info,toss_decision,field
info,winner,Sunrisers Hyderabad
info,winner_runs,35
info,target_overs,2,20
info,target_runs,2,208
info,player,Royal Challengers Bangalore,CH Gayle
...
info,registry,people,CH Gayle,db584dad
```

### Key Fields and Multi-Row Elements
- `info,team,<TeamName>`: Appears **twice per file** (Batting/Bowling teams).
- `info,player,<TeamName>,<PlayerName>`: Lists all participating players.
  - **Seasons 2008–2022:** Exactly 22 players per match (11 per team).
  - **Seasons 2023–2026:** 23 to 25 players per match due to the **Impact Player rule** and substitutions.
- `info,toss_winner,<TeamName>` & `info,toss_decision,<bat|field>`: 100% populated in all 1,243 matches.
- `info,winner,<TeamName>`: Populated for regular wins (1,218 matches).
- `info,outcome,tie` + `info,eliminator,<TeamName>`: For all 16 tied matches, the Super Over winner is recorded under `eliminator`.
- `info,outcome,no result`: Recorded for 9 abandoned matches.
- `info,method,D/L`: Recorded for 23 Duckworth-Lewis/Stern rain-affected matches.
- `info,target_runs,2,<TargetScore>`: Recorded in 1,237 matches (omitted only in 6 matches where 2nd innings never began).
- `info,registry,people,<PlayerName>,<UUID>`: Unique player registry identifier (useful for tracking player career histories).

---

## 5. Structure of Ball-by-Ball CSV Files (`all_matches.csv` & `[id].csv`)

The delivery-level data is stored in standard tabular format with **27 columns**:

```csv
match_id,season,start_date,venue,innings,ball,actual_delivery,batting_team,bowling_team,striker,non_striker,bowler,runs_off_bat,extras,wides,noballs,byes,legbyes,penalty,non_boundary,wicket_type,player_dismissed,other_wicket_type,other_player_dismissed,fielder_1,fielder_2,fielder_3
```

### Innings Breakdown across 295,732 deliveries
- **Innings 1:** 153,254 deliveries
- **Innings 2:** 142,303 deliveries
- **Innings 3:** 86 deliveries *(Super Over - Team 1)*
- **Innings 4:** 77 deliveries *(Super Over - Team 2)*
- **Innings 5:** 8 deliveries *(Double Super Over - Team 1, e.g. MI vs KXIP 2020)*
- **Innings 6:** 4 deliveries *(Double Super Over - Team 2)*

### Difference between `ball` and `actual_delivery`
- `ball`: Represents the sequential ball attempt within the over (e.g., `0.1`, `0.2`, `0.3`, ..., `0.7`, `0.8`). If extras occur, the over does not advance its legal count, but `ball` continues incrementing or stays as delivery attempts.
- `actual_delivery`: Represents the completed legal delivery count (e.g. `0.1`, `0.2`, `0.3`). In 35,647 deliveries, `ball != actual_delivery` due to wides, no-balls, and re-bowled deliveries.
- **For live ML prediction:** The legal ball count (`actual_delivery`) is crucial for computing remaining legal deliveries (out of 120) and Required Run Rate (RRR).

---

## 6. All Important Columns & Schema Definitions

| Column Name | Data Type | Missing Count (%) | Purpose & ML Significance |
| :--- | :--- | :--- | :--- |
| `match_id` | Integer | 0 (0.00%) | Unique match foreign key linking to `*_info.csv` |
| `season` | String | 0 (0.00%) | Season year identifier (temporal feature for train/test splits) |
| `start_date` | Date (YYYY-MM-DD) | 0 (0.00%) | Match date (avoids temporal data leakage) |
| `venue` | String | 0 (0.00%) | Ground location (ground dimensions, pitch behavior) |
| `innings` | Integer (1–6) | 0 (0.00%) | 1 = 1st innings, 2 = 2nd innings (chase), 3–6 = Super Overs |
| `ball` | Float | 0 (0.00%) | Delivery attempt in over notation |
| `actual_delivery` | Float | 0 (0.00%) | Legal ball notation (critical for balls remaining calculation) |
| `batting_team` | String | 0 (0.00%) | Team currently batting |
| `bowling_team` | String | 0 (0.00%) | Team currently bowling |
| `striker` | String | 0 (0.00%) | Batter facing delivery |
| `non_striker` | String | 0 (0.00%) | Non-striking batter |
| `bowler` | String | 0 (0.00%) | Current bowler |
| `runs_off_bat` | Integer (0–6) | 0 (0.00%) | Runs scored off the bat |
| `extras` | Integer (0–7) | 0 (0.00%) | Extra runs conceded on this delivery |
| `wides` | Float / Null | 285,856 (96.66%) | Wide runs awarded (legal ball count does not advance) |
| `noballs` | Float / Null | 294,506 (99.59%) | No-ball runs awarded |
| `byes` | Float / Null | 294,987 (99.75%) | Byes conceded |
| `legbyes` | Float / Null | 291,326 (98.51%) | Leg-byes conceded |
| `penalty` | Float / Null | 295,730 (100.0%) | Penalty runs (5 runs; occurred twice in dataset) |
| `wicket_type` | Categorical | 281,027 (95.03%) | Dismissal method (caught, bowled, lbw, run out, etc.) |
| `player_dismissed`| String | 281,027 (95.03%) | Name of dismissed player |
| `toss_winner`* | String | 0 (0.00%) | *(From info)* Team winning the coin toss |
| `toss_decision`* | String | 0 (0.00%) | *(From info)* `bat` or `field` |
| `winner`* | String | 25 (2.01%) | *(From info)* Match winning team |
| `target_runs`* | Integer | 6 (0.48%) | *(From info)* Target runs in 2nd innings |

*\*Note: Columns marked with an asterisk are extracted from `*_info.csv`.*

---

## 7. Missing Values Breakdown

### In Match-Info Files (1,243 matches):
1. **`winner` (25 matches missing / 2.01%):**
   - **16 matches are Ties:** Every single one of these 16 matches has a definitive winner recorded in the `info,eliminator` field (Super Over winner).
   - **9 matches are Abandoned / No Result:** True missing outcome where no winner exists due to weather (e.g., Match 1178424, 1359519, 501265, etc.).
2. **`city` (51 matches missing / 4.10%):**
   - 33 matches at `Dubai International Cricket Stadium`
   - 18 matches at `Sharjah Cricket Stadium`
   - *(Note: Both are easily imputed as "Dubai" and "Sharjah" respectively).*
3. **`target_runs` & `target_overs` (6 matches missing / 0.48%):**
   - Matches 1359519, 1473492, 1473495, 1527685, 501265, 829763: All 6 are matches where rain abandoned the game before Innings 2 could begin.
4. **`tv_umpire` (4 missing / 0.32%)**, **`reserve_umpire` (24 missing / 1.93%)**, **`player_of_match` (9 missing / 0.72%)**.

### In Ball-by-Ball Data (295,732 rows):
- Core identification fields (`match_id`, `season`, `start_date`, `venue`, `innings`, `ball`, `actual_delivery`, `batting_team`, `bowling_team`, `striker`, `bowler`, `runs_off_bat`, `extras`): **0 missing values (100% complete)**.
- **Sparse Event Fields:** Fields like `wides`, `noballs`, `byes`, `legbyes`, `penalty`, `wicket_type`, and `player_dismissed` are empty in 95%–99.9% of rows because wickets and extras only occur on a small fraction of balls. They should be filled with `0` (for extras) or empty strings (for wickets/dismissals).

---

## 8. Team-Name Inconsistencies

The dataset records **19 unique team names**. Several teams have undergone official franchise rebrandings or cosmetic name modifications:

| Raw Team Name in Dataset | Matches in Data | Active Seasons | Canonical Modern Team | Status / Category |
| :--- | :--- | :--- | :--- | :--- |
| **Delhi Daredevils** | 161 | 2008–2018 | **Delhi Capitals** | Renamed Dec 2018 |
| **Delhi Capitals** | 120 | 2019–2026 | **Delhi Capitals** | Current Name |
| **Kings XI Punjab** | 190 | 2008–2020 | **Punjab Kings** | Renamed Feb 2021 |
| **Punjab Kings** | 88 | 2021–2026 | **Punjab Kings** | Current Name |
| **Royal Challengers Bangalore** | 240 | 2008–2023 | **Royal Challengers Bengaluru** | Renamed Mar 2024 |
| **Royal Challengers Bengaluru** | 46 | 2024–2026 | **Royal Challengers Bengaluru** | Current Name |
| **Rising Pune Supergiants** | 14 | 2016 | **Rising Pune Supergiant** | Plural form in 2016 |
| **Rising Pune Supergiant** | 16 | 2017 | **Rising Pune Supergiant** | Singular form in 2017 |
| **Deccan Chargers** | 75 | 2008–2012 | **Sunrisers Hyderabad** (or keep distinct) | Franchise terminated; replaced by SRH in 2013 |
| **Sunrisers Hyderabad** | 211 | 2013–2026 | **Sunrisers Hyderabad** | Current Name |
| **Pune Warriors** | 46 | 2011–2013 | *Pune Warriors* | Defunct Franchise |
| **Gujarat Lions** | 30 | 2016–2017 | *Gujarat Lions* | Defunct Temporary Franchise |
| **Kochi Tuskers Kerala** | 14 | 2011 | *Kochi Tuskers Kerala* | Defunct Franchise |
| **Mumbai Indians** | 291 | 2008–2026 | **Mumbai Indians** | Consistent |
| **Kolkata Knight Riders** | 278 | 2008–2026 | **Kolkata Knight Riders** | Consistent |
| **Chennai Super Kings** | 266 | 2008–2026 | **Chennai Super Kings** | Consistent |
| **Rajasthan Royals** | 251 | 2008–2026 | **Rajasthan Royals** | Consistent |
| **Gujarat Titans** | 77 | 2022–2026 | **Gujarat Titans** | Consistent (Current) |
| **Lucknow Super Giants** | 72 | 2022–2026 | **Lucknow Super Giants** | Consistent (Current) |

> [!NOTE]
> If left unmapped, models will treat `Delhi Daredevils` and `Delhi Capitals`, or `Royal Challengers Bangalore` and `Royal Challengers Bengaluru`, as completely separate franchises, diluting team win-rates, head-to-head records, and historical priors.

---

## 9. Venue-Name Inconsistencies

The dataset records **60 unique venue strings**, representing only approximately **35 unique physical cricket grounds**. 

### Major Inconsistency Clusters:
1. **Bengaluru (M. Chinnaswamy Stadium):**
   - `M Chinnaswamy Stadium` (65 matches, City: Bangalore)
   - `M Chinnaswamy Stadium, Bengaluru` (24 matches, City: Bengaluru)
   - `M.Chinnaswamy Stadium` (15 matches, City: Bengaluru)
   - *Total: 104 matches at the same venue.*
2. **Kolkata (Eden Gardens):**
   - `Eden Gardens` (77 matches)
   - `Eden Gardens, Kolkata` (30 matches)
   - *Total: 107 matches.*
3. **Mumbai (Wankhede Stadium):**
   - `Wankhede Stadium` (73 matches)
   - `Wankhede Stadium, Mumbai` (59 matches)
   - *Total: 132 matches.*
4. **Delhi (Feroz Shah Kotla / Arun Jaitley Stadium):**
   - `Feroz Shah Kotla` (60 matches, 2008–2019)
   - `Arun Jaitley Stadium` (14 matches, 2018–2019)
   - `Arun Jaitley Stadium, Delhi` (30 matches, 2021–2026)
   - *Total: 104 matches at the same ground (officially renamed in September 2019).*
5. **Hyderabad (Rajiv Gandhi International Stadium):**
   - `Rajiv Gandhi International Stadium, Uppal` (49 matches)
   - `Rajiv Gandhi International Stadium, Uppal, Hyderabad` (26 matches)
   - `Rajiv Gandhi International Stadium` (15 matches)
   - *Total: 90 matches.*
6. **Chennai (MA Chidambaram Stadium / Chepauk):**
   - `MA Chidambaram Stadium, Chepauk` (48 matches)
   - `MA Chidambaram Stadium, Chepauk, Chennai` (41 matches)
   - `MA Chidambaram Stadium` (9 matches)
   - *Total: 98 matches.*
7. **Ahmedabad (Motera / Narendra Modi Stadium):**
   - `Sardar Patel Stadium, Motera` (12 matches, 2010–2015)
   - `Narendra Modi Stadium, Ahmedabad` (41 matches, 2021–2026)
   - *Total: 53 matches at the same venue (demolished, rebuilt, and renamed).*
8. **Mohali / Chandigarh (PCA Stadium):**
   - `Punjab Cricket Association Stadium, Mohali` (35 matches)
   - `Punjab Cricket Association IS Bindra Stadium, Mohali` (11 matches)
   - `Punjab Cricket Association IS Bindra Stadium` (10 matches)
   - `Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh` (5 matches)
   - *Total: 61 matches.*
9. **Mullanpur (New Chandigarh):**
   - `Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur` (10 matches)
   - `Maharaja Yadavindra Singh International Cricket Stadium, New Chandigarh` (7 matches)
   - *Total: 17 matches.*
10. **Pune (MCA Stadium / Subrata Roy Sahara Stadium):**
    - `Maharashtra Cricket Association Stadium` (22 matches)
    - `Maharashtra Cricket Association Stadium, Pune` (13 matches)
    - `Subrata Roy Sahara Stadium` (16 matches)
    - *Total: 51 matches at the exact same stadium.*
11. **Navi Mumbai & Mumbai Venues:**
    - `Dr DY Patil Sports Academy` (17 matches) vs `Dr DY Patil Sports Academy, Mumbai` (20 matches)
    - `Brabourne Stadium` (10 matches) vs `Brabourne Stadium, Mumbai` (17 matches)
12. **UAE Venues (Missing City):**
    - `Dubai International Cricket Stadium` (46 matches, 33 missing city)
    - `Sharjah Cricket Stadium` (28 matches, 18 missing city)
    - `Sheikh Zayed Stadium` (29 matches) vs `Zayed Cricket Stadium, Abu Dhabi` (8 matches)

> [!WARNING]
> Venue-based feature engineering (such as average 1st innings score, pitch pace, boundaries per over, chasing win percentage) will be fractured across 60 categories if venues are not consolidated into canonical stadium names.

---

## 10. Duplicate Records Verification

Thorough uniqueness checks were executed across all files:

1. **Delivery-Level Duplicates:**
   - Evaluated duplicate keys `(match_id, innings, ball, actual_delivery, striker, bowler)`: **0 duplicates**.
   - Evaluated exact row duplicates across all 27 columns: **0 duplicates**.
2. **Match-Level Duplicates:**
   - 1,243 unique `match_id` files in `*_info.csv`.
   - 1,243 unique `match_id` files in `*.csv`.
   - No redundant match IDs or clashing fixtures found.
3. **Parity Check:**
   - Every single line in `all_matches.csv` maps 1:1 to the individual match CSV files.

---

## 11. Dataset Sufficiency for Pre-Match and Live Predictions

### A. Pre-Match / Post-Toss Winning Probability Prediction: **SUFFICIENT & ROBUST**
Can we build a high-performance pre-match model before the first ball is bowled? **Yes.**

| Required Feature Category | Available Data in Dataset | Evaluation & Readiness |
| :--- | :--- | :--- |
| **Team Identities** | `batting_team`, `bowling_team`, `team` | 100% available across all 1,243 matches |
| **Toss Outcome & Decision** | `toss_winner`, `toss_decision` (`bat`/`field`) | 100% available across all 1,243 matches |
| **Ground & Conditions** | `venue`, `city` | 100% available (after standardizing venue aliases) |
| **Playing XI / Rosters** | `info,player,<team>,<player>` | Full 11 (or 12 with impact player) rosters listed |
| **Historical Form & Head-to-Head** | Chronological dates (`start_date`, `season`) | Fully computable via rolling historical metrics |
| **Target Variable (Winner)** | `winner` + `eliminator` | 1,234 valid labeled matches (drop 9 no-results) |

### B. Live / In-Match Winning Probability Prediction: **HIGHLY SUFFICIENT & GRANULAR**
Can we predict winning probability ball-by-ball or over-by-over as the game evolves? **Yes.**

| Live Match State Feature | Derivation from Dataset | Completeness & Quality |
| :--- | :--- | :--- |
| **Current Score & Overs** | Cumulative sum of `runs_off_bat + extras`, count of legal balls from `actual_delivery` | Computable at every single delivery (295,732 states) |
| **Wickets Lost** | Cumulative count where `wicket_type != ""` | Accurate across all 14,705 dismissals |
| **Target Runs & Balls Left** | `target_runs` in 2nd innings (or 1st innings total + 1), `120 - legal_balls_bowled` | Target available for 1,237 matches; D/L revised targets explicitly given |
| **Current Run Rate (CRR)** | `(cumulative_runs / legal_balls_bowled) * 6` | Real-time continuous feature |
| **Required Run Rate (RRR)** | `(runs_needed / balls_remaining) * 6` | Real-time chase feature in Innings 2 |
| **Phase of Play** | Categorized by over: Powerplay (0–5), Middle (6–14), Death (15–19) | Direct mapping from `actual_delivery` |
| **Momentum / Recent Form** | Rolling runs and wickets in last 6, 12, 18 balls | Fully computable from sequential ball order |
| **Batter & Bowler Live Stats** | Real-time batter runs, balls faced, bowler overs, economy, wickets in current match | Granularly trackable at ball level |

### Edge Cases & Modeling Caveats Identified:
1. **Duckworth-Lewis-Stern (D/L) Matches (23 matches):**
   - In rain-interrupted matches, the target and overs were revised mid-game. The dataset includes `info,target_runs` and `info,target_overs` for these matches. We can either train a dedicated handling for revised targets or exclude shortened matches (<15 overs) to prevent distorted run-rate distributions.
2. **Super Overs (Innings 3, 4, 5, 6):**
   - Super overs represent a separate mini-game (1 over per side). For normal in-match prediction, innings 1 and innings 2 should be modeled. The Super Over winner should be used as the match outcome label if the 2nd innings ended in a tie.
3. **Impact Player Rule (2023–2026):**
   - Higher team totals observed from 2023 onwards (teams batting deeper). Season-based or era-based normalization may be beneficial.

---

## Conclusion & Proposed Next Steps (Awaiting User Approval)

The dataset in `data/raw/cricsheet/` is clean, complete, and fully capable of supporting both pre-match and real-time live match prediction systems. 

**Recommended Preprocessing Pipeline (To be executed upon user approval):**
1. **Canonical Team Mapping:** Map all historical/rebranded team names (e.g. Delhi Daredevils $\rightarrow$ Delhi Capitals, Kings XI Punjab $\rightarrow$ Punjab Kings, RCBangalore $\rightarrow$ RCBengaluru).
2. **Canonical Venue & City Mapping:** Consolidate 60 venue strings into standard stadium names and fill the 51 missing city entries (Dubai and Sharjah).
3. **Winner Target Synthesis:** Synthesize binary outcome `target` using `winner` for regular matches and `eliminator` for tied matches, while filtering out the 9 abandoned "no result" matches.
4. **Feature Engineering Module:**
   - Generate match-level dataset for Pre-Match Model (`data/processed/match_level_prematch.csv`).
   - Generate delivery-level state dataset for In-Match Model (`data/processed/delivery_level_live.csv`) computing cumulative runs, wickets, balls remaining, CRR, RRR, and phase indicators.

*No changes have been made to the dataset or codebase. Awaiting user review and approval.*
