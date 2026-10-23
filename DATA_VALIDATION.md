# Data Validation & Integrity Report

**Date of Validation:** September 5, 2026  
**Clean Matches Dataset:** `data/processed/matches_clean.csv`  
**Clean Deliveries Dataset:** `data/processed/deliveries_clean.csv`  
**Validation Status:** **PASSED**  

---

## 1. Dataset Volume & Parity Verification

| Metric | Raw Cricsheet Dataset | Cleaned Processed Dataset | Discrepancy / Parity | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Total Matches** | 1243 | 1243 | 0 | **100% Match** |
| **Total Deliveries** | 295,732 | 295,732 | 0 | **100% Match** |
| **Unique Match IDs (Matches)** | 1243 | 1243 | 0 | **100% Match** |
| **Unique Match IDs (Deliveries)** | 1243 | 1243 | 0 | **100% Match** |
| **Super Over Deliveries** | 175 | 175 | 0 | **Preserved** |

---

## 2. Duplicate Records Analysis

- **Match-Level Primary Key (`match_id`) Duplicates:** 0
- **Match-Level Exact Row Duplicates:** 0
- **Delivery-Level Composite Key Duplicates:** 0
- **Delivery-Level Exact Row Duplicates:** 0

---

## 3. Match Outcome Classification

| Match Result Type | Count | Percentage | Winner Handling |
| :--- | :--- | :--- | :--- |
| **`normal`** | 1218 | 97.99% | Winner from regular match result |
| **`super_over`** | 16 | 1.29% | Winner resolved via Super Over eliminator |
| **`no_result`** | 9 | 0.72% | Abandoned matches; winner empty/null |
| **Total** | **1243** | **100.00%** | |

---

## 4. Team Name Standardization (19 Raw $\rightarrow$ 15 Canonical)

| Canonical Franchise Name | Raw Variations Mapped | Delivery Count | Matches Involved | Franchise Type |
| :--- | :--- | :--- | :--- | :--- |
| **Chennai Super Kings** | Chennai Super Kings | 32,101 | 266 | Active |
| **Deccan Chargers** | Deccan Chargers | 9,034 | 75 | Defunct |
| **Delhi Capitals** | Delhi Daredevils, Delhi Capitals | 33,107 | 281 | Active |
| **Gujarat Lions** | Gujarat Lions | 3,566 | 30 | Defunct |
| **Gujarat Titans** | Gujarat Titans | 9,379 | 77 | Active |
| **Kochi Tuskers Kerala** | Kochi Tuskers Kerala | 1,582 | 14 | Defunct |
| **Kolkata Knight Riders** | Kolkata Knight Riders | 32,463 | 278 | Active |
| **Lucknow Super Giants** | Lucknow Super Giants | 8,854 | 72 | Active |
| **Mumbai Indians** | Mumbai Indians | 34,990 | 291 | Active |
| **Pune Warriors** | Pune Warriors | 5,443 | 46 | Defunct |
| **Punjab Kings** | Kings XI Punjab, Punjab Kings | 33,100 | 278 | Active |
| **Rajasthan Royals** | Rajasthan Royals | 29,777 | 251 | Active |
| **Rising Pune Supergiant** | Rising Pune Supergiant, Rising Pune Supergiants | 3,480 | 30 | Defunct |
| **Royal Challengers Bengaluru** | Royal Challengers Bangalore, Royal Challengers Bengaluru | 33,572 | 286 | Active |
| **Sunrisers Hyderabad** | Sunrisers Hyderabad | 25,284 | 211 | Active |

---

## 5. Venue Standardization (60 Raw $\rightarrow$ 36 Canonical Stadiums)

- **Raw Unique Venues:** 60
- **Cleaned Canonical Stadiums:** 36
- **All 60 venue strings successfully mapped** without unmapped residuals.

| Canonical Stadium Name | Raw Delivery Count | Matches Hosted | Canonical City |
| :--- | :--- | :--- | :--- |
| **Arun Jaitley Stadium** | 24,566 | 104 | Standardized |
| **Barabati Stadium** | 1,695 | 7 | Standardized |
| **Barsapara Cricket Stadium** | 1,799 | 8 | Standardized |
| **Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium** | 6,904 | 29 | Standardized |
| **Brabourne Stadium** | 6,526 | 27 | Standardized |
| **Buffalo Park** | 715 | 3 | Standardized |
| **De Beers Diamond Oval** | 726 | 3 | Standardized |
| **Dr DY Patil Sports Academy** | 8,898 | 37 | Standardized |
| **Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium** | 4,000 | 17 | Standardized |
| **Dubai International Cricket Stadium** | 11,229 | 46 | Standardized |
| **Eden Gardens** | 24,936 | 107 | Standardized |
| **Green Park** | 921 | 4 | Standardized |
| **Himachal Pradesh Cricket Association Stadium** | 4,461 | 19 | Standardized |
| **Holkar Cricket Stadium** | 1,965 | 9 | Standardized |
| **JSCA International Stadium Complex** | 1,671 | 7 | Standardized |
| **Kingsmead** | 3,643 | 15 | Standardized |
| **M Chinnaswamy Stadium** | 24,128 | 104 | Standardized |
| **MA Chidambaram Stadium** | 23,851 | 98 | Standardized |
| **Maharaja Yadavindra Singh International Cricket Stadium** | 4,049 | 17 | Standardized |
| **Maharashtra Cricket Association Stadium** | 12,239 | 51 | Standardized |
| **Narendra Modi Stadium** | 12,671 | 53 | Standardized |
| **Nehru Stadium** | 1,155 | 5 | Standardized |
| **New Wanderers Stadium** | 1,940 | 8 | Standardized |
| **Newlands** | 1,539 | 7 | Standardized |
| **OUTsurance Oval** | 500 | 2 | Standardized |
| **Punjab Cricket Association IS Bindra Stadium** | 14,476 | 61 | Standardized |
| **Rajiv Gandhi International Stadium** | 21,416 | 90 | Standardized |
| **Saurashtra Cricket Association Stadium** | 2,381 | 10 | Standardized |
| **Sawai Mansingh Stadium** | 16,174 | 68 | Standardized |
| **Shaheed Veer Narayan Singh International Stadium** | 1,931 | 8 | Standardized |
| **Sharjah Cricket Stadium** | 6,672 | 28 | Standardized |
| **Sheikh Zayed Stadium** | 8,799 | 37 | Standardized |
| **St George's Park** | 1,677 | 7 | Standardized |
| **SuperSport Park** | 2,866 | 12 | Standardized |
| **Vidarbha Cricket Association Stadium, Jamtha** | 742 | 3 | Standardized |
| **Wankhede Stadium** | 31,871 | 132 | Standardized |

---

## 6. Missing Values Breakdown

### In `matches_clean.csv` (1,243 rows):

| Column Name | Missing Count | Missing % | Explanation |
| :--- | :--- | :--- | :--- |
| `match_id` | 0 | 0.00% | 100% complete |
| `season` | 0 | 0.00% | 100% complete |
| `date` | 0 | 0.00% | 100% complete |
| `venue` | 0 | 0.00% | 100% complete |
| `city` | 0 | 0.00% | 100% complete (imputed for Dubai & Sharjah). |
| `team_1` | 0 | 0.00% | 100% complete |
| `team_2` | 0 | 0.00% | 100% complete |
| `toss_winner` | 0 | 0.00% | 100% complete |
| `toss_decision` | 0 | 0.00% | 100% complete |
| `winner` | 9 | 0.72% | Intentionally null only for the 9 abandoned no-result matches. |
| `match_result_type` | 0 | 0.00% | 100% complete |
| `method` | 1220 | 98.15% | Recorded as 'D/L' for 23 rain-affected matches; empty for normal matches. |

### In `deliveries_clean.csv` (295,732 rows):

| Column Category | Field Names | Missing Count | Status & Logic |
| :--- | :--- | :--- | :--- |
| **Match & Ball ID** | `match_id`, `season`, `start_date`, `venue`, `innings`, `ball`, `actual_delivery` | 0 | 100% complete |
| **Teams & Players** | `batting_team`, `bowling_team`, `striker`, `non_striker`, `bowler` | 0 | 100% complete |
| **Runs & Extras** | `runs_off_bat`, `extras`, `wides`, `noballs`, `byes`, `legbyes`, `penalty`, `non_boundary` | 0 | 100% complete (missing extras imputed to 0) |
| **Dismissal Info** | `wicket_type`, `player_dismissed` | 281,027 | Sparse event (only populated on 14,705 wicket balls; intentionally preserved empty otherwise) |

---

## 7. Compliance with Architectural Constraints

- **Raw Files Integrity:** Completely untouched in `data/raw/cricsheet/` (0 modified, 0 deleted).
- **Feature Engineering:** Zero features computed (no CRR, RRR, rolling windows, or player stats).
- **Machine Learning Models:** Zero models trained.
- **Super Over Innings:** Properly preserved with explicit `is_super_over` tagging.
- **D/L Matches:** Fully preserved with method flag.