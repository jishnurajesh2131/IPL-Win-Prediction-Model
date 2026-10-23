# Final Preprocessing & Data Integrity Verification Report

**Project:** Cloud-Based IPL Match Winning Probability Prediction & Live Match Analytics System  
**Stage:** Pre-Feature Engineering Verification (Phase 1 Final Sign-off)  
**Mode:** READ-ONLY Verification (Zero files modified, overwritten, or regenerated)  
**Date of Audit:** September 5, 2026  

---

## Executive Summary & Verification Scorecard

An exhaustive, read-only audit of the preprocessed IPL datasets (`matches_clean.csv` and `deliveries_clean.csv`), source mapping code (`src/data_processing/`), raw Cricsheet files (`data/raw/cricsheet/`), and project documentation was conducted. 

Every verification check has been classified as **PASS**, **WARNING**, or **FAIL**:

| # | Verification Area | Target / Constraint | Result | Status |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **Match Dataset Parity** | 1,243 rows, 0 duplicate IDs, 0 missing IDs | 1,243 rows, 0 duplicates, 0 missing IDs | **PASS** |
| **2** | **Team Name Standardization** | 19 raw $\rightarrow$ 15 canonical teams; defunct isolated | 15 canonical teams; defunct isolated | **PASS** |
| **3** | **Venue Mapping (Data & Code)** | 60 raw strings $\rightarrow$ canonical physical grounds | 60 raw strings $\rightarrow$ 36 physical grounds | **PASS** |
| **4** | **Venue Documentation Parity** | Reconcile 33 vs 36 venue count statements | Heading says 33; table & data contain 36 | **WARNING** |
| **5** | **City Imputation & Completeness** | Exactly 51 missing (Dubai: 33, Sharjah: 18) $\rightarrow$ 0 missing | 51 imputed (Dubai: 33, Sharjah: 18); 0 missing | **PASS** |
| **6** | **Delivery Dataset Parity** | 295,732 rows, 0 duplicates, valid ranges | 295,732 rows, 0 duplicates, 0 invalid rows | **PASS** |
| **7** | **Referential Integrity** | Every delivery links to valid match & teams | 100% match & team referential integrity | **PASS** |
| **8** | **Super Over Logic** | 175 balls tagged; 16 ties resolved via eliminator | 175 balls tagged; 16 eliminator winners | **PASS** |
| **9** | **Duckworth-Lewis (D/L) Matches** | 23 D/L matches preserved with method flag | 23 matches preserved; result_type = normal | **PASS** |
| **10** | **Raw Dataset Integrity** | All 2,488 raw files completely untouched | 2,488 files untouched (0 modified, 0 deleted) | **PASS** |

---

## VENUE COUNT DISCREPANCY INVESTIGATION

### 1. Documented vs. Actual Venue Counts

| Source Document / Artifact | Location | Documented / Reported Count | Notes |
| :--- | :--- | :--- | :--- |
| `DATA_PREPROCESSING.md` | Section 2B & Section 4 | **33** canonical stadiums | States: *"Maps 60 raw venue strings to 33 canonical stadiums"* |
| `VENUE_MAPPING.md` | Section 2 Header | **33** canonical stadiums | Header reads: *"60 Raw Strings $\rightarrow$ 33 Canonical Stadiums"* |
| `VENUE_MAPPING.md` | Section 2 Table | **36** canonical stadiums | The table actually contains **36 unique canonical venue values** |
| `DATA_VALIDATION.md` | Section 5 Header | **33** canonical stadiums | Header reads: *"60 Raw $\rightarrow$ 33 Canonical Stadiums"* |
| `DATA_VALIDATION.md` | Section 5 Line 67 & Table | **36** canonical stadiums | Line 67 reads: `Cleaned Canonical Stadiums: 36` and table lists 36 rows |
| `src/data_processing/venue_mapping.py` | `CANONICAL_VENUE_MAP` | **36** unique canonical values | 60 raw keys map to **36 unique canonical venue values** |
| `data/processed/matches_clean.csv` | `venue` column | **36** unique canonical venues | **Actual ground truth in match-level dataset** |
| `data/processed/deliveries_clean.csv` | `venue` column | **36** unique canonical venues | **Actual ground truth in delivery-level dataset** |

---

### 2. Detailed Root-Cause Explanation for the Discrepancy

1. **Origin of the Number "33":**
   - During the preliminary exploratory analysis documented in `DATASET_ANALYSIS.md` (Line 229), it was observed that *"The dataset records 60 unique venue strings, representing only approximately 35 unique physical cricket grounds."*
   - In the initial implementation plan drafting phase, an early mental grouping grouped the venues into 17 consolidated multi-alias clusters and estimated 16 single/consistent grounds ($17 + 16 = 33$), inadvertently omitting 3 overseas/occasional grounds from the summary text header.
   - This preliminary string template (`"60 Raw → 33 Canonical Stadiums"`) was propagated into the markdown headers of `DATA_PREPROCESSING.md`, `VENUE_MAPPING.md`, and `DATA_VALIDATION.md`.

2. **The Actual Mathematical Reality (36 Grounds):**
   - When the concrete mapping table in `VENUE_MAPPING.md` and the Python dictionary `CANONICAL_VENUE_MAP` were constructed, every single physical stadium was comprehensively audited.
   - The actual count of distinct physical grounds that have hosted IPL matches across all 19 seasons (including South Africa 2009, UAE 2014/2020/2021, and Indian venues) is **36**:
     - **8 South African Stadiums (2009):** Buffalo Park (East London), De Beers Diamond Oval (Kimberley), Kingsmead (Durban), New Wanderers Stadium (Johannesburg), Newlands (Cape Town), OUTsurance Oval (Bloemfontein), St George's Park (Gqeberha), SuperSport Park (Centurion).
     - **3 UAE Stadiums (2014, 2020, 2021):** Dubai International Cricket Stadium, Sharjah Cricket Stadium, Sheikh Zayed Stadium (Abu Dhabi).
     - **25 Indian Stadiums (2008–2026):** Arun Jaitley Stadium, Barabati Stadium, Barsapara Cricket Stadium, Ekana Stadium (Lucknow), Brabourne Stadium, Dr DY Patil Sports Academy, ACA-VDCA Stadium (Visakhapatnam), Eden Gardens, Green Park, HPCA Stadium (Dharamsala), Holkar Cricket Stadium, JSCA Complex (Ranchi), M Chinnaswamy Stadium, MA Chidambaram Stadium, Maharaja Yadavindra Singh Stadium (Mullanpur), MCA Stadium (Pune), Narendra Modi Stadium, Nehru Stadium (Kochi), PCA IS Bindra Stadium (Mohali), Rajiv Gandhi Stadium (Hyderabad), Saurashtra Cricket Association Stadium, Sawai Mansingh Stadium, Shaheed Veer Narayan Singh Stadium (Raipur), VCA Stadium (Jamtha), Wankhede Stadium.
     - **Total: $8 + 3 + 25 = 36$ canonical physical stadiums.**

3. **Evaluation of Correctness:**
   - **Is 33 correct?** **NO.** 33 is an obsolete preliminary heading estimate that omitted 3 legitimate physical stadiums.
   - **Is 36 correct?** **YES.** 36 is the exact, true number of unique physical cricket stadiums in the dataset.
   - **Is another number correct?** **NO.** Exactly 36 physical grounds exist in the raw and processed data.
   - **Which documentation statements are inconsistent?**
     - `DATA_PREPROCESSING.md` Section 2B and Section 4 (text references to "33").
     - `VENUE_MAPPING.md` Section 2 title (reads "33 Canonical Stadiums", while the table directly underneath contains 36).
     - `DATA_VALIDATION.md` Section 5 title (reads "33 Canonical Stadiums", while line 67 states 36 and the table lists 36).
   - **Does the discrepancy affect the actual cleaned data or only documentation?**
     - **It affects ONLY documentation headings.**
     - The underlying datasets (`matches_clean.csv` and `deliveries_clean.csv`) and mapping code (`venue_mapping.py`) are **100% correct, complete, and uncorrupted**.
     - All 60 raw venue strings map to the 36 physical grounds without a single unmapped string or residual alias.
     - Distinct grounds within the same city (e.g., Wankhede vs. Brabourne vs. DY Patil in Mumbai) remain strictly separate.

---

### 3. Complete Alphabetical List of the 36 Canonical Venues

Every venue below was verified in `matches_clean.csv` and `deliveries_clean.csv`:

| # | Canonical Stadium Name | Matches Hosted | Deliveries Bowled | Primary City | Consolidated Raw Aliases |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **Arun Jaitley Stadium** | 104 | 24,566 | Delhi | `Feroz Shah Kotla`, `Arun Jaitley Stadium`, `Arun Jaitley Stadium, Delhi` |
| 2 | **Barabati Stadium** | 7 | 1,695 | Cuttack | `Barabati Stadium` |
| 3 | **Barsapara Cricket Stadium** | 8 | 1,799 | Guwahati | `Barsapara Cricket Stadium, Guwahati` |
| 4 | **Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium** | 29 | 6,904 | Lucknow | `Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow` |
| 5 | **Brabourne Stadium** | 27 | 6,526 | Mumbai | `Brabourne Stadium`, `Brabourne Stadium, Mumbai` |
| 6 | **Buffalo Park** | 3 | 715 | East London | `Buffalo Park` |
| 7 | **De Beers Diamond Oval** | 3 | 726 | Kimberley | `De Beers Diamond Oval` |
| 8 | **Dr DY Patil Sports Academy** | 37 | 8,898 | Mumbai | `Dr DY Patil Sports Academy`, `Dr DY Patil Sports Academy, Mumbai` |
| 9 | **Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium** | 17 | 4,000 | Visakhapatnam | `Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium`, `... Visakhapatnam` |
| 10 | **Dubai International Cricket Stadium** | 46 | 11,229 | Dubai | `Dubai International Cricket Stadium` *(33 missing cities imputed)* |
| 11 | **Eden Gardens** | 107 | 24,936 | Kolkata | `Eden Gardens`, `Eden Gardens, Kolkata` |
| 12 | **Green Park** | 4 | 921 | Kanpur | `Green Park` |
| 13 | **Himachal Pradesh Cricket Association Stadium** | 19 | 4,461 | Dharamsala | `Himachal Pradesh Cricket Association Stadium`, `... Dharamsala` |
| 14 | **Holkar Cricket Stadium** | 9 | 1,965 | Indore | `Holkar Cricket Stadium` |
| 15 | **JSCA International Stadium Complex** | 7 | 1,671 | Ranchi | `JSCA International Stadium Complex` |
| 16 | **Kingsmead** | 15 | 3,643 | Durban | `Kingsmead` |
| 17 | **M Chinnaswamy Stadium** | 104 | 24,128 | Bengaluru | `M Chinnaswamy Stadium`, `... Bengaluru`, `M.Chinnaswamy Stadium` |
| 18 | **MA Chidambaram Stadium** | 98 | 23,851 | Chennai | `MA Chidambaram Stadium`, `... Chepauk`, `... Chepauk, Chennai` |
| 19 | **Maharaja Yadavindra Singh International Cricket Stadium** | 17 | 4,049 | Mullanpur | `... Mullanpur`, `... New Chandigarh` |
| 20 | **Maharashtra Cricket Association Stadium** | 51 | 12,239 | Pune | `Maharashtra Cricket Association Stadium`, `... Pune`, `Subrata Roy Sahara Stadium` |
| 21 | **Narendra Modi Stadium** | 53 | 12,671 | Ahmedabad | `Narendra Modi Stadium, Ahmedabad`, `Sardar Patel Stadium, Motera` |
| 22 | **Nehru Stadium** | 5 | 1,155 | Kochi | `Nehru Stadium` |
| 23 | **New Wanderers Stadium** | 8 | 1,940 | Johannesburg | `New Wanderers Stadium` |
| 24 | **Newlands** | 7 | 1,539 | Cape Town | `Newlands` |
| 25 | **OUTsurance Oval** | 2 | 500 | Bloemfontein | `OUTsurance Oval` |
| 26 | **Punjab Cricket Association IS Bindra Stadium** | 61 | 14,476 | Mohali | `PCA Stadium, Mohali`, `PCA IS Bindra Stadium`, `... Mohali`, `... Mohali, Chandigarh` |
| 27 | **Rajiv Gandhi International Stadium** | 90 | 21,416 | Hyderabad | `Rajiv Gandhi International Stadium`, `... Uppal`, `... Uppal, Hyderabad` |
| 28 | **Saurashtra Cricket Association Stadium** | 10 | 2,381 | Rajkot | `Saurashtra Cricket Association Stadium` |
| 29 | **Sawai Mansingh Stadium** | 68 | 16,174 | Jaipur | `Sawai Mansingh Stadium`, `Sawai Mansingh Stadium, Jaipur` |
| 30 | **Shaheed Veer Narayan Singh International Stadium** | 8 | 1,931 | Raipur | `Shaheed Veer Narayan Singh International Stadium`, `... Raipur` |
| 31 | **Sharjah Cricket Stadium** | 28 | 6,672 | Sharjah | `Sharjah Cricket Stadium` *(18 missing cities imputed)* |
| 32 | **Sheikh Zayed Stadium** | 37 | 8,799 | Abu Dhabi | `Sheikh Zayed Stadium`, `Zayed Cricket Stadium, Abu Dhabi` |
| 33 | **St George's Park** | 7 | 1,677 | Port Elizabeth | `St George's Park` |
| 34 | **SuperSport Park** | 12 | 2,866 | Centurion | `SuperSport Park` |
| 35 | **Vidarbha Cricket Association Stadium, Jamtha** | 3 | 742 | Nagpur | `Vidarbha Cricket Association Stadium, Jamtha` |
| 36 | **Wankhede Stadium** | 132 | 31,871 | Mumbai | `Wankhede Stadium`, `Wankhede Stadium, Mumbai` |

---

## 1. Match Dataset Verification (`matches_clean.csv`)

- **Total Row Count:** Exactly **1,243 rows** (Matches documented: 1,243 $\rightarrow$ **100% Match**).
- **Duplicate Match IDs:** **0** (Primary key uniqueness verified).
- **Missing Match IDs:** **0** (All match IDs populated).
- **Date Range:** **2008/04/18** (First match: RCB vs KKR) to **2026/05/31** (2026 Final).
- **Match Result Types:**
  - `normal`: **1,218 matches** (97.99%)
  - `super_over`: **16 matches** (1.29%)
  - `no_result`: **9 matches** (0.72%)
  - Total: **1,243 matches**
- **Missing Values Audit:**
  - `match_id`, `season`, `date`, `venue`, `city`, `team_1`, `team_2`, `toss_winner`, `toss_decision`, `match_result_type`: **0 missing values (100% complete)**.
  - `winner`: Missing in exactly **9 matches** (100% corresponding to the 9 abandoned `no_result` matches).
  - `method`: Missing in 1,220 matches; populated with `D/L` in exactly **23 rain-affected matches**.

---

## 2. Team Name Mapping Verification

- **Total Unique Canonical Teams:** Exactly **15** across all matches and deliveries.
- **Verification of Specific Franchise Rebrands:**
  - `Delhi Daredevils` (161 matches) $\rightarrow$ `Delhi Capitals` (120 matches) = **281 matches** as `Delhi Capitals`.
  - `Kings XI Punjab` (190 matches) $\rightarrow$ `Punjab Kings` (88 matches) = **278 matches** as `Punjab Kings`.
  - `Royal Challengers Bangalore` (240 matches) $\rightarrow$ `Royal Challengers Bengaluru` (46 matches) = **286 matches** as `Royal Challengers Bengaluru`.
  - `Rising Pune Supergiants` (14 matches) $\rightarrow$ `Rising Pune Supergiant` (16 matches) = **30 matches** as `Rising Pune Supergiant`.
- **Verification of Franchise Independence Rules:**
  - `Deccan Chargers` (75 matches) is **strictly separated** from `Sunrisers Hyderabad` (211 matches).
  - `Pune Warriors` (46 matches) remains distinct.
  - `Gujarat Lions` (30 matches) remains distinct.
  - `Kochi Tuskers Kerala` (14 matches) remains distinct.
  - `Rising Pune Supergiant` (30 matches) remains distinct from `Pune Warriors`.
- **Active Unchanged Franchises:**
  - `Mumbai Indians` (291), `Kolkata Knight Riders` (278), `Chennai Super Kings` (266), `Rajasthan Royals` (251), `Gujarat Titans` (77), `Lucknow Super Giants` (72).

---

## 3. City Cleaning & Imputation Verification

- **Raw Matches with Missing City:** Exactly **51 matches** in `data/raw/cricsheet/`.
- **Imputations Applied:**
  - `Dubai International Cricket Stadium`: **33 matches** imputed as `Dubai`. Total Dubai matches in dataset: 46 (13 had city originally, 33 imputed $\rightarrow$ 100% now `Dubai`).
  - `Sharjah Cricket Stadium`: **18 matches** imputed as `Sharjah`. Total Sharjah matches in dataset: 28 (10 had city originally, 18 imputed $\rightarrow$ 100% now `Sharjah`).
- **Remaining Missing Cities:** **0** (100% complete across all 1,243 matches).
- **Suspicious City/Venue Pairings:** **None.** All canonical grounds map to expected cities.

---

## 4. Delivery Dataset Verification (`deliveries_clean.csv`)

- **Total Row Count:** Exactly **295,732 rows** (Documented raw count: 295,732 $\rightarrow$ **100% Match**).
- **Duplicate Rows:** **0**.
- **Composite Key Duplicates `(match_id, innings, ball, actual_delivery, striker, bowler)`:** **0**.
- **Innings Distribution:**
  - Innings 1: 153,254 deliveries
  - Innings 2: 142,303 deliveries
  - Innings 3: 86 deliveries (Super Over 1)
  - Innings 4: 77 deliveries (Super Over 2)
  - Innings 5: 8 deliveries (Double Super Over 1)
  - Innings 6: 4 deliveries (Double Super Over 2)
- **Super Over Tagging (`is_super_over`):**
  - Exactly **175 deliveries** have `is_super_over = 1` (Innings 3, 4, 5, 6).
  - Exactly **295,557 deliveries** have `is_super_over = 0` (Innings 1 and 2).
  - Inconsistent tags: **0**.
- **Data Integrity & Distributions:**
  - Negative runs off bat or extras: **0**.
  - Invalid innings numbers ($<1$ or $>6$): **0**.
  - Extras fields (`wides`, `noballs`, `byes`, `legbyes`, `penalty`): 100% integer, 0 nulls (empty values properly imputed to 0).
  - Dismissals: `wicket_type` and `player_dismissed` are populated on exactly 14,705 wicket deliveries and empty on 281,027 non-dismissal deliveries.

---

## 5. Referential Integrity Verification

- **Delivery $\rightarrow$ Match Linkage:** Every one of the 295,732 deliveries references a `match_id` present in `matches_clean.csv`. Missing foreign keys: **0**.
- **Team Consistency:** For every delivery, `batting_team` and `bowling_team` strictly belong to `{team_1, team_2}` of that match in `matches_clean.csv`. Mismatches: **0**.
- **Venue Consistency:** For every delivery, `venue` matches the canonical venue of that match in `matches_clean.csv`. Mismatches: **0**.
- **No Corruption:** File schemas, encodings, and row terminations are 100% valid.

---

## 6. Super Over Resolution Verification

All 16 tied matches were individually compared against the raw `*_info.csv` records:

| Match ID | Season / Date | Teams | Raw `info,eliminator` | Cleaned `winner` | Result Type |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **392190** | 2009/04/23 | KKR vs RR | Rajasthan Royals | Rajasthan Royals | `super_over` |
| **419121** | 2010/03/21 | CSK vs PBKS | Kings XI Punjab | Punjab Kings | `super_over` |
| **598004** | 2013/04/07 | SRH vs RCB | Sunrisers Hyderabad | Sunrisers Hyderabad | `super_over` |
| **598017** | 2013/04/16 | RCB vs DC | Royal Challengers Bangalore | Royal Challengers Bengaluru | `super_over` |
| **729315** | 2014/04/29 | KKR vs RR | Rajasthan Royals | Rajasthan Royals | `super_over` |
| **829741** | 2015/04/21 | RR vs PBKS | Kings XI Punjab | Punjab Kings | `super_over` |
| **1082625** | 2017/04/29 | GL vs MI | Mumbai Indians | Mumbai Indians | `super_over` |
| **1175365** | 2019/03/30 | KKR vs DC | Delhi Capitals | Delhi Capitals | `super_over` |
| **1178426** | 2019/05/02 | MI vs SRH | Mumbai Indians | Mumbai Indians | `super_over` |
| **1216493** | 2020/09/20 | DC vs PBKS | Delhi Capitals | Delhi Capitals | `super_over` |
| **1216512** | 2020/10/18 | KKR vs SRH | Kolkata Knight Riders | Kolkata Knight Riders | `super_over` |
| **1216517** | 2020/10/18 | MI vs PBKS | Kings XI Punjab | Punjab Kings | `super_over` |
| **1216547** | 2020/09/28 | RCB vs MI | Royal Challengers Bangalore | Royal Challengers Bengaluru | `super_over` |
| **1254077** | 2021/04/25 | DC vs SRH | Delhi Capitals | Delhi Capitals | `super_over` |
| **1473469** | 2025/04/16 | DC vs RR | Delhi Capitals | Delhi Capitals | `super_over` |
| **1529281** | 2026/04/26 | KKR vs LSG | Kolkata Knight Riders | Kolkata Knight Riders | `super_over` |

Outcome logic is **100% verified and flawless**.

---

## 7. Duckworth-Lewis (D/L) Matches Verification

- Exactly **23 matches** in `matches_clean.csv` have `method = 'D/L'`.
- All 23 matches have a valid winning team and `match_result_type = 'normal'`.
- Zero D/L matches were dropped or corrupted.

---

## Final Preprocessing Verdict & Required Action

### Analysis:
1. **Technical Datasets & Code:** **100% Flawless.** `matches_clean.csv` (1,243 rows) and `deliveries_clean.csv` (295,732 rows) are completely clean, verified against raw files, and referentially sound. Zero data changes or reprocessing needed.
2. **Documentation Inconsistency:** The markdown files `VENUE_MAPPING.md`, `DATA_PREPROCESSING.md`, and `DATA_VALIDATION.md` contain an outdated heading string stating "33 Canonical Stadiums" rather than the verified ground truth of "36 Canonical Stadiums".
3. Under the read-only mandate, no documentation files were modified during this verification. Therefore, an administrative documentation update (updating "33" to "36" in the three markdown files) is required to achieve complete parity before feature engineering is signed off.

---

FINAL PREPROCESSING STATUS:
FIX REQUIRED
