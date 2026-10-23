# Data Preprocessing & Standardization Methodology

**Project:** Cloud-Based IPL Match Winning Probability Prediction & Live Match Analytics System  
**Pipeline Stage:** Data Preprocessing & Cleaning (Phase 1)  
**Output Artifacts:** `data/processed/matches_clean.csv`, `data/processed/deliveries_clean.csv`  

---

## 1. Executive Summary & Goals

The objective of this stage is to convert the raw, fragmented Cricsheet IPL ball-by-ball and match metadata files (`data/raw/cricsheet/`) into two clean, standardized, and production-ready datasets:
1. `data/processed/matches_clean.csv` (Match-level dataset: 1,243 rows)
2. `data/processed/deliveries_clean.csv` (Delivery-level dataset: 295,732 rows)

All data cleaning followed strict mathematical and domain rules to preserve historical authenticity while eliminating categorical fractures.

---

## 2. What Was Changed and Why

### A. Team Name Standardization
- **The Issue:** Over 19 seasons (2008–2026), franchises underwent official brand renamings (e.g., *Delhi Daredevils* $\rightarrow$ *Delhi Capitals* in 2019, *Kings XI Punjab* $\rightarrow$ *Punjab Kings* in 2021, *Royal Challengers Bangalore* $\rightarrow$ *Royal Challengers Bengaluru* in 2024, and *Rising Pune Supergiants* $\rightarrow$ *Rising Pune Supergiant*).
- **The Rationale:** If unmapped, ML models and historical analyses would treat them as completely disjoint teams, artificially fracturing head-to-head records, franchise win rates, and team strength representations.
- **The Resolution:** All rebrands were consolidated to the canonical modern name. Full audit documented in `TEAM_NAME_MAPPING.md`.

### B. Venue Alias Consolidation
- **The Issue:** The raw data contained **60 distinct venue strings** representing **36 physical cricket grounds**. Variations arose from city suffixes (e.g., `Eden Gardens` vs `Eden Gardens, Kolkata`), punctuation (`M.Chinnaswamy Stadium` vs `M Chinnaswamy Stadium`), and official ground renamings (`Feroz Shah Kotla` $\rightarrow$ `Arun Jaitley Stadium`, `Sardar Patel Stadium, Motera` $\rightarrow$ `Narendra Modi Stadium`, `Subrata Roy Sahara Stadium` $\rightarrow$ `Maharashtra Cricket Association Stadium`).
- **The Rationale:** Ground dimensions, boundary sizes, surface characteristics, and altitude remain identical regardless of naming variations. Consolidating to 36 physical grounds prevents excessive categorical sparsity during ML modeling.
- **The Resolution:** All 60 raw strings were mapped to their canonical physical stadium names. Documented in `VENUE_MAPPING.md`.

### C. Conservative City Imputation
- **The Issue:** 51 match-info files omitted the `city` field.
- **The Resolution:** 33 matches at `Dubai International Cricket Stadium` were imputed as `Dubai`, and 18 matches at `Sharjah Cricket Stadium` were imputed as `Sharjah`. Both venues possess 100% unambiguous geographical locations. No ambiguous cities were guessed.

### D. Match Outcome & Tied Match Resolution
- **The Issue:** In 16 tied matches, Cricsheet leaves `winner` blank and records the Super Over winner under the `eliminator` key. In 9 abandoned matches, no result was possible due to persistent weather.
- **The Resolution:**
  - For normal matches: `match_result_type = 'normal'`, `winner = mapped winner`.
  - For tied matches: `match_result_type = 'super_over'`, `winner = mapped eliminator winner`.
  - For abandoned matches: `match_result_type = 'no_result'`, `winner = ''` (empty string).
  - All 1,243 matches are retained in `matches_clean.csv`. For downstream supervised classification, rows where `match_result_type == 'no_result'` can simply be excluded by filtering on this column.

### E. Numerical Extras Imputation & Sparse Dismissal Handling
- **The Issue:** Delivery-level extras (`wides`, `noballs`, `byes`, `legbyes`, `penalty`) were sparse/null whenever no extra occurred.
- **The Resolution:**
  - Empty extra values were imputed with `0`.
  - `wicket_type` and `player_dismissed` were intentionally left empty (`""`) on non-dismissal balls, reflecting genuine absence of a dismissal rather than missing data.

### F. Super Over Tagging
- **The Resolution:** Deliveries occurring in Super Overs (innings 3, 4, 5, and 6) were retained in `deliveries_clean.csv`, but explicitly tagged with `is_super_over = 1` (and `0` for normal innings 1 and 2). This ensures downstream feature engineering will not accidentally mix Super Over deliveries into standard 20-over run-rate curves.

---

## 3. What Was Intentionally NOT Changed

1. **Raw Files in `data/raw/cricsheet/`:**
   - Kept 100% immutable and untouched. No files were modified, renamed, or deleted.
2. **Defunct Franchises Kept Separate:**
   - *Deccan Chargers* was NOT merged into *Sunrisers Hyderabad*. They are treated as historically distinct franchises.
   - *Pune Warriors*, *Gujarat Lions*, and *Kochi Tuskers Kerala* were kept as independent defunct entities.
3. **Distinct Grounds in the Same City:**
   - In Mumbai, *Wankhede Stadium*, *Brabourne Stadium*, and *Dr DY Patil Sports Academy* were strictly kept as distinct stadiums.
4. **No Feature Engineering:**
   - Run rates (CRR, RRR), rolling batsman/bowler metrics, boundary percentages, and target differentials were NOT calculated at this stage. Clean data remains pure and free of model-specific assumptions.
5. **No Machine Learning Models:**
   - No models were trained or evaluated.
6. **Rain / Duckworth-Lewis Matches:**
   - D/L matches were preserved with a `method = 'D/L'` column rather than discarded.

---

## 4. Pipeline Execution & Reusability

The preprocessing pipeline is organized into modular components under `src/data_processing/`:
- `team_mapping.py`: Dictionaries and mapping functions for franchises.
- `venue_mapping.py`: Dictionaries and functions for stadiums and cities.
- `parser.py`: Low-level parser for key-value `*_info.csv` records.
- `clean_matches.py`: Batch processor for match-level metadata.
- `clean_deliveries.py`: Memory-efficient streaming processor for ball-by-ball deliveries.
- `validation.py`: Parity auditor checking rows, nulls, duplicates, and distributions.
- `run_pipeline.py`: Main orchestration script.

To regenerate both clean datasets and the validation report from scratch at any time:
```powershell
python -m src.data_processing.run_pipeline
```
