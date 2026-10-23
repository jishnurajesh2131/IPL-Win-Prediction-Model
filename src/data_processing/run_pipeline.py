"""
Data Preprocessing Pipeline Execution Runner.

Executes the full preprocessing pipeline:
1. Clean match-level data -> data/processed/matches_clean.csv
2. Clean delivery-level data -> data/processed/deliveries_clean.csv
3. Run comprehensive validation -> DATA_VALIDATION.md
"""

import time
from .clean_matches import clean_all_matches
from .clean_deliveries import clean_all_deliveries
from .validation import run_validation


def main():
    print("=" * 60)
    print("STARTING IPL DATA PREPROCESSING & CLEANING PIPELINE")
    print("=" * 60)

    start_time = time.time()

    # Step 1: Clean Match-Level Data
    print("\n[Step 1/3] Processing match-level info files...")
    matches = clean_all_matches(
        raw_dir="data/raw/cricsheet",
        output_path="data/processed/matches_clean.csv"
    )
    print(f" -> Successfully processed and saved {len(matches):,} matches to data/processed/matches_clean.csv")

    # Step 2: Clean Delivery-Level Data
    print("\n[Step 2/3] Processing ball-by-ball deliveries streamingly...")
    total_deliveries = clean_all_deliveries(
        raw_csv_path="data/raw/cricsheet/all_matches.csv",
        output_path="data/processed/deliveries_clean.csv"
    )
    print(f" -> Successfully processed and saved {total_deliveries:,} deliveries to data/processed/deliveries_clean.csv")

    # Step 3: Validation and Quality Assurance
    print("\n[Step 3/3] Running validation checks and generating audit report...")
    val_results = run_validation(
        raw_dir="data/raw/cricsheet",
        clean_matches_path="data/processed/matches_clean.csv",
        clean_deliveries_path="data/processed/deliveries_clean.csv",
        report_output_path="DATA_VALIDATION.md"
    )
    print(" -> Validation completed successfully! Report generated at DATA_VALIDATION.md")
    print(f" -> Duplicate matches: {val_results['clean_match_duplicates']}")
    print(f" -> Duplicate deliveries: {val_results['clean_deliv_duplicates']}")
    print(f" -> Outcomes breakdown: {val_results['result_types']}")

    elapsed = time.time() - start_time
    print(f"\nPipeline finished successfully in {elapsed:.2f} seconds.")
    print("=" * 60)


if __name__ == "__main__":
    main()
