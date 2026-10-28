"""
merging_output_v1.py
--------------------
Merges all flights_output_*.csv files produced by the scraper into a single
merged_output_with_lcc.csv, then populates LCCdum from the airlinelccdum.csv
lookup table.

Changes from merging_output_original.py:
- Removed intermediate airlines.csv export (not needed downstream)
- Removed merged_output_with_airline_id_head.csv debug export
- Cleaner output filename: merged_output_with_lcc.csv
- Added row counts per file and dedup report
"""

import glob
import pandas as pd


def main():
    # ------------------------------------------------------------------
    # 1. FIND & LOAD all flights_output_*.csv scraper files
    # ------------------------------------------------------------------
    csv_files = sorted(glob.glob("flights_output_*.csv"))

    if not csv_files:
        print("No flights_output_*.csv files found in current directory.")
        return

    print(f"Found {len(csv_files)} file(s) to merge:")
    dataframes = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            dataframes.append(df)
            print(f"  Loaded {len(df):,} rows from {f}")
        except Exception as e:
            print(f"  Error reading {f}: {e}")

    if not dataframes:
        print("No valid files to merge.")
        return

    # ------------------------------------------------------------------
    # 2. CONCATENATE
    # ------------------------------------------------------------------
    merged = pd.concat(dataframes, ignore_index=True)
    print(f"\nTotal rows before dedup: {len(merged):,}")

    # ------------------------------------------------------------------
    # 3. REMOVE EXACT DUPLICATES (same row appearing in multiple scrape files)
    # ------------------------------------------------------------------
    before = len(merged)
    merged = merged.drop_duplicates()
    print(f"Rows after exact-dedup:  {len(merged):,}  (dropped {before - len(merged):,})")

    # ------------------------------------------------------------------
    # 4. POPULATE LCCdum FROM LOOKUP TABLE
    # Expects airlinelccdum.csv with columns: airline, LCCdum
    # ------------------------------------------------------------------
    try:
        lcc_lookup = pd.read_csv("airlinelccdum.csv")
        lcc_lookup = lcc_lookup.set_index("airline")[["LCCdum"]].astype(int)

        merged = merged.drop(columns=["LCCdum"], errors="ignore")
        merged = merged.join(lcc_lookup, on="airline", how="left")

        missing_lcc = merged["LCCdum"].isna().sum()
        if missing_lcc > 0:
            print(f"Warning: {missing_lcc:,} rows have no LCCdum match — check airlinelccdum.csv")
    except FileNotFoundError:
        print("Warning: airlinelccdum.csv not found — LCCdum column will remain empty.")

    # ------------------------------------------------------------------
    # 5. SAVE
    # ------------------------------------------------------------------
    output_file = "merged_output_with_lcc.csv"
    merged.to_csv(output_file, index=False)
    print(f"\nSaved {len(merged):,} rows to '{output_file}'")
    print(merged.head(3).to_string())


if __name__ == "__main__":
    main()
