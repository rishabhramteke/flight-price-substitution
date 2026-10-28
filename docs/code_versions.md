# Code Versions

## merging_output

### merging_output_original.py
Original merging script. Concatenates all `flights_output_*.csv` scraper files, removes exact duplicates, exports an airlines list, and joins `LCCdum` from `airlinelccdum.csv`.

### merge_scrapes.py
**Changes from original:**
- Removed intermediate `airlines.csv` export (not needed downstream)
- Removed `merged_output_with_airline_id_head.csv` debug export
- Renamed output file to `merged_output_with_lcc.csv` (clearer name)
- Added per-file row count and dedup report to console output
- Warns explicitly if `airlinelccdum.csv` is missing rather than crashing

**Usage:**
```bash
python3 merge_scrapes.py
# Reads:  flights_output_*.csv  +  airlinelccdum.csv (optional)
# Writes: merged_output_with_lcc.csv
```

---

## Data_Preprocessing

### Data_Preprocessing_original.py
Original preprocessing script. Loads a single hardcoded 2026 file, renames `DaysBeforeDeparture`, cleans price, parses travel time, removes duplicates, computes LCC percentage, splits direct/indirect, builds a direct price benchmark using **minimum** price, merges it into the indirect dataset, and saves with log-transformed price and direct price.

**Known divergences from final model:**
- Direct price benchmark: used `min` — the model requires `mean` (`DirectPrice_avg`)
- `SoldOutDummy`: not implemented
- Missing log variables: `log(TravelTime)`, `[log(TravelTime)]²`, interaction `log(DirectPrice_avg) × DaysBeforeDeparture`
- `columns_to_drop` was a `[..., ...]` placeholder
- No `Year` column (needed for fixed effects)
- Hardcoded to a single 2026 filename; no support for 2023 (AMD.csv)

### preprocess.py
**Changes from original — aligned with the final methodology:**

| # | What changed | Why |
|---|---|---|
| 1 | Direct price benchmark: `min` → `mean` (`DirectPrice_avg`) | Spec step 4 specifies average direct price |
| 2 | `SoldOutDummy` added | Spec step 5: 1 when direct flight count drops below initial availability |
| 3 | Added `log_TravelTime`, `log_TravelTime_sq` | Spec step 8: non-linear quality proxy |
| 4 | Added `log_DirectPrice_x_Days` interaction | Spec step 8: time-varying direct price linkage |
| 5 | `P_direct` renamed to `DirectPrice_avg` throughout | Consistent with the model variable naming |
| 6 | Handles 2023 (AMD.csv) via column normalisation | AMD uses lowercase/no-space column names; price already numeric |
| 7 | `columns_to_drop` filled in (no longer a placeholder) | Removes columns not in final model |
| 8 | `Year` column added (2023 or 2026) | Required for year fixed effects in pooled regression |
| 9 | Dataset selected via `--dataset` argparse argument | Replaces hardcoded `DATASET = "2026"` variable |
| 10 | Itinerary-dedup via `groupby idxmin` instead of `sort_values` + `drop_duplicates` | More direct: select the min-price row per group without a full sort |
| 11 | `reset_index(drop=True)` after dedup | Prevents stale index issues downstream |

**Usage:**
```bash
python3 preprocess.py --dataset 2023
# Reads:  AMD.csv
# Writes: final_2023.csv

python3 preprocess.py --dataset 2026
# Reads:  merged_output_with_lcc.csv  (output of merge_scrapes.py)
# Writes: final_2026.csv
```

**Run results (preprocess.py):**

| Step | 2023 | 2026 (partial — 1 scrape day) |
|---|---|---|
| Rows loaded | 445,385 | 11,703 |
| Exact duplicates removed | 20,683 | 1 |
| Itinerary duplicates removed | 10,908 | 162 |
| Rows after dedup | 413,794 | 11,540 |
| Direct flights | 84,548 | 1,455 |
| Indirect flights | 329,246 | 10,085 |
| Indirect dropped (no benchmark) | 156,655 | 2,214 |
| **Final rows** | **172,591** | **7,871** |
| Missing values in output | 0 | 0 |

> Note: 2026 figures will change significantly as scraping continues and the booking horizon extends to 36 days.
