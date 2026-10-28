"""
Flight price preprocessing pipeline.

Run single year:
    python3 preprocess.py --dataset 2023
    python3 preprocess.py --dataset 2026

Run merged (stacks both years, within-year benchmarks, adds Year column for pooled regression):
    python3 preprocess.py --dataset merged

Override default input paths or output file:
    python3 preprocess.py --dataset merged \
        --input_2023 /path/to/2023.csv \
        --input_2026 /path/to/2026_clean.csv \
        --output my_output.csv
"""

import argparse, re, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PRICE_CAP  = 100000  # rows above this are scraper artifacts (time strings mis-parsed as price)
DEDUP_KEYS = ["FlightCode", "Route", "DepartureDate", "DaysBeforeDeparture", "TravelTime"]
DROP_COLS  = [
    "ScrapeDate", "DepartureTime", "ArrivalTime", "Arrival Time", "Origin", "Destination",
    "FareMean", "SoldOutDate", "PercentDeviationFromFareMean",
    "DeviationFromSoldOutDate", "HHI", "NumberOfCarrier", "IndirectDummy",
    "LCCdum", "LCCpercentage", "Unnamed: 11",
]
FINAL_COLS = [
    "FlightCode", "airline", "Route", "DepartureDate", "Year",
    "DaysBeforeDeparture", "price", "DirectPrice_avg", "TravelTimeMinutes",
    "ConnectionNumber", "SoldOutDummy", "InitialDirectCount", "log_price",
    "log_DirectPrice_avg", "log_TravelTime", "log_TravelTime_sq", "log_DirectPrice_x_Days",
]


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset",    choices=["2023", "2026", "merged"], required=True)
    p.add_argument("--input_2023", default="data/2023.csv")
    p.add_argument("--input_2026", default="data/2026_clean.csv")
    p.add_argument("--output",     default=None)
    p.add_argument("--price_cap",  type=float, default=PRICE_CAP)
    return p.parse_args()


def parse_travel_time(val):
    if pd.isna(val):
        return np.nan
    s = str(val)
    h = re.search(r"(\d+)\s*u", s)
    m = re.search(r"(\d+)\s*m", s)
    total = (int(h.group(1)) if h else 0) * 60 + (int(m.group(1)) if m else 0)
    return float(total) if total > 0 else np.nan


def drop_where(df, mask, label):
    before = len(df)
    df = df[mask].copy()
    print(f"  {label:<50} -{before - len(df):>7,}  →  {len(df):,}")
    return df


def load_csv(path, year):
    # 2023.csv uses semicolons; 2026_clean.csv uses commas — sniff from first line
    with open(path, encoding="latin-1") as f:
        sep = ";" if ";" in f.readline() else ","
    df = pd.read_csv(path, sep=sep, encoding="latin-1", low_memory=False)
    df = df.rename(columns={"Date before departure": "DaysBeforeDeparture"})
    df["price"] = df["price"].astype(str).str.replace(r"[^\d.]", "", regex=True)  # strip currency prefix
    df["Year"] = year
    return df


def build_benchmark(df_direct, bench_keys):
    return (
        df_direct.groupby(bench_keys)["price"].mean()
        .reset_index().rename(columns={"price": "DirectPrice_avg"})
    )


def build_soldout(df_direct, bench_keys, date_keys):
    # Count distinct direct flights per (route, date, day[, year])
    counts = (
        df_direct.groupby(bench_keys)["FlightCode"].nunique()
        .reset_index().rename(columns={"FlightCode": "DirectCount"})
    )
    # Baseline = count at the furthest scrape day (largest DaysBeforeDeparture)
    initial = (
        counts.sort_values("DaysBeforeDeparture", ascending=False)
        .groupby(date_keys).first().reset_index()[date_keys + ["DirectCount"]]
        .rename(columns={"DirectCount": "InitialDirectCount"})
    )
    counts = counts.merge(initial, on=date_keys, how="left")
    # SoldOutDummy = 1 when fewer direct flights are available than at the baseline
    counts["SoldOutDummy"] = (counts["DirectCount"] < counts["InitialDirectCount"]).astype(int)
    return counts[bench_keys + ["SoldOutDummy"]], initial


def clean(df, price_cap):
    print(f"\n  {'Step':<50} {'Dropped':>9}    Remaining")
    print("  " + "-" * 75)
    df["DaysBeforeDeparture"] = pd.to_numeric(df["DaysBeforeDeparture"], errors="coerce")
    df = drop_where(df, df["DaysBeforeDeparture"] >= 0, "Drop negative DaysBeforeDeparture")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = drop_where(df, df["price"].notna() & (df["price"] > 0), "Drop non-positive/missing price")
    df = drop_where(df, df["price"] <= price_cap, f"Drop price > {price_cap:,.0f} (schedule text)")
    df["TravelTimeMinutes"] = df["TravelTime"].apply(parse_travel_time)
    df = drop_where(df, df["TravelTimeMinutes"].notna() & (df["TravelTimeMinutes"] > 0), "Drop unparseable TravelTime")
    df = drop_where(df, ~df.duplicated(), "Remove exact duplicates")
    keys = [k for k in DEDUP_KEYS if k in df.columns]
    if "Year" in df.columns:
        keys = keys + ["Year"]
    df = df.loc[df.groupby(keys)["price"].idxmin()].reset_index(drop=True)
    print(f"  Itinerary dedup (keep cheapest)                    → {len(df):,}")
    return df


def add_log_vars(df):
    df["log_price"]              = np.log(df["price"])
    df["log_DirectPrice_avg"]    = np.log(df["DirectPrice_avg"])
    df["log_TravelTime"]         = np.log(df["TravelTimeMinutes"])
    df["log_TravelTime_sq"]      = df["log_TravelTime"] ** 2
    df["log_DirectPrice_x_Days"] = df["log_DirectPrice_avg"] * df["DaysBeforeDeparture"]
    return df


def run_pipeline(df, price_cap, is_merged=False):
    df = df.drop(columns=DROP_COLS, errors="ignore")
    df = clean(df, price_cap)

    df_direct   = df[df["ConnectionNumber"] == 0].copy()
    df_indirect = df[df["ConnectionNumber"] >  0].copy()
    print(f"\n  Direct: {len(df_direct):,}   Indirect: {len(df_indirect):,}")

    bench_keys = ["Route", "DepartureDate", "DaysBeforeDeparture"]
    date_keys  = ["Route", "DepartureDate"]
    if is_merged:
        bench_keys += ["Year"]
        date_keys  += ["Year"]

    direct_avg        = build_benchmark(df_direct, bench_keys)
    sold_out, initial = build_soldout(df_direct, bench_keys, date_keys)

    df_indirect = (
        df_indirect
        .merge(direct_avg,                              on=bench_keys, how="left")
        .merge(sold_out,                                on=bench_keys, how="left")
        .merge(initial[date_keys + ["InitialDirectCount"]], on=date_keys,  how="left")
    )

    before = len(df_indirect)
    df_indirect = df_indirect.dropna(subset=["DirectPrice_avg"])
    print(f"  Drop no-benchmark rows: -{before - len(df_indirect):,}  →  {len(df_indirect):,}")

    df_indirect["SoldOutDummy"] = df_indirect["SoldOutDummy"].fillna(0).astype(int)

    df_indirect = add_log_vars(df_indirect)

    if is_merged:
        # Strip year from date strings so C(DepartureDay) + C(Year) are separately identified
        # in the pooled regression (raw strings like "15-4-2023" vs "15/04/2026" are year-specific
        # and would make C(Year) collinear with C(DepartureDate))
        df_indirect["DepartureDay"] = pd.to_datetime(
            df_indirect["DepartureDate"], format="mixed", dayfirst=True, errors="coerce"
        ).dt.strftime("%m-%d")

    return df_indirect


def print_summary(df, label):
    print(f"\n=== {label} ===")
    print(f"  {len(df):,} rows  |  {df['Route'].nunique()} routes")
    for yr in sorted(df["Year"].unique()):
        sub = df[df["Year"] == yr]
        print(f"  {yr}: {len(sub):,} rows  mean €{sub['price'].mean():.0f}  SoldOut {sub['SoldOutDummy'].mean()*100:.1f}%")
    missing = {c: n for c, n in df[[c for c in FINAL_COLS if c in df]].isna().sum().items() if n > 0}
    if missing:
        print("  Missing:", {c: f"{n:,}" for c, n in missing.items()})


def main():
    args = parse_args()
    output = args.output or {"2023": "final_2023.csv", "2026": "final_2026_fixed.csv", "merged": "final_merged.csv"}[args.dataset]

    if args.dataset == "2023":
        print(f"Loading 2023: {args.input_2023}")
        df = load_csv(args.input_2023, 2023)
        is_merged = False

    elif args.dataset == "2026":
        print(f"Loading 2026: {args.input_2026}")
        df = load_csv(args.input_2026, 2026)
        is_merged = False

    else:
        print(f"Loading 2023: {args.input_2023}")
        df23 = load_csv(args.input_2023, 2023)
        print(f"Loading 2026: {args.input_2026}")
        df26 = load_csv(args.input_2026, 2026)
        df = pd.concat([df23, df26], ignore_index=True)
        print(f"Merged: {len(df):,} rows  (2023: {(df.Year==2023).sum():,}  2026: {(df.Year==2026).sum():,})")
        is_merged = True

    df_final = run_pipeline(df, price_cap=args.price_cap, is_merged=is_merged)

    keep = [c for c in FINAL_COLS if c in df_final.columns]
    if is_merged and "DepartureDay" in df_final.columns:
        keep += ["DepartureDay"]
    df_final = df_final[keep]

    print_summary(df_final, output)
    df_final.to_csv(output, index=False)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
