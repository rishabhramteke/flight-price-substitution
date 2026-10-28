"""
Regression: indirect flight price response to direct price, travel time, and sell-out pressure.

Model:
    log(price) = α1·log(DirectPrice_avg) + α2·log(DirectPrice_avg)×Days
               + α3·log(TravelTime) + α4·log(TravelTime)²
               + α5·SoldOutDummy + Route FE + DepartureDate FE [+ Year FE if merged]

Also produces the booking horizon curve: implied elasticity α1 + α2×d across days before departure.

Run single year:
    python3 regression.py --dataset 2023
    python3 regression.py --dataset 2026

Run merged (adds Year FE, plots 2023 vs 2026 horizon curves):
    python3 regression.py --dataset merged

Override input path or plot output directory:
    python3 regression.py --dataset merged --input /path/to/final_merged.csv --plots_dir plots
"""

import argparse, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

INPUT_FILES = {
    "2023":   "final_2023.csv",
    "2026":   "final_2026_fixed.csv",
    "merged": "final_merged.csv",
}

# Single-year formula: DepartureDate FE captures date-specific shocks
FORMULA = (
    "log_price ~ log_DirectPrice_avg + log_DirectPrice_x_Days + "
    "log_TravelTime + log_TravelTime_sq + SoldOutDummy + "
    "C(Route) + C(DepartureDate)"
)
# Merged formula: DepartureDay (MM-DD, year-stripped) keeps C(Year) separately identified
FORMULA_MERGED = (
    "log_price ~ log_DirectPrice_avg + log_DirectPrice_x_Days + "
    "log_TravelTime + log_TravelTime_sq + SoldOutDummy + "
    "C(Year) + C(Route) + C(DepartureDay)"
)

ALPHA_VARS = [
    ("log_DirectPrice_avg",    "α1  log(DirectPrice_avg)"),
    ("log_DirectPrice_x_Days", "α2  log(DirectPrice_avg) × Days"),
    ("log_TravelTime",         "α3  log(TravelTime)"),
    ("log_TravelTime_sq",      "α4  log(TravelTime)²"),
    ("SoldOutDummy",           "α5  SoldOutDummy"),
]


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset",   choices=["2023", "2026", "merged"], required=True)
    p.add_argument("--input",     default=None,    help="Override default input CSV path")
    p.add_argument("--output",    default="results", help="Output filename prefix")
    return p.parse_args()


def sig(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""


def cluster_ols(formula, df):
    return smf.ols(formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["Route"]}
    )


def load_data(dataset, path):
    df = pd.read_csv(path)
    print(f"Loaded {path}: {len(df):,} rows")
    if dataset == "merged":
        for yr in sorted(df["Year"].unique()):
            print(f"  {yr}: {(df.Year == yr).sum():,} rows")
    return df


def run_regression(df, dataset):
    """Returns dict of {label: result}. Merged runs per-year + pooled."""
    if dataset == "merged":
        results = {}
        for yr in [2023, 2026]:
            sub = df[df["Year"] == yr].copy()
            print(f"\nRunning {yr}  N={len(sub):,}  routes={sub['Route'].nunique()}")
            results[str(yr)] = cluster_ols(FORMULA, sub)
        print(f"\nRunning pooled  N={len(df):,}  routes={df['Route'].nunique()}")
        results["pooled"] = cluster_ols(FORMULA_MERGED, df)
        return results
    else:
        print(f"\nN={len(df):,}  routes={df['Route'].nunique()}")
        return {dataset: cluster_ols(FORMULA, df)}


def print_results(results, dataset):
    print(f"\n{'='*65}")
    print(f"RESULTS — dataset={dataset}   SEs clustered at route level")
    print(f"{'='*65}")
    for label, res in results.items():
        print(f"\n--- {label}  N={int(res.nobs):,}  R²={res.rsquared:.4f} ---")
        print(f"  {'Variable':<42} {'Coef':>8}  {'SE':>8}  {'t':>6}  {'p':>7}  Sig")
        print("  " + "-" * 60)
        for var, lbl in ALPHA_VARS:
            if var not in res.params:
                continue
            c, se, t, p = res.params[var], res.bse[var], res.tvalues[var], res.pvalues[var]
            print(f"  {lbl:<42} {c:>8.4f}  {se:>8.4f}  {t:>6.2f}  {p:>7.4f}  {sig(p)}")
        if label == "pooled" and "C(Year)[T.2026]" in res.params:
            var = "C(Year)[T.2026]"
            c, se, p = res.params[var], res.bse[var], res.pvalues[var]
            print(f"  {'λ_year (2026 vs 2023)':<42} {c:>8.4f}  {se:>8.4f}  {'':>6}  {p:>7.4f}  {sig(p)}")
    print(f"\n*** p<0.01  ** p<0.05  * p<0.1")


def save_results(results, dataset, prefix):
    rows = []
    for label, res in results.items():
        for var, lbl in ALPHA_VARS:
            if var not in res.params:
                continue
            p = res.pvalues[var]
            rows.append({
                "dataset": label, "variable": var, "label": lbl,
                "coef":    round(res.params[var], 6),
                "std_err": round(res.bse[var], 6),
                "t_stat":  round(res.tvalues[var], 4),
                "p_value": round(p, 6), "sig": sig(p),
                "n_obs":   int(res.nobs), "r2": round(res.rsquared, 6),
            })
        if label == "pooled" and "C(Year)[T.2026]" in res.params:
            var = "C(Year)[T.2026]"
            p = res.pvalues[var]
            rows.append({
                "dataset": label, "variable": var, "label": "λ_year (2026 vs 2023)",
                "coef":    round(res.params[var], 6),
                "std_err": round(res.bse[var], 6),
                "t_stat":  round(res.tvalues[var], 4),
                "p_value": round(p, 6), "sig": sig(p),
                "n_obs":   int(res.nobs), "r2": round(res.rsquared, 6),
            })
    fname = f"{prefix}_{dataset}.csv"
    pd.DataFrame(rows).to_csv(fname, index=False)
    print(f"Saved: {fname}")



def main():
    args    = parse_args()
    path    = args.input or INPUT_FILES[args.dataset]
    df      = load_data(args.dataset, path)
    results = run_regression(df, args.dataset)
    print_results(results, args.dataset)
    save_results(results, args.dataset, args.output)


if __name__ == "__main__":
    main()
