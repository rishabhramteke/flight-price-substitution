"""
run_merged_analyses.py
----------------------
Loads raw 2023.csv and 2026_clean.csv from data/,
runs the full preprocessing pipeline on the merged dataset (within-year
benchmarks), and executes Experiments 1, 3, 4.

Key purpose: estimate and characterise lambda_year — the structural
year-level price shift parameter between 2023 and 2026.

Experiments:
    Exp 1 — Substitution effect (α1) and lambda_year (Year FE)
    Exp 3 — Event study: indirect price response to sell-out, by year
    Exp 4 — LCC heterogeneity (α1, α5) by year

Note: Booking horizon curve (Exp 2) is produced by regression.py --dataset merged

Outputs:
    final_merged.csv            preprocessed merged dataset
    plots_merged/01_*.png  …    one plot per experiment
    results_merged_*.csv        regression tables

Usage:
    cd <repo>
    python3 run_merged_analyses.py
"""

import os, re, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
os.makedirs("plots_merged", exist_ok=True)

RAW_2023 = "data/2023.csv"
RAW_2026 = "data/2026_clean.csv"
AMD_FILE = "AMD.csv"        # LCCdum reference for 2026

PALETTE = {
    "2023":     "#2196F3",
    "2026":     "#FF5722",
    "pooled":   "#4CAF50",
    "lcc":      "#FF9800",
    "nonlcc":   "#607D8B",
    "treated":  "#F44336",
    "control":  "#90CAF9",
}

FORMULA_BASE = (
    "log_price ~ log_DirectPrice_avg + log_DirectPrice_x_Days + "
    "log_TravelTime + log_TravelTime_sq + SoldOutDummy + "
    "C(Route) + C(DepartureDate)"
)
# Pooled formula uses DepartureDay (MM-DD, year-stripped) so C(Year) is identified
# separately from C(DepartureDate), which would be perfectly collinear with C(Year)
# when all departure dates are year-specific strings.
FORMULA_YEAR = (
    "log_price ~ log_DirectPrice_avg + log_DirectPrice_x_Days + "
    "log_TravelTime + log_TravelTime_sq + SoldOutDummy + "
    "C(Year) + C(Route) + C(DepartureDay)"
)


def parse_travel_time(series: pd.Series) -> pd.Series:
    def _p(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip()
        h = re.search(r"(\d+)\s*u", s)
        m = re.search(r"(\d+)\s*m", s)
        total = (int(h.group(1)) if h else 0) * 60 + (int(m.group(1)) if m else 0)
        return total if total > 0 else np.nan
    return series.apply(_p)


def sig(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "n.s."


def cluster_ols(formula, data):
    return smf.ols(formula, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data["Route"]}
    )


def coef_row(res, var):
    """Return (coef, se, t, p) for a variable, or NaN tuple if absent."""
    if var not in res.params:
        return np.nan, np.nan, np.nan, np.nan
    return res.params[var], res.bse[var], res.tvalues[var], res.pvalues[var]


# =============================================================================
# STEP 1 — LOAD RAW FILES
# =============================================================================
print("=" * 65)
print("LOADING RAW DATA")
print("=" * 65)

print(f"Loading {RAW_2023} ...")
df23 = pd.read_csv(RAW_2023, sep=";", encoding="latin-1", low_memory=False)
df23 = df23.rename(columns={"Date before departure": "DaysBeforeDeparture"})
df23 = df23.drop(columns=["Unnamed: 11"], errors="ignore")
df23["Year"] = 2023
print(f"  {len(df23):,} rows  |  {df23['Route'].nunique()} routes")

print(f"Loading {RAW_2026} ...")
df26 = pd.read_csv(RAW_2026, encoding="latin-1", low_memory=False)
df26 = df26.rename(columns={
    "Date before departure": "DaysBeforeDeparture",
    "Arrival Time": "ArrivalTime",
})
df26["price"] = df26["price"].astype(str).str.replace(r"[^\d.]", "", regex=True)
df26["Year"] = 2026
print(f"  {len(df26):,} rows  |  {df26['Route'].nunique()} routes")

# Join LCCdum / LCCpercentage for 2026 from AMD.csv reference
print("Joining LCCdum for 2026 from AMD.csv ...")
lcc_ref = (
    pd.read_csv(AMD_FILE, usecols=["route", "lccdum", "lccpercentage"])
    .drop_duplicates("route")
    .rename(columns={"route": "Route", "lccdum": "LCCdum", "lccpercentage": "LCCpercentage"})
)
df26 = df26.drop(columns=["LCCdum", "LCCpercentage"], errors="ignore")
df26 = df26.merge(lcc_ref, on="Route", how="left")
print(f"  LCCdum matched: {df26['LCCdum'].notna().sum():,}/{len(df26):,}")


# =============================================================================
# STEP 2 — ALIGN COLUMNS AND CONCATENATE
# =============================================================================
KEEP_COLS = [
    "FlightCode", "airline", "LCCdum", "LCCpercentage",
    "Route", "DepartureDate", "ConnectionNumber",
    "TravelTime", "price", "DaysBeforeDeparture", "Year",
]
df23 = df23[[c for c in KEEP_COLS if c in df23.columns]].copy()
df26 = df26[[c for c in KEEP_COLS if c in df26.columns]].copy()

df = pd.concat([df23, df26], ignore_index=True)
print(f"\nMerged raw: {len(df):,} rows  "
      f"(2023: {(df.Year==2023).sum():,}  |  2026: {(df.Year==2026).sum():,})")


# =============================================================================
# STEP 3 — PREPROCESS
# =============================================================================
print("\n" + "=" * 65)
print("PREPROCESSING")
print("=" * 65)

# Numeric DaysBeforeDeparture, drop negatives
df["DaysBeforeDeparture"] = pd.to_numeric(df["DaysBeforeDeparture"], errors="coerce")
n = df["DaysBeforeDeparture"].lt(0).sum()
df = df[df["DaysBeforeDeparture"] >= 0].copy()
print(f"Dropped {n:,} rows with negative DaysBeforeDeparture")

# Numeric price, drop non-positive / missing
df["price"] = pd.to_numeric(df["price"], errors="coerce")
n = df["price"].isna().sum() + (df["price"] <= 0).sum()
df = df[df["price"].notna() & (df["price"] > 0)].copy()
print(f"Dropped {n:,} rows with bad price")

# TravelTime → minutes
df["TravelTimeMinutes"] = parse_travel_time(df["TravelTime"])
n = df["TravelTimeMinutes"].isna().sum()
df = df[df["TravelTimeMinutes"].notna() & (df["TravelTimeMinutes"] > 0)].copy()
print(f"Dropped {n:,} rows with unparseable TravelTime")

# Exact duplicates
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Exact-dedup removed: {before - len(df):,}")

# Itinerary dedup — keep cheapest per (FlightCode, Route, Date, Day, TT, Year)
before = len(df)
GKEYS = ["FlightCode", "Route", "DepartureDate", "DaysBeforeDeparture", "TravelTime", "Year"]
idx = df.groupby(GKEYS)["price"].idxmin()
df = df.loc[idx].reset_index(drop=True)
print(f"Itinerary-dedup removed: {before - len(df):,}  →  {len(df):,} remaining")

print(f"  2023: {(df.Year==2023).sum():,}  |  2026: {(df.Year==2026).sum():,}")


# =============================================================================
# STEP 4 — SPLIT DIRECT / INDIRECT
# =============================================================================
df_direct   = df[df["ConnectionNumber"] == 0].copy()
df_indirect = df[df["ConnectionNumber"] >  0].copy()
print(f"\nDirect: {len(df_direct):,}  |  Indirect: {len(df_indirect):,}")


# =============================================================================
# STEP 5 — DIRECT PRICE BENCHMARK (within-year)
# =============================================================================
direct_avg = (
    df_direct
    .groupby(["Route", "DepartureDate", "DaysBeforeDeparture", "Year"])["price"]
    .mean()
    .reset_index()
    .rename(columns={"price": "DirectPrice_avg"})
)


# =============================================================================
# STEP 6 — SOLD-OUT DUMMY (within-year)
# =============================================================================
direct_counts = (
    df_direct
    .groupby(["Route", "DepartureDate", "DaysBeforeDeparture", "Year"])["FlightCode"]
    .nunique()
    .reset_index()
    .rename(columns={"FlightCode": "DirectCount"})
)
initial_counts = (
    direct_counts
    .sort_values("DaysBeforeDeparture", ascending=False)
    .groupby(["Route", "DepartureDate", "Year"])
    .first()
    .reset_index()[["Route", "DepartureDate", "Year", "DirectCount"]]
    .rename(columns={"DirectCount": "InitialDirectCount"})
)
direct_counts = direct_counts.merge(
    initial_counts, on=["Route", "DepartureDate", "Year"], how="left"
)
direct_counts["SoldOutDummy"] = (
    direct_counts["DirectCount"] < direct_counts["InitialDirectCount"]
).astype(int)
direct_counts["SoldOutIntensity"] = (
    1 - direct_counts["DirectCount"] / direct_counts["InitialDirectCount"]
).clip(lower=0)

sold_out = direct_counts[[
    "Route", "DepartureDate", "DaysBeforeDeparture", "Year",
    "SoldOutDummy", "SoldOutIntensity",
]]


# =============================================================================
# STEP 7 — MERGE INTO INDIRECT; DROP NO-BENCHMARK
# =============================================================================
df_indirect = df_indirect.merge(
    direct_avg, on=["Route", "DepartureDate", "DaysBeforeDeparture", "Year"], how="left"
)
df_indirect = df_indirect.merge(
    sold_out, on=["Route", "DepartureDate", "DaysBeforeDeparture", "Year"], how="left"
)
df_indirect = df_indirect.merge(
    initial_counts[["Route", "DepartureDate", "Year", "InitialDirectCount"]],
    on=["Route", "DepartureDate", "Year"], how="left"
)

before = len(df_indirect)
df_indirect = df_indirect.dropna(subset=["DirectPrice_avg"])
print(f"Dropped {before - len(df_indirect):,} indirect rows with no direct benchmark")

df_indirect["SoldOutDummy"]     = df_indirect["SoldOutDummy"].fillna(0).astype(int)
df_indirect["SoldOutIntensity"] = df_indirect["SoldOutIntensity"].fillna(0)


# =============================================================================
# STEP 8 — LOG-TRANSFORM REGRESSION VARIABLES
# =============================================================================
df_indirect["log_price"]               = np.log(df_indirect["price"])
df_indirect["log_DirectPrice_avg"]     = np.log(df_indirect["DirectPrice_avg"])
df_indirect["log_TravelTime"]          = np.log(df_indirect["TravelTimeMinutes"])
df_indirect["log_TravelTime_sq"]       = df_indirect["log_TravelTime"] ** 2
df_indirect["log_DirectPrice_x_Days"]  = (
    df_indirect["log_DirectPrice_avg"] * df_indirect["DaysBeforeDeparture"]
)

df_m = df_indirect.copy()
n2023 = (df_m.Year == 2023).sum()
n2026 = (df_m.Year == 2026).sum()
routes_both = int(df_m.groupby("Route")["Year"].nunique().eq(2).sum())
print(f"\nFinal merged dataset: {len(df_m):,} rows")
print(f"  2023: {n2023:,}  |  2026: {n2026:,}")
print(f"  Routes: {df_m['Route'].nunique()}  |  In both years: {routes_both}")
print(f"  SoldOut=1: 2023={df_m[df_m.Year==2023]['SoldOutDummy'].mean()*100:.1f}%  "
      f"2026={df_m[df_m.Year==2026]['SoldOutDummy'].mean()*100:.1f}%")

df_m.to_csv("final_merged.csv", index=False)
print("Saved: final_merged.csv")

# Strip year from DepartureDate so C(DepartureDay) + C(Year) are separately identified
# "15-4-2023" and "15/04/2026" both → "04-15"
# format="mixed" is required because the two years use different separators
df_m["DepartureDay"] = pd.to_datetime(
    df_m["DepartureDate"], format="mixed", dayfirst=True, errors="coerce"
).dt.strftime("%m-%d")
n_missing_day = df_m["DepartureDay"].isna().sum()
if n_missing_day:
    print(f"  Warning: {n_missing_day:,} rows with unparseable DepartureDate → DepartureDay")
print(f"  DepartureDay values: {sorted(df_m['DepartureDay'].dropna().unique())}")


# =============================================================================
# EXP 1 — SUBSTITUTION EFFECT AND LAMBDA_YEAR
# =============================================================================
print("\n" + "=" * 65)
print("EXP 1: Substitution Effect + lambda_year")
print("=" * 65)

ALL_ALPHA_VARS = [
    ("log_DirectPrice_avg",    "α1  log(DirectPrice_avg)"),
    ("log_DirectPrice_x_Days", "α2  log(DP)×Days"),
    ("log_TravelTime",         "α3  log(TravelTime)"),
    ("log_TravelTime_sq",      "α4  log(TravelTime)²"),
    ("SoldOutDummy",           "α5  SoldOutDummy"),
]

# Run year-by-year and pooled regressions
results_by_year = {}
for yr, label in [(2023, "2023"), (2026, "2026")]:
    sub = df_m[df_m.Year == yr].copy()
    res = cluster_ols(FORMULA_BASE, sub)
    results_by_year[label] = res
    print(f"\n  {label}  N={int(res.nobs):,}  R²={res.rsquared:.4f}")
    for v, lbl in ALL_ALPHA_VARS:
        c, se, t, p = coef_row(res, v)
        print(f"    {lbl}: {c:.4f}  SE={se:.4f}  p={p:.4f}  {sig(p)}")

# Pooled with Year FE — C(DepartureDay) keeps Year separately identified
print(f"\n  Note: pooled model uses C(DepartureDay) [year-stripped] + C(Year)")
print(f"  so λ_year captures the year-level price shift net of route/day confounders.")
res_pooled = cluster_ols(FORMULA_YEAR, df_m)
print(f"\n  Pooled  N={int(res_pooled.nobs):,}  R²={res_pooled.rsquared:.4f}")
for v, lbl in ALL_ALPHA_VARS:
    c, se, t, p = coef_row(res_pooled, v)
    print(f"    {lbl}: {c:.4f}  SE={se:.4f}  p={p:.4f}  {sig(p)}")

# lambda_year
ly_key = "C(Year)[T.2026]"
ly_c, ly_se, ly_t, ly_p = coef_row(res_pooled, ly_key)
print(f"\n  λ_year (2026 vs 2023): {ly_c:.4f}  SE={ly_se:.4f}  p={ly_p:.4f}  {sig(ly_p)}")
print(f"  → 2026 indirect fares are e^{ly_c:.4f}−1 = {(np.exp(ly_c)-1)*100:.1f}% higher than 2023")

# Save results
rows_exp1 = []
KEY_VARS = ["log_DirectPrice_avg", "log_DirectPrice_x_Days",
            "log_TravelTime", "log_TravelTime_sq", "SoldOutDummy"]
for label, res in [("2023", results_by_year["2023"]),
                   ("2026", results_by_year["2026"]),
                   ("pooled", res_pooled)]:
    for v in KEY_VARS:
        c, se, t, p = coef_row(res, v)
        if np.isnan(c): continue
        rows_exp1.append({"dataset": label, "variable": v, "coef": round(c, 6),
                          "std_err": round(se, 6), "t_stat": round(t, 4),
                          "p_value": round(p, 6), "sig": sig(p),
                          "n_obs": int(res.nobs), "r2": round(res.rsquared, 6)})
    if label == "pooled" and not np.isnan(ly_c):
        rows_exp1.append({"dataset": "pooled", "variable": "lambda_year",
                          "coef": round(ly_c, 6), "std_err": round(ly_se, 6),
                          "t_stat": round(ly_t, 4), "p_value": round(ly_p, 6),
                          "sig": sig(ly_p), "n_obs": int(res_pooled.nobs),
                          "r2": round(res_pooled.rsquared, 6)})
pd.DataFrame(rows_exp1).to_csv("results_merged_exp1.csv", index=False)

# ── Plot 1a: Year comparison of all 5 alphas ─────────────────────────────
PLOT_VARS = [v for v, _ in ALL_ALPHA_VARS]
VLABELS   = [lbl for _, lbl in ALL_ALPHA_VARS]
datasets  = ["2023", "2026", "pooled"]
colors    = [PALETTE["2023"], PALETTE["2026"], PALETTE["pooled"]]
res_map   = {"2023": results_by_year["2023"],
             "2026": results_by_year["2026"],
             "pooled": res_pooled}

x  = np.arange(len(PLOT_VARS))
w  = 0.22
fig, ax = plt.subplots(figsize=(15, 5))
for i, (ds, col) in enumerate(zip(datasets, colors)):
    coefs = []
    errs  = []
    for v in PLOT_VARS:
        c, se, *_ = coef_row(res_map[ds], v)
        coefs.append(c)
        errs.append(se * 1.96)
    bars = ax.bar(x + (i - 1) * w, coefs, w, yerr=errs, capsize=4,
                  color=col, alpha=0.88, label=ds.capitalize(),
                  error_kw={"elinewidth": 1.2})
    sigs_ = [sig(res_map[ds].pvalues[v]) if v in res_map[ds].pvalues else ""
             for v in PLOT_VARS]
    for bar, c, e, s in zip(bars, coefs, errs, sigs_):
        yt = c + e + 0.003 if c >= 0 else c - e - 0.012
        ax.text(bar.get_x() + bar.get_width() / 2, yt, s,
                ha="center", va="bottom", fontsize=8, color=col)

ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xticks(x)
ax.set_xticklabels(VLABELS, fontsize=9)
ax.set_ylabel("Coefficient (log-log scale)", fontsize=10)
ax.legend(fontsize=10, framealpha=0.9)
ax.set_title("Exp 1: All Coefficients (α1–α5) — 2023 vs 2026 vs Pooled\n"
             "Year effect captured by λ_year in pooled model  |  "
             "95% CI  |  SEs clustered at route level", fontsize=11)
ax.grid(axis="y", linestyle=":", alpha=0.4)
fig.tight_layout()
fig.savefig("plots_merged/01a_substitution_year_comparison.png", dpi=150)
plt.close()
print("  Plot: plots_merged/01a_substitution_year_comparison.png (all α1–α5)")

# ── Plot 1b: lambda_year with context ─────────────────────────────────────
# Per-route mean log-price shift (2026 − 2023) to characterise lambda_year
route_yr = (
    df_m.groupby(["Route", "Year"])["log_price"]
    .mean()
    .unstack("Year")
    .dropna()
)
if 2023 in route_yr.columns and 2026 in route_yr.columns:
    route_yr["delta"] = route_yr[2026] - route_yr[2023]
    route_yr = route_yr.sort_values("delta")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: horizontal bar — per-route year shift
    ax = axes[0]
    ypos = np.arange(len(route_yr))
    colors_bar = [PALETTE["2026"] if d >= 0 else PALETTE["2023"] for d in route_yr["delta"]]
    ax.barh(ypos, route_yr["delta"], color=colors_bar, alpha=0.8, height=0.7)
    ax.axvline(ly_c, color="black", linewidth=1.5, linestyle="--",
               label=f"λ_year = {ly_c:.3f}")
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_yticks(ypos[::max(1, len(route_yr)//20)])
    ax.set_yticklabels(route_yr.index[::max(1, len(route_yr)//20)], fontsize=7)
    ax.set_xlabel("Δ log(IndirectPrice)  (2026 − 2023)", fontsize=10)
    ax.set_title("Per-Route Year Effect\n(raw mean log-price shift)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    # Right: histogram of route-level deltas
    ax = axes[1]
    ax.hist(route_yr["delta"], bins=20, color=PALETTE["2026"], alpha=0.8, edgecolor="white")
    ax.axvline(ly_c, color="black", linewidth=1.8, linestyle="--",
               label=f"λ_year (Year FE) = {ly_c:.3f}")
    ax.axvline(route_yr["delta"].mean(), color=PALETTE["2023"], linewidth=1.5,
               linestyle=":", label=f"Mean route shift = {route_yr['delta'].mean():.3f}")
    ax.set_xlabel("Δ log(IndirectPrice)  (2026 − 2023)", fontsize=10)
    ax.set_ylabel("Number of routes", fontsize=10)
    ax.set_title("Distribution of Year-Level Price Shifts\nAcross Common Routes", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(linestyle=":", alpha=0.4)

    fig.suptitle(f"Exp 1: Characterising λ_year = {ly_c:.4f}  "
                 f"(≈ {(np.exp(ly_c)-1)*100:.1f}% price increase 2023→2026)\n"
                 f"Based on {len(route_yr)} common routes", fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig("plots_merged/01b_lambda_year.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Plot: plots_merged/01b_lambda_year.png")


# =============================================================================
# EXP 3 — EVENT STUDY BY YEAR
# =============================================================================
print("\n" + "=" * 65)
print("EXP 3: Event Study — Merged (by Year)")
print("=" * 65)

def find_event_day(grp):
    grp = grp.sort_values("DaysBeforeDeparture", ascending=False)
    sold = grp["SoldOutDummy"].values
    days = grp["DaysBeforeDeparture"].values
    for i in range(1, len(sold)):
        if sold[i - 1] == 0 and sold[i] == 1:
            return days[i]
    return np.nan

WINDOW = 7
ev_all   = []
ctrl_all = []

for yr in [2023, 2026]:
    sub = df_m[df_m.Year == yr].copy()

    event_days_yr = (
        sub.groupby(["Route", "DepartureDate"])
           .apply(find_event_day, include_groups=False)
           .reset_index()
           .rename(columns={0: "EventDay"})
    )
    sub = sub.merge(event_days_yr, on=["Route", "DepartureDate"], how="left")
    sub["Treated"] = sub["EventDay"].notna()

    df_treated = sub[sub["Treated"]].copy()
    df_control = sub[sub["SoldOutDummy"] == 0].copy()

    n_t = int(sub.groupby(["Route", "DepartureDate"])["Treated"].first().sum())
    n_c = int((~sub.groupby(["Route", "DepartureDate"])["SoldOutDummy"].any()).sum())
    print(f"  {yr}: treated route-dates={n_t}  |  control route-dates={n_c}")

    df_treated["RelDay"] = df_treated["DaysBeforeDeparture"] - df_treated["EventDay"]
    pre_means = (
        df_treated[df_treated["RelDay"] > 0]
        .groupby(["Route", "DepartureDate"])["log_price"]
        .mean()
        .rename("PreMean")
    )
    df_treated = df_treated.merge(pre_means, on=["Route", "DepartureDate"], how="left")
    df_treated["NormLogPrice"] = df_treated["log_price"] - df_treated["PreMean"]

    ev_yr = (
        df_treated[df_treated["RelDay"].between(-WINDOW, WINDOW)]
        .groupby("RelDay")
        .agg(MeanNormLogPrice=("NormLogPrice", "mean"), N=("NormLogPrice", "count"))
        .reset_index()
    )
    ev_yr["group"] = str(yr)
    ev_all.append(ev_yr)

    # Control group
    pseudo_event = int(event_days_yr["EventDay"].dropna().median())
    df_control["RelDay"] = df_control["DaysBeforeDeparture"] - pseudo_event
    pre_ctrl = (
        df_control[df_control["RelDay"] > 0]
        .groupby(["Route", "DepartureDate"])["log_price"]
        .mean()
        .rename("PreMean")
    )
    df_control = df_control.merge(pre_ctrl, on=["Route", "DepartureDate"], how="left")
    df_control["NormLogPrice"] = df_control["log_price"] - df_control["PreMean"]
    ctrl_yr = (
        df_control[df_control["RelDay"].between(-WINDOW, WINDOW)]
        .groupby("RelDay")
        .agg(MeanNormLogPrice=("NormLogPrice", "mean"), N=("NormLogPrice", "count"))
        .reset_index()
    )
    ctrl_yr["group"] = f"control_{yr}"
    ctrl_all.append(ctrl_yr)

    at_ev = ev_yr.loc[ev_yr["RelDay"] == 0, "MeanNormLogPrice"].values
    if len(at_ev):
        print(f"    At event day (RelDay=0): {at_ev[0]:.4f}  "
              f"≈ {(np.exp(at_ev[0])-1)*100:.1f}% price change")

ev_df   = pd.concat(ev_all, ignore_index=True)
ctrl_df = pd.concat(ctrl_all, ignore_index=True)
pd.concat([ev_df, ctrl_df], ignore_index=True).to_csv(
    "results_merged_event_study.csv", index=False
)

fig, ax = plt.subplots(figsize=(11, 5))
for yr, col, ls, mk in [(2023, PALETTE["2023"], "-", "o"), (2026, PALETTE["2026"], "-", "s")]:
    ev_yr  = ev_df[ev_df["group"] == str(yr)]
    ctrl_yr = ctrl_df[ctrl_df["group"] == f"control_{yr}"]
    ax.plot(ev_yr["RelDay"], ev_yr["MeanNormLogPrice"],
            color=col, linewidth=2.2, marker=mk, markersize=5,
            linestyle=ls, label=f"Treated {yr}")
    ax.plot(ctrl_yr["RelDay"], ctrl_yr["MeanNormLogPrice"],
            color=col, linewidth=1.5, marker=mk, markersize=4,
            linestyle="--", alpha=0.5, label=f"Control {yr}")

ax.axvline(0, color="black", linewidth=1.2, linestyle="--", label="Event day (sell-out)")
ax.axhline(0, color="grey", linewidth=0.7, linestyle=":")
ax.fill_between(range(-WINDOW, 1), -0.3, 0.3, alpha=0.04, color="grey")
ax.set_xlabel("Days relative to sell-out event  (0 = first day direct disappears)", fontsize=10)
ax.set_ylabel("Mean normalised log(IndirectPrice)\n(relative to pre-event mean)", fontsize=10)
ax.set_title("Exp 3: Event Study — Indirect Price Response to Sell-Out\nMerged 2023 + 2026",
             fontsize=11)
ax.legend(fontsize=9, framealpha=0.9, ncol=2)
ax.set_xlim(-WINDOW - 0.5, WINDOW + 0.5)
ax.set_xticks(range(-WINDOW, WINDOW + 1))
ax.set_xticklabels([str(x) if x % 2 == 0 else "" for x in range(-WINDOW, WINDOW + 1)])
ax.grid(linestyle=":", alpha=0.4)
fig.tight_layout()
fig.savefig("plots_merged/03_event_study_merged.png", dpi=150)
plt.close()
print("  Plot: plots_merged/03_event_study_merged.png")


# =============================================================================
# EXP 4 — LCC HETEROGENEITY BY YEAR
# =============================================================================
print("\n" + "=" * 65)
print("EXP 4: LCC Heterogeneity by Year")
print("=" * 65)

df_lcc = df_m.dropna(subset=["LCCdum"]).copy()
df_lcc["LCCdum"] = df_lcc["LCCdum"].astype(int)
print(f"  N (with LCCdum): {len(df_lcc):,}  "
      f"|  LCC=1: {df_lcc['LCCdum'].sum():,} ({df_lcc['LCCdum'].mean()*100:.1f}%)")

FORMULA_LCC_YEAR = (
    "log_price ~ "
    "log_DirectPrice_avg + log_DirectPrice_x_Days + "
    "log_TravelTime + log_TravelTime_sq + SoldOutDummy + "
    "LCCdum + LCCdum:log_DirectPrice_avg + LCCdum:SoldOutDummy + "
    "C(Year) + C(Route) + C(DepartureDate)"
)

# Run per-year and pooled LCC models
lcc_res = {}
for yr_lbl, sub in [("2023", df_lcc[df_lcc.Year == 2023]),
                     ("2026", df_lcc[df_lcc.Year == 2026]),
                     ("pooled", df_lcc)]:
    if len(sub) < 200:
        print(f"  {yr_lbl}: too few rows, skipping")
        continue
    formula = FORMULA_LCC_YEAR if yr_lbl == "pooled" else (
        "log_price ~ "
        "log_DirectPrice_avg + log_DirectPrice_x_Days + "
        "log_TravelTime + log_TravelTime_sq + SoldOutDummy + "
        "LCCdum + LCCdum:log_DirectPrice_avg + LCCdum:SoldOutDummy + "
        "C(Route) + C(DepartureDate)"
    )
    res = smf.ols(formula, data=sub).fit(
        cov_type="cluster", cov_kwds={"groups": sub["Route"]}
    )
    lcc_res[yr_lbl] = res

    a1   = res.params.get("log_DirectPrice_avg", 0)
    b2k  = ("log_DirectPrice_avg:LCCdum" if "log_DirectPrice_avg:LCCdum" in res.params
            else "LCCdum:log_DirectPrice_avg")
    b2   = res.params.get(b2k, 0)
    a5   = res.params.get("SoldOutDummy", 0)
    b3k  = "SoldOutDummy:LCCdum" if "SoldOutDummy:LCCdum" in res.params else "LCCdum:SoldOutDummy"
    b3   = res.params.get(b3k, 0)

    print(f"\n  {yr_lbl}  N={int(res.nobs):,}  R²={res.rsquared:.4f}")
    print(f"    α1 non-LCC: {a1:.4f}  |  LCC: {a1+b2:.4f}  (β2={b2:.4f}  {sig(res.pvalues.get(b2k, 1))})")
    print(f"    α5 non-LCC: {a5:.4f}  |  LCC: {a5+b3:.4f}  (β3={b3:.4f}  {sig(res.pvalues.get(b3k, 1))})")

# ── Plot 4: 2×2 LCC comparison by year ──────────────────────────────────────
if len(lcc_res) >= 2:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    group_labels = ["Non-LCC", "LCC"]
    years_plot   = [k for k in ["2023", "2026"] if k in lcc_res]
    bar_colors   = [PALETTE["2023"], PALETTE["2026"]]
    x            = np.arange(2)
    width        = 0.32

    for ax_idx, (metric_key, metric_title, y_lbl) in enumerate([
        ("alpha1", "Substitution Elasticity (α1)\nNon-LCC vs LCC by Year",
         "Coef on log(DirectPrice_avg)"),
        ("alpha5", "Scarcity Premium (α5)\nNon-LCC vs LCC by Year",
         "Coef on SoldOutDummy"),
    ]):
        ax = axes[ax_idx]
        for i, (yr_lbl, col) in enumerate(zip(years_plot, bar_colors)):
            res = lcc_res[yr_lbl]
            a1  = res.params.get("log_DirectPrice_avg", 0)
            b2k = ("log_DirectPrice_avg:LCCdum" if "log_DirectPrice_avg:LCCdum" in res.params
                   else "LCCdum:log_DirectPrice_avg")
            b2  = res.params.get(b2k, 0)
            a5  = res.params.get("SoldOutDummy", 0)
            b3k = "SoldOutDummy:LCCdum" if "SoldOutDummy:LCCdum" in res.params else "LCCdum:SoldOutDummy"
            b3  = res.params.get(b3k, 0)

            vals = [a1, a1 + b2] if metric_key == "alpha1" else [a5, a5 + b3]
            se1  = res.bse.get("log_DirectPrice_avg", 0)
            se2k = "log_DirectPrice_avg:LCCdum" if "log_DirectPrice_avg:LCCdum" in res.bse else "LCCdum:log_DirectPrice_avg"
            se2  = res.bse.get(se2k, 0)
            se5  = res.bse.get("SoldOutDummy", 0)
            se3k = "SoldOutDummy:LCCdum" if "SoldOutDummy:LCCdum" in res.bse else "LCCdum:SoldOutDummy"
            se3  = res.bse.get(se3k, 0)
            # approximate SE for combined (ignores covariance — illustrative)
            errs = ([se1 * 1.96, np.sqrt(se1**2 + se2**2) * 1.96]
                    if metric_key == "alpha1"
                    else [se5 * 1.96, np.sqrt(se5**2 + se3**2) * 1.96])

            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, vals, width, yerr=errs, capsize=4,
                          color=col, alpha=0.85, label=yr_lbl,
                          error_kw={"elinewidth": 1.2})
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        v + (0.004 if v >= 0 else -0.012),
                        f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top",
                        fontsize=8.5, fontweight="bold", color=col)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(group_labels, fontsize=10)
        ax.set_ylabel(y_lbl, fontsize=9)
        ax.set_title(metric_title, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.suptitle("Exp 4: LCC vs Non-LCC Price Mechanisms — 2023 vs 2026\n"
                 "95% CI bars  |  SEs clustered at route level", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig("plots_merged/04_lcc_by_year.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Plot: plots_merged/04_lcc_by_year.png")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 65)
print("ALL MERGED ANALYSES COMPLETE")
print(f"  lambda_year = {ly_c:.4f}  (SE={ly_se:.4f}  p={ly_p:.4f}  {sig(ly_p)})")
print(f"  Interpretation: 2026 indirect fares are "
      f"{(np.exp(ly_c)-1)*100:.1f}% higher than 2023 on identical routes/dates")
print(f"  Plots saved to: plots_merged/")
print(f"  Data saved to:  final_merged.csv")
print("=" * 65)
