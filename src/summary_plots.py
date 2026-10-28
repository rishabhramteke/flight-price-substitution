"""
summary_plots.py
---------------
All summary plots in one place. Reads pre-computed CSVs — no regression re-runs.

Required inputs — generate with:
    python3 preprocess.py --dataset merged   →  final_merged.csv
    python3 regression.py --dataset merged           →  results_merged.csv

Run:
    python3 summary_plots.py

Outputs → figures/summary/
    01_coefficients.png       α1–α5 bar chart: 2023 vs 2026 vs pooled
    02_booking_horizon.png    implied elasticity α1+α2·d over the booking horizon
    03_price_distributions.png direct vs indirect price histograms
    04_soldout_effect.png     median price by SoldOut=0/1 per year
    05_traveltime.png         travel time vs price with model-implied curve
    06_lambda_year.png        per-route year shift distribution vs λ_year
    07_direct_indirect_scatter.png  direct vs indirect price correlation
    08_soldout_rate_horizon.png     sell-out rate over the booking horizon
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.makedirs("figures/summary", exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
# To change year colours, edit these values. Keys match dataset labels in CSVs.
COLORS = {
    "2023":   "#2196F3",   # blue
    "2026":   "#FF5722",   # deep orange
    "pooled": "#4CAF50",   # green
}

ALPHA_VARS = [
    ("log_DirectPrice_avg",    "α1"),
    ("log_DirectPrice_x_Days", "α2  DP×Days"),
    ("log_TravelTime",         "α3"),
    ("log_TravelTime_sq",      "α4  TT²"),
    ("SoldOutDummy",           "α5  SoldOut"),
]


def load(path, required=True):
    if not os.path.exists(path):
        msg = f"Missing: {path}"
        if required:
            raise FileNotFoundError(msg + " — run the required script first (see docstring)")
        print(f"  Warning: {msg}, skipping dependent plots")
        return None
    df = pd.read_csv(path)
    print(f"  Loaded {path}: {len(df):,} rows")
    return df


def get(res, dataset, var, col="coef"):
    """Look up one value from the results dataframe by dataset + variable name."""
    row = res[(res["dataset"] == dataset) & (res["variable"] == var)]
    return row.iloc[0][col] if not row.empty else np.nan


def save(fname):
    plt.savefig(f"figures/summary/{fname}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: figures/summary/{fname}")


# ── Plot 1: Coefficient comparison ───────────────────────────────────────────
# Shows all α1–α5 side by side for each dataset — the main results summary.
# Fine-tuning:
#   w = bar width (default 0.25 — reduce if labels overlap, increase for fatter bars)
#   alpha = bar transparency (0–1)
#   fontsize in ax.set_xticklabels = tick label size
#   error bars: capsize = cap width; elinewidth in error_kw = line thickness

def plot_coefficients(res):
    vars_  = [v for v, _ in ALPHA_VARS]
    labels = [l for _, l in ALPHA_VARS]
    x, w   = np.arange(len(vars_)), 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, ds in enumerate(["2023", "2026", "pooled"]):
        coefs = [get(res, ds, v) for v in vars_]
        errs  = [get(res, ds, v, "std_err") * 1.96 for v in vars_]
        sigs  = [get(res, ds, v, "sig") for v in vars_]
        bars  = ax.bar(x + (i-1)*w, coefs, w, yerr=errs, capsize=3,
                       color=COLORS[ds], alpha=0.85, label=ds,
                       error_kw={"elinewidth": 1.2})
        for bar, c, e, s in zip(bars, coefs, errs, sigs):
            if np.isnan(c): continue
            yt = c + e + 0.003 if c >= 0 else c - e - 0.015
            ax.text(bar.get_x() + bar.get_width()/2, yt, s or "",
                    ha="center", va="bottom", fontsize=7, color=COLORS[ds])
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Coefficient")
    ax.set_title("Regression Coefficients α1–α5 — 2023 vs 2026 vs Pooled\n"
                 "95% CI  |  SEs clustered at route level")
    ax.legend()
    ax.grid(axis="y", ls=":", alpha=0.4)
    save("01_coefficients.png")


# ── Plot 2: Booking horizon curve ─────────────────────────────────────────────
# Left: α1+α2·d — the model's answer to "how does elasticity evolve over time?"
# Right: raw observed mean log_price per day before departure.
# Fine-tuning:
#   day_range = horizon range (0–36 by default, change upper bound if data differs)
#   lw = line width; ls = line style ("--" for pooled, "-" for year-specific)
#   invert_xaxis() makes x go far-out→last-minute (left→right); remove to reverse

def plot_booking_horizon(res, df):
    day_range = np.arange(0, 37)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for ds in ["2023", "2026", "pooled"]:
        a1, a2 = get(res, ds, "log_DirectPrice_avg"), get(res, ds, "log_DirectPrice_x_Days")
        if np.isnan(a1): continue
        ax.plot(day_range, a1 + a2*day_range, color=COLORS[ds], lw=2,
                ls="--" if ds == "pooled" else "-",
                label=f"{ds}  α1={a1:.3f}  α2={a2:.4f}")
    ax.axhline(0, color="black", lw=0.7, ls=":")
    ax.set_xlabel("Days Before Departure")
    ax.set_ylabel("∂ log(IndirectPrice) / ∂ log(DirectPrice)")
    ax.set_title("Implied Substitution Elasticity vs Booking Horizon")
    ax.legend(fontsize=9)
    ax.grid(ls=":", alpha=0.4)
    ax.invert_xaxis()

    ax = axes[1]
    for yr in sorted(df["Year"].unique()):
        sub = df[df["Year"] == yr]
        m   = sub.groupby("DaysBeforeDeparture")["log_price"].mean()
        ax.plot(m.index, m.values, color=COLORS[str(yr)], lw=1.8,
                label=f"{yr} (N={len(sub):,})")
    ax.set_xlabel("Days Before Departure")
    ax.set_ylabel("Mean log(IndirectPrice)")
    ax.set_title("Observed Mean Price by Booking Horizon\n(raw, not residualised)")
    ax.legend(fontsize=9)
    ax.grid(ls=":", alpha=0.4)
    ax.invert_xaxis()

    fig.suptitle("Booking Horizon Dynamics — 2023 vs 2026", fontsize=12)
    save("02_booking_horizon.png")


# ── Plot 3: Price distributions ───────────────────────────────────────────────
# Histograms of indirect and direct prices per year.
# Fine-tuning:
#   bins = number of histogram bins (default 61)
#   alpha = transparency of each year's bars (lower = more see-through, helps overlap)
#   density=True → y-axis is density not count; set to False for raw counts

def plot_price_distributions(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, col, title in [
        (axes[0], "price",           "Indirect Flight Price (€)"),
        (axes[1], "DirectPrice_avg", "Direct Flight Avg Price (€)"),
    ]:
        for yr in sorted(df["Year"].unique()):
            sub  = df[df["Year"] == yr][col].dropna()
            bins = np.linspace(sub.quantile(0.01), sub.quantile(0.99), 61)
            ax.hist(sub, bins=bins, alpha=0.55, color=COLORS[str(yr)],
                    edgecolor="white", lw=0.3, label=str(yr), density=True)
        ax.set_xlabel(title)
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", ls=":", alpha=0.4)
    fig.suptitle("Price Distributions — 2023 vs 2026", fontsize=13)
    save("03_price_distributions.png")


# ── Plot 4: SoldOut effect ────────────────────────────────────────────────────
# Median indirect price when direct flights are available vs sold out, per year.
# Fine-tuning:
#   bar colors: first element = SoldOut=0 color, second = SoldOut=1 color
#   width = bar width
#   ylim multiplier 1.2 = headroom above highest bar for value labels

def plot_soldout_effect(df):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, yr in zip(axes, sorted(df["Year"].unique())):
        sub  = df[df["Year"] == yr]
        grp  = sub.groupby("SoldOutDummy")["price"].median()
        v0, v1 = grp.get(0, np.nan), grp.get(1, np.nan)
        ax.bar(["Direct available\n(SoldOut=0)", "Direct sold out\n(SoldOut=1)"],
               [v0, v1], color=["#90CAF9", "#F44336"], width=0.5, edgecolor="white")
        for x_, v in enumerate([v0, v1]):
            ax.text(x_, v + 3, f"€{v:.0f}", ha="center", va="bottom",
                    fontsize=11, fontweight="bold")
        ax.set_title(f"{yr}  (+{(v1-v0)/v0*100:.1f}% when sold out)")
        ax.set_ylabel("Median indirect price (€)")
        ax.set_ylim(0, max(v0, v1) * 1.2)
        ax.grid(axis="y", ls=":", alpha=0.4)
    fig.suptitle("Indirect Price by Direct Flight Availability (α5 effect)", fontsize=13)
    save("04_soldout_effect.png")


# ── Plot 5: Travel time vs price ──────────────────────────────────────────────
# Observed binned medians (scatter) + model-implied curve (twin y-axis).
# Fine-tuning:
#   bins=12 → number of travel-time bins for the scatter; more bins = finer granularity
#   quantile(0.02/0.98) → trims outlier travel times; adjust to include/exclude extremes
#   twinx creates a secondary y-axis for the model curve; remove ax2 lines for single axis

def plot_traveltime(res, df):
    a3 = get(res, "pooled", "log_TravelTime")
    a4 = get(res, "pooled", "log_TravelTime_sq")
    df["TT_bin"] = pd.cut(df["TravelTimeMinutes"], bins=12)
    tt_grp = (df.groupby("TT_bin", observed=True)["price"].median()
                .reset_index())
    tt_grp["mid"] = tt_grp["TT_bin"].apply(lambda x: x.mid).astype(float)

    tt_range = np.linspace(df["TravelTimeMinutes"].quantile(0.02),
                           df["TravelTimeMinutes"].quantile(0.98), 200)
    log_tt, mid_log = np.log(tt_range), np.log(df["TravelTimeMinutes"].median())
    implied = df["price"].median() * np.exp(
        a3*(log_tt - mid_log) + a4*(log_tt**2 - mid_log**2)
    )

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.scatter(tt_grp["mid"]/60, tt_grp["price"], color="#9C27B0",
               s=80, zorder=5, label="Median observed price (binned)")
    ax2 = ax.twinx()
    ax2.plot(tt_range/60, implied, color=COLORS["pooled"], lw=2, alpha=0.8,
             label=f"Model curve  (α3={a3:.3f}, α4={a4:.3f})")
    ax2.set_ylabel("Model-implied price (€)", color=COLORS["pooled"])
    ax2.tick_params(axis="y", labelcolor=COLORS["pooled"])
    ax.set_xlabel("Travel Time (hours)")
    ax.set_ylabel("Median indirect price (€)")
    ax.set_title("Travel Time vs Indirect Price — Pooled Dataset\n(α3/α4 illustration)")
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1+l2, lb1+lb2, fontsize=9)
    ax.grid(ls=":", alpha=0.4)
    save("05_traveltime.png")


# ── Plot 6: λ_year distribution ───────────────────────────────────────────────
# Histogram of raw per-route mean log-price shift (2026−2023), with λ_year overlaid.
# Shows that almost all routes have a positive year effect and how spread it is.
# Fine-tuning:
#   bins=20 → number of histogram bins; increase for more granularity
#   vertical lines: black dashed = Year FE estimate, blue dotted = simple mean of route shifts

def plot_lambda_year(res, df):
    ly = get(res, "pooled", "C(Year)[T.2026]")
    route_yr = (df.groupby(["Route", "Year"])["log_price"].mean()
                  .unstack("Year").dropna())
    if 2023 not in route_yr.columns or 2026 not in route_yr.columns:
        print("  Warning: cannot compute route shifts, skipping plot 07")
        return
    route_yr["delta"] = route_yr[2026] - route_yr[2023]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(route_yr["delta"], bins=20, color=COLORS["2026"], alpha=0.8, edgecolor="white")
    if not np.isnan(ly):
        ax.axvline(ly, color="black", lw=2, ls="--",
                   label=f"λ_year (Year FE) = {ly:.3f}")
    ax.axvline(route_yr["delta"].mean(), color=COLORS["2023"], lw=1.5, ls=":",
               label=f"Mean route shift = {route_yr['delta'].mean():.3f}")
    ax.set_xlabel("Δ log(IndirectPrice)  (2026 − 2023)")
    ax.set_ylabel("Number of routes")
    pct = (np.exp(ly)-1)*100 if not np.isnan(ly) else float("nan")
    ax.set_title(f"Per-Route Year-Level Price Shifts\n"
                 f"λ_year ≈ {pct:.1f}% structural price increase 2023→2026")
    ax.legend(fontsize=9)
    ax.grid(ls=":", alpha=0.4)
    save("06_lambda_year.png")


# ── Plot 8: Direct vs indirect price scatter ──────────────────────────────────
# Useful to visually confirm the price linkage.
# Points coloured by DaysBeforeDeparture — shows if the relationship shifts over the horizon.
# Fine-tuning:
#   n_sample = number of points to plot (too many = slow rendering; 6000 is a good balance)
#   alpha and s control point transparency and size
#   cmap = matplotlib colormap for DaysBeforeDeparture gradient; try "viridis" or "plasma"

def plot_direct_indirect_scatter(df):
    sample = (df.dropna(subset=["log_DirectPrice_avg", "log_price"])
                .sample(n=min(6000, len(df)), random_state=42))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sc = ax.scatter(sample["log_DirectPrice_avg"], sample["log_price"],
                    c=sample["DaysBeforeDeparture"], cmap="RdYlBu_r",
                    alpha=0.35, s=8, linewidths=0)
    cb = fig.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label("Days Before Departure", fontsize=10)
    m, b = np.polyfit(df["log_DirectPrice_avg"], df["log_price"], 1)
    x_line = np.linspace(df["log_DirectPrice_avg"].min(), df["log_DirectPrice_avg"].max(), 200)
    ax.plot(x_line, m*x_line + b, color="black", lw=2, label=f"OLS  slope={m:.3f}")
    ax.set_xlabel("log(DirectPrice_avg)")
    ax.set_ylabel("log(IndirectPrice)")
    ax.set_title("Direct vs Indirect Price Correlation — Pooled\n"
                 "Colour = Days Before Departure (red = far out, blue = near departure)")
    ax.legend(fontsize=10)
    ax.grid(ls=":", alpha=0.3)
    save("07_direct_indirect_scatter.png")


# ── Plot 9: Sell-out rate over the booking horizon ────────────────────────────
# Shows WHEN during the booking window direct flights tend to sell out.
# Complements Plot 6 (event study) by showing the distribution of event days.
# Fine-tuning:
#   groupby("DaysBeforeDeparture")["SoldOutDummy"].mean() → fraction of obs where SoldOut=1
#   lw and marker per year; invert_xaxis() so far-out is on the left

def plot_soldout_rate_horizon(df):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for yr in sorted(df["Year"].unique()):
        sub  = df[df["Year"] == yr]
        rate = sub.groupby("DaysBeforeDeparture")["SoldOutDummy"].mean()
        ax.plot(rate.index, rate.values * 100, color=COLORS[str(yr)], lw=2,
                label=f"{yr}  (overall {sub['SoldOutDummy'].mean()*100:.1f}%)")
    ax.set_xlabel("Days Before Departure")
    ax.set_ylabel("% of observations with SoldOutDummy = 1")
    ax.set_title("Direct Flight Sell-Out Rate Over the Booking Horizon\n"
                 "High near departure = flights sell out as the date approaches")
    ax.legend(fontsize=10)
    ax.grid(ls=":", alpha=0.4)
    ax.invert_xaxis()
    save("08_soldout_rate_horizon.png")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Loading inputs...")
    res = load("results_merged.csv")
    df  = load("final_merged.csv")

    print("\nGenerating plots → figures/summary/")
    plot_coefficients(res)
    plot_booking_horizon(res, df)
    plot_price_distributions(df)
    plot_soldout_effect(df)
    plot_traveltime(res, df)
    plot_lambda_year(res, df)
    plot_direct_indirect_scatter(df)
    plot_soldout_rate_horizon(df)
    print("\nDone. All plots saved to figures/summary/")


if __name__ == "__main__":
    main()
