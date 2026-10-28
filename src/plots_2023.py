"""
plots_v1.py
-----------
Visualisations for the 2023 regression experiments.
Reads final_2023.csv and results CSVs produced by regression.py.

Usage:
    python3 plots_v1.py
Outputs:
    plots/01_coef_comparison.png
    plots/02_price_distributions.png
    plots/03_substitution_elasticity_over_time.png
    plots/04_soldout_price_effect.png
    plots/05_traveltime_price_curve.png
    plots/06_direct_vs_indirect_scatter.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

os.makedirs("plots", exist_ok=True)

PALETTE = {
    "full":    "#2196F3",   # blue  — full window
    "window11":"#FF5722",   # deep orange — 11-day window
    "direct":  "#4CAF50",   # green
    "indirect":"#9C27B0",   # purple
    "sold_yes":"#F44336",   # red
    "sold_no": "#90CAF9",   # light blue
    "neutral": "#607D8B",   # blue-grey
}

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("final_2023.csv")

res_full = pd.read_csv("results_2023_full.csv").set_index("variable")
res_11   = pd.read_csv("results_2023_11.csv").set_index("variable")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Coefficient comparison: full window vs 11-day window
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 1: coefficient comparison...")

KEY_VARS  = ["log_DirectPrice_avg", "log_DirectPrice_x_Days",
             "log_TravelTime", "log_TravelTime_sq", "SoldOutDummy"]
TICK_LABELS = [
    "α1  log(DirectPrice)", "α2  log(DirectPrice)×Days",
    "α3  log(TravelTime)", "α4  log(TravelTime)²", "α5  SoldOutDummy"
]

coef_full = [res_full.loc[v, "coef"] for v in KEY_VARS]
err_full  = [res_full.loc[v, "std_err"]*1.96 for v in KEY_VARS]
coef_11   = [res_11.loc[v, "coef"] for v in KEY_VARS]
err_11    = [res_11.loc[v, "std_err"]*1.96 for v in KEY_VARS]

x = np.arange(len(KEY_VARS))
w = 0.32

fig, ax = plt.subplots(figsize=(11, 5.5))
bars1 = ax.bar(x - w/2, coef_full, w, yerr=err_full, capsize=4,
               color=PALETTE["full"],     label="2023 full (1–36 days)",  alpha=0.88)
bars2 = ax.bar(x + w/2, coef_11,   w, yerr=err_11,   capsize=4,
               color=PALETTE["window11"], label="2023 window (1–11 days)", alpha=0.88)

ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xticks(x)
ax.set_xticklabels(TICK_LABELS, fontsize=10)
ax.set_ylabel("Coefficient estimate", fontsize=11)
ax.set_title("Regression Coefficients: Full vs 11-Day Booking Window (2023)\n"
             "Error bars = 95% CI  |  Std errors clustered at route level", fontsize=12)
ax.legend(fontsize=10, framealpha=0.9)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
ax.grid(axis="y", linestyle=":", alpha=0.5)
fig.tight_layout()
fig.savefig("plots/01_coef_comparison.png", dpi=150)
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Price distribution: direct vs indirect
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 2: price distributions...")

# Load raw 2023 to get direct prices — use final_2023 for indirect only
# DirectPrice_avg column is the mean direct price per route/date/day
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

for ax, col, color, label in [
    (axes[0], "price",           PALETTE["indirect"], "Indirect flight price (€)"),
    (axes[1], "DirectPrice_avg", PALETTE["direct"],   "Direct flight avg price (€)"),
]:
    data = df[col].dropna()
    bin_edges = np.linspace(data.min(), data.quantile(0.99), 61)
    ax.hist(data.values, bins=bin_edges, color=color, edgecolor="white", linewidth=0.3, alpha=0.85)
    ax.axvline(data.median(), color="black",  linestyle="--", linewidth=1.5, label=f"Median €{data.median():.0f}")
    ax.axvline(data.mean(),   color="#FF9800", linestyle=":",  linewidth=1.5, label=f"Mean €{data.mean():.0f}")
    ax.set_xlabel(label, fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(label, fontsize=11)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_xlim(0, None)

fig.suptitle("Price Distributions — 2023 Dataset (full window)", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig("plots/02_price_distributions.png", dpi=150, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Substitution elasticity over booking horizon (implied from model)
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 3: elasticity over booking horizon...")

days_range = np.arange(1, 37)

a1_full = res_full.loc["log_DirectPrice_avg",    "coef"]
a2_full = res_full.loc["log_DirectPrice_x_Days", "coef"]
a1_11   = res_11.loc["log_DirectPrice_avg",    "coef"]
a2_11   = res_11.loc["log_DirectPrice_x_Days", "coef"]

elast_full = a1_full + a2_full * days_range
elast_11   = a1_11   + a2_11   * np.clip(days_range, 1, 11)

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(days_range, elast_full, color=PALETTE["full"],
        linewidth=2.2, label="Full window model (1–36 days)")
ax.plot(np.arange(1, 12), elast_full[:11], color=PALETTE["full"],
        linewidth=4, alpha=0.35)  # highlight shared zone

# 11-day model: only valid for 1-11 days
days_11 = np.arange(1, 12)
elast_11_vals = a1_11 + a2_11 * days_11
ax.plot(days_11, elast_11_vals, color=PALETTE["window11"],
        linewidth=2.2, linestyle="--", label="11-day window model (1–11 days)")

ax.axvline(11, color="grey", linestyle=":", linewidth=1, label="11-day cutoff")
ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
ax.fill_between(days_range, elast_full, 0,
                where=(elast_full > 0), alpha=0.08, color=PALETTE["full"])

ax.set_xlabel("Days Before Departure", fontsize=11)
ax.set_ylabel("Implied elasticity of indirect\nprice to direct price", fontsize=11)
ax.set_title("Substitution Elasticity Over the Booking Horizon\n"
             "Elasticity = α1 + α2 × DaysBeforeDeparture", fontsize=12)
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(linestyle=":", alpha=0.4)
ax.invert_xaxis()   # departure on the right → booking horizon left-to-right
ax.set_xlim(36, 1)
fig.tight_layout()
fig.savefig("plots/03_substitution_elasticity_over_time.png", dpi=150)
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — SoldOutDummy: median indirect price by sold-out status and window
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 4: sold-out price effect...")

df_11 = df[df["DaysBeforeDeparture"] <= 11].copy()

summary = {}
for label, subset in [("Full (1–36 d)", df), ("11-day (1–11 d)", df_11)]:
    grp = subset.groupby("SoldOutDummy")["price"].median()
    summary[label] = grp

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
for ax, (label, grp) in zip(axes, summary.items()):
    vals  = [grp.get(0, np.nan), grp.get(1, np.nan)]
    bars  = ax.bar(["Direct available\n(SoldOut=0)", "Direct sold out\n(SoldOut=1)"],
                   vals, color=[PALETTE["sold_no"], PALETTE["sold_yes"]],
                   edgecolor="white", width=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 3, f"€{val:.0f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    pct_diff = (vals[1] - vals[0]) / vals[0] * 100
    ax.set_title(f"{label}\n+{pct_diff:.1f}% when sold out", fontsize=11)
    ax.set_ylabel("Median indirect price (€)", fontsize=10)
    ax.set_ylim(0, max(vals)*1.15)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

fig.suptitle("Indirect Flight Prices by Direct Flight Availability (2023)",
             fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig("plots/04_soldout_price_effect.png", dpi=150, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Plot 5 — Travel time vs price (binned medians)
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 5: travel time vs price...")

df["TravelTimeBin"] = pd.cut(df["TravelTimeMinutes"], bins=12)
tt_grp = df.groupby("TravelTimeBin", observed=True)["price"].median().reset_index()
tt_grp["mid"] = tt_grp["TravelTimeBin"].apply(lambda x: x.mid).astype(float)

# Model-implied curve from full-window regression
a3 = res_full.loc["log_TravelTime", "coef"]
a4 = res_full.loc["log_TravelTime_sq", "coef"]
tt_range = np.linspace(df["TravelTimeMinutes"].quantile(0.02),
                       df["TravelTimeMinutes"].quantile(0.98), 200)
log_tt = np.log(tt_range)
# Relative price (centred at median)
mid_log = np.log(df["TravelTimeMinutes"].median())
rel_curve = np.exp(a3*(log_tt - mid_log) + a4*(log_tt**2 - mid_log**2))
# Scale to data median
baseline = df["price"].median()
implied_price = baseline * rel_curve

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.scatter(tt_grp["mid"]/60, tt_grp["price"], color=PALETTE["indirect"],
           s=80, zorder=5, label="Median observed price (binned)")
ax2 = ax.twinx()
ax2.plot(tt_range/60, implied_price, color=PALETTE["full"],
         linewidth=2, label="Model-implied curve (α3 + α4 terms)", alpha=0.8)
ax2.set_ylabel("Model-implied relative price (€)", color=PALETTE["full"], fontsize=10)
ax2.tick_params(axis="y", labelcolor=PALETTE["full"])

ax.set_xlabel("Travel Time (hours)", fontsize=11)
ax.set_ylabel("Median indirect price (€)", fontsize=11)
ax.set_title("Travel Time vs Indirect Flight Price — 2023 Full Window\n"
             "Model implies a U-shaped curve (long connections eventually cost more)", fontsize=11)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, framealpha=0.9)
ax.grid(linestyle=":", alpha=0.4)
fig.tight_layout()
fig.savefig("plots/05_traveltime_price_curve.png", dpi=150)
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Plot 6 — Scatter: log(DirectPrice_avg) vs log(IndirectPrice) by DaysBeforeDeparture
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 6: direct vs indirect scatter...")

sample = df.dropna(subset=["log_DirectPrice_avg", "log_price"]).sample(
    n=min(8000, len(df)), random_state=42)

cmap = plt.cm.RdYlBu_r
norm = plt.Normalize(sample["DaysBeforeDeparture"].min(),
                     sample["DaysBeforeDeparture"].max())
colors_scatter = cmap(norm(sample["DaysBeforeDeparture"]))

fig, ax = plt.subplots(figsize=(9, 5.5))
sc = ax.scatter(sample["log_DirectPrice_avg"], sample["log_price"],
                c=sample["DaysBeforeDeparture"], cmap="RdYlBu_r",
                alpha=0.35, s=8, linewidths=0)
cb = fig.colorbar(sc, ax=ax, pad=0.01)
cb.set_label("Days Before Departure", fontsize=10)

# OLS line across full sample
m, b = np.polyfit(df["log_DirectPrice_avg"], df["log_price"], 1)
x_line = np.linspace(df["log_DirectPrice_avg"].min(), df["log_DirectPrice_avg"].max(), 200)
ax.plot(x_line, m*x_line + b, color="black", linewidth=2,
        label=f"OLS: slope={m:.3f}")

ax.set_xlabel("log(DirectPrice_avg)", fontsize=11)
ax.set_ylabel("log(IndirectPrice)", fontsize=11)
ax.set_title("Direct vs Indirect Price Relationship — 2023 Full Window\n"
             "Colour = Days Before Departure (red = far, blue = near)", fontsize=11)
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(linestyle=":", alpha=0.3)
fig.tight_layout()
fig.savefig("plots/06_direct_vs_indirect_scatter.png", dpi=150)
plt.close()

print("\nAll plots saved to plots/ directory.")
print("  01_coef_comparison.png")
print("  02_price_distributions.png")
print("  03_substitution_elasticity_over_time.png")
print("  04_soldout_price_effect.png")
print("  05_traveltime_price_curve.png")
print("  06_direct_vs_indirect_scatter.png")
