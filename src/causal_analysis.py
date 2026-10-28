"""
causal_analysis_v1.py
---------------------
Two supplementary analyses to move closer to causation:

  Option A — Event study around SoldOutDummy activation
    For each (Route, DepartureDate), find the day when the first direct
    flight disappears (SoldOut switches 0→1). Track indirect prices in a
    ±5 day window around that event. Compare to route-dates that never
    sell out (control group). A price jump at event day 0 is consistent
    with a causal scarcity mechanism.

  Option C — LCC heterogeneity
    Add LCCdum × log(DirectPrice_avg) and LCCdum × SoldOutDummy
    interactions to the main regression. If the substitution effect or
    scarcity effect is structurally different on LCC routes, the mechanism
    differs by market type.

Usage:
    python3 causal_analysis_v1.py
Outputs:
    results_event_study.csv
    results_lcc_interaction.csv
    plots/07_event_study.png
    plots/08_lcc_coef_comparison.png
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("final_2023.csv")

# Join LCCdum from AMD.csv using FlightCode
raw = pd.read_csv("AMD.csv")[["flightcode", "lccdum", "lccpercentage"]].drop_duplicates("flightcode")
df = df.merge(raw.rename(columns={"flightcode": "FlightCode"}),
              on="FlightCode", how="left")

print(f"LCCdum join: {df['lccdum'].notna().sum():,} matched / {len(df):,} total")
print(f"LCC routes (lccdum=1): {df['lccdum'].mean()*100:.1f}% of observations")


# =============================================================================
# OPTION A — EVENT STUDY
# =============================================================================
print("\n" + "="*60)
print("OPTION A: Event Study around SoldOutDummy activation")
print("="*60)

# Step 1: For each (Route, DepartureDate), find event day = first
# DaysBeforeDeparture where SoldOutDummy first becomes 1 (moving toward
# departure, i.e. the largest DaysBeforeDeparture where SoldOut=1).
# We want the transition point: last day with SoldOut=0 before it fires.

df_sorted = df.sort_values(["Route", "DepartureDate", "DaysBeforeDeparture"],
                            ascending=[True, True, False])

def find_event_day(grp):
    """Return the DaysBeforeDeparture at which SoldOut first fires (0→1 transition)."""
    grp = grp.sort_values("DaysBeforeDeparture", ascending=False)  # furthest first
    sold = grp["SoldOutDummy"].values
    days = grp["DaysBeforeDeparture"].values
    for i in range(1, len(sold)):
        if sold[i-1] == 0 and sold[i] == 1:
            return days[i]   # first day SoldOut=1
    return np.nan

event_days = (df.groupby(["Route", "DepartureDate"])
                .apply(find_event_day, include_groups=False)
                .reset_index()
                .rename(columns={0: "EventDay"}))

df = df.merge(event_days, on=["Route", "DepartureDate"], how="left")

# Step 2: Treated = route-dates with a 0→1 switch; control = always SoldOut=0
df["Treated"] = df["EventDay"].notna()
df_treated  = df[df["Treated"]].copy()
df_control  = df[df["SoldOutDummy"] == 0].copy()  # control: never sold out

print(f"Treated route-dates (0→1 switch): {df['Treated'].groupby([df['Route'], df['DepartureDate']]).first().sum()}")
print(f"Control route-dates (SoldOut always 0): {(~df.groupby(['Route','DepartureDate'])['SoldOutDummy'].any()).sum()}")

# Step 3: Compute relative days (days relative to event)
df_treated["RelDay"] = df_treated["DaysBeforeDeparture"] - df_treated["EventDay"]

# Step 4: Normalise log prices within route-date to pre-event mean (RelDay > 0)
pre_means = (df_treated[df_treated["RelDay"] > 0]
             .groupby(["Route", "DepartureDate"])["log_price"]
             .mean()
             .rename("PreMean"))
df_treated = df_treated.merge(pre_means, on=["Route", "DepartureDate"], how="left")
df_treated["NormLogPrice"] = df_treated["log_price"] - df_treated["PreMean"]

# Step 5: Aggregate by relative day (window ±7)
window = 7
ev = (df_treated[df_treated["RelDay"].between(-window, window)]
      .groupby("RelDay")
      .agg(MeanNormLogPrice=("NormLogPrice", "mean"),
           N=("NormLogPrice", "count"))
      .reset_index())

# Control: normalise similarly using median DaysBeforeDeparture as pseudo-event
# Use median event day (~8) as the pseudo-event for control routes
pseudo_event = int(event_days["EventDay"].median())
df_control["RelDay"] = df_control["DaysBeforeDeparture"] - pseudo_event
pre_means_ctrl = (df_control[df_control["RelDay"] > 0]
                  .groupby(["Route", "DepartureDate"])["log_price"]
                  .mean()
                  .rename("PreMean"))
df_control = df_control.merge(pre_means_ctrl, on=["Route", "DepartureDate"], how="left")
df_control["NormLogPrice"] = df_control["log_price"] - df_control["PreMean"]

ctrl = (df_control[df_control["RelDay"].between(-window, window)]
        .groupby("RelDay")
        .agg(MeanNormLogPrice=("NormLogPrice", "mean"),
             N=("NormLogPrice", "count"))
        .reset_index())

# Save results
ev["group"] = "Treated (sells out)"
ctrl["group"] = "Control (never sells out)"
ev_out = pd.concat([ev, ctrl], ignore_index=True)
ev_out.to_csv("results_event_study.csv", index=False)
print("\nEvent study results saved to results_event_study.csv")
print(ev[["RelDay", "MeanNormLogPrice", "N"]].to_string(index=False))

# Print the key statistic: price change at event day 0
pre_price  = ev.loc[ev["RelDay"] > 0, "MeanNormLogPrice"].mean()
post_price = ev.loc[ev["RelDay"] <= 0, "MeanNormLogPrice"].mean()
at_event   = ev.loc[ev["RelDay"] == 0, "MeanNormLogPrice"].values
print(f"\nKey result:")
print(f"  Mean normalised log price, pre-event  (RelDay > 0): {pre_price:.4f}")
print(f"  Mean normalised log price, post-event (RelDay ≤ 0): {post_price:.4f}")
if len(at_event):
    print(f"  At event day (RelDay = 0): {at_event[0]:.4f}  → "
          f"implied price change ≈ {(np.exp(at_event[0])-1)*100:.1f}%")

# ── Plot 07 ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(ev["RelDay"],   ev["MeanNormLogPrice"],
        color="#F44336", linewidth=2.2, marker="o", markersize=5,
        label="Treated: route-dates where directs sell out")
ax.plot(ctrl["RelDay"], ctrl["MeanNormLogPrice"],
        color="#90CAF9", linewidth=2.2, marker="s", markersize=5,
        linestyle="--", label="Control: route-dates, directs never sell out")

ax.axvline(0, color="black", linewidth=1.2, linestyle="--", label="Event: first sell-out day")
ax.axhline(0, color="grey",  linewidth=0.8, linestyle=":")
ax.fill_betweenx([-0.15, 0.15], -window, 0, alpha=0.06, color="#F44336")

ax.set_xlabel("Days relative to sell-out event  (0 = first day direct disappears)", fontsize=11)
ax.set_ylabel("Mean normalised log(IndirectPrice)\n(relative to pre-event mean)", fontsize=11)
ax.set_title("Event Study: Indirect Price Response to Direct Flight Sell-Out\n"
             "2023 Dataset — Full Window", fontsize=12)
ax.legend(fontsize=10, framealpha=0.9)
ax.set_xlim(-window - 0.5, window + 0.5)
ax.set_xticks(range(-window, window+1))
ax.set_xticklabels([str(x) if x % 2 == 0 else "" for x in range(-window, window+1)])
ax.grid(linestyle=":", alpha=0.4)
fig.tight_layout()
fig.savefig("plots/07_event_study.png", dpi=150)
plt.close()
print("Plot saved: plots/07_event_study.png")


# =============================================================================
# OPTION C — LCC HETEROGENEITY
# =============================================================================
print("\n" + "="*60)
print("OPTION C: LCC Heterogeneity — Interaction Regression")
print("="*60)

df_lcc = df.dropna(subset=["lccdum"]).copy()
df_lcc["lccdum"] = df_lcc["lccdum"].astype(int)

# Run full window
print(f"\nN (with LCCdum): {len(df_lcc):,}")
print(f"LCC=1: {df_lcc['lccdum'].sum():,} ({df_lcc['lccdum'].mean()*100:.1f}%)")

formula_lcc = (
    "log_price ~ "
    "log_DirectPrice_avg + "
    "log_DirectPrice_x_Days + "
    "log_TravelTime + "
    "log_TravelTime_sq + "
    "SoldOutDummy + "
    "lccdum + "
    "lccdum:log_DirectPrice_avg + "   # Does substitution differ on LCC routes?
    "lccdum:SoldOutDummy + "           # Does scarcity effect differ on LCC routes?
    "C(Route) + C(DepartureDate)"
)

print("Running LCC interaction regression...")
res_lcc = smf.ols(formula_lcc, data=df_lcc).fit(
    cov_type="cluster",
    cov_kwds={"groups": df_lcc["Route"]},
)

# Extract key coefficients
INTEREST_VARS = [
    "log_DirectPrice_avg",
    "log_DirectPrice_x_Days",
    "log_TravelTime",
    "log_TravelTime_sq",
    "SoldOutDummy",
    "lccdum",
    "log_DirectPrice_avg:lccdum",
    "SoldOutDummy:lccdum",
]

LABELS = {
    "log_DirectPrice_avg":         "α1  log(DirectPrice_avg)  [non-LCC baseline]",
    "log_DirectPrice_x_Days":      "α2  log(DirectPrice_avg) × Days",
    "log_TravelTime":              "α3  log(TravelTime)",
    "log_TravelTime_sq":           "α4  log(TravelTime)²",
    "SoldOutDummy":                "α5  SoldOutDummy           [non-LCC baseline]",
    "lccdum":                      "β1  LCCdum (level effect)",
    "log_DirectPrice_avg:lccdum":  "β2  LCCdum × log(DirectPrice_avg)  [LCC premium/discount on α1]",
    "SoldOutDummy:lccdum":         "β3  LCCdum × SoldOutDummy          [LCC premium/discount on α5]",
}

print(f"\nR² = {res_lcc.rsquared:.4f}  |  Adj R² = {res_lcc.rsquared_adj:.4f}")
print(f"N  = {int(res_lcc.nobs):,}\n")
print(f"{'Variable':<55} {'Coef':>8}  {'SE':>8}  {'t':>6}  {'p':>7}  {'Sig'}")
print("-"*90)

rows = []
for var in INTEREST_VARS:
    if var not in res_lcc.params:
        # try flipped interaction name
        alt = ":".join(var.split(":")[::-1])
        if alt in res_lcc.params:
            var = alt
        else:
            continue
    coef = res_lcc.params[var]
    se   = res_lcc.bse[var]
    t    = res_lcc.tvalues[var]
    p    = res_lcc.pvalues[var]
    sig  = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
    label = LABELS.get(var, var)
    print(f"{label:<55} {coef:>8.4f}  {se:>8.4f}  {t:>6.2f}  {p:>7.4f}  {sig}")
    rows.append({"variable": var, "label": label, "coef": round(coef,6),
                 "std_err": round(se,6), "t_stat": round(t,4),
                 "p_value": round(p,6), "sig": sig,
                 "n_obs": int(res_lcc.nobs), "r2": round(res_lcc.rsquared,6)})

print("-"*90)
print("Significance: *** p<0.01  ** p<0.05  * p<0.1")

# Implied total coefficients for LCC vs non-LCC
a1     = res_lcc.params["log_DirectPrice_avg"]
b2_key = "log_DirectPrice_avg:lccdum" if "log_DirectPrice_avg:lccdum" in res_lcc.params else "lccdum:log_DirectPrice_avg"
b2     = res_lcc.params.get(b2_key, 0)
a5     = res_lcc.params["SoldOutDummy"]
b3_key = "SoldOutDummy:lccdum" if "SoldOutDummy:lccdum" in res_lcc.params else "lccdum:SoldOutDummy"
b3     = res_lcc.params.get(b3_key, 0)

print(f"\nImplied total coefficients:")
print(f"  Substitution elasticity (non-LCC): α1        = {a1:.4f}")
print(f"  Substitution elasticity (LCC):     α1 + β2   = {a1+b2:.4f}")
print(f"  Scarcity effect (non-LCC):         α5        = {a5:.4f}  ({(np.exp(a5)-1)*100:.1f}%)")
print(f"  Scarcity effect (LCC):             α5 + β3   = {a5+b3:.4f}  ({(np.exp(a5+b3)-1)*100:.1f}%)")

pd.DataFrame(rows).to_csv("results_lcc_interaction.csv", index=False)
print("\nLCC interaction results saved to results_lcc_interaction.csv")

# ── Plot 08 ──────────────────────────────────────────────────────────────────
groups   = ["Non-LCC", "LCC"]
colors   = ["#2196F3", "#FF5722"]
subst    = [a1, a1 + b2]
scarcity = [a5, a5 + b3]

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

for ax, vals, title, ylabel in [
    (axes[0], subst,    "Substitution Elasticity (α1)\nby Route Type",
               "Coefficient on log(DirectPrice_avg)"),
    (axes[1], scarcity, "Scarcity Effect (α5)\nby Route Type",
               "Coefficient on SoldOutDummy"),
]:
    bars = ax.bar(groups, vals, color=colors, edgecolor="white", width=0.45)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                val + (0.003 if val >= 0 else -0.008),
                f"{val:.4f}", ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=11, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_ylim(min(vals)*1.3 - 0.02, max(vals)*1.3 + 0.02)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

fig.suptitle("LCC vs Non-LCC Routes: How Pricing Mechanisms Differ\n"
             "2023 Dataset — Full Window  |  Std errors clustered at route level",
             fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig("plots/08_lcc_coef_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Plot saved: plots/08_lcc_coef_comparison.png")

print("\nDone.")
