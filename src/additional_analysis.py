"""
additional_analysis_v1.py
--------------------------
Seven supplementary analyses (Q1–Q7) on the 2023 dataset.

Q1  — 1-stop vs 2-stop substitution
Q2  — Continuous sell-out intensity
Q3  — LCC route share (continuous lccpercentage)
Q4  — Asymmetric price response (up vs down)
Q5  — Quantile regression (cheap vs expensive indirect flights)
Q6  — Which routes sell out earliest?
Q7  — Price gap (indirect − direct) over the booking horizon

REPLICATION ON 2026 DATA:
  1. Change INPUT_FILE to "final_2026.csv"
  2. Q2: AMD_FILE-based intensity won't apply — compute SoldOutIntensity
     directly in preprocess.py and include in final_2026.csv
  3. Q3/Q4: lccpercentage for 2026 comes from airlinelccdum.csv via
     merging_output_v1.py — join by airline name instead of by route
  4. Everything else runs identically.

Usage:
    python3 additional_analysis_v1.py
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
os.makedirs("plots", exist_ok=True)

INPUT_FILE = "final_2023.csv"
AMD_FILE   = "AMD.csv"

PALETTE = {
    "one_stop":  "#2196F3",
    "two_stop":  "#FF5722",
    "pos":       "#4CAF50",
    "neg":       "#F44336",
    "q25":       "#9C27B0",
    "q50":       "#2196F3",
    "q75":       "#FF9800",
    "gap":       "#00796B",
    "neutral":   "#607D8B",
}

def cluster_ols(formula, data):
    m = smf.ols(formula, data=data)
    return m.fit(cov_type="cluster", cov_kwds={"groups": data["Route"]})

def sig(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""

def print_coef(res, vars_):
    print(f"  {'Variable':<50} {'Coef':>8}  {'SE':>8}  {'t':>6}  {'p':>7}  Sig")
    print("  " + "-"*85)
    for v in vars_:
        if v not in res.params: continue
        c, se, t, p = res.params[v], res.bse[v], res.tvalues[v], res.pvalues[v]
        print(f"  {v:<50} {c:>8.4f}  {se:>8.4f}  {t:>6.2f}  {p:>7.4f}  {sig(p)}")

# ── Load and enrich data ─────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(INPUT_FILE)

raw = pd.read_csv(AMD_FILE)
lcc_flight = raw[["flightcode","lccdum"]].drop_duplicates("flightcode")
df = df.merge(lcc_flight.rename(columns={"flightcode":"FlightCode"}),
              on="FlightCode", how="left")

lcc_route = raw.groupby("route")["lccpercentage"].first().reset_index()
lcc_route.columns = ["Route","lccpercentage"]
df = df.merge(lcc_route, on="Route", how="left")

FORMULA_BASE = (
    "log_price ~ log_DirectPrice_avg + log_DirectPrice_x_Days + "
    "log_TravelTime + log_TravelTime_sq + SoldOutDummy + "
    "C(Route) + C(DepartureDate)"
)

print(f"  {len(df):,} rows | {df['Route'].nunique()} routes | "
      f"lccpercentage joined: {df['lccpercentage'].notna().sum():,}/{len(df):,}")


# =============================================================================
# Q1 — 1-stop vs 2-stop substitution
# =============================================================================
print("\n" + "="*65)
print("Q1: 1-stop vs 2-stop substitution")
print("="*65)

results_q1 = {}
for n, label, color in [(1,"1-stop",PALETTE["one_stop"]),
                         (2,"2-stop",PALETTE["two_stop"])]:
    sub = df[df["ConnectionNumber"] == n].copy()
    res = cluster_ols(FORMULA_BASE, sub)
    results_q1[label] = res
    print(f"\n  {label}  N={len(sub):,}  R²={res.rsquared:.4f}")
    print_coef(res, ["log_DirectPrice_avg","log_DirectPrice_x_Days",
                     "log_TravelTime","log_TravelTime_sq","SoldOutDummy"])

# Plot
KEY = ["log_DirectPrice_avg","log_DirectPrice_x_Days",
       "log_TravelTime","log_TravelTime_sq","SoldOutDummy"]
XLABELS = ["α1 log(DirectPrice)","α2 ×Days","α3 log(TravelTime)",
           "α4 log(TravelTime)²","α5 SoldOutDummy"]
x = np.arange(len(KEY)); w = 0.35

fig, ax = plt.subplots(figsize=(11,5))
for i,(label,color) in enumerate([("1-stop",PALETTE["one_stop"]),
                                    ("2-stop",PALETTE["two_stop"])]):
    res = results_q1[label]
    coefs = [res.params[v] for v in KEY]
    errs  = [res.bse[v]*1.96 for v in KEY]
    ax.bar(x + (i-0.5)*w, coefs, w, yerr=errs, capsize=4,
           color=color, alpha=0.88, label=label)
ax.axhline(0,color="black",linewidth=0.8,linestyle="--")
ax.set_xticks(x); ax.set_xticklabels(XLABELS, fontsize=10)
ax.set_ylabel("Coefficient"); ax.legend(fontsize=10)
ax.set_title("Q1: Coefficients by Connection Depth\n"
             "2023 Full Window | 95% CI | SEs clustered at route level", fontsize=11)
ax.grid(axis="y",linestyle=":",alpha=0.5)
fig.tight_layout()
fig.savefig("plots/09_connection_depth.png", dpi=150)
plt.close()
print("\n  Plot saved: plots/09_connection_depth.png")


# =============================================================================
# Q2 — Continuous sell-out intensity
# =============================================================================
print("\n" + "="*65)
print("Q2: Continuous sell-out intensity")
print("="*65)

direct = raw[raw["connectionnumber"] == 0].copy()
direct.rename(columns={"route":"Route","departuredate":"DepartureDate",
                        "daysbeforedeparture":"DaysBeforeDeparture",
                        "flightcode":"FlightCode"}, inplace=True)

counts = (direct.groupby(["Route","DepartureDate","DaysBeforeDeparture"])["FlightCode"]
                .nunique().reset_index(name="DirectCount"))

# Initial count = count at maximum DaysBeforeDeparture per (Route, DepartureDate)
init = (counts.sort_values("DaysBeforeDeparture", ascending=False)
              .groupby(["Route","DepartureDate"])["DirectCount"]
              .first()
              .rename("InitialCount")
              .reset_index())
counts = counts.merge(init, on=["Route","DepartureDate"])
counts["SoldOutIntensity"] = (
    1 - counts["DirectCount"] / counts["InitialCount"]
).clip(lower=0)

df2 = df.drop(columns=["SoldOutIntensity"], errors="ignore").merge(
    counts[["Route","DepartureDate","DaysBeforeDeparture","SoldOutIntensity"]],
    on=["Route","DepartureDate","DaysBeforeDeparture"], how="left"
)
df2["SoldOutIntensity"] = df2["SoldOutIntensity"].fillna(0)
print(f"  SoldOutIntensity: mean={df2['SoldOutIntensity'].mean():.4f} "
      f"| >0: {(df2['SoldOutIntensity']>0).mean()*100:.1f}%"
      f"| =1 (fully sold out): {(df2['SoldOutIntensity']==1).mean()*100:.1f}%")

formula_q2 = (
    "log_price ~ log_DirectPrice_avg + log_DirectPrice_x_Days + "
    "log_TravelTime + log_TravelTime_sq + SoldOutIntensity + "
    "C(Route) + C(DepartureDate)"
)
res_q2 = cluster_ols(formula_q2, df2)
print(f"\n  N={int(res_q2.nobs):,}  R²={res_q2.rsquared:.4f}")
print_coef(res_q2, ["log_DirectPrice_avg","log_DirectPrice_x_Days",
                    "log_TravelTime","log_TravelTime_sq","SoldOutIntensity"])

# Binned plot: mean indirect price by intensity bucket
df2["IntensityBin"] = pd.cut(df2["SoldOutIntensity"],
                              bins=[-0.01,0,0.25,0.5,0.75,1.01],
                              labels=["0% (none)","1–25%","26–50%","51–75%","76–100%"])
binned = df2.groupby("IntensityBin", observed=True)["price"].median()

fig, axes = plt.subplots(1,2,figsize=(12,4.5))
axes[0].bar(binned.index, binned.values,
            color=[PALETTE["neutral"],"#64B5F6","#1E88E5","#1565C0","#0D47A1"],
            edgecolor="white")
for i,(idx,val) in enumerate(binned.items()):
    axes[0].text(i, val+2, f"€{val:.0f}", ha="center", fontsize=9, fontweight="bold")
axes[0].set_xlabel("Sell-out intensity (% of direct flights gone)")
axes[0].set_ylabel("Median indirect price (€)")
axes[0].set_title("Median Indirect Price by Sell-Out Intensity")
axes[0].grid(axis="y",linestyle=":",alpha=0.4)

coef_q2  = res_q2.params["SoldOutIntensity"]
coef_bin = res_q2.params["log_DirectPrice_avg"]
x_int = np.linspace(0,1,100)
y_int = coef_q2 * x_int
axes[1].plot(x_int*100, y_int, color="#0D47A1", linewidth=2.2)
axes[1].axhline(0, color="grey", linewidth=0.8, linestyle="--")
axes[1].fill_between(x_int*100, 0, y_int, alpha=0.15, color="#0D47A1")
axes[1].set_xlabel("Sell-out intensity (%)")
axes[1].set_ylabel("Effect on log(IndirectPrice)")
axes[1].set_title(f"Model-implied effect of intensity\n"
                  f"(coef={coef_q2:.4f}, p={res_q2.pvalues['SoldOutIntensity']:.4f})")
axes[1].grid(linestyle=":",alpha=0.4)
fig.suptitle("Q2: Continuous Sell-Out Intensity vs Indirect Price — 2023 Full Window",
             fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig("plots/10_soldout_intensity.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Plot saved: plots/10_soldout_intensity.png")


# =============================================================================
# Q3 — lccpercentage continuous
# =============================================================================
print("\n" + "="*65)
print("Q3: lccpercentage (continuous) route competition")
print("="*65)

df3 = df.dropna(subset=["lccpercentage"]).copy()
df3["lccpct_x_logDP"] = df3["lccpercentage"] * df3["log_DirectPrice_avg"]
df3["lccpct_x_SoldOut"] = df3["lccpercentage"] * df3["SoldOutDummy"]

formula_q3 = (
    "log_price ~ log_DirectPrice_avg + log_DirectPrice_x_Days + "
    "log_TravelTime + log_TravelTime_sq + SoldOutDummy + "
    "lccpercentage + lccpct_x_logDP + lccpct_x_SoldOut + "
    "C(Route) + C(DepartureDate)"
)
res_q3 = cluster_ols(formula_q3, df3)
print(f"\n  N={int(res_q3.nobs):,}  R²={res_q3.rsquared:.4f}")
print_coef(res_q3, ["log_DirectPrice_avg","log_DirectPrice_x_Days",
                    "log_TravelTime","log_TravelTime_sq","SoldOutDummy",
                    "lccpercentage","lccpct_x_logDP","lccpct_x_SoldOut"])

a1   = res_q3.params["log_DirectPrice_avg"]
b_dp = res_q3.params["lccpct_x_logDP"]
a5   = res_q3.params["SoldOutDummy"]
b_so = res_q3.params["lccpct_x_SoldOut"]
lcc_range = np.linspace(0, 1, 100)

fig, axes = plt.subplots(1,2,figsize=(12,4.5))
axes[0].plot(lcc_range*100, a1 + b_dp*lcc_range,
             color=PALETTE["one_stop"], linewidth=2.2)
axes[0].set_xlabel("LCC route share (%)")
axes[0].set_ylabel("Implied substitution elasticity α1")
axes[0].set_title(f"Substitution elasticity vs LCC share\n"
                  f"gradient coef={b_dp:.4f} (p={res_q3.pvalues['lccpct_x_logDP']:.4f})")
axes[0].grid(linestyle=":",alpha=0.4)

axes[1].plot(lcc_range*100, (np.exp(a5 + b_so*lcc_range)-1)*100,
             color=PALETTE["two_stop"], linewidth=2.2)
axes[1].set_xlabel("LCC route share (%)")
axes[1].set_ylabel("Implied scarcity effect on price (%)")
axes[1].set_title(f"Scarcity effect vs LCC share\n"
                  f"gradient coef={b_so:.4f} (p={res_q3.pvalues['lccpct_x_SoldOut']:.4f})")
axes[1].grid(linestyle=":",alpha=0.4)
fig.suptitle("Q3: How LCC Route Share Modulates the Pricing Mechanism — 2023",
             fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig("plots/11_lccpercentage_interaction.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Plot saved: plots/11_lccpercentage_interaction.png")


# =============================================================================
# Q4 — Asymmetric price response
# =============================================================================
print("\n" + "="*65)
print("Q4: Asymmetric price response (up vs down)")
print("="*65)

# Within-route-date mean of log_DirectPrice_avg as reference
rd_mean = (df.groupby(["Route","DepartureDate"])["log_DirectPrice_avg"]
             .mean().rename("logDP_mean").reset_index())
df4 = df.merge(rd_mean, on=["Route","DepartureDate"])
df4["logDP_dev"]  = df4["log_DirectPrice_avg"] - df4["logDP_mean"]
df4["logDP_up"]   = df4["logDP_dev"].clip(lower=0)
df4["logDP_down"] = (-df4["logDP_dev"]).clip(lower=0)
# Interaction with Days
df4["logDP_up_x_Days"]   = df4["logDP_up"]   * df4["DaysBeforeDeparture"]
df4["logDP_down_x_Days"] = df4["logDP_down"] * df4["DaysBeforeDeparture"]

formula_q4 = (
    "log_price ~ logDP_up + logDP_down + "
    "logDP_up_x_Days + logDP_down_x_Days + "
    "log_TravelTime + log_TravelTime_sq + SoldOutDummy + "
    "C(Route) + C(DepartureDate)"
)
res_q4 = cluster_ols(formula_q4, df4)
print(f"\n  N={int(res_q4.nobs):,}  R²={res_q4.rsquared:.4f}")
print_coef(res_q4, ["logDP_up","logDP_down",
                    "logDP_up_x_Days","logDP_down_x_Days",
                    "log_TravelTime","log_TravelTime_sq","SoldOutDummy"])

coef_up   = res_q4.params["logDP_up"]
coef_down = res_q4.params["logDP_down"]

# Wald test: are up and down coefficients equal?
from statsmodels.stats.contrast import ContrastResults
# Simple t-test approximation using clustered SEs
diff  = coef_up - coef_down
se_up   = res_q4.bse["logDP_up"]
se_down = res_q4.bse["logDP_down"]
se_diff = np.sqrt(se_up**2 + se_down**2)   # approximate (ignores covariance)
t_diff  = diff / se_diff
p_diff  = 2*(1 - __import__("scipy").stats.t.cdf(abs(t_diff), df=res_q4.df_resid))
print(f"\n  Up coef={coef_up:.4f}  Down coef={coef_down:.4f}")
print(f"  Difference = {diff:.4f}  t={t_diff:.2f}  p≈{p_diff:.4f}  {sig(p_diff)}")
print(f"  → {'Significant upward asymmetry' if p_diff<0.05 and diff>0 else 'Upward asymmetry' if diff>0 else 'Symmetric or downward'}")

fig, ax = plt.subplots(figsize=(7,5))
vals   = [coef_up, coef_down]
labels = ["Price above route-date\naverage (UP)", "Price below route-date\naverage (DOWN)"]
colors = [PALETTE["pos"], PALETTE["neg"]]
bars = ax.bar(labels, vals, color=colors, edgecolor="white", width=0.45)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.002 if val>=0 else val-0.005,
            f"{val:.4f}", ha="center", va="bottom" if val>=0 else "top",
            fontsize=12, fontweight="bold")
ax.axhline(0,color="black",linewidth=0.8,linestyle="--")
ax.set_ylabel("Coefficient on log(DirectPrice) deviation")
ax.set_title(f"Q4: Asymmetric Price Response\n"
             f"Up={coef_up:.4f} vs Down={coef_down:.4f}  "
             f"(diff p≈{p_diff:.3f} {sig(p_diff)})")
ax.grid(axis="y",linestyle=":",alpha=0.4)
fig.tight_layout()
fig.savefig("plots/12_asymmetric_response.png", dpi=150)
plt.close()
print("  Plot saved: plots/12_asymmetric_response.png")


# =============================================================================
# Q5 — Quantile regression
# =============================================================================
print("\n" + "="*65)
print("Q5: Quantile regression (cheap vs expensive indirect flights)")
print("="*65)

# Within-group demeaning to absorb Route + DepartureDate FE
# (avoids having to pass 130 dummy variables to quantreg)
qvars = ["log_price","log_DirectPrice_avg","log_DirectPrice_x_Days",
         "log_TravelTime","log_TravelTime_sq","SoldOutDummy"]
df5 = df[qvars + ["Route","DepartureDate"]].dropna().copy()
for v in qvars:
    gm = df5.groupby(["Route","DepartureDate"])[v].transform("mean")
    df5[v+"_dm"] = df5[v] - gm

formula_q5 = (
    "log_price_dm ~ log_DirectPrice_avg_dm + log_DirectPrice_x_Days_dm + "
    "log_TravelTime_dm + log_TravelTime_sq_dm + SoldOutDummy_dm - 1"
)

results_q5 = {}
for q, color in [(0.25, PALETTE["q25"]), (0.50, PALETTE["q50"]), (0.75, PALETTE["q75"])]:
    print(f"  Running quantile regression τ={q}...", end=" ", flush=True)
    res = smf.quantreg(formula_q5, data=df5).fit(q=q, max_iter=5000)
    results_q5[q] = res
    print(f"done  pseudo-R²={res.prsquared:.4f}")
    print_coef(res, ["log_DirectPrice_avg_dm","log_DirectPrice_x_Days_dm",
                     "SoldOutDummy_dm"])

# Plot: α1 and α5 across quantiles
quantiles = [0.25, 0.50, 0.75]
a1_q = [results_q5[q].params["log_DirectPrice_avg_dm"] for q in quantiles]
a5_q = [results_q5[q].params["SoldOutDummy_dm"] for q in quantiles]
a1_ci = [(results_q5[q].conf_int().loc["log_DirectPrice_avg_dm"]) for q in quantiles]
a5_ci = [(results_q5[q].conf_int().loc["SoldOutDummy_dm"]) for q in quantiles]

fig, axes = plt.subplots(1,2,figsize=(11,4.5))
colors_q = [PALETTE["q25"], PALETTE["q50"], PALETTE["q75"]]
labels_q = ["Q25\n(cheap)","Q50\n(median)","Q75\n(expensive)"]

for ax, coefs, cis, ols_val, title, ylabel in [
    (axes[0], a1_q, a1_ci, 0.3421,
     "Substitution Elasticity (α1) by Price Quantile",
     "Coef on log(DirectPrice_avg)"),
    (axes[1], a5_q, a5_ci, 0.0484,
     "Scarcity Effect (α5) by Price Quantile",
     "Coef on SoldOutDummy"),
]:
    bars = ax.bar(labels_q, coefs, color=colors_q, edgecolor="white", width=0.45)
    for bar,(lo,hi) in zip(bars,cis):
        mid = bar.get_x()+bar.get_width()/2
        ax.plot([mid,mid],[lo,hi],color="black",linewidth=1.5)
    for bar,val in zip(bars,coefs):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.003,
                f"{val:.4f}", ha="center", fontsize=9, fontweight="bold")
    ax.axhline(ols_val, color="grey", linewidth=1.5, linestyle="--",
               label=f"OLS (full) = {ols_val:.4f}")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y",linestyle=":",alpha=0.4)

fig.suptitle("Q5: Are Cheap and Expensive Indirect Flights Equally Responsive?\n"
             "Quantile regression with within-FE demeaning | 2023 Full Window",
             fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig("plots/13_quantile_regression.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Plot saved: plots/13_quantile_regression.png")


# =============================================================================
# Q6 — Which routes sell out earliest?
# =============================================================================
print("\n" + "="*65)
print("Q6: What predicts sell-out timing?")
print("="*65)

# Recompute EventDay (first DaysBeforeDeparture where SoldOut fires)
def find_event_day(grp):
    grp = grp.sort_values("DaysBeforeDeparture", ascending=False)
    sold = grp["SoldOutDummy"].values
    days = grp["DaysBeforeDeparture"].values
    for i in range(1, len(sold)):
        if sold[i-1] == 0 and sold[i] == 1:
            return days[i]
    return np.nan

event_days = (df.groupby(["Route","DepartureDate"])
                .apply(find_event_day, include_groups=False)
                .reset_index()
                .rename(columns={0:"EventDay"}))
event_days = event_days[event_days["EventDay"].notna()].copy()

# Route-level characteristics
route_chars = (df.groupby("Route")
                 .agg(lccpercentage=("lccpercentage","first"),
                      MeanDirectPrice=("DirectPrice_avg","mean"),
                      MeanTravelTime=("TravelTimeMinutes","mean"))
                 .reset_index())

# Initial direct flight count per route-date (from AMD.csv)
init_counts = (direct.groupby(["Route","DepartureDate"])
               .apply(lambda g: g.loc[g["DaysBeforeDeparture"].idxmax(), "FlightCode"]
                      if len(g) else np.nan, include_groups=False)
               .reset_index(name="ignore"))
init_dc = (direct.groupby(["Route","DepartureDate"])
                 .agg(InitialDirectCount=("FlightCode","nunique"))
                 .reset_index())

df6 = (event_days
       .merge(route_chars, on="Route", how="left")
       .merge(init_dc, on=["Route","DepartureDate"], how="left"))
df6["log_MeanDirectPrice"] = np.log(df6["MeanDirectPrice"])
df6 = df6.dropna(subset=["EventDay","lccpercentage","log_MeanDirectPrice"])

print(f"  Treated route-dates with EventDay: {len(df6)}")
print(f"  EventDay: mean={df6['EventDay'].mean():.1f}  "
      f"std={df6['EventDay'].std():.1f}  "
      f"range={df6['EventDay'].min():.0f}–{df6['EventDay'].max():.0f}")

formula_q6 = "EventDay ~ lccpercentage + log_MeanDirectPrice + InitialDirectCount + C(DepartureDate)"
res_q6 = smf.ols(formula_q6, data=df6).fit()  # route-level, no need to cluster
print(f"\n  R²={res_q6.rsquared:.4f}  N={int(res_q6.nobs)}")
print_coef(res_q6, ["lccpercentage","log_MeanDirectPrice","InitialDirectCount"])

fig, axes = plt.subplots(1,3,figsize=(14,4.5))
for ax, xcol, xlabel in [
    (axes[0], "lccpercentage",      "LCC route share"),
    (axes[1], "log_MeanDirectPrice","log(Mean direct price)"),
    (axes[2], "InitialDirectCount", "Initial direct flight count"),
]:
    ax.scatter(df6[xcol], df6["EventDay"],
               alpha=0.5, s=25, color=PALETTE["neutral"], edgecolors="none")
    m,b = np.polyfit(df6[xcol].dropna(), df6.loc[df6[xcol].notna(),"EventDay"], 1)
    xr = np.linspace(df6[xcol].min(), df6[xcol].max(), 100)
    ax.plot(xr, m*xr+b, color=PALETTE["two_stop"], linewidth=2)
    ax.set_xlabel(xlabel); ax.set_ylabel("Event Day (DaysBeforeDeparture)")
    ax.set_title(xlabel); ax.grid(linestyle=":",alpha=0.4)
fig.suptitle("Q6: What Predicts When Direct Flights Sell Out?\n"
             "Higher EventDay = sells out earlier (more days before departure)",
             fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig("plots/14_sellout_timing.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Plot saved: plots/14_sellout_timing.png")


# =============================================================================
# Q7 — Price gap over the booking horizon
# =============================================================================
print("\n" + "="*65)
print("Q7: Price gap (indirect − direct) over booking horizon")
print("="*65)

df7 = df.copy()
df7["PriceGap"]    = df7["price"] - df7["DirectPrice_avg"]
df7["LogPriceGap"] = df7["log_price"] - df7["log_DirectPrice_avg"]

# Aggregate by DaysBeforeDeparture
gap_by_day = (df7.groupby("DaysBeforeDeparture")
                  .agg(MedianGap=("PriceGap","median"),
                       MeanGap=("PriceGap","mean"),
                       MeanLogGap=("LogPriceGap","mean"),
                       N=("PriceGap","count"))
                  .reset_index())

# Sold-out vs not
gap_soldout = (df7.groupby(["DaysBeforeDeparture","SoldOutDummy"])
                   .agg(MedianGap=("PriceGap","median"))
                   .reset_index())

print(f"  Overall median gap: €{df7['PriceGap'].median():.1f}")
print(f"  At 36 days: €{df7[df7['DaysBeforeDeparture']==36]['PriceGap'].median():.1f}")
print(f"  At 1 day:   €{df7[df7['DaysBeforeDeparture']==1]['PriceGap'].median():.1f}")

# Regression: how does gap change per day, controlling for route FE
formula_q7 = "LogPriceGap ~ DaysBeforeDeparture + C(Route) + C(DepartureDate)"
res_q7 = cluster_ols(formula_q7, df7)
gap_coef = res_q7.params["DaysBeforeDeparture"]
gap_p    = res_q7.pvalues["DaysBeforeDeparture"]
print(f"\n  Log-gap ~ Days: coef={gap_coef:.5f}  p={gap_p:.4f}  {sig(gap_p)}")
print(f"  → Each day further from departure: gap changes by {gap_coef*100:.3f}% of price")

# Also test: is gap larger when sold out?
formula_q7b = "LogPriceGap ~ DaysBeforeDeparture + SoldOutDummy + C(Route) + C(DepartureDate)"
res_q7b = cluster_ols(formula_q7b, df7)
so_coef = res_q7b.params["SoldOutDummy"]
so_p    = res_q7b.pvalues["SoldOutDummy"]
print(f"  SoldOutDummy on gap: coef={so_coef:.4f}  p={so_p:.4f}  {sig(so_p)}")
print(f"  → When directs sell out, gap widens by {(np.exp(so_coef)-1)*100:.1f}%")

fig, axes = plt.subplots(1,2,figsize=(13,5))
axes[0].plot(gap_by_day["DaysBeforeDeparture"], gap_by_day["MedianGap"],
             color=PALETTE["gap"], linewidth=2.2, label="Median gap (€)")
axes[0].fill_between(gap_by_day["DaysBeforeDeparture"],
                     gap_by_day["MedianGap"] - gap_by_day["MeanGap"].std()*0.3,
                     gap_by_day["MedianGap"] + gap_by_day["MeanGap"].std()*0.3,
                     alpha=0.15, color=PALETTE["gap"])
axes[0].axhline(df7["PriceGap"].median(), color="grey",
                linewidth=1, linestyle="--", label=f"Overall median €{df7['PriceGap'].median():.0f}")
axes[0].set_xlabel("Days Before Departure")
axes[0].set_ylabel("Median price gap: indirect − direct (€)")
axes[0].set_title("Price Gap Over Booking Horizon\n(positive = indirect more expensive)")
axes[0].invert_xaxis(); axes[0].legend(fontsize=9)
axes[0].grid(linestyle=":",alpha=0.4)

for sold_val, color, label in [(0, PALETTE["one_stop"], "Direct available (SoldOut=0)"),
                                (1, PALETTE["two_stop"], "Direct sold out (SoldOut=1)")]:
    sub = gap_soldout[gap_soldout["SoldOutDummy"]==sold_val]
    axes[1].plot(sub["DaysBeforeDeparture"], sub["MedianGap"],
                 color=color, linewidth=2.2, label=label)
axes[1].axhline(0, color="black", linewidth=0.6, linestyle="--")
axes[1].set_xlabel("Days Before Departure")
axes[1].set_ylabel("Median price gap: indirect − direct (€)")
axes[1].set_title(f"Price Gap by Sell-Out Status\nSoldOut widens gap by {(np.exp(so_coef)-1)*100:.1f}% (p={so_p:.4f})")
axes[1].invert_xaxis(); axes[1].legend(fontsize=9)
axes[1].grid(linestyle=":",alpha=0.4)

fig.suptitle("Q7: How Does the Price Gap Between Indirect and Direct Flights Evolve? — 2023",
             fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig("plots/15_price_gap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Plot saved: plots/15_price_gap.png")

print("\n\nAll Q1–Q7 analyses complete.")
