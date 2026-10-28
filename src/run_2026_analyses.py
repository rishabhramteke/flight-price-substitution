"""
run_2026_analyses.py
--------------------
Runs all supplementary analyses (Exp3, Exp4, Q1–Q7, T1–T6) on 2026 data.
Saves plots to plots_2026/, results to *_2026.csv.
Does NOT modify any 2023 outputs.

Usage:
    python3 run_2026_analyses.py
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import grangercausalitytests
from linearmodels import PanelOLS, RandomEffects

warnings.filterwarnings("ignore")
os.makedirs("plots_2026", exist_ok=True)

INPUT_FILE = "final_2026_fixed.csv"

# 2026 full-window OLS baseline values (from regression.py run)
OLS_ALPHA1 = 0.2648
OLS_ALPHA5 = 0.0637

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
    "main":      "#2196F3",
    "accent":    "#FF5722",
    "light":     "#90CAF9",
}

def sig(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "n.s."

def cluster_ols(formula, data):
    return smf.ols(formula, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data["Route"]}
    )

def print_coef(res, vars_):
    print(f"  {'Variable':<52} {'Coef':>8}  {'SE':>8}  {'t':>6}  {'p':>7}  Sig")
    print("  " + "-"*87)
    for v in vars_:
        if v not in res.params:
            continue
        c, se, t, p = res.params[v], res.bse[v], res.tvalues[v], res.pvalues[v]
        print(f"  {v:<52} {c:>8.4f}  {se:>8.4f}  {t:>6.2f}  {p:>7.4f}  {sig(p)}")

FORMULA_BASE = (
    "log_price ~ log_DirectPrice_avg + log_DirectPrice_x_Days + "
    "log_TravelTime + log_TravelTime_sq + SoldOutDummy + "
    "C(Route) + C(DepartureDate)"
)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading 2026 data...")
df = pd.read_csv(INPUT_FILE)
# LCCdum and LCCpercentage already in file from preprocessing
print(f"  {len(df):,} rows | {df['Route'].nunique()} routes")
print(f"  LCCdum present: {df['LCCdum'].notna().sum():,}/{len(df):,}")
print(f"  LCC=1: {(df['LCCdum']==1).sum():,} ({(df['LCCdum']==1).mean()*100:.1f}%)")
print(f"  SoldOutIntensity present: {df['SoldOutIntensity'].notna().sum():,}/{len(df):,}")
df = df.rename(columns={"LCCdum": "lccdum", "LCCpercentage": "lccpercentage"})

print("\nRunning baseline OLS for diagnostics...")
res_base = cluster_ols(FORMULA_BASE, df)
print(f"  Baseline R²={res_base.rsquared:.4f}  N={int(res_base.nobs):,}")


# =============================================================================
# EXP 3 — Event study around SoldOutDummy activation
# =============================================================================
print("\n" + "="*65)
print("EXP 3: Event Study — 2026")
print("="*65)

def find_event_day(grp):
    grp = grp.sort_values("DaysBeforeDeparture", ascending=False)
    sold = grp["SoldOutDummy"].values
    days = grp["DaysBeforeDeparture"].values
    for i in range(1, len(sold)):
        if sold[i-1] == 0 and sold[i] == 1:
            return days[i]
    return np.nan

event_days = (df.groupby(["Route", "DepartureDate"])
                .apply(find_event_day, include_groups=False)
                .reset_index()
                .rename(columns={0: "EventDay"}))
df = df.merge(event_days, on=["Route", "DepartureDate"], how="left")
df["Treated"] = df["EventDay"].notna()
df_treated = df[df["Treated"]].copy()
df_control = df[df["SoldOutDummy"] == 0].copy()

n_treated_rd = int(df.groupby(["Route","DepartureDate"])["Treated"].first().sum())
n_control_rd = int((~df.groupby(["Route","DepartureDate"])["SoldOutDummy"].any()).sum())
total_rd     = df.groupby(["Route","DepartureDate"]).ngroups
print(f"  Route-dates with sell-out (treated): {n_treated_rd} / {total_rd} ({n_treated_rd/total_rd*100:.1f}%)")
print(f"  Route-dates never sold out (control): {n_control_rd}")

df_treated["RelDay"] = df_treated["DaysBeforeDeparture"] - df_treated["EventDay"]
pre_means = (df_treated[df_treated["RelDay"] > 0]
             .groupby(["Route", "DepartureDate"])["log_price"]
             .mean().rename("PreMean"))
df_treated = df_treated.merge(pre_means, on=["Route", "DepartureDate"], how="left")
df_treated["NormLogPrice"] = df_treated["log_price"] - df_treated["PreMean"]

window = 7
ev = (df_treated[df_treated["RelDay"].between(-window, window)]
      .groupby("RelDay")
      .agg(MeanNormLogPrice=("NormLogPrice","mean"), N=("NormLogPrice","count"))
      .reset_index())

pseudo_event = int(event_days["EventDay"].median())
df_control["RelDay"] = df_control["DaysBeforeDeparture"] - pseudo_event
pre_means_ctrl = (df_control[df_control["RelDay"] > 0]
                  .groupby(["Route","DepartureDate"])["log_price"]
                  .mean().rename("PreMean"))
df_control = df_control.merge(pre_means_ctrl, on=["Route","DepartureDate"], how="left")
df_control["NormLogPrice"] = df_control["log_price"] - df_control["PreMean"]
ctrl = (df_control[df_control["RelDay"].between(-window, window)]
        .groupby("RelDay")
        .agg(MeanNormLogPrice=("NormLogPrice","mean"), N=("NormLogPrice","count"))
        .reset_index())

ev["group"]   = "Treated (sells out)"
ctrl["group"] = "Control (never sells out)"
ev_out = pd.concat([ev, ctrl], ignore_index=True)
ev_out.to_csv("results_event_study_2026.csv", index=False)

pre_price  = ev.loc[ev["RelDay"] > 0, "MeanNormLogPrice"].mean()
post_price = ev.loc[ev["RelDay"] <= 0, "MeanNormLogPrice"].mean()
at_event   = ev.loc[ev["RelDay"] == 0, "MeanNormLogPrice"].values
print(f"\n  Pre-event mean (RelDay > 0):    {pre_price:.4f}")
print(f"  Post-event mean (RelDay ≤ 0):   {post_price:.4f}")
if len(at_event):
    print(f"  At event day (RelDay = 0):      {at_event[0]:.4f}  → "
          f"implied price change ≈ {(np.exp(at_event[0])-1)*100:.1f}%")
print(f"  Jump (post − pre):              {post_price - pre_price:.4f}  ≈ {(np.exp(post_price-pre_price)-1)*100:.1f}%")
print("\n  Event study table:")
print(ev[["RelDay","MeanNormLogPrice","N"]].to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(ev["RelDay"], ev["MeanNormLogPrice"],
        color="#F44336", linewidth=2.2, marker="o", markersize=5,
        label="Treated: route-dates where directs sell out")
ax.plot(ctrl["RelDay"], ctrl["MeanNormLogPrice"],
        color="#90CAF9", linewidth=2.2, marker="s", markersize=5,
        linestyle="--", label="Control: route-dates, directs never sell out")
ax.axvline(0, color="black", linewidth=1.2, linestyle="--", label="Event: first sell-out day")
ax.axhline(0, color="grey",  linewidth=0.8, linestyle=":")
ax.fill_betweenx([-0.15, 0.25], -window, 0, alpha=0.06, color="#F44336")
ax.set_xlabel("Days relative to sell-out event  (0 = first day direct disappears)", fontsize=11)
ax.set_ylabel("Mean normalised log(IndirectPrice)\n(relative to pre-event mean)", fontsize=11)
ax.set_title("Event Study: Indirect Price Response to Direct Flight Sell-Out\n"
             "2026 Dataset — Full Window", fontsize=12)
ax.legend(fontsize=10, framealpha=0.9)
ax.set_xlim(-window - 0.5, window + 0.5)
ax.set_xticks(range(-window, window+1))
ax.set_xticklabels([str(x) if x % 2 == 0 else "" for x in range(-window, window+1)])
ax.grid(linestyle=":", alpha=0.4)
fig.tight_layout()
fig.savefig("plots_2026/07_event_study_2026.png", dpi=150)
plt.close()
print("  Plot: plots_2026/07_event_study_2026.png")


# =============================================================================
# EXP 4 — LCC heterogeneity interaction
# =============================================================================
print("\n" + "="*65)
print("EXP 4: LCC Heterogeneity — 2026")
print("="*65)

df_lcc = df.dropna(subset=["lccdum"]).copy()
df_lcc["lccdum"] = df_lcc["lccdum"].astype(int)
print(f"  N (with LCCdum): {len(df_lcc):,}")
print(f"  LCC=1: {df_lcc['lccdum'].sum():,} ({df_lcc['lccdum'].mean()*100:.1f}%)")

formula_lcc = (
    "log_price ~ "
    "log_DirectPrice_avg + log_DirectPrice_x_Days + "
    "log_TravelTime + log_TravelTime_sq + SoldOutDummy + "
    "lccdum + lccdum:log_DirectPrice_avg + lccdum:SoldOutDummy + "
    "C(Route) + C(DepartureDate)"
)
res_lcc = smf.ols(formula_lcc, data=df_lcc).fit(
    cov_type="cluster", cov_kwds={"groups": df_lcc["Route"]}
)
print(f"\n  R²={res_lcc.rsquared:.4f}  N={int(res_lcc.nobs):,}")

INTEREST_VARS = [
    "log_DirectPrice_avg", "log_DirectPrice_x_Days",
    "log_TravelTime", "log_TravelTime_sq", "SoldOutDummy", "lccdum",
    "log_DirectPrice_avg:lccdum", "SoldOutDummy:lccdum",
]
rows_lcc = []
print(f"\n  {'Variable':<55} {'Coef':>8}  {'SE':>8}  {'t':>6}  {'p':>7}  Sig")
print("  " + "-"*90)
for var in INTEREST_VARS:
    if var not in res_lcc.params:
        alt = ":".join(var.split(":")[::-1])
        if alt in res_lcc.params:
            var = alt
        else:
            continue
    c = res_lcc.params[var]; se = res_lcc.bse[var]
    t = res_lcc.tvalues[var]; p = res_lcc.pvalues[var]
    s = sig(p)
    print(f"  {var:<55} {c:>8.4f}  {se:>8.4f}  {t:>6.2f}  {p:>7.4f}  {s}")
    rows_lcc.append({"variable":var,"coef":round(c,6),"std_err":round(se,6),
                     "t_stat":round(t,4),"p_value":round(p,6),"sig":s,
                     "n_obs":int(res_lcc.nobs),"r2":round(res_lcc.rsquared,6)})

a1 = res_lcc.params["log_DirectPrice_avg"]
b2_key = "log_DirectPrice_avg:lccdum" if "log_DirectPrice_avg:lccdum" in res_lcc.params else "lccdum:log_DirectPrice_avg"
b2 = res_lcc.params.get(b2_key, 0)
a5 = res_lcc.params["SoldOutDummy"]
b3_key = "SoldOutDummy:lccdum" if "SoldOutDummy:lccdum" in res_lcc.params else "lccdum:SoldOutDummy"
b3 = res_lcc.params.get(b3_key, 0)
print(f"\n  Implied total coefficients:")
print(f"    Substitution (non-LCC): α1      = {a1:.4f}")
print(f"    Substitution (LCC):     α1+β2   = {a1+b2:.4f}")
print(f"    Scarcity (non-LCC):     α5      = {a5:.4f}  ({(np.exp(a5)-1)*100:.1f}%)")
print(f"    Scarcity (LCC):         α5+β3   = {a5+b3:.4f}  ({(np.exp(a5+b3)-1)*100:.1f}%)")

pd.DataFrame(rows_lcc).to_csv("results_lcc_interaction_2026.csv", index=False)

fig, axes = plt.subplots(1,2,figsize=(11,5))
groups = ["Non-LCC","LCC"]; colors = ["#2196F3","#FF5722"]
for ax, vals, title, ylabel in [
    (axes[0], [a1, a1+b2], "Substitution Elasticity (α1)\nby Route Type",
     "Coefficient on log(DirectPrice_avg)"),
    (axes[1], [a5, a5+b3], "Scarcity Effect (α5)\nby Route Type",
     "Coefficient on SoldOutDummy"),
]:
    bars = ax.bar(groups, vals, color=colors, edgecolor="white", width=0.45)
    for bar,val in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2, val+(0.003 if val>=0 else -0.008),
                f"{val:.4f}", ha="center", va="bottom" if val>=0 else "top",
                fontsize=11, fontweight="bold")
    ax.axhline(0,color="black",linewidth=0.8,linestyle="--")
    ax.set_title(title,fontsize=11); ax.set_ylabel(ylabel,fontsize=10)
    ax.set_ylim(min(vals)*1.4-0.02, max(vals)*1.4+0.02)
    ax.grid(axis="y",linestyle=":",alpha=0.4)
fig.suptitle("LCC vs Non-LCC Routes: How Pricing Mechanisms Differ\n"
             "2026 Dataset — Full Window  |  SEs clustered at route level",
             fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig("plots_2026/08_lcc_coef_comparison_2026.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Plot: plots_2026/08_lcc_coef_comparison_2026.png")


# =============================================================================
# Q1 — 1-stop vs 2-stop
# =============================================================================
print("\n" + "="*65)
print("Q1: 1-stop vs 2-stop substitution — 2026")
print("="*65)

results_q1 = {}
for n, label in [(1,"1-stop"),(2,"2-stop")]:
    sub = df[df["ConnectionNumber"]==n].copy()
    if len(sub) < 100:
        print(f"  {label}: only {len(sub)} rows, skipping")
        continue
    res = cluster_ols(FORMULA_BASE, sub)
    results_q1[label] = res
    print(f"\n  {label}  N={len(sub):,}  R²={res.rsquared:.4f}")
    print_coef(res, ["log_DirectPrice_avg","log_DirectPrice_x_Days",
                     "log_TravelTime","log_TravelTime_sq","SoldOutDummy"])

if len(results_q1) >= 2:
    KEY = ["log_DirectPrice_avg","log_DirectPrice_x_Days",
           "log_TravelTime","log_TravelTime_sq","SoldOutDummy"]
    XLABELS = ["α1 log(DirectPrice)","α2 ×Days","α3 log(TravelTime)","α4 log(TT)²","α5 SoldOut"]
    x = np.arange(len(KEY)); w = 0.35
    fig, ax = plt.subplots(figsize=(11,5))
    for i,(label,color) in enumerate([("1-stop",PALETTE["one_stop"]),("2-stop",PALETTE["two_stop"])]):
        if label not in results_q1: continue
        res = results_q1[label]
        coefs = [res.params.get(v,0) for v in KEY]
        errs  = [res.bse.get(v,0)*1.96 for v in KEY]
        ax.bar(x+(i-0.5)*w, coefs, w, yerr=errs, capsize=4,
               color=color, alpha=0.88, label=label)
    ax.axhline(0,color="black",linewidth=0.8,linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels(XLABELS,fontsize=10)
    ax.set_ylabel("Coefficient"); ax.legend(fontsize=10)
    ax.set_title("Q1: Coefficients by Connection Depth\n2026 Full Window | 95% CI | SEs clustered at route level",fontsize=11)
    ax.grid(axis="y",linestyle=":",alpha=0.5)
    fig.tight_layout()
    fig.savefig("plots_2026/09_connection_depth_2026.png",dpi=150)
    plt.close()
    print("  Plot: plots_2026/09_connection_depth_2026.png")


# =============================================================================
# Q2 — Continuous sell-out intensity (already in file)
# =============================================================================
print("\n" + "="*65)
print("Q2: Continuous sell-out intensity — 2026")
print("="*65)

print(f"  SoldOutIntensity: mean={df['SoldOutIntensity'].mean():.4f} "
      f"| >0: {(df['SoldOutIntensity']>0).mean()*100:.1f}%"
      f"| =1 (fully sold out): {(df['SoldOutIntensity']==1).mean()*100:.1f}%")

formula_q2 = (
    "log_price ~ log_DirectPrice_avg + log_DirectPrice_x_Days + "
    "log_TravelTime + log_TravelTime_sq + SoldOutIntensity + "
    "C(Route) + C(DepartureDate)"
)
res_q2 = cluster_ols(formula_q2, df)
print(f"\n  N={int(res_q2.nobs):,}  R²={res_q2.rsquared:.4f}")
print_coef(res_q2, ["log_DirectPrice_avg","log_DirectPrice_x_Days",
                    "log_TravelTime","log_TravelTime_sq","SoldOutIntensity"])

df["IntensityBin"] = pd.cut(df["SoldOutIntensity"],
                             bins=[-0.01,0,0.25,0.5,0.75,1.01],
                             labels=["0% (none)","1–25%","26–50%","51–75%","76–100%"])
binned = df.groupby("IntensityBin", observed=True)["price"].median()

fig, axes = plt.subplots(1,2,figsize=(12,4.5))
axes[0].bar(binned.index, binned.values,
            color=[PALETTE["neutral"],"#64B5F6","#1E88E5","#1565C0","#0D47A1"],
            edgecolor="white")
for i,(idx,val) in enumerate(binned.items()):
    axes[0].text(i, val+2, f"€{val:.0f}", ha="center",fontsize=9,fontweight="bold")
axes[0].set_xlabel("Sell-out intensity (% of direct flights gone)")
axes[0].set_ylabel("Median indirect price (€)")
axes[0].set_title("Median Indirect Price by Sell-Out Intensity")
axes[0].grid(axis="y",linestyle=":",alpha=0.4)

coef_q2 = res_q2.params["SoldOutIntensity"]
x_int = np.linspace(0,1,100)
y_int = coef_q2 * x_int
axes[1].plot(x_int*100, y_int, color="#0D47A1",linewidth=2.2)
axes[1].axhline(0,color="grey",linewidth=0.8,linestyle="--")
axes[1].fill_between(x_int*100,0,y_int,alpha=0.15,color="#0D47A1")
axes[1].set_xlabel("Sell-out intensity (%)"); axes[1].set_ylabel("Effect on log(IndirectPrice)")
axes[1].set_title(f"Model-implied effect\n(coef={coef_q2:.4f}, p={res_q2.pvalues['SoldOutIntensity']:.4f})")
axes[1].grid(linestyle=":",alpha=0.4)
fig.suptitle("Q2: Continuous Sell-Out Intensity vs Indirect Price — 2026 Full Window",
             fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig("plots_2026/10_soldout_intensity_2026.png",dpi=150,bbox_inches="tight")
plt.close()
print("  Plot: plots_2026/10_soldout_intensity_2026.png")


# =============================================================================
# Q3 — lccpercentage continuous
# =============================================================================
print("\n" + "="*65)
print("Q3: lccpercentage (continuous) route competition — 2026")
print("="*65)

df3 = df.dropna(subset=["lccpercentage"]).copy()
df3["lccpct_x_logDP"]    = df3["lccpercentage"] * df3["log_DirectPrice_avg"]
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

a1_q3   = res_q3.params["log_DirectPrice_avg"]
b_dp    = res_q3.params["lccpct_x_logDP"]
a5_q3   = res_q3.params["SoldOutDummy"]
b_so    = res_q3.params["lccpct_x_SoldOut"]
lcc_range = np.linspace(0,1,100)

fig, axes = plt.subplots(1,2,figsize=(12,4.5))
axes[0].plot(lcc_range*100, a1_q3+b_dp*lcc_range, color=PALETTE["one_stop"],linewidth=2.2)
axes[0].set_xlabel("LCC route share (%)"); axes[0].set_ylabel("Implied substitution elasticity α1")
axes[0].set_title(f"Substitution elasticity vs LCC share\ngradient coef={b_dp:.4f} (p={res_q3.pvalues['lccpct_x_logDP']:.4f})")
axes[0].grid(linestyle=":",alpha=0.4)
axes[1].plot(lcc_range*100, (np.exp(a5_q3+b_so*lcc_range)-1)*100, color=PALETTE["two_stop"],linewidth=2.2)
axes[1].set_xlabel("LCC route share (%)"); axes[1].set_ylabel("Implied scarcity effect (%)")
axes[1].set_title(f"Scarcity effect vs LCC share\ngradient coef={b_so:.4f} (p={res_q3.pvalues['lccpct_x_SoldOut']:.4f})")
axes[1].grid(linestyle=":",alpha=0.4)
fig.suptitle("Q3: LCC Route Share vs Pricing Mechanism — 2026",fontsize=12,y=1.01)
fig.tight_layout()
fig.savefig("plots_2026/11_lccpercentage_interaction_2026.png",dpi=150,bbox_inches="tight")
plt.close()
print("  Plot: plots_2026/11_lccpercentage_interaction_2026.png")


# =============================================================================
# Q4 — Asymmetric price response
# =============================================================================
print("\n" + "="*65)
print("Q4: Asymmetric price response (up vs down) — 2026")
print("="*65)

rd_mean = (df.groupby(["Route","DepartureDate"])["log_DirectPrice_avg"]
             .mean().rename("logDP_mean").reset_index())
df4 = df.merge(rd_mean, on=["Route","DepartureDate"])
df4["logDP_dev"]  = df4["log_DirectPrice_avg"] - df4["logDP_mean"]
df4["logDP_up"]   = df4["logDP_dev"].clip(lower=0)
df4["logDP_down"] = (-df4["logDP_dev"]).clip(lower=0)
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
print_coef(res_q4, ["logDP_up","logDP_down","logDP_up_x_Days","logDP_down_x_Days",
                    "log_TravelTime","log_TravelTime_sq","SoldOutDummy"])

coef_up   = res_q4.params["logDP_up"]
coef_down = res_q4.params["logDP_down"]
diff   = coef_up - coef_down
se_up  = res_q4.bse["logDP_up"]
se_dn  = res_q4.bse["logDP_down"]
se_diff = np.sqrt(se_up**2 + se_dn**2)
t_diff  = diff / se_diff
p_diff  = 2*(1 - __import__("scipy").stats.t.cdf(abs(t_diff), df=res_q4.df_resid))
print(f"\n  Up coef={coef_up:.4f}  Down coef={coef_down:.4f}")
print(f"  Difference = {diff:.4f}  t={t_diff:.2f}  p≈{p_diff:.4f}  {sig(p_diff)}")
print(f"  → {'Significant upward asymmetry' if p_diff<0.05 and diff>0 else 'No significant asymmetry'}")

fig, ax = plt.subplots(figsize=(7,5))
vals   = [coef_up, coef_down]
labels = ["Price above route-date\naverage (UP)","Price below route-date\naverage (DOWN)"]
colors = [PALETTE["pos"],PALETTE["neg"]]
bars = ax.bar(labels,vals,color=colors,edgecolor="white",width=0.45)
for bar,val in zip(bars,vals):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.002 if val>=0 else val-0.005,
            f"{val:.4f}", ha="center", va="bottom" if val>=0 else "top",
            fontsize=12, fontweight="bold")
ax.axhline(0,color="black",linewidth=0.8,linestyle="--")
ax.set_ylabel("Coefficient on log(DirectPrice) deviation")
ax.set_title(f"Q4: Asymmetric Price Response — 2026\nUp={coef_up:.4f} vs Down={coef_down:.4f}  (diff p≈{p_diff:.3f} {sig(p_diff)})")
ax.grid(axis="y",linestyle=":",alpha=0.4)
fig.tight_layout()
fig.savefig("plots_2026/12_asymmetric_response_2026.png",dpi=150)
plt.close()
print("  Plot: plots_2026/12_asymmetric_response_2026.png")


# =============================================================================
# Q5 — Quantile regression
# =============================================================================
print("\n" + "="*65)
print("Q5: Quantile regression (cheap vs expensive) — 2026")
print("="*65)

qvars = ["log_price","log_DirectPrice_avg","log_DirectPrice_x_Days",
         "log_TravelTime","log_TravelTime_sq","SoldOutDummy"]
df5 = df[qvars+["Route","DepartureDate"]].dropna().copy()
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
    print_coef(res, ["log_DirectPrice_avg_dm","SoldOutDummy_dm"])

quantiles = [0.25, 0.50, 0.75]
a1_q = [results_q5[q].params["log_DirectPrice_avg_dm"] for q in quantiles]
a5_q = [results_q5[q].params["SoldOutDummy_dm"] for q in quantiles]
a1_ci = [results_q5[q].conf_int().loc["log_DirectPrice_avg_dm"] for q in quantiles]
a5_ci = [results_q5[q].conf_int().loc["SoldOutDummy_dm"] for q in quantiles]

fig, axes = plt.subplots(1,2,figsize=(11,4.5))
colors_q = [PALETTE["q25"],PALETTE["q50"],PALETTE["q75"]]
labels_q = ["Q25\n(cheap)","Q50\n(median)","Q75\n(expensive)"]
for ax, coefs, cis, ols_val, title, ylabel in [
    (axes[0], a1_q, a1_ci, OLS_ALPHA1,
     "Substitution Elasticity (α1) by Price Quantile", "Coef on log(DirectPrice_avg)"),
    (axes[1], a5_q, a5_ci, OLS_ALPHA5,
     "Scarcity Effect (α5) by Price Quantile",         "Coef on SoldOutDummy"),
]:
    bars = ax.bar(labels_q,coefs,color=colors_q,edgecolor="white",width=0.45)
    for bar,(lo,hi) in zip(bars,cis):
        mid = bar.get_x()+bar.get_width()/2
        ax.plot([mid,mid],[lo,hi],color="black",linewidth=1.5)
    for bar,val in zip(bars,coefs):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.003,
                f"{val:.4f}", ha="center", fontsize=9, fontweight="bold")
    ax.axhline(ols_val,color="grey",linewidth=1.5,linestyle="--",label=f"OLS (full)={ols_val:.4f}")
    ax.axhline(0,color="black",linewidth=0.6)
    ax.set_title(title,fontsize=10); ax.set_ylabel(ylabel,fontsize=10)
    ax.legend(fontsize=9); ax.grid(axis="y",linestyle=":",alpha=0.4)
fig.suptitle("Q5: Are Cheap and Expensive Indirect Flights Equally Responsive?\n"
             "Quantile regression with within-FE demeaning | 2026 Full Window",
             fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig("plots_2026/13_quantile_regression_2026.png",dpi=150,bbox_inches="tight")
plt.close()
print("  Plot: plots_2026/13_quantile_regression_2026.png")


# =============================================================================
# Q6 — Which routes sell out earliest?
# =============================================================================
print("\n" + "="*65)
print("Q6: What predicts sell-out timing? — 2026")
print("="*65)

event_days_q6 = (df.groupby(["Route","DepartureDate"])
                   .apply(find_event_day, include_groups=False)
                   .reset_index().rename(columns={0:"EventDay"}))
event_days_q6 = event_days_q6[event_days_q6["EventDay"].notna()].copy()
total_rd_q6   = df.groupby(["Route","DepartureDate"]).ngroups
pct_sold = len(event_days_q6)/total_rd_q6*100

print(f"  Route-dates with sell-out: {len(event_days_q6)} / {total_rd_q6} ({pct_sold:.1f}%)")
print(f"  EventDay: mean={event_days_q6['EventDay'].mean():.1f}  "
      f"std={event_days_q6['EventDay'].std():.1f}  "
      f"range={event_days_q6['EventDay'].min():.0f}–{event_days_q6['EventDay'].max():.0f}")

route_chars = (df.groupby("Route")
                 .agg(lccpercentage=("lccpercentage","first"),
                      MeanDirectPrice=("DirectPrice_avg","mean"),
                      MeanTravelTime=("TravelTimeMinutes","mean"))
                 .reset_index())
init_dc = (df.groupby(["Route","DepartureDate"])
             .agg(InitialDirectCount=("InitialDirectCount","first"))
             .reset_index())

df6 = (event_days_q6
       .merge(route_chars, on="Route", how="left")
       .merge(init_dc, on=["Route","DepartureDate"], how="left"))
df6["log_MeanDirectPrice"] = np.log(df6["MeanDirectPrice"])
df6 = df6.dropna(subset=["EventDay","lccpercentage","log_MeanDirectPrice"])

formula_q6 = "EventDay ~ lccpercentage + log_MeanDirectPrice + InitialDirectCount + C(DepartureDate)"
res_q6 = smf.ols(formula_q6, data=df6).fit()
print(f"\n  R²={res_q6.rsquared:.4f}  N={int(res_q6.nobs)}")
print_coef(res_q6, ["lccpercentage","log_MeanDirectPrice","InitialDirectCount"])

fig, axes = plt.subplots(1,3,figsize=(14,4.5))
for ax, xcol, xlabel in [
    (axes[0], "lccpercentage",       "LCC route share"),
    (axes[1], "log_MeanDirectPrice", "log(Mean direct price)"),
    (axes[2], "InitialDirectCount",  "Initial direct flight count"),
]:
    valid = df6[[xcol,"EventDay"]].dropna()
    ax.scatter(valid[xcol], valid["EventDay"],
               alpha=0.5,s=25,color=PALETTE["neutral"],edgecolors="none")
    m,b = np.polyfit(valid[xcol],valid["EventDay"],1)
    xr = np.linspace(valid[xcol].min(),valid[xcol].max(),100)
    ax.plot(xr, m*xr+b, color=PALETTE["two_stop"],linewidth=2)
    ax.set_xlabel(xlabel); ax.set_ylabel("Event Day")
    ax.set_title(xlabel); ax.grid(linestyle=":",alpha=0.4)
fig.suptitle("Q6: What Predicts When Direct Flights Sell Out? — 2026\n"
             "Higher EventDay = sells out earlier (more days before departure)",
             fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig("plots_2026/14_sellout_timing_2026.png",dpi=150,bbox_inches="tight")
plt.close()
print("  Plot: plots_2026/14_sellout_timing_2026.png")


# =============================================================================
# Q7 — Price gap over booking horizon
# =============================================================================
print("\n" + "="*65)
print("Q7: Price gap (indirect − direct) over booking horizon — 2026")
print("="*65)

df7 = df.copy()
df7["PriceGap"]    = df7["price"] - df7["DirectPrice_avg"]
df7["LogPriceGap"] = df7["log_price"] - df7["log_DirectPrice_avg"]

gap_by_day = (df7.groupby("DaysBeforeDeparture")
                  .agg(MedianGap=("PriceGap","median"),
                       MeanGap=("PriceGap","mean"),
                       N=("PriceGap","count"))
                  .reset_index())

print(f"  Overall median gap: €{df7['PriceGap'].median():.1f}")
# Check extremes robustly
if 36 in df7["DaysBeforeDeparture"].values:
    v36 = df7[df7["DaysBeforeDeparture"]==36]["PriceGap"].median()
    print(f"  At 36 days: €{v36:.1f}")
if 1 in df7["DaysBeforeDeparture"].values:
    v1 = df7[df7["DaysBeforeDeparture"]==1]["PriceGap"].median()
    print(f"  At 1 day:   €{v1:.1f}")

formula_q7  = "LogPriceGap ~ DaysBeforeDeparture + C(Route) + C(DepartureDate)"
res_q7      = cluster_ols(formula_q7, df7)
gap_coef    = res_q7.params["DaysBeforeDeparture"]
gap_p       = res_q7.pvalues["DaysBeforeDeparture"]
formula_q7b = "LogPriceGap ~ DaysBeforeDeparture + SoldOutDummy + C(Route) + C(DepartureDate)"
res_q7b     = cluster_ols(formula_q7b, df7)
so_coef     = res_q7b.params["SoldOutDummy"]
so_p        = res_q7b.pvalues["SoldOutDummy"]
print(f"\n  Log-gap ~ Days: coef={gap_coef:.5f}  p={gap_p:.4f}  {sig(gap_p)}")
print(f"  SoldOut on gap: coef={so_coef:.4f}  p={so_p:.4f}  {sig(so_p)}")
print(f"  → Sell-out widens gap by {(np.exp(so_coef)-1)*100:.1f}%")

gap_soldout = (df7.groupby(["DaysBeforeDeparture","SoldOutDummy"])
                   .agg(MedianGap=("PriceGap","median"))
                   .reset_index())
fig, axes = plt.subplots(1,2,figsize=(13,5))
axes[0].plot(gap_by_day["DaysBeforeDeparture"], gap_by_day["MedianGap"],
             color=PALETTE["gap"],linewidth=2.2,label="Median gap (€)")
axes[0].axhline(df7["PriceGap"].median(), color="grey",linewidth=1,linestyle="--",
                label=f"Overall median €{df7['PriceGap'].median():.0f}")
axes[0].set_xlabel("Days Before Departure"); axes[0].set_ylabel("Median price gap: indirect − direct (€)")
axes[0].set_title("Price Gap Over Booking Horizon\n(positive = indirect more expensive)")
axes[0].invert_xaxis(); axes[0].legend(fontsize=9); axes[0].grid(linestyle=":",alpha=0.4)

for sold_val,color,label in [(0,PALETTE["one_stop"],"Direct available (SoldOut=0)"),
                              (1,PALETTE["two_stop"],"Direct sold out (SoldOut=1)")]:
    sub = gap_soldout[gap_soldout["SoldOutDummy"]==sold_val]
    axes[1].plot(sub["DaysBeforeDeparture"],sub["MedianGap"],color=color,linewidth=2.2,label=label)
axes[1].axhline(0,color="black",linewidth=0.6,linestyle="--")
axes[1].set_xlabel("Days Before Departure"); axes[1].set_ylabel("Median price gap: indirect − direct (€)")
axes[1].set_title(f"Price Gap by Sell-Out Status\nSoldOut widens gap by {(np.exp(so_coef)-1)*100:.1f}% (p={so_p:.4f})")
axes[1].invert_xaxis(); axes[1].legend(fontsize=9); axes[1].grid(linestyle=":",alpha=0.4)
fig.suptitle("Q7: Price Gap (Indirect − Direct) Over Booking Horizon — 2026",fontsize=11,y=1.02)
fig.tight_layout()
fig.savefig("plots_2026/15_price_gap_2026.png",dpi=150,bbox_inches="tight")
plt.close()
print("  Plot: plots_2026/15_price_gap_2026.png")

print("\n\n=== ROBUSTNESS TESTS ===")

# =============================================================================
# TEST 1 — First differences
# =============================================================================
print("\n" + "="*65)
print("TEST 1: First differences regression — 2026")
print("="*65)

dp_wide = (df.groupby(["Route","DepartureDate","DaysBeforeDeparture"])["log_DirectPrice_avg"]
             .mean().reset_index()
             .sort_values(["Route","DepartureDate","DaysBeforeDeparture"]))
dp_wide["logDP_lag"]    = dp_wide.groupby(["Route","DepartureDate"])["log_DirectPrice_avg"].shift(1)
dp_wide["day_lag_dp"]   = dp_wide.groupby(["Route","DepartureDate"])["DaysBeforeDeparture"].shift(1)
dp_wide["delta_logDP"]  = dp_wide["log_DirectPrice_avg"] - dp_wide["logDP_lag"]
dp_wide = dp_wide[dp_wide["DaysBeforeDeparture"] - dp_wide["day_lag_dp"] == 1].dropna(subset=["delta_logDP"])

so_wide = (df.groupby(["Route","DepartureDate","DaysBeforeDeparture"])["SoldOutDummy"]
             .mean().reset_index()
             .sort_values(["Route","DepartureDate","DaysBeforeDeparture"]))
so_wide["SoldOut_lag"]    = so_wide.groupby(["Route","DepartureDate"])["SoldOutDummy"].shift(1)
so_wide["day_lag_so"]     = so_wide.groupby(["Route","DepartureDate"])["DaysBeforeDeparture"].shift(1)
so_wide["delta_SoldOut"]  = so_wide["SoldOutDummy"] - so_wide["SoldOut_lag"]
so_wide = so_wide[so_wide["DaysBeforeDeparture"] - so_wide["day_lag_so"] == 1]

it_id = ["Route","DepartureDate","TravelTimeMinutes","ConnectionNumber"]
df1 = df.sort_values(it_id+["DaysBeforeDeparture"]).copy()
df1["logprice_lag"] = df1.groupby(it_id)["log_price"].shift(1)
df1["day_lag"]      = df1.groupby(it_id)["DaysBeforeDeparture"].shift(1)
df1["delta_logprice"] = df1["log_price"] - df1["logprice_lag"]
df1 = df1[df1["DaysBeforeDeparture"]-df1["day_lag"]==1].dropna(subset=["delta_logprice"])

df1 = df1.merge(dp_wide[["Route","DepartureDate","DaysBeforeDeparture","delta_logDP"]],
                on=["Route","DepartureDate","DaysBeforeDeparture"], how="inner")
df1 = df1.merge(so_wide[["Route","DepartureDate","DaysBeforeDeparture","delta_SoldOut"]],
                on=["Route","DepartureDate","DaysBeforeDeparture"], how="left")
df1 = df1.dropna(subset=["delta_logDP","delta_SoldOut"])

print(f"\n  First-difference pairs: {len(df1):,}")
formula_fd = "delta_logprice ~ delta_logDP + delta_SoldOut + C(Route) + C(DepartureDate)"
res_fd = cluster_ols(formula_fd, df1)
print(f"  R²={res_fd.rsquared:.4f}  N={int(res_fd.nobs):,}")
print_coef(res_fd, ["delta_logDP","delta_SoldOut"])

fig, axes = plt.subplots(1,2,figsize=(12,4.5))
sample = df1.sample(n=min(5000,len(df1)), random_state=42)
axes[0].scatter(sample["delta_logDP"],sample["delta_logprice"],
                alpha=0.2,s=8,color=PALETTE["main"],edgecolors="none")
x_line = np.linspace(sample["delta_logDP"].quantile(0.01),sample["delta_logDP"].quantile(0.99),100)
coef_fd = res_fd.params["delta_logDP"]
axes[0].plot(x_line, coef_fd*x_line, color=PALETTE["accent"],linewidth=2,
             label=f"slope={coef_fd:.4f} {sig(res_fd.pvalues['delta_logDP'])}")
axes[0].set_xlabel("Δ log(DirectPrice)"); axes[0].set_ylabel("Δ log(IndirectPrice)")
axes[0].set_title("First Differences: Day-over-Day Changes")
axes[0].legend(fontsize=10); axes[0].grid(linestyle=":",alpha=0.4)
bins60 = np.linspace(-0.5,0.5,61)
axes[1].hist(df1["delta_logprice"].clip(-0.5,0.5).values,bins=bins60,
             color=PALETTE["main"],alpha=0.7,edgecolor="white",label="Indirect Δ")
axes[1].hist(df1["delta_logDP"].clip(-0.5,0.5).values,bins=bins60,
             color=PALETTE["accent"],alpha=0.5,edgecolor="white",label="Direct Δ")
axes[1].axvline(0,color="black",linewidth=1,linestyle="--")
axes[1].set_xlabel("Day-over-day log price change"); axes[1].set_ylabel("Count")
axes[1].set_title("Distribution of Daily Price Changes")
axes[1].legend(fontsize=10); axes[1].grid(linestyle=":",alpha=0.4)
fig.suptitle("Test 1: First Differences Regression — 2026 Full Window",fontsize=12,y=1.01)
fig.tight_layout()
fig.savefig("plots_2026/16_first_differences_2026.png",dpi=150,bbox_inches="tight")
plt.close()
print("  Plot: plots_2026/16_first_differences_2026.png")


# =============================================================================
# TEST 2 — SoldOut × DirectPrice interaction
# =============================================================================
print("\n" + "="*65)
print("TEST 2: SoldOut × DirectPrice interaction — 2026")
print("="*65)

formula_int = (
    "log_price ~ log_DirectPrice_avg + log_DirectPrice_x_Days + "
    "log_TravelTime + log_TravelTime_sq + SoldOutDummy + "
    "SoldOutDummy:log_DirectPrice_avg + C(Route) + C(DepartureDate)"
)
res_int = cluster_ols(formula_int, df)
int_key = next((k for k in res_int.params.index
                if "SoldOutDummy" in k and "log_DirectPrice_avg" in k
                and k not in ["log_DirectPrice_avg","SoldOutDummy"]), None)
print(f"\n  R²={res_int.rsquared:.4f}  N={int(res_int.nobs):,}")
print_coef(res_int, ["log_DirectPrice_avg","SoldOutDummy",int_key,
                     "log_DirectPrice_x_Days","log_TravelTime","log_TravelTime_sq"])

a1_t2  = res_int.params["log_DirectPrice_avg"]
a5_t2  = res_int.params["SoldOutDummy"]
inter  = res_int.params[int_key]
p_int  = res_int.pvalues[int_key]
print(f"\n  Implied elasticity when SoldOut=0: α1        = {a1_t2:.4f}")
print(f"  Implied elasticity when SoldOut=1: α1+inter  = {a1_t2+inter:.4f}")
print(f"  Scarcity premium at avg price:     α5        = {(np.exp(a5_t2)-1)*100:.1f}%")

fig, ax = plt.subplots(figsize=(8,5))
groups  = ["SoldOut=0\n(directs available)","SoldOut=1\n(directs gone)"]
elasts  = [a1_t2, a1_t2+inter]
bars = ax.bar(groups,elasts,color=[PALETTE["main"],PALETTE["accent"]],edgecolor="white",width=0.45)
for bar,val in zip(bars,elasts):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.002,
            f"{val:.4f}",ha="center",fontsize=12,fontweight="bold")
ax.axhline(0,color="black",linewidth=0.8,linestyle="--")
ax.set_ylabel("Implied substitution elasticity")
ax.set_title(f"Test 2: SoldOut × DirectPrice Interaction — 2026\n"
             f"Interaction coef={inter:.4f}  p={p_int:.4f}  {sig(p_int)}")
ax.grid(axis="y",linestyle=":",alpha=0.4)
fig.tight_layout()
fig.savefig("plots_2026/17_soldout_x_directprice_2026.png",dpi=150)
plt.close()
print("  Plot: plots_2026/17_soldout_x_directprice_2026.png")


# =============================================================================
# TEST 3 — Hausman + specification stability
# =============================================================================
print("\n" + "="*65)
print("TEST 3: Hausman + specification stability — 2026")
print("="*65)

df_panel = df.copy()
df_panel["entity"] = df_panel["Route"]+"|"+df_panel["DepartureDate"].astype(str)
df_panel = df_panel.set_index(["entity","DaysBeforeDeparture"])
XVARS = ["log_DirectPrice_avg","log_DirectPrice_x_Days",
         "log_TravelTime","log_TravelTime_sq","SoldOutDummy"]

fe_mod = PanelOLS.from_formula(
    "log_price ~ log_DirectPrice_avg + log_DirectPrice_x_Days + "
    "log_TravelTime + log_TravelTime_sq + SoldOutDummy + EntityEffects",
    data=df_panel
).fit(cov_type="clustered",cluster_entity=True)

re_mod = RandomEffects.from_formula(
    "log_price ~ log_DirectPrice_avg + log_DirectPrice_x_Days + "
    "log_TravelTime + log_TravelTime_sq + SoldOutDummy",
    data=df_panel
).fit()

b_fe = fe_mod.params[XVARS].values
b_re = re_mod.params[XVARS].values
v_fe = fe_mod.cov.loc[XVARS,XVARS].values
v_re = re_mod.cov.loc[XVARS,XVARS].values
diff_h = b_fe - b_re
v_diff = v_fe - v_re
try:
    H_stat = float(diff_h @ np.linalg.inv(v_diff) @ diff_h)
    from scipy.stats import chi2
    H_pval = 1 - chi2.cdf(H_stat, df=len(XVARS))
    print(f"\n  Hausman stat={H_stat:.3f}  df={len(XVARS)}  p={H_pval:.6f}  {sig(H_pval)}")
    print(f"  → {'Reject RE: FE is consistent. Use FE.' if H_pval<0.05 else 'Cannot reject RE.'}")
except np.linalg.LinAlgError:
    H_stat, H_pval = np.nan, np.nan
    print("  Hausman: singular matrix — FE/RE comparison instead")

specs = {
    "Pooled OLS (no FE)":          "log_price ~ log_DirectPrice_avg + log_DirectPrice_x_Days + log_TravelTime + log_TravelTime_sq + SoldOutDummy",
    "Route FE only":               "log_price ~ log_DirectPrice_avg + log_DirectPrice_x_Days + log_TravelTime + log_TravelTime_sq + SoldOutDummy + C(Route)",
    "Date FE only":                "log_price ~ log_DirectPrice_avg + log_DirectPrice_x_Days + log_TravelTime + log_TravelTime_sq + SoldOutDummy + C(DepartureDate)",
    "Route + Date FE (baseline)":  FORMULA_BASE,
}
spec_results = {}
print(f"\n  {'Specification':<35} {'α1':>8}  {'SE':>8}  {'α5':>8}  {'R²':>6}")
print("  " + "-"*65)
for label, formula in specs.items():
    r = smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups":df["Route"]})
    a1v = r.params["log_DirectPrice_avg"]
    a5v = r.params["SoldOutDummy"]
    spec_results[label] = r
    print(f"  {label:<35} {a1v:>8.4f}  {r.bse['log_DirectPrice_avg']:>8.4f}  {a5v:>8.4f}  {r.rsquared:>6.4f}")

labels_short = ["No FE","Route FE","Date FE","Route+Date\n(baseline)"]
colors_s = [PALETTE["neutral"],PALETTE["light"],PALETTE["accent"],PALETTE["main"]]
fig, axes = plt.subplots(1,2,figsize=(12,4.5))
for ax, key, title, ols_val in [
    (axes[0],"log_DirectPrice_avg","α1: Substitution elasticity",OLS_ALPHA1),
    (axes[1],"SoldOutDummy",       "α5: Scarcity effect",        OLS_ALPHA5),
]:
    vals = [spec_results[l].params[key] for l in specs]
    errs = [spec_results[l].bse[key]*1.96 for l in specs]
    axes[list(axes).index(ax)].bar(labels_short,vals,color=colors_s,
                                   edgecolor="white",width=0.55,yerr=errs,capsize=4)
    ax.axhline(ols_val,color="grey",linewidth=1.5,linestyle="--",label=f"Baseline={ols_val}")
    ax.set_title(title,fontsize=10); ax.set_ylabel("Coefficient")
    ax.legend(fontsize=9); ax.grid(axis="y",linestyle=":",alpha=0.4)
hausman_str = f"{H_stat:.2f}" if not np.isnan(H_stat) else "n/a"
hpval_str   = f"{H_pval:.4f}" if not np.isnan(H_pval) else "n/a"
fig.suptitle(f"Test 3: Coefficient Stability — 2026\nHausman stat={hausman_str} p={hpval_str}",
             fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig("plots_2026/18_specification_stability_2026.png",dpi=150,bbox_inches="tight")
plt.close()
print("  Plot: plots_2026/18_specification_stability_2026.png")


# =============================================================================
# TEST 4 — Residual diagnostics
# =============================================================================
print("\n" + "="*65)
print("TEST 4: Residual diagnostics — 2026")
print("="*65)

X_bp = smf.ols(FORMULA_BASE, data=df).fit()
bp_stat, bp_p, f_stat, f_p = sms.het_breuschpagan(X_bp.resid, X_bp.model.exog)
print(f"\n  (a) Breusch-Pagan: LM={bp_stat:.2f}  p={bp_p:.6f}  {sig(bp_p)}")

vif_data = df[["log_DirectPrice_avg","log_DirectPrice_x_Days",
               "log_TravelTime","log_TravelTime_sq","SoldOutDummy"]].dropna()
vif_data = (vif_data - vif_data.mean()) / vif_data.std()
vif_data["const"] = 1
print(f"\n  (b) VIF:")
vif_vals = {}
for i,col in enumerate(["log_DirectPrice_avg","log_DirectPrice_x_Days",
                         "log_TravelTime","log_TravelTime_sq","SoldOutDummy"]):
    v = variance_inflation_factor(vif_data.values, i)
    vif_vals[col] = v
    flag = " ← HIGH (>10)" if v>10 else " ← MODERATE (5–10)" if v>5 else ""
    print(f"      {col:<40} VIF={v:.2f}{flag}")

resid_df = df[["Route","DepartureDate","DaysBeforeDeparture"]].copy()
resid_df["resid"] = X_bp.resid
resid_df = resid_df.sort_values(["Route","DepartureDate","DaysBeforeDeparture"])
resid_df["resid_lag"] = resid_df.groupby(["Route","DepartureDate"])["resid"].shift(1)
resid_df = resid_df.dropna(subset=["resid_lag"])
sc_res  = smf.ols("resid ~ resid_lag", data=resid_df).fit()
sc_coef = sc_res.params["resid_lag"]
sc_p    = sc_res.pvalues["resid_lag"]
print(f"\n  (c) Serial correlation: lag-1 coef={sc_coef:.4f}  p={sc_p:.6f}  {sig(sc_p)}")

resids = X_bp.resid; fitted = X_bp.fittedvalues
fig = plt.figure(figsize=(14,4))
gs  = gridspec.GridSpec(1,3)
ax1 = fig.add_subplot(gs[0])
ax1.scatter(fitted[:5000],resids[:5000],alpha=0.15,s=4,color=PALETTE["main"])
ax1.axhline(0,color="red",linewidth=1)
ax1.set_xlabel("Fitted"); ax1.set_ylabel("Residuals")
ax1.set_title(f"Residuals vs Fitted\nBP p={bp_p:.2e}")
ax1.grid(linestyle=":",alpha=0.3)
ax2 = fig.add_subplot(gs[1])
_r = resids.clip(-2,2)
ax2.hist(_r.values,bins=np.linspace(_r.min(),_r.max(),81),
         color=PALETTE["main"],alpha=0.8,edgecolor="white")
ax2.set_xlabel("Residual"); ax2.set_ylabel("Count")
ax2.set_title("Residual Distribution"); ax2.grid(linestyle=":",alpha=0.3)
ax3 = fig.add_subplot(gs[2])
vif_color = [PALETTE["neg"] if v>10 else PALETTE["accent"] if v>5 else PALETTE["pos"]
             for v in vif_vals.values()]
ax3.barh(list(vif_vals.keys()),list(vif_vals.values()),color=vif_color,edgecolor="white")
ax3.axvline(5,color="orange",linestyle="--",linewidth=1,label="VIF=5")
ax3.axvline(10,color="red",linestyle="--",linewidth=1,label="VIF=10")
ax3.set_xlabel("VIF"); ax3.set_title("Variance Inflation Factors")
ax3.legend(fontsize=8); ax3.grid(axis="x",linestyle=":",alpha=0.4)
fig.suptitle("Test 4: Residual Diagnostics — 2026 Full Window OLS",fontsize=12)
fig.tight_layout()
fig.savefig("plots_2026/19_residual_diagnostics_2026.png",dpi=150,bbox_inches="tight")
plt.close()
print("  Plot: plots_2026/19_residual_diagnostics_2026.png")


# =============================================================================
# TEST 5 — Granger causality
# =============================================================================
print("\n" + "="*65)
print("TEST 5: Granger causality — 2026")
print("="*65)

agg = (df.groupby(["Route","DepartureDate","DaysBeforeDeparture"])
         .agg(mean_logprice=("log_price","mean"),
              log_dp=("log_DirectPrice_avg","first"))
         .reset_index()
         .sort_values(["Route","DepartureDate","DaysBeforeDeparture"]))

min_obs = 15
pvals_dp, pvals_rev = [], []
n_eligible = 0
for (route,date), grp in agg.groupby(["Route","DepartureDate"]):
    grp = grp.sort_values("DaysBeforeDeparture", ascending=False)
    if len(grp) < min_obs:
        continue
    n_eligible += 1
    ts = grp[["log_dp","mean_logprice"]].values
    try:
        res_gc  = grangercausalitytests(ts[:,[1,0]], maxlag=2, verbose=False)
        pvals_dp.append(min(res_gc[1][0]["ssr_ftest"][1],res_gc[2][0]["ssr_ftest"][1]))
        res_rev = grangercausalitytests(ts[:,[0,1]], maxlag=2, verbose=False)
        pvals_rev.append(min(res_rev[1][0]["ssr_ftest"][1],res_rev[2][0]["ssr_ftest"][1]))
    except Exception:
        continue

pvals_dp  = np.array(pvals_dp)
pvals_rev = np.array(pvals_rev)
print(f"\n  Eligible route-dates (≥{min_obs} obs): {n_eligible}")
print(f"  Direct → Indirect: {(pvals_dp<0.05).sum()}/{len(pvals_dp)} ({(pvals_dp<0.05).mean()*100:.1f}%)")
print(f"    Median p-value: {np.median(pvals_dp):.4f}")
print(f"  Indirect → Direct (reverse): {(pvals_rev<0.05).sum()}/{len(pvals_rev)} ({(pvals_rev<0.05).mean()*100:.1f}%)")
print(f"    Median p-value: {np.median(pvals_rev):.4f}")

fig, axes = plt.subplots(1,2,figsize=(11,4.5))
for ax,pv,title,color in [
    (axes[0],pvals_dp, "Direct → Indirect\n(hypothesised direction)",PALETTE["main"]),
    (axes[1],pvals_rev,"Indirect → Direct\n(reverse / should be weaker)",PALETTE["accent"]),
]:
    if len(pv)==0:
        ax.text(0.5,0.5,"No eligible route-dates",ha="center",transform=ax.transAxes)
        continue
    ax.hist(pv,bins=30,color=color,alpha=0.8,edgecolor="white")
    ax.axvline(0.05,color="red",linewidth=1.5,linestyle="--",label="p=0.05")
    ax.set_xlabel("Granger p-value"); ax.set_ylabel("Route-date count")
    ax.set_title(f"{title}\n{(pv<0.05).mean()*100:.1f}% of route-dates: p < 0.05")
    ax.legend(fontsize=9); ax.grid(linestyle=":",alpha=0.4)
fig.suptitle("Test 5: Granger Causality — 2026 Full Window",fontsize=11,y=1.02)
fig.tight_layout()
fig.savefig("plots_2026/20_granger_causality_2026.png",dpi=150,bbox_inches="tight")
plt.close()
print("  Plot: plots_2026/20_granger_causality_2026.png")


# =============================================================================
# TEST 6 — Placebo test
# =============================================================================
print("\n" + "="*65)
print("TEST 6: Placebo test — shuffled SoldOutDummy — 2026")
print("="*65)

n_iter = 100
rng = np.random.default_rng(42)
placebo_coefs = []
print(f"  Running {n_iter} placebo iterations...")
so_values   = df["SoldOutDummy"].values.copy()
date_groups = df["DepartureDate"].values
unique_dates = np.unique(date_groups)
for i in range(n_iter):
    shuffled = so_values.copy()
    for d in unique_dates:
        mask = date_groups == d
        idx  = np.where(mask)[0]
        shuffled[idx] = rng.permutation(shuffled[idx])
    df_p = df.copy()
    df_p["SoldOutDummy"] = shuffled
    r = smf.ols(FORMULA_BASE, data=df_p).fit()
    placebo_coefs.append(r.params["SoldOutDummy"])
    if (i+1) % 25 == 0:
        print(f"    {i+1}/{n_iter} done...")

placebo_arr = np.array(placebo_coefs)
true_coef   = res_base.params["SoldOutDummy"]
placebo_p   = (np.abs(placebo_arr) >= np.abs(true_coef)).mean()
pct_rank    = (placebo_arr < true_coef).mean()*100

print(f"\n  True α5 (SoldOutDummy): {true_coef:.4f}")
print(f"  Placebo mean:           {placebo_arr.mean():.4f}")
print(f"  Placebo std:            {placebo_arr.std():.4f}")
print(f"  z-score:                {(true_coef-placebo_arr.mean())/placebo_arr.std():.1f}σ")
print(f"  Placebo p-value:        {placebo_p:.4f}  {sig(placebo_p)}")
print(f"  True coef at {pct_rank:.1f}th percentile of placebo distribution")

fig, ax = plt.subplots(figsize=(8,5))
ax.hist(placebo_arr,bins=30,color=PALETTE["neutral"],alpha=0.8,edgecolor="white",
        label="Placebo α5 (shuffled SoldOutDummy)")
ax.axvline(true_coef,color=PALETTE["accent"],linewidth=2.5,
           label=f"True α5 = {true_coef:.4f}")
ax.axvline(0,color="black",linewidth=0.8,linestyle="--")
ax.set_xlabel("Coefficient on SoldOutDummy")
ax.set_ylabel("Count (out of 100 placebo runs)")
ax.set_title(f"Test 6: Placebo Test — Shuffled SoldOutDummy — 2026\n"
             f"True coef at {pct_rank:.1f}th percentile  |  Placebo p={placebo_p:.4f}  {sig(placebo_p)}")
ax.legend(fontsize=10); ax.grid(linestyle=":",alpha=0.4)
fig.tight_layout()
fig.savefig("plots_2026/21_placebo_test_2026.png",dpi=150)
plt.close()
print("  Plot: plots_2026/21_placebo_test_2026.png")

print("\n\n" + "="*65)
print("ALL 2026 ANALYSES COMPLETE")
print("="*65)
print(f"Plots saved to: plots_2026/")
print(f"Results saved to: results_event_study_2026.csv, results_lcc_interaction_2026.csv")
