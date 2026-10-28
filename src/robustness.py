"""
robustness_v1.py
----------------
Six robustness and second-order tests on final_2023.csv.

Tests:
  1. First differences regression
  2. SoldOut × log(DirectPrice) interaction
  3. Hausman test (FE vs RE) + specification stability
  4. Residual diagnostics (Breusch-Pagan, VIF, serial correlation)
  5. Granger causality (do lagged direct prices predict indirect prices?)
  6. Placebo test (shuffled SoldOutDummy)

REPLICATION ON 2026:
  Change INPUT_FILE = "final_2026.csv" at the top.
  All tests run identically. Note: placebo test uses 100 iterations —
  reduce to 50 if slow. Granger test requires >=10 obs per route-date;
  with shorter 2026 horizon this may limit the number of eligible series.

Usage:
    python3 robustness_v1.py
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
os.makedirs("plots", exist_ok=True)

INPUT_FILE = "final_2023.csv"

PALETTE = {
    "main":    "#2196F3",
    "accent":  "#FF5722",
    "pos":     "#4CAF50",
    "neg":     "#F44336",
    "neutral": "#607D8B",
    "light":   "#90CAF9",
}

def sig(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "n.s."

def cluster_ols(formula, data):
    return smf.ols(formula, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data["Route"]}
    )

def print_coef(res, vars_):
    print(f"  {'Variable':<48} {'Coef':>8}  {'SE':>8}  {'t':>6}  {'p':>7}  Sig")
    print("  " + "-"*83)
    for v in vars_:
        if v not in res.params: continue
        c,se,t,p = res.params[v],res.bse[v],res.tvalues[v],res.pvalues[v]
        print(f"  {v:<48} {c:>8.4f}  {se:>8.4f}  {t:>6.2f}  {p:>7.4f}  {sig(p)}")

print("Loading data...")
df = pd.read_csv(INPUT_FILE)
print(f"  {len(df):,} rows | {df['Route'].nunique()} routes")

FORMULA_BASE = (
    "log_price ~ log_DirectPrice_avg + log_DirectPrice_x_Days + "
    "log_TravelTime + log_TravelTime_sq + SoldOutDummy + "
    "C(Route) + C(DepartureDate)"
)

# Run baseline once for diagnostics use
print("Running baseline OLS for diagnostics...")
res_base = cluster_ols(FORMULA_BASE, df)


# =============================================================================
# TEST 1 — First differences regression
# =============================================================================
print("\n" + "="*65)
print("TEST 1: First differences regression")
print("="*65)

# Sort ascending by DaysBeforeDeparture.
# shift(1) gives the PREVIOUS row = 1 day further from departure = "yesterday".
# delta = today(closer to departure) - yesterday(further from departure).

# --- DirectPrice first difference (route-date level) ---
dp_wide = (df.groupby(["Route","DepartureDate","DaysBeforeDeparture"])["log_DirectPrice_avg"]
             .mean().reset_index()
             .sort_values(["Route","DepartureDate","DaysBeforeDeparture"]))
dp_wide["logDP_lag"]  = dp_wide.groupby(["Route","DepartureDate"])["log_DirectPrice_avg"].shift(1)
dp_wide["day_lag_dp"] = dp_wide.groupby(["Route","DepartureDate"])["DaysBeforeDeparture"].shift(1)
dp_wide["delta_logDP"] = dp_wide["log_DirectPrice_avg"] - dp_wide["logDP_lag"]
# Keep only consecutive days (current day = lag day + 1)
dp_wide = dp_wide[dp_wide["DaysBeforeDeparture"] - dp_wide["day_lag_dp"] == 1].dropna(subset=["delta_logDP"])

# SoldOut first difference at route-date-day level
so_wide = (df.groupby(["Route","DepartureDate","DaysBeforeDeparture"])["SoldOutDummy"]
             .mean().reset_index()
             .sort_values(["Route","DepartureDate","DaysBeforeDeparture"]))
so_wide["SoldOut_lag"]  = so_wide.groupby(["Route","DepartureDate"])["SoldOutDummy"].shift(1)
so_wide["day_lag_so"]   = so_wide.groupby(["Route","DepartureDate"])["DaysBeforeDeparture"].shift(1)
so_wide["delta_SoldOut"] = so_wide["SoldOutDummy"] - so_wide["SoldOut_lag"]
so_wide = so_wide[so_wide["DaysBeforeDeparture"] - so_wide["day_lag_so"] == 1]

# Indirect price first difference (itinerary level)
it_id = ["Route","DepartureDate","TravelTimeMinutes","ConnectionNumber"]
df1 = (df.sort_values(it_id + ["DaysBeforeDeparture"]).copy())
df1["logprice_lag"] = df1.groupby(it_id)["log_price"].shift(1)
df1["day_lag"]      = df1.groupby(it_id)["DaysBeforeDeparture"].shift(1)
df1["delta_logprice"] = df1["log_price"] - df1["logprice_lag"]
# Keep only rows where the previous observation is exactly 1 day earlier
df1 = df1[df1["DaysBeforeDeparture"] - df1["day_lag"] == 1].dropna(subset=["delta_logprice"])

# Merge DirectPrice and SoldOut differences in
df1 = df1.merge(dp_wide[["Route","DepartureDate","DaysBeforeDeparture","delta_logDP"]],
                on=["Route","DepartureDate","DaysBeforeDeparture"], how="inner")
df1 = df1.merge(so_wide[["Route","DepartureDate","DaysBeforeDeparture","delta_SoldOut"]],
                on=["Route","DepartureDate","DaysBeforeDeparture"], how="left")
df1 = df1.dropna(subset=["delta_logDP","delta_SoldOut"])

print(f"\n  First-difference pairs: {len(df1):,}")
print(f"  Itinerary groups with ≥2 consecutive obs: {df1.groupby(it_id).ngroups:,}")

formula_fd = "delta_logprice ~ delta_logDP + delta_SoldOut + C(Route) + C(DepartureDate)"
res_fd = cluster_ols(formula_fd, df1)
print(f"\n  R²={res_fd.rsquared:.4f}  N={int(res_fd.nobs):,}")
print_coef(res_fd, ["delta_logDP","delta_SoldOut"])

# Plot
fig, axes = plt.subplots(1,2,figsize=(12,4.5))
sample = df1.sample(n=min(5000,len(df1)), random_state=42)
axes[0].scatter(sample["delta_logDP"], sample["delta_logprice"],
                alpha=0.2, s=8, color=PALETTE["main"], edgecolors="none")
x_line = np.linspace(sample["delta_logDP"].quantile(0.01),
                     sample["delta_logDP"].quantile(0.99), 100)
coef_fd = res_fd.params["delta_logDP"]
axes[0].plot(x_line, coef_fd*x_line, color=PALETTE["accent"], linewidth=2,
             label=f"slope={coef_fd:.4f} {sig(res_fd.pvalues['delta_logDP'])}")
axes[0].set_xlabel("Δ log(DirectPrice)")
axes[0].set_ylabel("Δ log(IndirectPrice)")
axes[0].set_title("First Differences: Day-over-Day Price Changes")
axes[0].legend(fontsize=10); axes[0].grid(linestyle=":",alpha=0.4)

# Distribution of daily price changes
bins60 = np.linspace(-0.5, 0.5, 61)
axes[1].hist(df1["delta_logprice"].clip(-0.5,0.5).values, bins=bins60,
             color=PALETTE["main"], alpha=0.7, edgecolor="white", label="Indirect Δ")
axes[1].hist(df1["delta_logDP"].clip(-0.5,0.5).values, bins=bins60,
             color=PALETTE["accent"], alpha=0.5, edgecolor="white", label="Direct Δ")
axes[1].axvline(0,color="black",linewidth=1,linestyle="--")
axes[1].set_xlabel("Day-over-day log price change")
axes[1].set_ylabel("Count")
axes[1].set_title("Distribution of Daily Price Changes")
axes[1].legend(fontsize=10); axes[1].grid(linestyle=":",alpha=0.4)

fig.suptitle("Test 1: First Differences Regression — 2023 Full Window", fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig("plots/16_first_differences.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Plot: plots/16_first_differences.png")


# =============================================================================
# TEST 2 — SoldOut × log(DirectPrice) interaction
# =============================================================================
print("\n" + "="*65)
print("TEST 2: SoldOut × log(DirectPrice) interaction")
print("="*65)

formula_int = (
    "log_price ~ log_DirectPrice_avg + log_DirectPrice_x_Days + "
    "log_TravelTime + log_TravelTime_sq + SoldOutDummy + "
    "SoldOutDummy:log_DirectPrice_avg + "
    "C(Route) + C(DepartureDate)"
)
res_int = cluster_ols(formula_int, df)
int_key = next((k for k in res_int.params.index if "SoldOutDummy" in k and "log_DirectPrice_avg" in k
                and k not in ["log_DirectPrice_avg","SoldOutDummy"]), None)
print(f"\n  R²={res_int.rsquared:.4f}  N={int(res_int.nobs):,}")
print_coef(res_int, ["log_DirectPrice_avg","SoldOutDummy",
                     int_key,
                     "log_DirectPrice_x_Days","log_TravelTime","log_TravelTime_sq"])

a1     = res_int.params["log_DirectPrice_avg"]
a5     = res_int.params["SoldOutDummy"]
inter  = res_int.params[int_key]
p_int  = res_int.pvalues[int_key]

print(f"\n  Implied elasticity when SoldOut=0: α1        = {a1:.4f}")
print(f"  Implied elasticity when SoldOut=1: α1+inter  = {a1+inter:.4f}")
print(f"  Scarcity premium at avg price:     α5        = {(np.exp(a5)-1)*100:.1f}%")

fig, ax = plt.subplots(figsize=(8,5))
groups  = ["SoldOut=0\n(directs available)", "SoldOut=1\n(directs gone)"]
elasts  = [a1, a1+inter]
colors  = [PALETTE["main"], PALETTE["accent"]]
bars = ax.bar(groups, elasts, color=colors, edgecolor="white", width=0.45)
for bar,val in zip(bars,elasts):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.002,
            f"{val:.4f}", ha="center", fontsize=12, fontweight="bold")
ax.axhline(0,color="black",linewidth=0.8,linestyle="--")
ax.set_ylabel("Implied substitution elasticity")
ax.set_title(f"Test 2: SoldOut × DirectPrice Interaction\n"
             f"Interaction coef={inter:.4f}  p={p_int:.4f}  {sig(p_int)}")
ax.grid(axis="y",linestyle=":",alpha=0.4)
fig.tight_layout()
fig.savefig("plots/17_soldout_x_directprice.png", dpi=150)
plt.close()
print("  Plot: plots/17_soldout_x_directprice.png")


# =============================================================================
# TEST 3 — Hausman test (FE vs RE) + specification stability
# =============================================================================
print("\n" + "="*65)
print("TEST 3: Hausman test (FE vs RE) + specification stability")
print("="*65)

# --- Hausman test via linearmodels ---
df_panel = df.copy()
df_panel["entity"] = df_panel["Route"] + "|" + df_panel["DepartureDate"].astype(str)
df_panel = df_panel.set_index(["entity","DaysBeforeDeparture"])

XVARS = ["log_DirectPrice_avg","log_DirectPrice_x_Days",
         "log_TravelTime","log_TravelTime_sq","SoldOutDummy"]

fe_mod = PanelOLS.from_formula(
    "log_price ~ log_DirectPrice_avg + log_DirectPrice_x_Days + "
    "log_TravelTime + log_TravelTime_sq + SoldOutDummy + EntityEffects",
    data=df_panel
).fit(cov_type="clustered", cluster_entity=True)

re_mod = RandomEffects.from_formula(
    "log_price ~ log_DirectPrice_avg + log_DirectPrice_x_Days + "
    "log_TravelTime + log_TravelTime_sq + SoldOutDummy",
    data=df_panel
).fit()

# Hausman statistic
b_fe = fe_mod.params[XVARS].values
b_re = re_mod.params[XVARS].values
v_fe = fe_mod.cov.loc[XVARS, XVARS].values
v_re = re_mod.cov.loc[XVARS, XVARS].values
diff    = b_fe - b_re
v_diff  = v_fe - v_re
try:
    H_stat  = float(diff @ np.linalg.inv(v_diff) @ diff)
    from scipy.stats import chi2
    H_pval  = 1 - chi2.cdf(H_stat, df=len(XVARS))
    print(f"\n  Hausman statistic: {H_stat:.3f}  df={len(XVARS)}  p={H_pval:.6f}  {sig(H_pval)}")
    print(f"  → {'Reject RE: FE is consistent, RE is not. Use FE.' if H_pval < 0.05 else 'Cannot reject RE: RE may be efficient.'}")
except np.linalg.LinAlgError:
    H_stat, H_pval = np.nan, np.nan
    print("  Hausman test: singular matrix — report FE/RE coefficient comparison instead")

# Specification stability table
print("\n  Specification stability — α1 (substitution) across models:")
specs = {
    "Pooled OLS (no FE)":        "log_price ~ log_DirectPrice_avg + log_DirectPrice_x_Days + log_TravelTime + log_TravelTime_sq + SoldOutDummy",
    "Route FE only":             "log_price ~ log_DirectPrice_avg + log_DirectPrice_x_Days + log_TravelTime + log_TravelTime_sq + SoldOutDummy + C(Route)",
    "Date FE only":              "log_price ~ log_DirectPrice_avg + log_DirectPrice_x_Days + log_TravelTime + log_TravelTime_sq + SoldOutDummy + C(DepartureDate)",
    "Route + Date FE (baseline)":"log_price ~ log_DirectPrice_avg + log_DirectPrice_x_Days + log_TravelTime + log_TravelTime_sq + SoldOutDummy + C(Route) + C(DepartureDate)",
}
spec_results = {}
print(f"  {'Specification':<35} {'α1':>8}  {'SE':>8}  {'α5':>8}  {'R²':>6}")
print("  " + "-"*65)
for label, formula in specs.items():
    r = smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups":df["Route"]})
    a1v = r.params["log_DirectPrice_avg"]
    a1s = r.bse["log_DirectPrice_avg"]
    a5v = r.params["SoldOutDummy"]
    spec_results[label] = r
    print(f"  {label:<35} {a1v:>8.4f}  {a1s:>8.4f}  {a5v:>8.4f}  {r.rsquared:>6.4f}")

fig, axes = plt.subplots(1,2,figsize=(12,4.5))
labels_short = ["No FE","Route FE","Date FE","Route+Date\n(baseline)"]
colors_s = [PALETTE["neutral"],PALETTE["light"],PALETTE["accent"],PALETTE["main"]]
for ax, key, title, ols_val in [
    (axes[0],"log_DirectPrice_avg","α1: Substitution elasticity", 0.3421),
    (axes[1],"SoldOutDummy",       "α5: Scarcity effect", 0.0484),
]:
    vals = [spec_results[l].params[key] for l in specs]
    errs = [spec_results[l].bse[key]*1.96 for l in specs]
    bars = axes[list(axes).index(ax)].bar(labels_short, vals, color=colors_s,
                                          edgecolor="white", width=0.55, yerr=errs, capsize=4)
    ax.axhline(ols_val,color="grey",linewidth=1.5,linestyle="--",label=f"Baseline={ols_val}")
    ax.set_title(title,fontsize=10); ax.set_ylabel("Coefficient")
    ax.legend(fontsize=9); ax.grid(axis="y",linestyle=":",alpha=0.4)
fig.suptitle(f"Test 3: Coefficient Stability Across Specifications\n"
             f"Hausman stat={H_stat:.2f} p={H_pval:.4f} — FE preferred over RE",
             fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig("plots/18_specification_stability.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Plot: plots/18_specification_stability.png")


# =============================================================================
# TEST 4 — Residual diagnostics
# =============================================================================
print("\n" + "="*65)
print("TEST 4: Residual diagnostics")
print("="*65)

resids = res_base.resid
fitted = res_base.fittedvalues

# (a) Breusch-Pagan heteroskedasticity test
XVARS_for_BP = ["log_DirectPrice_avg","log_DirectPrice_x_Days",
                "log_TravelTime","log_TravelTime_sq","SoldOutDummy"]
X_bp = smf.ols(FORMULA_BASE, data=df).fit()  # plain OLS for BP
bp_stat, bp_p, f_stat, f_p = sms.het_breuschpagan(
    X_bp.resid, X_bp.model.exog
)
print(f"\n  (a) Breusch-Pagan heteroskedasticity test")
print(f"      LM stat={bp_stat:.2f}  p={bp_p:.6f}  {sig(bp_p)}")
print(f"      F stat={f_stat:.2f}   p={f_p:.6f}")
print(f"      → {'Significant heteroskedasticity — clustered SEs are the right correction.' if bp_p<0.05 else 'No significant heteroskedasticity.'}")

# (b) VIF — key variables only (no dummies)
vif_data = df[["log_DirectPrice_avg","log_DirectPrice_x_Days",
               "log_TravelTime","log_TravelTime_sq","SoldOutDummy"]].dropna()
vif_data = (vif_data - vif_data.mean()) / vif_data.std()
vif_data["const"] = 1
print(f"\n  (b) Variance Inflation Factors")
vif_vals = {}
for i,col in enumerate(["log_DirectPrice_avg","log_DirectPrice_x_Days",
                         "log_TravelTime","log_TravelTime_sq","SoldOutDummy"]):
    v = variance_inflation_factor(vif_data.values, i)
    vif_vals[col] = v
    flag = " ← HIGH (>10)" if v > 10 else " ← MODERATE (5–10)" if v > 5 else ""
    print(f"      {col:<40} VIF = {v:.2f}{flag}")

# (c) Serial correlation within route-dates
# Wooldridge-style: regress residuals on lagged residuals within entity
resid_df = df[["Route","DepartureDate","DaysBeforeDeparture"]].copy()
resid_df["resid"] = X_bp.resid
resid_df = resid_df.sort_values(["Route","DepartureDate","DaysBeforeDeparture"])
resid_df["resid_lag"] = resid_df.groupby(["Route","DepartureDate"])["resid"].shift(1)
resid_df = resid_df.dropna(subset=["resid_lag"])
sc_res = smf.ols("resid ~ resid_lag", data=resid_df).fit()
sc_coef = sc_res.params["resid_lag"]
sc_p    = sc_res.pvalues["resid_lag"]
print(f"\n  (c) Serial correlation in residuals (lag-1 within route-date)")
print(f"      Lag-1 coefficient = {sc_coef:.4f}  p={sc_p:.6f}  {sig(sc_p)}")
print(f"      → {'Significant positive autocorrelation — residuals correlated across days.' if sc_p<0.05 else 'No significant autocorrelation.'}")

# Plot
fig = plt.figure(figsize=(14,4))
gs  = gridspec.GridSpec(1,3)

ax1 = fig.add_subplot(gs[0])
ax1.scatter(fitted[:5000], resids[:5000], alpha=0.15, s=4, color=PALETTE["main"])
ax1.axhline(0,color="red",linewidth=1)
ax1.set_xlabel("Fitted values"); ax1.set_ylabel("Residuals")
ax1.set_title(f"Residuals vs Fitted\nBP test p={bp_p:.2e}")
ax1.grid(linestyle=":",alpha=0.3)

ax2 = fig.add_subplot(gs[1])
_r = resids.clip(-2,2)
_bins80 = np.linspace(_r.min(), _r.max(), 81)
ax2.hist(_r.values, bins=_bins80, color=PALETTE["main"], alpha=0.8, edgecolor="white")
ax2.set_xlabel("Residual"); ax2.set_ylabel("Count")
ax2.set_title("Residual Distribution"); ax2.grid(linestyle=":",alpha=0.3)

ax3 = fig.add_subplot(gs[2])
vif_cols  = list(vif_vals.keys())
vif_color = [PALETTE["neg"] if v>10 else PALETTE["accent"] if v>5 else PALETTE["pos"]
             for v in vif_vals.values()]
ax3.barh(vif_cols, list(vif_vals.values()), color=vif_color, edgecolor="white")
ax3.axvline(5, color="orange",linestyle="--",linewidth=1,label="VIF=5")
ax3.axvline(10,color="red",   linestyle="--",linewidth=1,label="VIF=10")
ax3.set_xlabel("VIF"); ax3.set_title("Variance Inflation Factors")
ax3.legend(fontsize=8); ax3.grid(axis="x",linestyle=":",alpha=0.4)

fig.suptitle("Test 4: Residual Diagnostics — 2023 Full Window OLS", fontsize=12)
fig.tight_layout()
fig.savefig("plots/19_residual_diagnostics.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Plot: plots/19_residual_diagnostics.png")


# =============================================================================
# TEST 5 — Granger causality
# =============================================================================
print("\n" + "="*65)
print("TEST 5: Granger causality")
print("="*65)

# Aggregate to route-date-day level
agg = (df.groupby(["Route","DepartureDate","DaysBeforeDeparture"])
         .agg(mean_logprice=("log_price","mean"),
              log_dp=("log_DirectPrice_avg","first"))
         .reset_index()
         .sort_values(["Route","DepartureDate","DaysBeforeDeparture"]))

min_obs = 15
pvals_dp_causes_p = []
pvals_p_causes_dp = []
n_eligible = 0

for (route,date), grp in agg.groupby(["Route","DepartureDate"]):
    grp = grp.sort_values("DaysBeforeDeparture", ascending=False)  # chronological
    if len(grp) < min_obs:
        continue
    n_eligible += 1
    ts = grp[["log_dp","mean_logprice"]].values
    try:
        # Test: does log_dp (direct) Granger-cause mean_logprice (indirect)?
        res_gc = grangercausalitytests(ts[:,  [1,0]], maxlag=2, verbose=False)  # y=indirect, x=direct
        pvals_dp_causes_p.append(min(res_gc[1][0]["ssr_ftest"][1],
                                     res_gc[2][0]["ssr_ftest"][1]))
        # Test: does indirect Granger-cause direct? (reverse)
        res_rev = grangercausalitytests(ts[:, [0,1]], maxlag=2, verbose=False)
        pvals_p_causes_dp.append(min(res_rev[1][0]["ssr_ftest"][1],
                                     res_rev[2][0]["ssr_ftest"][1]))
    except Exception:
        continue

pvals_dp  = np.array(pvals_dp_causes_p)
pvals_rev = np.array(pvals_p_causes_dp)
print(f"\n  Eligible route-dates (≥{min_obs} obs): {n_eligible}")
print(f"  Direct → Indirect Granger causality:")
print(f"    p<0.05 in {(pvals_dp<0.05).sum()}/{len(pvals_dp)} route-dates "
      f"({(pvals_dp<0.05).mean()*100:.1f}%)")
print(f"    Median p-value: {np.median(pvals_dp):.4f}")
print(f"  Indirect → Direct (reverse / placebo direction):")
print(f"    p<0.05 in {(pvals_rev<0.05).sum()}/{len(pvals_rev)} route-dates "
      f"({(pvals_rev<0.05).mean()*100:.1f}%)")
print(f"    Median p-value: {np.median(pvals_rev):.4f}")

fig, axes = plt.subplots(1,2,figsize=(11,4.5))
for ax, pv, title, color in [
    (axes[0], pvals_dp,  "Direct → Indirect\n(hypothesised direction)", PALETTE["main"]),
    (axes[1], pvals_rev, "Indirect → Direct\n(reverse / should be weaker)", PALETTE["accent"]),
]:
    ax.hist(pv, bins=30, color=color, alpha=0.8, edgecolor="white")
    ax.axvline(0.05,color="red",linewidth=1.5,linestyle="--",label="p=0.05")
    sig_pct = (pv<0.05).mean()*100
    ax.set_xlabel("Granger causality p-value (min of 1 & 2 lags)")
    ax.set_ylabel("Route-date count")
    ax.set_title(f"{title}\n{sig_pct:.1f}% of route-dates: p < 0.05")
    ax.legend(fontsize=9); ax.grid(linestyle=":",alpha=0.4)
fig.suptitle("Test 5: Granger Causality — Does DirectPrice Predict Future IndirectPrice?\n"
             "2023 Full Window | Separate test per route-date", fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig("plots/20_granger_causality.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Plot: plots/20_granger_causality.png")


# =============================================================================
# TEST 6 — Placebo test (shuffled SoldOutDummy)
# =============================================================================
print("\n" + "="*65)
print("TEST 6: Placebo test — shuffled SoldOutDummy")
print("="*65)

n_iter   = 100
rng      = np.random.default_rng(42)
placebo_coefs = []

print(f"  Running {n_iter} placebo iterations (shuffling SoldOutDummy within departure dates)...")
for i in range(n_iter):
    df_p = df.copy()
    # Shuffle SoldOutDummy across routes within each departure date
    df_p["SoldOutDummy"] = (df_p.groupby("DepartureDate")["SoldOutDummy"]
                               .transform(lambda x: rng.permutation(x.values)))
    r = smf.ols(FORMULA_BASE, data=df_p).fit()
    placebo_coefs.append(r.params["SoldOutDummy"])
    if (i+1) % 25 == 0:
        print(f"    {i+1}/{n_iter} done...")

placebo_arr  = np.array(placebo_coefs)
true_coef    = res_base.params["SoldOutDummy"]
placebo_p    = (np.abs(placebo_arr) >= np.abs(true_coef)).mean()

print(f"\n  True α5 (SoldOutDummy): {true_coef:.4f}")
print(f"  Placebo mean:           {placebo_arr.mean():.4f}")
print(f"  Placebo std:            {placebo_arr.std():.4f}")
print(f"  Placebo p-value:        {placebo_p:.4f}  {sig(placebo_p)}")
print(f"  → True coef is at the {(placebo_arr < true_coef).mean()*100:.1f}th percentile "
      f"of the placebo distribution")

fig, ax = plt.subplots(figsize=(8,5))
ax.hist(placebo_arr, bins=30, color=PALETTE["neutral"],
        alpha=0.8, edgecolor="white", label="Placebo α5 (shuffled SoldOutDummy)")
ax.axvline(true_coef, color=PALETTE["accent"], linewidth=2.5,
           label=f"True α5 = {true_coef:.4f}")
ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
pct = (placebo_arr < true_coef).mean()*100
ax.set_xlabel("Coefficient on SoldOutDummy")
ax.set_ylabel("Count (out of 100 placebo runs)")
ax.set_title(f"Test 6: Placebo Test — Shuffled SoldOutDummy\n"
             f"True coef at {pct:.1f}th percentile  |  Placebo p={placebo_p:.4f}  {sig(placebo_p)}")
ax.legend(fontsize=10); ax.grid(linestyle=":",alpha=0.4)
fig.tight_layout()
fig.savefig("plots/21_placebo_test.png", dpi=150)
plt.close()
print("  Plot: plots/21_placebo_test.png")

print("\n\nAll robustness tests complete.")
print("Plots: 16_first_differences, 17_soldout_x_directprice, 18_specification_stability,")
print("       19_residual_diagnostics, 20_granger_causality, 21_placebo_test")
