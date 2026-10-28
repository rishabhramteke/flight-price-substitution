"""
make_figures.py
---------------
Clean, self-explanatory figures for the write-up (docs/FINDINGS.md).

Inputs
  results/*.csv            coefficient tables and event-study series (in this repo)
  final_merged.csv         optional, only for the three descriptive figures (D1-D3)

Numbers that exist only in docs/experiments.md (Q1-Q7, T1-T6) are typed in the
NUMBERS block below, each with the section it was copied from.

Run from the repo root:  python src/make_figures.py
Output: figures/clean/*.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "figures/clean"
os.makedirs(OUT, exist_ok=True)

C23, C26, CPOOL = "#1f6fb4", "#e8590c", "#5f8d4e"     # one colour per year, everywhere
GREY, RED = "#6b7280", "#c0392b"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 11, "legend.fontsize": 10, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
})


def finish(fig, name, title, takeaway=None):
    fig.subplots_adjust(top=0.74 if takeaway else 0.84, bottom=0.2 if fig.legends else None, wspace=0.35)
    fig.text(0.01, 0.99, title, ha="left", va="top", fontsize=14, fontweight="bold")
    if takeaway:
        fig.text(0.01, 0.925, takeaway, ha="left", va="top", fontsize=11, color=GREY, wrap=True)
    fig.savefig(f"{OUT}/{name}.png")
    plt.close(fig)
    print("saved", name)


def stars(p):
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else "n.s."


# ----------------------------------------------------------------------------
# NUMBERS copied from docs/experiments.md (section named in each comment)
# ----------------------------------------------------------------------------
Q1 = dict(  # "Q1" 2023 table and the 2023-vs-2026 comparison table
    a1={"2023": (0.3442, 0.2803), "2026": (0.26, 0.15)},       # (1-stop, 2-stop)
    a5_2023=(0.0412, 0.1385), a5_2026_2stop=0.08)
Q2 = dict(coef={"2023": 0.2020, "2026": 0.238}, binary={"2023": 0.0484, "2026": 0.064})
Q3 = [("LCC share (level)", -0.4190, 0.3675),                  # "Q3" 2023 table: coef, SE
      ("LCC share x direct price", 0.0476, 0.0845),
      ("LCC share x sell-out", 0.1813, 0.1305)]
Q4 = dict(up={"2023": 0.5507, "2026": 0.74}, down={"2023": 0.0397, "2026": 0.35},
          se_2023=(0.0463, 0.0826))
Q6 = dict(share_sell_out={"2023": 47.3, "2026": 55.5}, mean_day={"2023": 22.7, "2026": 8.9})
Q7 = dict(gap={"2023": (18, 37), "2026": (55, 111)})           # median EUR gap, (36 days out, 1 day out)
T1 = dict(dp={"2023": 0.063, "2026": 0.108}, so={"2023": 0.017, "2026": 0.034})
T3 = dict(specs=["No fixed effects", "Route FE only", "Date FE only", "Route + date FE\n(main model)"],
          a1=[0.3505, 0.3943, 0.3328, 0.3421], a5=[-0.0454, 0.0461, -0.0382, 0.0484])
T5 = dict(forward={"2023": 35.1, "2026": 46}, reverse={"2023": 21.8, "2026": 27})
T6 = dict(sigma={"2023": 21, "2026": 28.8})
YEAR_EFFECT = [("All routes,\n0-36 day window", 0.3229), ("Matched routes,\n11-day window", 0.2382)]

# ----------------------------------------------------------------------------
# 01  main coefficients
# ----------------------------------------------------------------------------
res = pd.read_csv("results/results_merged.csv")
res = res[res.variable != "C(Year)[T.2026]"]
panels = [("log_DirectPrice_avg", "a1  Substitution\n(direct price up 1% -> indirect up a1 %)"),
          ("log_DirectPrice_x_Days", "a2  Change in a1 per extra\nday before departure"),
          ("SoldOutDummy", "a5  Sell-out premium\n(log points, ~ % for small values)"),
          ("log_TravelTime", "a3  Travel time\n(log points)"),
          ("log_TravelTime_sq", "a4  Travel time squared")]
fig, axes = plt.subplots(1, 5, figsize=(17, 4.2))
for ax, (v, ttl) in zip(axes, panels):
    for i, (ds, col) in enumerate([("2023", C23), ("2026", C26), ("pooled", CPOOL)]):
        r = res[(res.dataset == ds) & (res.variable == v)].iloc[0]
        ax.errorbar(i, r.coef, yerr=1.96 * r.std_err, fmt="o", color=col, capsize=5, ms=8, lw=2)
        ax.annotate(stars(r.p_value), (i, r.coef + 1.96 * r.std_err), xytext=(0, 5),
                    textcoords="offset points", ha="center", fontsize=10, color=col, fontweight="bold")
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["2023", "2026", "Both\npooled"])
    ax.set_xlim(-.6, 2.6); ax.margins(y=.18); ax.set_title(ttl, fontsize=11, pad=10)
fig.legend([plt.Line2D([], [], color=c, marker="o", lw=2) for c in (C23, C26, CPOOL)],
           ["2023", "2026", "2023+2026 pooled"], loc="lower center", ncol=3, bbox_to_anchor=(.5, -.08))
finish(fig, "01_main_coefficients", "The main regression, all five coefficients",
       "Dot = estimate, bar = 95% confidence interval. A bar that touches zero (n.s.) means no detectable effect.")

# ----------------------------------------------------------------------------
# 02  substitution strength across the booking window
# ----------------------------------------------------------------------------
def coef(ds, v): return res[(res.dataset == ds) & (res.variable == v)].coef.iloc[0]
d = np.arange(36, -1, -1)
fig, ax = plt.subplots(figsize=(9, 4.6))
for ds, col in [("2023", C23), ("2026", C26)]:
    ax.plot(d, coef(ds, "log_DirectPrice_avg") + coef(ds, "log_DirectPrice_x_Days") * d, color=col, lw=3, label=ds)
ax.set_xlim(36, 0); ax.set_ylim(0, 0.42)
ax.set_xlabel("Days before departure  (time moves to the right)")
ax.set_ylabel("How much indirect fares follow direct fares\n(elasticity = a1 + a2 x days)")
ax.legend(loc="lower right", title="Year")
ax.annotate("Far from departure:\nweaker link, esp. 2026", (34, .19), fontsize=10, color=GREY)
ax.annotate("At departure:\nlink is strongest", (7, .375), fontsize=10, color=GREY)
finish(fig, "02_substitution_over_time", "Indirect fares follow direct fares more closely as departure nears",
       "A value of 0.3 means: direct fare +10% -> indirect fare +3% on the same route and date.")

# ----------------------------------------------------------------------------
# 03  event study, time running left -> right
# ----------------------------------------------------------------------------
def load_es(path):
    e = pd.read_csv(path)
    e["x"] = -e.RelDay                                  # RelDay>0 is BEFORE the sell-out; flip so time moves right
    e["pct"] = (np.exp(e.MeanNormLogPrice) - 1) * 100
    return e.sort_values("x")
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
for ax, (path, yr, col) in zip(axes, [("results/results_event_study.csv", "2023", C23),
                                      ("results/results_event_study_2026.csv", "2026", C26)]):
    e = load_es(path)
    t, c = e[e.group.str.startswith("Treated")], e[e.group.str.startswith("Control")]
    ax.axvspan(0, 8, color="#f3f4f6", zorder=0)
    ax.plot(t.x, t.pct, "-o", color=col, lw=2.8, label="Routes where a direct flight sells out")
    ax.plot(c.x, c.pct, "--s", color=GREY, lw=1.8, ms=5, label="Routes where directs never sell out\n(pseudo sell-out day)")
    ax.axvline(0, color="k", lw=1.5)
    ax.text(0.15, ax.get_ylim()[1] * .02 if False else -3, "  after the sell-out", fontsize=10, color=GREY)
    ax.set_title(yr); ax.set_xlabel("Days since the first direct flight disappeared (0 = sell-out day)")
    ax.set_xlim(-7.5, 7.5)
axes[0].set_ylabel("Indirect fare vs. its own pre-sell-out average (%)")
axes[0].legend(loc="upper left")
finish(fig, "03_event_study", "What happens to indirect fares on the day a direct flight sells out?",
       "2023: flat, then a jump on day 0 while the control stays flat. 2026: fares were already rising and the control rises too, so the 2026 evidence is weaker.")

# ----------------------------------------------------------------------------
# 04  LCC vs non-LCC (binary), both years
# ----------------------------------------------------------------------------
l23 = pd.read_csv("results/results_lcc_interaction.csv").set_index("variable")
l26 = pd.read_csv("results/results_lcc_interaction_2026.csv").set_index("variable")
def lcc_pair(t, v, inter):  # (non-LCC, LCC)
    return t.loc[v, "coef"], t.loc[v, "coef"] + t.loc[inter, "coef"]
a1 = {"2023": lcc_pair(l23, "log_DirectPrice_avg", "lccdum:log_DirectPrice_avg"),
      "2026": lcc_pair(l26, "log_DirectPrice_avg", "lccdum:log_DirectPrice_avg")}
a5 = {"2023": lcc_pair(l23, "SoldOutDummy", "lccdum:SoldOutDummy"),
      "2026": lcc_pair(l26, "SoldOutDummy", "lccdum:SoldOutDummy")}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
for ax, data, ttl in [(axes[0], a1, "Substitution a1\n(do indirects follow directs?)"),
                      (axes[1], a5, "Sell-out premium a5\n(log points)")]:
    x = np.arange(2); w = .36
    for k, (yr, col) in enumerate([("2023", C23), ("2026", C26)]):
        b = ax.bar(x + (k - .5) * w, data[yr], w, color=col, label=yr)
        for xi, val in zip(x + (k - .5) * w, data[yr]): ax.text(xi, val + .004, f"{val:.2f}", ha="center", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(["Non-LCC routes", "LCC routes"]); ax.set_title(ttl, fontsize=11)
axes[0].legend(title="Year")
finish(fig, "04_lcc_vs_non_lcc", "Low-cost vs full-service routes: no statistically reliable difference",
       "Bars differ, but none of the LCC interaction terms is significant in either year (p > 0.2), so treat the gaps as noise.")

# ----------------------------------------------------------------------------
# 05  Q1  1-stop vs 2-stop
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
ax = axes[0]; x = np.arange(2); w = .36
for k, (yr, col) in enumerate([("2023", C23), ("2026", C26)]):
    vals = Q1["a1"][yr]; ax.bar(x + (k - .5) * w, vals, w, color=col, label=yr)
    for xi, v in zip(x + (k - .5) * w, vals): ax.text(xi, v + .005, f"{v:.2f}", ha="center")
ax.set_xticks(x); ax.set_xticklabels(["1 stop", "2 stops"]); ax.set_title("Substitution a1", fontsize=11); ax.legend(title="Year")
ax = axes[1]
ax.bar(x - w / 2, Q1["a5_2023"], w, color=C23, label="2023")
ax.bar([1 + w / 2], [Q1["a5_2026_2stop"]], w, color=C26, label="2026 (2-stop only reported)")
for xi, v in [(-w / 2, Q1["a5_2023"][0]), (1 - w / 2, Q1["a5_2023"][1]), (1 + w / 2, Q1["a5_2026_2stop"])]:
    ax.text(xi, v + .004, f"{v:.2f}", ha="center")
ax.set_xticks(x); ax.set_xticklabels(["1 stop", "2 stops"]); ax.set_title("Sell-out premium a5 (log points)", fontsize=11); ax.legend()
finish(fig, "05_one_vs_two_stops", "2-stop itineraries track directs less, but pay a bigger premium when directs sell out",
       "Q1. Same pattern in both years.")

# ----------------------------------------------------------------------------
# 06  Q2  dose-response
# ----------------------------------------------------------------------------
x = np.linspace(0, 1, 50)
fig, ax = plt.subplots(figsize=(9, 4.8))
for yr, col in [("2023", C23), ("2026", C26)]:
    ax.plot(x * 100, (np.exp(Q2["coef"][yr] * x) - 1) * 100, color=col, lw=3, label=f"{yr}  (slope {Q2['coef'][yr]:.2f})")
    ax.axhline((np.exp(Q2["binary"][yr]) - 1) * 100, color=col, ls=":", lw=1.6)
ax.text(100, 1.5, "dotted lines = what the yes/no sell-out dummy reports", fontsize=9.5, color=GREY, ha="right")
ax.set_xlabel("Share of the route's direct flights that have disappeared (%)")
ax.set_ylabel("Indirect fare increase (%)"); ax.legend(loc="upper left")
finish(fig, "06_dose_response", "The more direct flights disappear, the higher indirect fares go",
       "Q2. Losing all directs would imply about +22% (2023) and +27% (2026); the yes/no dummy understates this.")

# ----------------------------------------------------------------------------
# 07  Q3  LCC share
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 3.8))
for i, (lab, c, se) in enumerate(Q3):
    ax.errorbar(c, i, xerr=1.96 * se, fmt="o", color=C23, capsize=5, ms=8, lw=2)
    ax.text(c, i + .22, f"{c:+.2f}  (p>{0.15:.2f})", ha="center", fontsize=10)
ax.axvline(0, color="k", lw=1); ax.set_yticks(range(3)); ax.set_yticklabels([q[0] for q in Q3])
ax.set_ylim(-.6, 2.7); ax.set_xlabel("Coefficient with 95% confidence interval (2023)")
finish(fig, "07_lcc_share", "How much of a route is low-cost does not change the mechanism",
       "Q3. Every interval crosses zero: no significant moderation.")

# ----------------------------------------------------------------------------
# 08  Q4  asymmetry
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4.6)); x = np.arange(2); w = .36
for k, (yr, col) in enumerate([("2023", C23), ("2026", C26)]):
    vals = [Q4["up"][yr], Q4["down"][yr]]; ax.bar(x + (k - .5) * w, vals, w, color=col, label=yr)
    for xi, v in zip(x + (k - .5) * w, vals): ax.text(xi, v + .015, f"{v:.2f}", ha="center", fontweight="bold")
ax.errorbar([-.18], [Q4["up"]["2023"]], yerr=[1.96 * Q4["se_2023"][0]], color="k", capsize=4)
ax.errorbar([1 - .18], [Q4["down"]["2023"]], yerr=[1.96 * Q4["se_2023"][1]], color="k", capsize=4)
ax.set_xticks(x); ax.set_xticklabels(["Direct fare ABOVE its\nroute-date average", "Direct fare BELOW its\nroute-date average"])
ax.set_ylabel("How strongly the indirect fare responds"); ax.legend(title="Year (black bar = 95% CI, 2023)", loc="upper center", bbox_to_anchor=(.5,-.18), ncol=2)
finish(fig, "08_asymmetric_response", "Indirect fares follow direct fares up, much less so down",
       "Q4. 2023: no response to falling directs. 2026: some response, still about half as strong.")

# ----------------------------------------------------------------------------
# 09  Q6  sell-out timing
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, key, ttl, yl in [(axes[0], "share_sell_out", "Route-dates where a direct sells out", "% of route-dates"),
                         (axes[1], "mean_day", "Average sell-out timing", "Days before departure")]:
    vals = [Q6[key]["2023"], Q6[key]["2026"]]
    ax.bar(["2023", "2026"], vals, color=[C23, C26], width=.5)
    for i, v in enumerate(vals): ax.text(i, v + .8, f"{v:.1f}", ha="center", fontweight="bold")
    ax.set_title(ttl, fontsize=11); ax.set_ylabel(yl)
finish(fig, "09_sell_out_timing", "About half of route-dates see a direct flight sell out; 2026 sell-outs come later",
       "Q6. 2023 regression: LCC-heavy routes sell out about 17 days earlier (p<0.001), and busier routes with more direct flights sell out earlier.")

# ----------------------------------------------------------------------------
# 10  Q7  price gap
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 4.6))
for yr, col in [("2023", C23), ("2026", C26)]:
    g = Q7["gap"][yr]; ax.plot([36, 1], g, "-o", color=col, lw=3, ms=9, label=yr)
    ax.text(36, g[0] + 4, f"EUR {g[0]}", ha="center", color=col, fontweight="bold")
    ax.text(1, g[1] + 4, f"EUR {g[1]}", ha="center", color=col, fontweight="bold")
ax.set_xlim(38, -1); ax.set_ylim(0, 130)
ax.set_xlabel("Days before departure (time moves to the right)"); ax.set_ylabel("Median (indirect fare - average direct fare), EUR")
ax.legend(title="Year")
finish(fig, "10_price_gap", "The euro gap between indirect and direct fares roughly doubles toward departure",
       "Q7. Only the two end points of each line are reported in the write-up, so the straight line is a guide, not data.")

# ----------------------------------------------------------------------------
# 11  T1  first differences
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4.4)); x = np.arange(2); w = .36
for k, (yr, col) in enumerate([("2023", C23), ("2026", C26)]):
    vals = [T1["dp"][yr], T1["so"][yr]]; ax.bar(x + (k - .5) * w, vals, w, color=col, label=yr)
    for xi, v in zip(x + (k - .5) * w, vals): ax.text(xi, v + .003, f"{v:.3f}", ha="center")
ax.set_xticks(x); ax.set_xticklabels(["Day-to-day change in direct fare\n(effect on indirect fare)", "Day a direct flight is lost\n(jump in indirect fare)"])
ax.set_ylabel("Coefficient (all significant, p<0.001)"); ax.legend(title="Year")
finish(fig, "11_first_differences", "Check 1: the effects survive when only day-to-day changes are used",
       "T1. This removes every route and date fixed effect, so slow-moving differences cannot explain the result.")

# ----------------------------------------------------------------------------
# 12  T3  specification stability
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
for ax, key, ttl in [(axes[0], "a1", "Substitution a1: stable"), (axes[1], "a5", "Sell-out premium a5: needs route fixed effects")]:
    cols = [RED if (key == "a5" and v < 0) else C23 for v in T3[key]]
    ax.bar(range(4), T3[key], color=cols)
    for i, v in enumerate(T3[key]):
        if v >= 0: ax.text(i, v + .004 * (key == "a5") + .008 * (key == "a1"), f"{v:+.3f}", ha="center")
        else: ax.text(i, v / 2, f"{v:+.3f}", ha="center", va="center", color="white", fontweight="bold")
    ax.axhline(0, color="k", lw=1); ax.set_xticks(range(4)); ax.set_xticklabels(T3["specs"], fontsize=9); ax.margins(y=.2); ax.set_title(ttl, fontsize=11, pad=10)
finish(fig, "12_specification_stability", "Check 2: same data, four model set-ups (2023)",
       "T3. a1 stays between 0.33 and 0.39. a5 turns negative (red) without route fixed effects, so the main model keeps them. A Hausman test (p ~ 0) also rejects random effects.")

# ----------------------------------------------------------------------------
# 13  T5  Granger
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4.4)); x = np.arange(2); w = .36
for k, (lab, d, col) in enumerate([("Direct fares help predict indirect fares", T5["forward"], C23),
                                   ("Indirect fares help predict direct fares", T5["reverse"], GREY)]):
    vals = [d["2023"], d["2026"]]; ax.bar(x + (k - .5) * w, vals, w, color=col, label=lab)
    for xi, v in zip(x + (k - .5) * w, vals): ax.text(xi, v + .8, f"{v:.0f}%", ha="center", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(["2023", "2026"]); ax.set_ylabel("% of route-dates with a significant test (p<0.05)"); ax.legend(loc="upper left")
finish(fig, "13_granger", "Check 3: direct fares lead indirect fares more often than the reverse",
       "T5. About 1.6x (2023) and 1.7x (2026), but the reverse direction is not rare, so this is supportive, not conclusive.")

# ----------------------------------------------------------------------------
# 14  T6  placebo
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.4))
ax.bar(["2023", "2026"], [T6["sigma"]["2023"], T6["sigma"]["2026"]], color=[C23, C26], width=.5)
for i, yr in enumerate(["2023", "2026"]): ax.text(i, T6["sigma"][yr] + .6, f"{T6['sigma'][yr]:.0f} sigma", ha="center", fontweight="bold")
ax.axhline(2, color=RED, ls="--"); ax.text(1.45, 2.6, "usual 'significant' cut-off (2 sigma)", ha="right", color=RED, fontsize=10)
ax.set_ylabel("True sell-out effect, in standard deviations\nabove 100 shuffled-dummy placebo runs")
finish(fig, "14_placebo", "Check 4: shuffling the sell-out dummy makes the effect vanish",
       "T6. The real estimate beat all 100 placebo runs in both years; the placebo average was about 0.")

# ----------------------------------------------------------------------------
# 15  year effect
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.4))
vals = [(np.exp(v) - 1) * 100 for _, v in YEAR_EFFECT]
ax.bar([l for l, _ in YEAR_EFFECT], vals, color=C26, width=.5)
for i, v in enumerate(vals): ax.text(i, v + .8, f"+{v:.0f}%", ha="center", fontweight="bold")
ax.set_ylabel("2026 indirect fares vs 2023 (%)")
finish(fig, "15_year_effect", "After controlling for everything in the model, 2026 fares are 27-38% higher",
       "Pooled regression year effect (0.24 and 0.32 log points). The range comes from the sample, not from noise.")

# ----------------------------------------------------------------------------
# Descriptive figures from the regression-ready data (optional file)
# ----------------------------------------------------------------------------
DATA = "final_merged.csv"
if os.path.exists(DATA):
    df = pd.read_csv(DATA, usecols=["Year", "price", "DirectPrice_avg", "TravelTimeMinutes", "DaysBeforeDeparture", "SoldOutDummy"])
    # D1 share of indirect observations where a direct has sold out, by days before departure
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for yr, col in [(2023, C23), (2026, C26)]:
        s = df[df.Year == yr].groupby("DaysBeforeDeparture").SoldOutDummy.mean() * 100
        s = s[s.index >= 1]
        ax.plot(s.index, s.values, color=col, lw=3, label=f"{yr}  (overall {df[df.Year == yr].SoldOutDummy.mean() * 100:.1f}%)")
    ax.set_xlim(36, 1); ax.set_xlabel("Days before departure (time moves to the right)"); ax.set_ylabel("% of observations where a direct has sold out")
    ax.legend(title="Year")
    finish(fig, "D1_sell_out_rate", "Sell-outs pile up close to departure", "Share of indirect observations that sit in a route-date where a direct flight has already disappeared.")
    # D2 travel time vs price
    df["hours"] = df.TravelTimeMinutes / 60
    df = df[(df.hours >= 1) & (df.hours <= 40)]
    df["bin"] = pd.cut(df.hours, [1, 3, 5, 7, 9, 12, 16, 20, 25, 30, 40])
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for yr, col in [(2023, C23), (2026, C26)]:
        g = df[df.Year == yr].groupby("bin", observed=True).agg(h=("hours", "median"), p=("price", "median"), n=("price", "size"))
        g = g[g.n >= 200]
        ax.plot(g.h, g.p, "-o", color=col, lw=2.5, label=str(yr))
    ax.set_xlabel("Total travel time of the indirect itinerary (hours)"); ax.set_ylabel("Median indirect fare (EUR)"); ax.legend(title="Year")
    finish(fig, "D2_travel_time_vs_price", "Longer itineraries are cheaper in 2023; in 2026 that pattern is gone", "Raw medians, not model output. Bins with fewer than 200 observations are dropped.")
    # D3 fares by year
    fig, ax = plt.subplots(figsize=(9, 4.4)); x = np.arange(2); w = .36
    for k, (yr, col) in enumerate([(2023, C23), (2026, C26)]):
        d_ = df[df.Year == yr]; vals = [d_.price.median(), d_.DirectPrice_avg.median()]
        ax.bar(x + (k - .5) * w, vals, w, color=col, label=str(yr))
        for xi, v in zip(x + (k - .5) * w, vals): ax.text(xi, v + 4, f"EUR {v:.0f}", ha="center", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(["Indirect fare", "Average direct fare\non the same route-date"]); ax.set_ylabel("Median (EUR)"); ax.legend(title="Year")
    finish(fig, "D3_median_fares", "Fares are higher in 2026 for both types of flight", "Medians over the regression sample.")
else:
    print(f"{DATA} not found: skipped descriptive figures D1-D3")
