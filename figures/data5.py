"""Numbers of paper 5, read from the finished ROAM-CHO study in this repository (no new simulation).

Sources: RESULTS_judge.json (verdict, primary tests, ablations, overfit), sel_final.json (selected settings and
validation budgets) and the 180 per-world test files runs\\test\\<card>_<speed>_<seed>.json.  Paired confidence
intervals repeat the stratified bootstrap of scripts\\report500.py (seed 500, 10 000 resamples), so that every
interval equals the one in RESULTS.md.
"""
import json
import os

import numpy as np
from scipy.stats import t as student_t
from scipy.stats import wilcoxon

STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
CARDS = ("D", "F1", "F2")
SPEEDS = ("ped", "v30", "v60")
SEEDS = tuple(range(1, 21))
RULES = ("A3ref", "A3", "CHO", "RA3", "MCHO", "RCHO", "RCHO-H", "RCHO-A", "RCHO-D", "RCHO-S")
B = 10000


def _load_json(name):
    with open(os.path.join(STUDY, name), encoding="utf-8") as f:
        return json.load(f)


JUDGE = _load_json("RESULTS_judge.json")
SEL = _load_json("sel_final.json")


def load_rows():
    rows = []
    for c in CARDS:
        for s in SPEEDS:
            for k in SEEDS:
                with open(os.path.join(STUDY, "runs", "test", "%s_%s_%d.json" % (c, s, k)), encoding="utf-8") as f:
                    d = json.load(f)
                rows.append(((c, s, k), {n: d["rules"][n]["metrics"] for n in RULES}))
    return rows


ROWS = load_rows()


def sub(card=None, speed=None):
    return [r for r in ROWS if (card is None or r[0][0] == card) and (speed is None or r[0][1] == speed)]


def arr(rows, rule, m):
    return np.array([r[1][rule][m] for r in rows], dtype=float)


def paired_ci(rows, a, b, m, seed=500):
    """Mean of a-b, two-sided 95 % bootstrap CI stratified by card x speed, two-sided Wilcoxon p."""
    d = arr(rows, a, m) - arr(rows, b, m)
    st = {}
    for i, r in enumerate(rows):
        st.setdefault(r[0][:2], []).append(i)
    rng = np.random.default_rng(seed)
    tot = np.zeros(B)
    for idx in st.values():
        idx = np.array(idx)
        tot += d[idx][rng.integers(0, len(idx), size=(B, len(idx)))].sum(1)
    tot /= len(d)
    p = 1.0 if np.all(d == 0) else float(wilcoxon(arr(rows, a, m), arr(rows, b, m)).pvalue)
    base = arr(rows, b, m).mean()
    return {"mu": d.mean(), "lo": float(np.quantile(tot, 0.025)), "hi": float(np.quantile(tot, 0.975)), "p": p,
            "base": base, "rel": (d.mean() / base) if base else None}


def seed_ci(rows, rule, m):
    """Mean over the seeds of one card x speed and its 95 % t interval half-width."""
    x = arr(rows, rule, m)
    n = len(x)
    h = student_t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    return x.mean(), h


# ----------------------------------------------------------------------------- formatting helpers
def fp(p, digits=1):
    """p-value in LaTeX with the significant digits of RESULTS.md (1 for paired tables, 2 for the primary tests)."""
    if p >= 0.01:
        return ("%." + str(digits) + "g") % p
    s = ("%." + str(digits - 1) + "e") % p
    m, e = s.split("e")
    return r"%s\times10^{%d}" % (m, int(e))


def facts():
    """Flat dict of every number used in the text (strings, ready for LaTeX)."""
    F = {}
    P = JUDGE["pooled"]
    F["verdict"] = JUDGE["verdict"]
    F["n_worlds"] = "%d" % P["n"]
    F["n_seeds"] = "%d" % len(SEEDS)

    def prim(prefix, x):
        F[prefix + "uho_cho"] = "%.3f" % x["uho_cho"]
        F[prefix + "uho_rcho"] = "%.3f" % x["uho_rcho"]
        F[prefix + "uho_red"] = "%.1f" % (100 * x["uho_reduction"])
        F[prefix + "uho_red0"] = "%.0f" % (100 * x["uho_reduction"])
        F[prefix + "uho_inc"] = "%.1f" % (-100 * x["uho_reduction"])
        F[prefix + "p1"] = fp(x["p_wilcoxon"], 2)
        F[prefix + "fail_cho"] = "%.3f" % x["fail_cho"]
        F[prefix + "fail_rcho"] = "%.3f" % x["fail_rcho"]
        F[prefix + "fail_cho4"] = "%.4f" % x["fail_cho"]
        F[prefix + "fail_rcho4"] = "%.4f" % x["fail_rcho"]
        F[prefix + "fail_diff"] = "%+.3f" % x["fail_diff"]
        F[prefix + "fail_diff4"] = "%+.4f" % x["fail_diff"]
        F[prefix + "up95"] = "%+.3f" % x["fail_diff_upper95"]
        F[prefix + "up95_4"] = "%+.4f" % x["fail_diff_upper95"]
        F[prefix + "delta"] = "%.3f" % x["delta"]
        F[prefix + "delta4"] = "%.4f" % x["delta"]
        F[prefix + "fail_ratio"] = "%.1f" % (x["fail_rcho"] / x["fail_cho"])
        F[prefix + "H1a"] = "holds" if x["H1a"] else "does not hold"
        F[prefix + "H1b"] = "holds" if x["H1b"] else "does not hold"
    prim("pool_", P)
    for c in CARDS:
        prim(c + "_", JUDGE["per_card"][c])

    # paired comparisons RCHO vs baselines, per card and pooled (same numbers as RESULTS.md)
    KEY = ("unnecessary_pm", "failures_pm", "prepared_cells_mean", "rrc_messages_pm", "pingpong_pm",
           "interruption_s_pm", "rate_mbps")
    short = {"unnecessary_pm": "uho", "failures_pm": "fail", "prepared_cells_mean": "prep", "rrc_messages_pm": "rrc",
             "pingpong_pm": "pp", "interruption_s_pm": "int", "rate_mbps": "rate"}
    CMP = {}
    for b in ("CHO", "MCHO", "A3", "RA3"):
        for c in CARDS + ("all",):
            rows = ROWS if c == "all" else sub(c)
            for m in KEY:
                r = paired_ci(rows, "RCHO", b, m)
                CMP[(b, c, m)] = r
                tag = "%s_%s_%s" % (b, c, short[m])
                F["rel_" + tag] = ("%+.0f" % (100 * r["rel"])) if r["rel"] is not None else "--"
                F["absrel_" + tag] = ("%.0f" % abs(100 * r["rel"])) if r["rel"] is not None else "--"
                F["p_" + tag] = fp(r["p"])
                F["mu_" + tag] = "%+.3f" % r["mu"]
    # rate and interruption relative changes with one decimal where needed
    F["fail_ratio_F1"] = "%.1f" % (JUDGE["per_card"]["F1"]["fail_rcho"] / JUDGE["per_card"]["F1"]["fail_cho"])

    # ablations
    for a, v in JUDGE["ablations"].items():
        k = a.replace("RCHO-", "")
        F["abl_%s_uho" % k] = "%+.1f" % (100 * v["uho_change_when_removed"])
        F["abl_%s_uho_abs" % k] = "%.1f" % abs(100 * v["uho_change_when_removed"])
        F["abl_%s_p" % k] = fp(v["p_holm"], 2)
        F["abl_%s_fail" % k] = "%+.4f" % v["fail_change"]
        F["abl_%s_helps" % k] = "yes" if v["helps"] else "no"
    F["abl_delta"] = "%.3f" % max(0.10 * JUDGE["means"]["RCHO"]["failures_pm"], 0.002)

    # budgets and overfit
    ratios = []
    for c in CARDS:
        F["budget_%s" % c] = "%.3f" % SEL[c]["budget_failures_pm"]
        for m, x in JUDGE["overfit"][c].items():
            ratios.append((x["ratio"], c, m))
    for c in CARDS:
        for m, x in JUDGE["overfit"][c].items():
            F["ratio_%s_%s" % (c, m)] = "%.2f" % x["ratio"]
            F["testfail_%s_%s" % (c, m)] = "%.3f" % x["test_fail"]
            for k, v in SEL[c][m]["params"].items():
                F["sel_%s_%s_%s" % (c, m, k)] = ("%d" % round(v * 1000)) if k in ("ttt", "ttt_exec") else ("%g" % v)
    rmax = max(ratios)
    F["overfit_max"] = "%.2f" % rmax[0]
    F["overfit_max_who"] = "%s in card %s" % (rmax[2], rmax[1])
    F["n_infeasible"] = "%d" % sum(1 for c in CARDS for m in JUDGE["overfit"][c] if not JUDGE["overfit"][c][m]["val_feasible"])

    # preparation counts and too-late handovers per card (test means)
    for c in CARDS:
        for r in ("CHO", "MCHO", "RCHO"):
            F["preps_%s_%s" % (c, r)] = "%.1f" % arr(sub(c), r, "preparations_pm").mean()
            F["unused_%s_%s" % (c, r)] = "%.1f" % arr(sub(c), r, "unused_preparations_pm").mean()
            F["late_%s_%s" % (c, r)] = "%.2f" % arr(sub(c), r, "too_late_pm").mean()
    return F, CMP


def check_against_results_md(CMP):
    """Guard: a few intervals must match RESULTS.md exactly (same bootstrap)."""
    r = CMP[("CHO", "D", "unnecessary_pm")]
    assert "%+.3f [%+.3f, %+.3f]" % (r["mu"], r["lo"], r["hi"]) == "-0.040 [-0.051, -0.029]", r
    r = CMP[("CHO", "all", "prepared_cells_mean")]
    assert "%+.3f [%+.3f, %+.3f]" % (r["mu"], r["lo"], r["hi"]) == "-0.239 [-0.242, -0.235]", r
    r = CMP[("MCHO", "all", "rrc_messages_pm")]
    assert "%+.3f" % r["mu"] == "-96.469", r
    txt = open(os.path.join(STUDY, "RESULTS.md"), encoding="utf-8").read()
    assert "**FAIL**" in txt


if __name__ == "__main__":
    F, CMP = facts()
    check_against_results_md(CMP)
    for k in sorted(F):
        if not k.startswith(("rel_", "absrel_", "p_", "mu_")):
            print(k, F[k])
    for k in ("rel_CHO_all_prep", "rel_CHO_D_prep", "rel_CHO_F1_prep", "rel_CHO_F2_prep", "rel_CHO_all_rrc",
              "rel_CHO_D_rrc", "rel_CHO_F2_rrc", "rel_CHO_F1_rrc", "rel_MCHO_all_uho", "rel_MCHO_all_fail",
              "rel_MCHO_all_prep", "rel_MCHO_all_rrc", "rel_RA3_all_uho", "rel_RA3_all_fail", "rel_RA3_all_prep",
              "rel_A3_all_uho", "rel_A3_all_fail", "rel_A3_all_rrc"):
        print(k, F[k], F[k.replace("rel_", "p_")])
