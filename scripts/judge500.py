"""Judge the test stage against PLAN_roamcho.md Section 5.  Writes RESULTS_judge.json and RESULTS.md."""
import json
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common500 import CARDS, GRIDS, METHODS, ROOT, SPEEDS, TEST_SEEDS, out_path  # noqa: E402

RULES = ("A3ref",) + METHODS + ("RCHO-H", "RCHO-A", "RCHO-D", "RCHO-S")
MET = ("unnecessary_pm", "failures_pm", "pingpong_pm", "handovers_pm", "prepared_cells_mean", "preparations_pm",
       "rrc_messages_pm", "interruption_s_pm", "rate_mbps", "too_late_pm", "too_early_pm", "recoveries_pm")
B = 10000


def load():
    rows = []                                     # (card, speed, seed) -> {rule: metrics}
    for c in CARDS:
        for s in SPEEDS:
            for k in TEST_SEEDS:
                with open(out_path("test", c, s, k), encoding="utf-8") as f:
                    d = json.load(f)
                rows.append(((c, s, k), {n: d["rules"][n]["metrics"] for n in RULES}))
    return rows


def arr(rows, rule, m):
    return np.array([r[1][rule][m] for r in rows], dtype=float)


def strat_boot_upper(rows, a, b, m, seed=500):
    """One-sided 95 % upper bound of mean(a) - mean(b), bootstrap within card x speed strata."""
    d = arr(rows, a, m) - arr(rows, b, m)
    strata = {}
    for i, r in enumerate(rows):
        strata.setdefault(r[0][:2], []).append(i)
    rng = np.random.default_rng(seed)
    n = len(d)
    tot = np.zeros(B)
    for idx in strata.values():
        idx = np.array(idx)
        draw = rng.integers(0, len(idx), size=(B, len(idx)))
        tot += d[idx][draw].sum(1)
    return float(np.quantile(tot / n, 0.95)), float(d.mean())


def wil_less(x, y):
    """One-sided Wilcoxon p for x < y (paired)."""
    d = x - y
    if np.all(d == 0):
        return 1.0
    return float(wilcoxon(x, y, alternative="less", zero_method="wilcox").pvalue)


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj.tolist()


def primary(rows):
    u_r, u_c = arr(rows, "RCHO", "unnecessary_pm"), arr(rows, "CHO", "unnecessary_pm")
    f_c = arr(rows, "CHO", "failures_pm")
    red = 1 - u_r.mean() / u_c.mean() if u_c.mean() > 0 else float("nan")
    p = wil_less(u_r, u_c)
    up, dm = strat_boot_upper(rows, "RCHO", "CHO", "failures_pm")
    delta = max(0.10 * f_c.mean(), 0.002)
    return {"n": len(rows), "uho_rcho": u_r.mean(), "uho_cho": u_c.mean(), "uho_reduction": red, "p_wilcoxon": p,
            "H1a": bool(p < 0.05 and red >= 0.10), "H1a_sig_only": bool(p < 0.05 and red > 0),
            "fail_rcho": arr(rows, "RCHO", "failures_pm").mean(), "fail_cho": f_c.mean(), "fail_diff": dm,
            "fail_diff_upper95": up, "delta": delta, "H1b": bool(up <= delta)}


def compare(rows, a, b):
    out = {}
    ps = []
    for m in MET:
        x, y = arr(rows, a, m), arr(rows, b, m)
        d = x - y
        p = 1.0 if np.all(d == 0) else float(wilcoxon(x, y, zero_method="wilcox").pvalue)
        ps.append(p)
        out[m] = {"a": x.mean(), "b": y.mean(), "rel": (x.mean() / y.mean() - 1) if y.mean() else None, "p": p}
    for m, q in zip(MET, holm(ps)):
        out[m]["p_holm"] = q
    return out


def ablations(rows):
    delta = max(0.10 * arr(rows, "RCHO", "failures_pm").mean(), 0.002)
    res, ps = {}, []
    for a in ("RCHO-H", "RCHO-A", "RCHO-D", "RCHO-S"):
        ua, ur = arr(rows, a, "unnecessary_pm"), arr(rows, "RCHO", "unnecessary_pm")
        p = wil_less(ur, ua)                     # removing the part raises UHO <=> full < ablation
        ps.append(p)
        fa, fr = arr(rows, a, "failures_pm").mean(), arr(rows, "RCHO", "failures_pm").mean()
        res[a] = {"uho_ablation": ua.mean(), "uho_full": ur.mean(), "uho_change_when_removed": ua.mean() / ur.mean() - 1
                  if ur.mean() else None, "p": p, "fail_ablation": fa, "fail_full": fr, "fail_change": fa - fr}
    for a, q in zip(res, holm(ps)):
        r = res[a]
        r["p_holm"] = q
        r["helps"] = bool((q < 0.05 and r["fail_change"] >= -delta) or r["fail_change"] > delta)
    return res


def overfit():
    sel = json.load(open(os.path.join(ROOT, "sel_final.json"), encoding="utf-8"))
    out = {}
    for c in CARDS:
        rows = [r for r in load_cache if r[0][0] == c]
        out[c] = {}
        for m in METHODS:
            ft = arr(rows, m, "failures_pm").mean()
            b = sel[c]["budget_failures_pm"]
            edges = [k for k, v in sel[c][m]["params"].items()
                     if k in GRIDS[m] and len(GRIDS[m][k]) > 2 and v in (GRIDS[m][k][0], GRIDS[m][k][-1])]
            out[c][m] = {"test_fail": ft, "budget": b, "ratio": ft / b if b else None,
                         "flag": bool(b and ft / b > 1.3), "val_feasible": sel[c][m]["feasible"],
                         "grid_edges": edges, "params": sel[c][m]["params"]}
    return out


if __name__ == "__main__":
    load_cache = rows = load()
    R = {"pooled": primary(rows)}
    R["per_card"] = {}
    for c in CARDS:
        rc = [r for r in rows if r[0][0] == c]
        pc = primary(rc)
        R["per_card"][c] = pc
        R["per_card"][c]["per_speed"] = {s: {m: {k: arr([r for r in rc if r[0][1] == s], k, m).mean()
                                                 for k in RULES} for m in MET}
                                         for s in SPEEDS}
    P = R["pooled"]
    allc = all(R["per_card"][c]["uho_reduction"] > 0 and R["per_card"][c]["H1b"] for c in CARDS)
    if P["H1a"] and P["H1b"]:
        R["verdict"] = "PASS in all cards" if allc else "PASS"
    elif P["H1a_sig_only"] and P["H1b"]:
        R["verdict"] = "WEAK"
    else:
        R["verdict"] = "FAIL"
    R["vs"] = {b: compare(rows, "RCHO", b) for b in ("CHO", "A3", "RA3", "MCHO")}
    R["ablations"] = ablations(rows)
    R["overfit"] = overfit()
    R["means"] = {n: {m: arr(rows, n, m).mean() for m in MET} for n in RULES}
    with open(os.path.join(ROOT, "RESULTS_judge.json"), "w", encoding="utf-8") as f:
        json.dump(R, f, indent=1, default=float)
    print("verdict", R["verdict"])
    print(json.dumps(P, default=float))
    for c in CARDS:
        pc = R["per_card"][c]
        print(c, "uho %.4f->%.4f (%.1f%%) p=%.3g fail %.4f->%.4f up=%.4f delta=%.4f" % (
            pc["uho_cho"], pc["uho_rcho"], 100 * pc["uho_reduction"], pc["p_wilcoxon"], pc["fail_cho"],
            pc["fail_rcho"], pc["fail_diff_upper95"], pc["delta"]))
