"""Write RESULTS.md from the test stage (after judge500.py); no new decisions, only the Section 5 numbers."""
import json
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import judge500 as J  # noqa: E402
from common500 import CARDS, ROOT, SPEEDS  # noqa: E402

rows = J.load()
J.load_cache = rows
R = json.load(open(os.path.join(ROOT, "RESULTS_judge.json"), encoding="utf-8"))
KEY = ("unnecessary_pm", "failures_pm", "prepared_cells_mean", "rrc_messages_pm", "pingpong_pm",
       "interruption_s_pm", "rate_mbps")


def ci(rows, a, b, m, seed=500, B=10000):
    """Paired mean difference a-b with a two-sided 95 % bootstrap CI stratified by card x speed."""
    d = J.arr(rows, a, m) - J.arr(rows, b, m)
    st = {}
    for i, r in enumerate(rows):
        st.setdefault(r[0][:2], []).append(i)
    rng = np.random.default_rng(seed)
    tot = np.zeros(B)
    for idx in st.values():
        idx = np.array(idx)
        tot += d[idx][rng.integers(0, len(idx), size=(B, len(idx)))].sum(1)
    tot /= len(d)
    p = 1.0 if np.all(d == 0) else float(wilcoxon(J.arr(rows, a, m), J.arr(rows, b, m)).pvalue)
    return d.mean(), np.quantile(tot, 0.025), np.quantile(tot, 0.975), p


L = ["# ROAM-CHO study: test-stage results (PLAN_roamcho.md Section 5)", "",
     "180 paired test worlds (3 cards x 3 speeds x seeds 1-20). Numbers per UE-minute unless noted.", ""]
P = R["pooled"]
L += ["## Verdict (pre-registered): **%s**" % R["verdict"], "",
      "| | UHO CHO | UHO RCHO | reduction | p (1-sided) | H1a | fail CHO | fail RCHO | diff upper95 | delta | H1b |",
      "|---|---|---|---|---|---|---|---|---|---|---|"]
for name, x in [("pooled", P)] + [(c, R["per_card"][c]) for c in CARDS]:
    L.append("| %s | %.4f | %.4f | %.1f %% | %.2g | %s | %.4f | %.4f | %+.4f | %.4f | %s |" % (
        name, x["uho_cho"], x["uho_rcho"], 100 * x["uho_reduction"], x["p_wilcoxon"], x["H1a"], x["fail_cho"],
        x["fail_rcho"], x["fail_diff_upper95"], x["delta"], x["H1b"]))
L += ["", "## RCHO minus baseline, paired mean difference [95 % stratified bootstrap CI], two-sided Wilcoxon p", ""]
for b in ("CHO", "MCHO", "A3"):
    L += ["### vs %s" % b, "", "| card | " + " | ".join(KEY) + " |", "|---" * (len(KEY) + 1) + "|"]
    for c in list(CARDS) + ["all"]:
        rc = rows if c == "all" else [r for r in rows if r[0][0] == c]
        cells = []
        for m in KEY:
            mu, lo, hi, p = ci(rc, "RCHO", b, m)
            base = J.arr(rc, b, m).mean()
            rel = " (%+.0f %%)" % (100 * mu / base) if base else ""
            cells.append("%+.3f [%+.3f, %+.3f]%s p=%.1g" % (mu, lo, hi, rel, p))
        L.append("| %s | %s |" % (c, " | ".join(cells)))
    L.append("")
L += ["## Per card and speed (means of 20 seeds)", "",
      "| card | speed | UHO A3/CHO/MCHO/RCHO | fail A3/CHO/MCHO/RCHO | prep CHO/MCHO/RCHO | RRC CHO/RCHO |",
      "|---|---|---|---|---|---|"]
for c in CARDS:
    for s in SPEEDS:
        q = R["per_card"][c]["per_speed"][s]
        f = lambda m, ks: "/".join("%.3f" % q[m][k] for k in ks)
        L.append("| %s | %s | %s | %s | %s | %s |" % (c, s, f("unnecessary_pm", ("A3", "CHO", "MCHO", "RCHO")),
                 f("failures_pm", ("A3", "CHO", "MCHO", "RCHO")), f("prepared_cells_mean", ("CHO", "MCHO", "RCHO")),
                 "/".join("%.0f" % q["rrc_messages_pm"][k] for k in ("CHO", "RCHO"))))
L += ["", "## Ablations (each removes one part from the selected RCHO setting; not retuned)", "",
      "| ablation | UHO change when removed | p Holm | failure change | helps |", "|---|---|---|---|---|"]
for a, v in R["ablations"].items():
    L.append("| %s | %+.1f %% | %.2g | %+.4f | %s |" % (a, 100 * v["uho_change_when_removed"], v["p_holm"],
                                                      v["fail_change"], v["helps"]))
L += ["", "## Overfit check (test failures / validation budget) and grid-edge parameters", "",
      "| card | method | test fail | budget | ratio | flag | feasible on val | edge params |", "|---|---|---|---|---|---|---|---|"]
for c, v in R["overfit"].items():
    for m, x in v.items():
        L.append("| %s | %s | %.3f | %.3f | %.2f | %s | %s | %s |" % (c, m, x["test_fail"], x["budget"], x["ratio"] or 0,
                 x["flag"], x["val_feasible"], ", ".join(x["grid_edges"]) or "-"))
with open(os.path.join(ROOT, "RESULTS.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")
print("ok")
