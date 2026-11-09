"""Scenarios, seeds, rule specs and grids of PLAN_roamcho.md."""
import json
import os

import numpy as np

import handoverlab as hl
from handoverlab.rules.builtin import A3, CHO

from rules500 import RoamA3, RoamCHO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
CARDS = {
    "D": ("urban-grid-10ghz", 240.0, 40.0),
    "F1": ("urban-macro-tr37885-4ghz", 180.0, 30.0),
    "F2": ("urban-grid-28ghz-beams", 120.0, 20.0),
}
SPEEDS = ("ped", "v30", "v60")
VAL_SEEDS = (901, 902, 903)
TEST_SEEDS = tuple(range(1, 21))


def scenario(card, speed, duration=None):
    name, dur, warm = CARDS[card]
    return hl.builtin_scenario(name).with_overrides({
        "mobility.profile": speed, "run.duration_s": float(duration or dur),
        "run.warmup_s": float(warm if duration is None else min(warm, duration / 4)), "run.backend": "numba"})


# ---------------------------------------------------------------- grids (Section 4)
def _r(a, b, s):
    return [round(float(v), 3) for v in np.arange(a, b + s / 2, s)]


TTT_A3 = [0.0, 0.04, 0.064, 0.08, 0.1, 0.128, 0.16, 0.256, 0.32, 0.48, 0.64, 1.024]
TTT_CHO = [0.0, 0.04, 0.08, 0.16, 0.256, 0.32, 0.48, 0.64]
GRIDS = {
    "A3": {"off": _r(-1, 8, 0.5), "ttt": TTT_A3},
    "CHO": {"o_prep": _r(-6, 6, 0.5), "o_exec": _r(-1, 10, 0.5), "ttt_exec": TTT_CHO, "n_max": [1, 2, 3, 4]},
    "RA3": {"h0": _r(-1, 8, 0.5), "ttt": TTT_A3, "eta_H": _r(0, 16, 1), "eta_L": _r(0, 12, 1),
            "eta_A": _r(0, 10, 1), "eta_S": _r(0, 6, 0.5)},
}
GRIDS["MCHO"] = dict(GRIDS["CHO"], eta_H=_r(0, 16, 1), eta_L=_r(0, 12, 1), eta_A=_r(0, 10, 1), rho=[0.0, 0.5, 1.0])
GRIDS["RCHO"] = dict(GRIDS["MCHO"], eta_S=_r(0, 6, 0.5), eta_D=_r(0, 8, 0.5))
METHODS = ("A3", "CHO", "RA3", "MCHO", "RCHO")
N_RANDOM, N_REFINE = 24, 8


def make_rule(method, p, name):
    if method == "A3ref":
        return A3(name=name)
    if method == "A3":
        return A3(off_db=p["off"], ttt_s=p["ttt"], name=name)
    if method == "CHO":
        return CHO(o_prep_db=p["o_prep"], o_exec_db=p["o_exec"], ttt_exec_s=p["ttt_exec"], n_max=p["n_max"],
                   name=name)
    if method == "RA3":
        return RoamA3(h0=p["h0"], ttt=p["ttt"], eta_H=p["eta_H"], eta_L=p["eta_L"], eta_A=p["eta_A"],
                      eta_S=p["eta_S"], name=name)
    if method in ("MCHO", "RCHO", "RCHO-H", "RCHO-A", "RCHO-D", "RCHO-S"):
        q = dict(p)
        q.setdefault("eta_S", 0.0)
        q.setdefault("eta_D", 0.0)
        mem = True
        if method == "RCHO-H":
            mem = False
        elif method == "RCHO-A":
            q["eta_A"] = 0.0
        elif method == "RCHO-D":
            q["eta_D"] = 0.0
        elif method == "RCHO-S":
            q["eta_S"] = 0.0
        return RoamCHO(o_prep=q["o_prep"], o_exec=q["o_exec"], ttt_exec=q["ttt_exec"], n_max=q["n_max"],
                       eta_H=q["eta_H"], eta_L=q["eta_L"], eta_A=q["eta_A"], eta_S=q["eta_S"], eta_D=q["eta_D"],
                       rho=q["rho"], memory=mem, name=name)
    raise KeyError(method)


def random_settings(method, card):
    """The 24 seeded random draws of a method and card (Section 4)."""
    g = GRIDS[method]
    rng = np.random.default_rng([500, METHODS.index(method), list(CARDS).index(card)])
    out = []
    for _ in range(N_RANDOM):
        out.append({k: g[k][int(rng.integers(len(g[k])))] for k in g})
    return out


def refine_settings(method, card, best):
    """8 seeded neighbours of ``best``: one or two parameters moved by one grid step."""
    g = GRIDS[method]
    keys = list(g)
    rng = np.random.default_rng([501, METHODS.index(method), list(CARDS).index(card)])
    out, seen = [], {json.dumps(best, sort_keys=True)}
    tries = 0
    while len(out) < N_REFINE and tries < 1000:
        tries += 1
        q = dict(best)
        for k in rng.choice(keys, size=int(rng.integers(1, 3)), replace=False):
            vals = g[k]
            i = vals.index(q[k]) if q[k] in vals else int(np.argmin([abs(v - q[k]) for v in vals]))
            j = int(np.clip(i + (1 if rng.random() < 0.5 else -1), 0, len(vals) - 1))
            q[k] = vals[j]
        s = json.dumps(q, sort_keys=True)
        if s not in seen:
            seen.add(s)
            out.append(q)
    return out


KEEP = ("unnecessary_pm", "failures_pm", "pingpong_pm", "handovers_pm", "attempts_pm", "prepared_cells_mean",
        "preparations_pm", "unused_preparations_pm", "rrc_messages_pm", "reports_pm", "interruption_s_pm",
        "rate_mbps", "too_late_pm", "too_early_pm", "wrong_cell_pm", "recoveries_pm", "unnecessary_short_and_low_pm",
        "outage_pct", "sinr_db")


def slim(metrics):
    return {k: (float(metrics[k]) if k in metrics else None) for k in KEEP}


def out_path(stage, card, speed, seed):
    d = os.path.join(ROOT, "runs", stage)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "%s_%s_%d.json" % (card, speed, seed))
