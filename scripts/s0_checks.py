"""S0 implementation checks of PLAN_roamcho.md Section 6 (seed 7001, short runs; no result is judged)."""
import json
import os
import sys
import time

import numpy as np

import handoverlab as hl
from handoverlab.rules.builtin import A3, CHO
from handoverlab.simulate import run_world

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common500 import scenario          # noqa: E402
from rules500 import RoamA3, RoamCHO, service_columns   # noqa: E402

card = sys.argv[1]
dur = float(sys.argv[2]) if len(sys.argv) > 2 else 40.0
sc = scenario(card, "v60", duration=dur)
t0 = time.perf_counter()
w = hl.build_world(sc, 7001)


class Ctx:
    scenario, seed = sc, 7001


idx, tmin, ws, wbar = service_columns(Ctx)
out = {"card": card, "service_match": bool(np.array_equal(idx, w.services.index)), "w_bar": wbar,
       "n_cells": int(w.n_cells)}
base = dict(o_prep=-2.0, o_exec=3.0, ttt_exec=0.08, n_max=4)
rules = [CHO(o_prep_db=-2.0, o_exec_db=3.0, ttt_exec_s=0.08, n_max=4, name="cho"),
         RoamCHO(**base, name="rcho0"), RoamCHO(**base, memory=False, name="rcho0_nomem"),
         A3(3.0, 0.16, name="a3"), RoamA3(h0=3.0, ttt=0.16, eta_H=0, eta_L=0, eta_A=0, eta_S=0, t_hold=0, name="ra30"),
         RoamCHO(o_prep=-2.0, o_exec=3.0, ttt_exec=0.08, n_max=3, eta_H=8, eta_L=4, eta_A=4, eta_S=2, eta_D=3,
                 rho=0.5, name="rcho_full"),
         RoamA3(name="ra3_paper")]
t1 = time.perf_counter()
res = {r.rule: r for r in run_world(w, rules)}
t2 = time.perf_counter()
fp = {k: v.fingerprint for k, v in res.items()}
out["eq_cho"] = fp["cho"] == fp["rcho0"] == fp["rcho0_nomem"]
out["eq_a3"] = fp["a3"] == fp["ra30"]
out["world_s"], out["run_s"] = round(t1 - t0, 1), round(t2 - t1, 1)
out["full_runs"] = {k: round(res[k].metrics["handovers_pm"], 3) for k in ("rcho_full", "ra3_paper", "cho")}
print(json.dumps(out))
