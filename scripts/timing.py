"""Timing probe: one seed per card, A3 + CHO, before the plan is frozen (no results used)."""
import sys, time
import handoverlab as hl
from handoverlab.rules.builtin import A3, CHO
card, prof, dur = sys.argv[1], sys.argv[2], float(sys.argv[3])
sc = hl.builtin_scenario(card).with_overrides({"mobility.profile": prof, "run.duration_s": dur,
                                               "run.backend": "numba"})
t0 = time.perf_counter()
w = hl.build_world(sc, 7001) if hasattr(hl, "build_world") else None
t1 = time.perf_counter()
from handoverlab.simulate import run_world
res = run_world(w, [A3(), CHO()])
t2 = time.perf_counter()
print(card, prof, dur, "world %.1fs run %.1fs" % (t1 - t0, t2 - t1), res[0].timing if hasattr(res[0], "timing") else "")
