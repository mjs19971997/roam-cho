"""Driver of PLAN_roamcho.md.

  run500.py world <stage> <card> <speed> <seed>     one world, all rules of the stage (one process)
  run500.py batch <stage> <workers>                 every missing world of a stage, at most <workers> at once
  run500.py select <stage>                          selection after tune1 (sel1.json) or tune2 (sel_final.json)
  run500.py pipeline <workers>                      tune1 -> select -> tune2 -> select -> test (resumable)
"""
import ctypes
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common500 import (CARDS, METHODS, ROOT, SPEEDS, TEST_SEEDS, VAL_SEEDS, make_rule, out_path,  # noqa: E402
                       random_settings, refine_settings, slim)

LOG = os.path.join(ROOT, "logs")
os.makedirs(LOG, exist_ok=True)
PY = sys.executable
BELOW_NORMAL = 0x4000
AFFINITY = 0x3F


def log(stage, msg):
    with open(os.path.join(LOG, "batch_%s.log" % stage), "a", encoding="utf-8") as f:
        f.write("%s %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg))


def load(name):
    with open(os.path.join(ROOT, name), encoding="utf-8") as f:
        return json.load(f)


def stage_rules(stage, card):
    """[(name, method, params)] of a stage."""
    out = [("A3ref", "A3ref", {})]
    if stage == "tune1":
        for m in METHODS:
            out += [("%s#%02d" % (m, i), m, p) for i, p in enumerate(random_settings(m, card))]
    elif stage == "tune2":
        sel = load("sel1.json")[card]
        for m in METHODS:
            out += [("%s#%02d" % (m, 24 + i), m, p)
                    for i, p in enumerate(refine_settings(m, card, sel[m]["params"]))]
    elif stage == "test":
        sel = load("sel_final.json")[card]
        for m in METHODS:
            out.append((m, m, sel[m]["params"]))
        for a in ("RCHO-H", "RCHO-A", "RCHO-D", "RCHO-S"):
            out.append((a, a, sel["RCHO"]["params"]))
    else:
        raise KeyError(stage)
    return out


def run_one(stage, card, speed, seed):
    import handoverlab as hl
    from handoverlab.simulate import run_world
    from common500 import scenario
    t0 = time.perf_counter()
    specs = stage_rules(stage, card)
    w = hl.build_world(scenario(card, speed), seed)
    rules = [make_rule(m, p, n) for n, m, p in specs]
    res = run_world(w, rules)
    rec = {}
    for (n, m, p), r in zip(specs, res):
        assert r.rule == n, (r.rule, n)
        rec[n] = {"method": m, "params": p, "metrics": slim(r.metrics), "fingerprint": r.fingerprint}
    out = {"stage": stage, "card": card, "speed": speed, "seed": seed, "wall_s": round(time.perf_counter() - t0, 1),
           "rules": rec}
    path = out_path(stage, card, speed, seed)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(out, f)
    os.replace(path + ".tmp", path)


def jobs(stage):
    seeds = TEST_SEEDS if stage == "test" else VAL_SEEDS
    order = ("F2", "F1", "D")                    # slowest card first
    return [(c, s, k) for c in order for s in SPEEDS for k in seeds]


def lower_priority():
    k = ctypes.windll.kernel32
    k.GetCurrentProcess.restype = ctypes.c_void_p
    k.SetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
    k.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
    h = k.GetCurrentProcess()
    k.SetPriorityClass(h, BELOW_NORMAL)
    k.SetProcessAffinityMask(h, AFFINITY)


def batch(stage, workers):
    lower_priority()
    todo = [j for j in jobs(stage) if not os.path.exists(out_path(stage, *j))]
    log(stage, "BATCH start %d worlds, %d workers" % (len(todo), workers))
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1",
               NUMBA_NUM_THREADS="1")

    def work(j):
        c, s, k = j
        tag = "%s_%s_%d" % (c, s, k)
        log(stage, "START " + tag)
        with open(os.path.join(LOG, "%s_%s.out" % (stage, tag)), "w", encoding="utf-8") as f:
            rc = subprocess.run([PY, __file__, "world", stage, c, s, str(k)], stdout=f,
                                stderr=subprocess.STDOUT, env=env, creationflags=BELOW_NORMAL).returncode
        log(stage, "END %s rc=%d" % (tag, rc))
        return rc

    with ThreadPoolExecutor(workers) as ex:
        rcs = list(ex.map(work, todo))
    missing = [j for j in jobs(stage) if not os.path.exists(out_path(stage, *j))]
    log(stage, "BATCH end, failed rc %d, missing %d" % (sum(r != 0 for r in rcs), len(missing)))
    return not missing


# ---------------------------------------------------------------- selection (Section 4)
def pooled(stages, card):
    """{name: (method, params, mean metrics over the 9 validation worlds)}."""
    acc = {}
    for st in stages:
        for c, s, k in jobs(st):
            if c != card:
                continue
            with open(out_path(st, c, s, k), encoding="utf-8") as f:
                d = json.load(f)
            for n, r in d["rules"].items():
                if st != stages[0] and n == "A3ref":
                    continue
                a = acc.setdefault(n, {"method": r["method"], "params": r["params"], "m": []})
                a["m"].append(r["metrics"])
    out = {}
    for n, a in acc.items():
        keys = a["m"][0].keys()
        out[n] = (a["method"], a["params"],
                  {k: sum(x[k] for x in a["m"]) / len(a["m"]) for k in keys if a["m"][0][k] is not None},
                  len(a["m"]))
    return out


def choose(cands, budget, method):
    """cands: [(name, params, metrics)] -> chosen entry (Section 4 rule)."""
    feas = [c for c in cands if c[2]["failures_pm"] <= budget]
    if not feas:
        return min(cands, key=lambda c: c[2]["failures_pm"]), False
    u = min(c[2]["unnecessary_pm"] for c in feas)
    eq = [c for c in feas if c[2]["unnecessary_pm"] <= u * 1.05 + 1e-12]
    tie = "prepared_cells_mean" if method not in ("A3", "RA3") else "handovers_pm"
    return min(eq, key=lambda c: (c[2][tie], c[2]["failures_pm"])), True


def select(stage):
    stages = ["tune1"] if stage == "tune1" else ["tune1", "tune2"]
    res = {}
    for card in CARDS:
        P = pooled(stages, card)
        budget = P["A3ref"][2]["failures_pm"]
        res[card] = {"budget_failures_pm": budget, "n_worlds": P["A3ref"][3],
                     "A3ref": P["A3ref"][2]}
        for m in METHODS:
            cands = [(n, p, mt) for n, (mm, p, mt, _) in P.items() if mm == m]
            (n, p, mt), feasible = choose(cands, budget, m)
            res[card][m] = {"name": n, "params": p, "feasible": feasible, "n_settings": len(cands), "val": mt}
    name = "sel1.json" if stage == "tune1" else "sel_final.json"
    with open(os.path.join(ROOT, name), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    log(stage, "SELECT wrote " + name)


def pipeline(workers):
    lower_priority()
    for st in ("tune1", "tune2", "test"):
        if st == "tune2" and not os.path.exists(os.path.join(ROOT, "sel1.json")):
            select("tune1")
        if st == "test" and not os.path.exists(os.path.join(ROOT, "sel_final.json")):
            select("tune2")
        ok = batch(st, workers)
        if not ok:
            ok = batch(st, workers)           # one retry of missing worlds
        if not ok:
            log(st, "PIPELINE stopped: missing worlds")
            open(os.path.join(ROOT, "PIPELINE_FAILED.txt"), "w").write(st)
            return
    open(os.path.join(ROOT, "PIPELINE_DONE.txt"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "world":
        run_one(a[1], a[2], a[3], int(a[4]))
    elif a[0] == "batch":
        batch(a[1], int(a[2]))
    elif a[0] == "select":
        select(a[1])
    elif a[0] == "pipeline":
        pipeline(int(a[1]))
