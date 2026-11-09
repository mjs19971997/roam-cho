# ROAM-CHO: history- and dwell-aware conditional handover

Study plan, scripts, raw outputs and results of the article

> M. Jalili Sanjarani, Y. Elmi Sola, H. Hasanpour, A. A. Neghabi, *Fewer prepared cells at equal reliability in dense
> small-cell networks: a pre-registered study of history- and dwell-aware conditional handover*.

ROAM-CHO (RCHO) moves the four parts of the ROAM handover policy onto 3GPP Release-16 conditional handover (CHO):
outcome memory per route and per place (H), the overlap of the predicted path with each neighbour (A), a short-stay
term from the predicted dwell time (D) and service awareness (S). Together they shift, per user and candidate cell,
the preparation threshold and the execution offset that the conditional configuration already carries
(TS 38.331 `CondTriggerConfig-r16`). With all weights at zero it is standard CHO.

All runs use the open simulator [HandoverLab](https://github.com/mjs19971997/handoverlab) 2.11.1
(DOI [10.5281/zenodo.23409131](https://doi.org/10.5281/zenodo.23409131)).

## Design

The plan `PLAN_roamcho.md` was frozen before any tuning run; `plan_freeze.txt` holds the SHA-256 of the plan and of
its Addendum 1. Three scenario cards (D: dense urban grid, 10 GHz; F1: TR 37.885 urban macro, 4 GHz; F2: urban grid,
28 GHz beams) x three speeds (pedestrian, 30 km/h, 60 km/h). Every tuned method (A3, CHO, ROAM on A3, a memory-and-path
CHO "MCHO", RCHO) got the same budget of 32 settings on validation seeds 901-903; the selected settings were then run
once on test seeds 1-20 (180 paired worlds), together with four ablations of RCHO (one part removed each).

## Results in brief

Pre-registered verdict: **FAIL**. Unnecessary handovers fell by 26.9 % against tuned CHO (one-sided Wilcoxon
p = 2.9e-13), but failure non-inferiority failed pooled, because of card F1.

RCHO against tuned CHO, paired relative change of the means (full tables with 95 % CIs in `RESULTS.md`):

| Card | Unnecessary handovers | Failures | Prepared cells | RRC messages |
|---|---|---|---|---|
| D (dense urban) | -30 % | -43 % | -63 % | -24 % |
| F1 (FR1 macro) | -56 % | +131 % | -40 % | +17 % |
| F2 (FR2 beams) | +3 % (not significant) | -1 % (not significant) | -62 % | -29 % |
| All 180 worlds | -27 % | +12 % | -55 % | -13 % |

Ablations: removing the memory (H) or the dwell term (D) raises unnecessary handovers by 5.8 % and 9.1 %; removing
the area (A) or service (S) term changes nothing measurable.

## Files

| Path | Content |
|---|---|
| `PLAN_roamcho.md`, `plan_freeze.txt` | frozen study plan and its hashes |
| `scripts/rules500.py` | the RCHO and ROAM-on-A3 rules (HandoverLab rule API) |
| `scripts/common500.py` | cards, seeds, tuning grids, rule factory |
| `scripts/run500.py` | runner: tuning stages, selection, test stage |
| `scripts/judge500.py`, `scripts/report500.py` | pre-registered tests -> `RESULTS_judge.json`, `RESULTS.md` |
| `scripts/s0_checks.py`, `logs/s0_*.txt` | implementation checks run before tuning (zero weights reproduce CHO and A3) |
| `scripts/timing.py` | timing probe used to size the runs |
| `sel1.json`, `sel_final.json` | selected settings after the two tuning stages |
| `runs/tune1`, `runs/tune2`, `runs/test` | per-world metrics of every rule (one JSON file per card, speed and seed) |
| `figures/data5.py`, `figures/figs5.py` | numbers and figures of the article, from the files above |

## Reproduce

Python 3.12 (64-bit):

```bash
pip install -r requirements.txt
cd scripts
python s0_checks.py D          # implementation checks (also F1, F2)
python run500.py pipeline 6    # tune1 -> select -> tune2 -> select -> test, 6 worlds at once
python judge500.py             # pre-registered verdict -> RESULTS_judge.json
python report500.py            # tables -> RESULTS.md
cd ../figures
python figs5.py                # figures of the article -> figures/out
```

The outputs of all stages are included, so `judge500.py`, `report500.py` and `figs5.py` run without any simulation.
`run500.py` skips worlds whose output file already exists; remove `runs/` (and `sel1.json`, `sel_final.json`) to
rerun from scratch.

## License

MIT, see `LICENSE`.
