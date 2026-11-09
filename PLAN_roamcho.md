# ROAM on conditional handover (ROAM-CHO) in HandoverLab: pre-registered plan and pass criteria

Written 2026-11-08, before any result run. Base: HandoverLab `main` e3266f6 (version 2.11.1 plus the per-ray
option, which stays off). The library is not changed: every rule below is study code in `code\hl500\scripts`
that uses the public rule API (`handoverlab.rules.base.ConditionalRule` / `BaselineRule`). The SHA-256 of this
file is recorded in `plan_freeze.txt` before the first tuning run; later changes are dated addenda at the end,
each with its new SHA-256 in `plan_freeze.txt`. Only one run was made before this file: a timing probe
(`scripts\timing.py`, seed 7001, A3 and CHO at library defaults, wall time only; no metric was looked at).

## 0. Standard basis (read in `code\std3gpp`)

- TS 38.300 (R18 extract) 9.2.3.4.1: a CHO is "executed by the UE when one or more handover execution conditions
  are met"; the CHO configuration holds the candidate cells (from the candidate gNBs) and the execution condition(s)
  (from the source gNB); an execution condition is one or two trigger conditions, CHO events A3/A5.
- TS 38.300 9.2.3.4.2 steps 2-3 and 6: "The source gNB decides to use CHO" and requests CHO for one or more
  candidate cells; it sends the RRCReconfiguration with the candidate cells and their execution conditions. Step 8:
  the UE evaluates the conditions and executes towards a candidate that satisfies its condition; step 8c: the
  other candidates are cancelled.
- TS 38.331 V18.11.0 `CondTriggerConfig-r16`: `condEventA3` carries its own `a3-Offset`
  (`MeasTriggerQuantityOffset`, INTEGER (-30..30), "actual value is field value * 0.5 dB"), `hysteresis` and
  `timeToTrigger` per conditional reconfiguration, i.e. per candidate cell. `TimeToTrigger` ENUMERATED
  {ms0, ms40, ms64, ms80, ms100, ms128, ms160, ms256, ms320, ms480, ms512, ms640, ms1024, ms1280, ms2560,
  ms5120}. `maxNrofCondCells-r16` = 8.
- TS 38.331 5.3.5.13.4 (evaluation) and 5.3.5.13.5 (execution): if more than one cell is triggered, "it is up to
  UE implementation which one to select".

Consequences used below: (a) which cells are prepared and with which execution offset is the source gNB's
decision, so it may differ per UE and per candidate; (b) the execution offset is fixed in the configuration
when it is sent (a change needs a new reconfiguration); (c) execution offsets are multiples of 0.5 dB in
[-15, 15] dB; (d) time-to-trigger values come from the enumeration above. HandoverLab models (a)-(d) as
follows: per-(UE, cell) `prep_offsets` and `exec_offsets`, the execution offset frozen when the cell is prepared
(`engine/procedures.py`), a scalar `ttt_exec_s`, a scalar reporting offset, a global `n_max` <= 8, and among the
triggered cells the one with the best gap executes.

## 1. Method: ROAM-CHO

ROAM's four parts, as in the ROAM paper (`roam-handover/roam/policies.py`, paper Table S1), now set the Rel-16
CHO configuration of every UE u, serving cell c and neighbour j at each preparation instant.

1. **Outcome memory (HHT).** Network-wide discounted counts of the four observable outcomes USEFUL /
   UNNECESSARY / EARLY / LATE that HandoverLab feeds back to a learning rule (`on_outcome`; the unnecessary label
   is the engine's observable one: no counterfactual source failure, and a stay shorter than T_min^s or a mean
   gain below theta_Q^s, from L3 measurements). Route table (c, j) and location table (bin b, j); square bins of
   d_b = 40 m of the reported position (the memory key of an attempt is the bin at preparation).
   P_x = (N_x + beta) / (sum N + 4 beta) on the route; P_x^loc = (N^loc_x + kappa_L P_x) / (sum N^loc + kappa_L);
   H = alpha P_un + (1 - alpha) P_early, L = P_late; beta = 1, kappa_L = 3, alpha = 0.6; counts discounted by
   exp(-dt / tau_f) once per second with tau_f = 600 s. A radio-link failure without a handover
   (`on_link_failure`) adds a LATE count to route (source, re-establishment cell) and to the bin where T310 started.
   An empty table gives H = L = 1/4.
2. **Path-area term.** f_k = share of the predicted motion sector (radius max(v T_p, r0), half-angle phi around
   the reported heading; 48-point quadrature: 6 equal-area rings x 8 angles) whose nominal best cell
   (`RuleContext.nominal_cell`, the operator's cell plan) is k. T_p = 5 s, phi = 0.6 rad, r0 = 10 m. Motion
   confidence gamma = 1 / (1 + kappa Omega T_p), Omega an exponential average (weight 0.1 per instant) of
   |heading change| / dt, kappa = 1. f is refreshed every 0.3 s per UE.
3. **Dwell time / short-stay filter.** For every neighbour j the predicted stay T^_j is the time the straight
   predicted path (reported position and heading, speed v, horizon 10 s, steps of 0.25 s) spends in j's nominal
   area, counted from its first entry until it leaves. Short-stay indicator s_j = clip(1 - T^_j / T_min^s, 0, 1),
   with T_min^s of the UE's service (the same 5QI thresholds as the metric). Refreshed with f.
4. **Service awareness.** Interruption weight w_s of the UE's service class (ROAM: voice 0.35, video call 0.55,
   gaming 0.70, streaming 0.20, web 0.10). The service of each UE is the network's knowledge of the bearer's 5QI;
   the rule obtains it by redrawing the world's own service assignment (`streams(seed, "world", 3)[1]`), which
   the smoke check S0 compares with `world.services`.

Per-(u, j) adjustment (centred, so that with all weights 0 ROAM-CHO is exactly tuned CHO):

  D_uj = eta_H (H_uj - 1/4) - eta_L (L_uj - 1/4) + eta_S (w_s(u) - w_bar) - eta_A gamma_u (f_uj - f_uc) + eta_D s_uj,
  clipped to [-6, 6] dB; w_bar = population mean of w_s (0.33 with the default shares).

- **Execution offset** (condEventA3 a3-Offset of candidate j): O_exec = q(o_exec + D_uj), q = rounding to 0.5 dB
  and clipping to [-15, 15] dB; it is computed when j is prepared and stays fixed (Section 0 (b)).
- **Preparation threshold:** cell j is prepared when gap_uj - O_prep_uj >= 0 with O_prep_uj = o_prep + rho D_uj;
  the reporting offset is the lowest reachable value o_prep - 6 rho, so that every candidate the network may want
  is reported.
- **Candidate choice:** the prepared set is the top n_max cells by gap_uj - O_prep_uj (engine rule with the
  default stick bonus and release hysteresis), so memory, area, dwell and service reorder the candidates; at
  execution the engine takes the triggered candidate with the best gap.
- Tuned: o_prep, o_exec, ttt_exec, n_max, eta_H, eta_L, eta_A, eta_S, eta_D, rho. Fixed: everything else above.

## 2. Methods and baselines (same tuning budget for each tuned method)

| Key | Method | Tuned parameters (grids in Section 4) |
|---|---|---|
| A3 | Event A3 handover (HandoverLab `A3`) | off, ttt |
| CHO | Rel-16 CHO (HandoverLab `CHO`, rel_hys 3 dB) | o_prep, o_exec, ttt_exec, n_max |
| RA3 | ROAM on A3, the published method: CIO m_uj = h0 + eta_H H - eta_L L + eta_S w_s - eta_A gamma (f_j - f_c), clipped to [0, 9] dB, TTT, hold T_hold = 1 s; the base-layer term epsilon[j = a_init] is left out (epsilon = 0; no K-means/tree base layer) | h0, ttt, eta_H, eta_L, eta_A, eta_S |
| MCHO | MemCHO-style ablation: ROAM-CHO with memory and path only (eta_S = eta_D = 0) | o_prep, o_exec, ttt_exec, n_max, eta_H, eta_L, eta_A, rho |
| RCHO | ROAM-CHO (Section 1) | all ten |

Ablations of RCHO (not retuned; RCHO's selected setting with one part removed): RCHO-H (memory off: H = L = 1/4
always), RCHO-A (eta_A = 0), RCHO-D (eta_D = 0), RCHO-S (eta_S = 0). Reference rule for the failure budget:
A3ref = HandoverLab `A3()` defaults (3 dB, 160 ms). All rules run with `learns` where they learn; every rule gets
the same worlds (shared radio samples per seed).

## 3. Scenarios, seeds, metrics

- Cards (HandoverLab built-ins, defaults otherwise; `run.backend: numba`, bit-identical to NumPy):
  - **D** `urban-grid-10ghz`: dense urban small cells, TR 37.885 grid, 76 sites, 10 GHz (upper mid-band /
    beyond 5G), 40 UEs; run 240 s, warm-up 40 s.
  - **F1** `urban-macro-tr37885-4ghz`: 5G FR1 urban macro, 35 sites (105 cells), 4 GHz, 100 UEs; run 180 s,
    warm-up 30 s.
  - **F2** `urban-grid-28ghz-beams`: 5G FR2 small cells with SSB beams and beam failure recovery, 228 cells,
    28 GHz, 40 UEs; run 120 s, warm-up 20 s (card default).
- Speeds (`mobility.profile`): `ped` (1.2-1.8 m/s on the pavement), `v30` (30 km/h), `v60` (60 km/h).
- Seeds: validation 901, 902, 903 (tuning only); test 1-20 (judging only). Never mixed.
- Metrics (HandoverLab `RunResult.metrics`, per UE-minute unless stated): **unnecessary_pm** (main), failures_pm
  (all radio-link and handover failures), pingpong_pm (return within 1 s), handovers_pm, prepared_cells_mean,
  preparations_pm, unused_preparations_pm, rrc_messages_pm (signalling), interruption_s_pm, rate_mbps
  (throughput). The unnecessary label is the engine's (stay < T_min^s or mean SINR gain < theta_Q^s, without a
  counterfactual source failure), the same definition as the ROAM paper.

## 4. Tuning (validation seeds only; same budget for every tuned method)

- Budget: 32 settings per method and card: 24 random draws from the grid (seeded, `numpy.random.default_rng(500)`
  per method and card), then 8 refinements: neighbours of the best feasible setting so far, each changing one or
  two parameters by one grid step (seeded).
- Objective per card, pooled over the 3 speeds x 3 validation seeds (9 worlds, mean of per-world values):
  feasible = mean failures_pm <= budget, budget = mean failures_pm of A3ref on the same 9 worlds. Among feasible
  settings, the lowest unnecessary_pm; settings within 5 % of it are equal, then the fewest prepared_cells_mean
  (CHO variants) or handovers_pm (A3 variants), then the fewest failures. If none is feasible: fewest failures.
- Grids: offsets in 0.5 dB steps. A3: off -1..8 dB, ttt {0, 40, 64, 80, 100, 128, 160, 256, 320, 480, 640,
  1024} ms. CHO and CHO variants: o_prep -6..6 dB, o_exec -1..10 dB, ttt_exec {0, 40, 80, 160, 256, 320, 480,
  640} ms, n_max {1, 2, 3, 4}. ROAM weights: eta_H 0..16 (step 1), eta_L 0..12 (1), eta_A 0..10 (1), eta_S 0..6
  (0.5), eta_D 0..8 (0.5), rho {0, 0.5, 1}. RA3: h0 -1..8 dB, ttt as A3. A grid edge reached by a selected
  setting is reported (MemCHO lesson); grids are not widened after the test run.
- One setting per method and card (not per speed), to limit the overfit that per-site-distance tuning on two
  seeds caused in the MemCHO study.

## 5. Pass criteria (fixed now)

Unit of pairing: one world = (card, speed, test seed), 3 x 3 x 20 = 180 worlds; RCHO vs tuned CHO on the same
world.

- **H1a (primary, fewer unnecessary handovers):** pooled mean unnecessary_pm of RCHO is lower than CHO's,
  one-sided Wilcoxon signed-rank test over the 180 paired differences p < 0.05, and the pooled relative
  reduction is at least 10 %.
- **H1b (primary, failures not worse):** non-inferiority of failures: the one-sided 95 % upper bound (stratified
  bootstrap over worlds within each card x speed, 10 000 resamples, seed 500) of mean failures_pm(RCHO) - mean
  failures_pm(CHO) is <= delta, delta = max(0.10 x mean failures_pm(CHO), 0.002 per UE-minute).
- **Verdict:** PASS if H1a and H1b hold pooled; "PASS in all cards" if in addition, in each card, the point
  reduction of unnecessary_pm is > 0 and the failure difference's upper bound is <= that card's delta;
  WEAK if H1a holds only without the 10 % size (significant but smaller) and H1b holds; else FAIL.
- **Overfit check (reported with the verdict):** for every tuned method and card, test-seed mean failures_pm of the
  selected setting divided by the validation budget; above 1.3 is flagged as overfit.
- **Secondary (reported, not part of the verdict):** RCHO vs A3, RA3 and MCHO on all metrics (paired Wilcoxon,
  Holm over the metrics of each comparison); per-card and per-speed tables; signalling (rrc_messages_pm),
  prepared cells, interruption, throughput, ping-pong.
- **Which parts help:** for each ablation, the paired change of unnecessary_pm and failures_pm against RCHO; a
  part "helps" if removing it raises unnecessary_pm significantly (one-sided Wilcoxon, Holm over the four parts,
  alpha 0.05) without lowering failures_pm by more than delta, or if removing it raises failures_pm beyond delta.

## 6. Checks before the runs (implementation, not results)

- S0: RCHO with all weights 0 and rho = 0 gives the same fingerprint and metrics as CHO with the same base
  parameters (same process, same world); MCHO likewise; RA3 with all weights 0, h0 = off and hold 0 equals A3.
- S0: the rule's service assignment equals `world.services` on one seed of every card.
- S0: the rule raises no error on all three cards with all parts on (short run).

## 7. Runs

Windows scheduled task, one driver that keeps at most 6 worker processes, each worker at BelowNormal priority and
affinity limited to 6 cores (0x3F); single-threaded BLAS/Numba. Never launched through WMI. Resume skips worlds with
a complete output file. Outputs: `code\hl500\runs\<stage>\<card>_<speed>_<seed>.json`.

## Addendum 1 (2026-11-08): S0 results and implementation notes, before any tuning run

- S0 passed on D, F1 and F2 (seed 7001, 40 s, v60; `scripts\s0_checks.py`, logs `logs\s0_*.txt`): RCHO with all
  weights 0 and rho = 0 (with and without memory) has the same event-log fingerprint as `CHO(-2, 3, 80 ms, 4)`;
  RA3 with all weights 0, h0 = 3 dB and no hold has the same fingerprint as `A3(3 dB, 160 ms)`; the rule's
  service draw equals `world.services.index`; the full RCHO and the paper-valued RA3 run without error. Only
  handovers_pm was printed for those two (to see that they act); no comparison was made.
- Correction of a number in Section 1: w_bar with the default service shares is 0.36 (0.2 x 0.35 + 0.2 x 0.55 +
  0.15 x 0.70 + 0.3 x 0.20 + 0.15 x 0.10), not 0.33. The definition (population mean) is unchanged; the code
  computes it from the shares.
- Implementation notes: Omega (heading-change rate) and the HHT discount are updated at the instants when the
  engine calls the rule (reports and preparations for CHO rules, every measurement for RA3), with the averaging
  weight 0.9^(dt / 0.1 s), which equals ROAM's per-0.1 s average. Code: `scripts\rules500.py`,
  `scripts\common500.py`, `scripts\run500.py` (stages tune1 -> select -> tune2 -> select -> test, as one
  resumable pipeline). Stage tune2 repeats A3ref; the budget is taken from tune1 only (same worlds, same value).
  The random draws use `default_rng([500, method index, card index])`, the refinements
  `default_rng([501, method index, card index])` (a fixed version of Section 4's "seeded").
