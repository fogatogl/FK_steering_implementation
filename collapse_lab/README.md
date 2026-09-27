# `collapse_lab/`: why the four final images descend from a single x_T

Investigation folder, kept apart from the repo: nothing here is imported by `smc/`,
`scripts/` or `tests/`, and nothing here has modified `smc/`. The scripts read
`results/*.json` and call `smc.fk` / `smc.weights` read-only.

The conclusions are in `FINDINGS.md`. This file only says how to replay.

## Without GPU, a few seconds each

    python collapse_lab/a_step0_ess.py      # is the logged ESS the one lam*r implies
    python collapse_lab/b_comb.py           # how many ancestors the comb lets live
    python collapse_lab/c_information.py    # does the ranking at t=80 predict the final ranking
    python collapse_lab/d_forms.py          # do max and difference coincide at the first step
    python collapse_lab/e_variants.py       # what the 19 variants on disk say
    python collapse_lab/g_equivalence.py    # is the `floor` arm the authors' potential
    python collapse_lab/h_invariances.py    # what can move the ESS, and what cannot
    python collapse_lab/i_root100.py        # the root kept, on 100 prompts
    python collapse_lab/j_decomposition.py  # where fk4's gain over bon4 comes from
    python collapse_lab/l_target_ess.py     # how many particles the target carries, at each lambda
    python collapse_lab/r_solutions.py      # the three criteria of "solved", per arm, bootstrap CI
    python collapse_lab/n_coalescence.py    # the lineages predicted from the weights alone, against the observed
    python collapse_lab/t_sessionC.py       # session C: findings 3 and 5 bis on the valid per-slot pairing
    python collapse_lab/ref/parse_authors.py  # the authors' code against bon4, ctl and table 1
    python collapse_lab/z_sessionG.py       # session G: the authors' code, bon4 and FK on three seeds
    python collapse_lab/p_two_rewards.py    # F7: ir_max and mean ir paired against ctl (venv ddpm)
    python collapse_lab/o_ancestry_fig.py    # F4: the ancestry of F5's prompt, the free sampler against FK
    python collapse_lab/q_image_grid.py     # F5: the four finals per arm, framed by root (--appendix: the ten of A.3)
    python collapse_lab/s_coalescence_fig.py  # F6: roots replayed from the weights against observed
    python scripts/fig_three_scales.py      # F8: CIFAR / CelebA / SD (venv ddpm)

The authors' code is cloned outside the repo (`/home/onyxia/work/fkd_ref/`) and launched
from `ref/run_authors.py` in the `sd` venv with `ref/shim` on the path (a module their code
imports and never calls). The night runs are `nuit*.sh`; the predictions written before
each run and their verdicts are in `ASSESSMENT.md` and in `docs/protocol_sd.md`.

The launchers, one per GPU session, kept as they ran (2026):

| launcher | when | what it ran |
|---|---|---|
| `nuit.sh` | 21-22/09 | session A: `lam0`, `ctl`, `floor`, `lam2`, `floor2`, `adapt`, `fadapt` on the A2 |
| `nuit2.sh` | 22-23/09 | the reference night: the released code (`R0`), then the collapse arms |
| `nuit2a.sh` to `nuit2i.sh` | 22-23/09 | its continuations: the corrections at equal n, the latents test, the released code through this repository's generator (`R0g`, `R0g24`), its pipeline without FK, and the `_b1` image replays |
| `nuit3.sh` | 23/09 | session C: `ctl`, `lam0`, `floor2` at 100 prompts in one process, then the released code at seed 2024 |
| `nuit4.sh` | 23/09 | the determinism test: `ctl` on two prompts in three processes |
| `nuitD.sh`, `nuitD_hps.sh` | 23-24/09 | session D on the A2, with every image and Tweedie estimate saved, then HPS v2.1 of its finals |
| `nuitE.sh` | 25/09 | session E on the T4: the machine check, then the released code at seed 2024 on the 100 prompts |
| `nuitF.sh` | 25/09 | session F on the T4: the released code at its commit before the max-potential fix |
| `nuitG.sh`, `nuitG_fk.sh` | 26-27/09 | session G on the A2: seeds 2025 and 2026 of best-of-4, both released versions, then FK |

`commun.py` carries the rule against double counting: `fk4_stat.json` and
`fk4_diff.json` are the same 20 prompts at the same x_T, and their first step is
identical, to within zero (finding 4). Stacking them would divide the standard errors by
root 2 without adding one observation.

## With GPU

    /home/onyxia/work/.venvs/sd/bin/python collapse_lab/probe.py --arms ctl floor --limit 40
    /home/onyxia/work/.venvs/sd/bin/python collapse_lab/f_probe.py
    /home/onyxia/work/.venvs/sd/bin/python collapse_lab/k_figure.py

`nuit.sh` runs the arms in order: `lam0` first, because it decides whether the pairing
with `bon4` holds, then the arms that carry the correction. It resumes from
`out/probe.json`; the (prompt, arm) pairs already done are skipped.

`probe.py` runs `fk_steer` on the same prompts, the same seeds and therefore the
same x_T as `scripts/run_sd_baseline.py`, and keeps in addition the ancestor
matrix, which no file in `results/` preserves. It checks the pairing before
spending a second of GPU: the prompt file must be
`data/imagereward-benchmark-prompts.json`, in that order, otherwise
`seed_effective` does not match `sd_baseline.json` and nothing is comparable
(`prompts_subset_40.json` has another order, and that is the trap).

The arms:

| arm | lambda | floor | what it isolates |
|---|---|---|---|
| `ctl` | 10 | no | the reference `fk4`, control for the steering |
| `floor` | 10 | yes | the released code's `reward_min_value = 0.0`, alone |
| `lam2` | 2 | no | the strength of the tilt alone |
| `floor2` | 2 | yes | both |
| `lam0` | 0 | no | pairing control: must return `bon4` slot by slot |
| `adapt` | 10 max | no | lambda bisected to ESS = k/2 at the non-terminal steps |
| `fadapt` | 10 max | yes | the floor and the adaptive lambda together |

`probe.py --arms` also takes the arms added later (`late`, `stat0`, `multi`, `vae`,
`idx`, `R1`, `thr05`, `rise`); tables A and B of `ASSESSMENT.md` say what each one
changes.

For `adapt` and `fadapt`, 10 is both the ceiling and the lambda of the terminal step:
`bisect_lambda` returns the ceiling as soon as the ESS there is already above the
target, so lambda_t can only go down. These two arms test "less hard early", not the
rising profile of finding 8; "harder late" needs `lam_max = 100` and is another arm
(`rise`).

The floor goes through the reward (`max(r, 0)` before `fk_steer`), not through
`smc/fk.py`: `g_equivalence.py` checks to 9e-7 that this is the authors' `max`
potential, floor included.

## What this folder does not do

It fixes nothing in `smc/`. `FINDINGS.md` section 9 says where the corrections
would live and what trade-off each one carries; the code is the author's to write.
