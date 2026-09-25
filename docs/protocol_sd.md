# SD v1.5 reduced protocol (19/09)

Written before the run, as the plan requires, and amended only to record what was
actually launched. The target is the SD v1.5 row of table 1 of the FK Steering paper:
ImageReward 0.187 / 0.737 / 0.898 and HPS 0.245 / 0.265 / 0.263 for k = 1,
best-of-4 and FK at lambda = 10, k = 4.

## Prompts

All 100 prompts of the ImageReward benchmark
(`data/imagereward-benchmark-prompts.json`, taken verbatim from the THUDM repository,
which is where the list lives; it is not on the Hub).

The reduced protocol first planned 40 of them, drawn at seed 2026 into
`data/prompts_subset_40.json` by `scripts/sample_prompts.py`. Measuring the two
baselines at 73.6 s per prompt showed the full benchmark fits in a night, so the run
uses all 100. The 40-prompt subset and its script stay versioned: they are what the FK
row will fall back to if its own run does not fit.

The prompt's index in the loaded file feeds the generator seed, so the prompt file is
part of the experiment's identity, not a convenience. Switching files renumbers the
prompts and changes every `x_T`. The partial measurement taken on the 40-prompt file
was therefore discarded rather than merged.

## Sampler

SD v1.5 fp16 (`stable-diffusion-v1-5/stable-diffusion-v1-5`, the `runwayml` repo is
404 since 2024), DDIM with `eta = 1`, 100 steps, classifier-free guidance 7.5,
512 px, no attention slicing (peak VRAM measured at 2.16 GiB out of 15).

## Configurations

| name | what | status |
|---|---|---|
| `k1` | one sample per prompt | measured |
| `bon4` | four samples, the best one by ImageReward | measured |
| `fk4` | lambda = 10, k = 4, MAX potential, fixed schedule `[0, 20, 40, 60, 80]` | measured 20/09, 07:18 to 12:36 |

The first two do not go through `fk_steer` at all: best-of-N here is generate four,
score, take the max, which is why they could be measured before the SD wrapper existed.
The FK row plugs into the same harness as a third sampler, `sample_fk` in
`run_sd_baseline.py`. Its schedule is the paper's, in reverse-process time with 0 the
terminal step, converted to loop indices as `steps - 1 - t`: `[19, 39, 59, 79, 99]`
on 100 steps. "Fixed" means the resampling threshold is 1.0, so the particles are
resampled at every scheduled step as soon as the weights are not uniform; the
adaptive ESS < k/2 variant is not run. The guide decodes with `sd-vae-ft-mse`
(decision 6); the four final latents are decoded with the pipeline's own VAE, the same
decoder the judge saw on `k1` and `bon4`. Each record carries the schedule in both
conventions, the ESS at the five scheduled steps, the number of resamplings and the
guide's ImageReward on the final particles.

## Seeds

**Three seeds**, 2024, 2025 and 2026. The generator is seeded at
`base * 1000 + prompt index`, so each prompt gets its own `x_T` and the draws are not
correlated through a shared initial noise. The base seed and the effective seed both go
into every record.

The loop is prompt-major and finishes all three seeds of a prompt before moving to the
next, so a run cut short by an ephemeral instance leaves complete prompts rather than
a ragged grid. The table reports how many prompts it aggregated.

## Outputs

`results/sd_baseline.json`, one flat record per (prompt, sampler, seed): the per
particle ImageReward and HPS lists, `ir_max` and `hps_at_ir_max`, the UNet counters,
wall time and the generation config. Rewriting the whole file after each finished run
makes the script idempotent: it reindexes `(prompt_id, sampler, seed)` on start and
skips what is already there, which is what the ephemeral Onyxia instances require.

`hps_at_ir_max` and not `max(hps)`: the paper reports the best particle under the
reward that guides, so the judge is read at the guide's argmax. Taking the max of HPS
would be scoring the judge on its own preferred sample and would inflate the FK row.

## Matched budget

On `n_unet_rows`, the number of sample rows that crossed the UNet, not on
`n_unet_calls`. The N particles go through one batched forward, so the call counter
reads 100 for every configuration while the rows read 200 for `k1` and 800 for `bon4`,
a ratio of exactly 4. FK at k = 4 has to land on 800 as well. Both counters are
recorded by a forward hook on `pipe.unet`, reset before each run, so the figure rests
on a measurement and not on an assumption.

## What is missing, and will be said in the write-up

The VAE is
`stabilityai/sd-vae-ft-mse` where the paper does not say which decoder it used. The
current run does not keep the images: the image grid of C7 regenerates the one or two
prompts it needs, which is cheap because the seeds are recorded.

## Follow-up: closing the gap to the paper (20/09, written before running)

The first FK row lands at +0.062 over best-of-4 where the paper has +0.161. The
diagnostics in `docs/results.md` point at premature collapse: median ESS 1.18
out of 4 at the first scheduled step (t = 80), when twenty of the hundred
denoising steps have run. The variants below each move one knob.

**Screening design.** The first 20 prompts of the benchmark (`--limit 20`), seed
2024, `fk4` only, one `--out` file per variant. The prompt index and the seed
are unchanged, so every variant run shares its four $x_T$ with the `k1`, `bon4`
and `fk4` records already in `results/sd_baseline.json`, and the comparison is
paired. A separate file per variant is required, not a convenience: the resume
key is `(prompt_id, sampler, seed)`, and a variant written into the main file
would be skipped as already done.

Cost: 20 runs at ~63 s, about 21 minutes per variant. On 20 prompts the paired
standard error of a difference is about 0.06 (it was 0.026 on 100), so a variant
has to gain about +0.10 over the current `fk4` to show at two standard errors,
which is also the size of the gap to the paper. The screen can rank the
variants; it cannot settle them. The one or two that come out ahead go to 100
prompts x 3 seeds (5.3 h, a night), and only that run enters the table.

| tag | flags | tests | code change |
|---|---|---|---|
| `S60` | `--fk-schedule 0 20 40 60` | drop the t = 80 step, where the collapse happens | none |
| `S40` | `--fk-schedule 0 20 40` | drop t = 80 and t = 60 | none |
| `L2`, `L5`, `L20` | `--lam 2` / `5` / `20` | weight sharpness; the paper's value is 10 | none |
| `D10` | `--fk-schedule 0 10 20 30 40 50 60 70 80 90` | a denser running max, decision 1 | none, +5 decodes per particle |
| `A05` | resample when ESS < k/2 | decision 2; predicted not to help since 1.18 < 2 | a `--fk-threshold` flag |
| `K8` | `fk8` against `bon8` | whether the paper's number needs k > 4 | two `SAMPLERS` entries; twice the cost |

`S60` first: it is the cheapest, it targets the failure the diagnostics show,
and it is the paper's schedule minus one point, so it stays a reproduction with
one stated deviation. A lambda other than 10 is a deviation from the paper's
setting; a change of schedule is a deviation from its interval. Both are
reported as such if kept.

A tempered $\lambda_t$ growing with the denoising progress is the natural fix for
a guide that is noise early and signal late; it changes the potential, so it is
`smc/` and gets its own section below.

## A time-dependent lambda (20/09)

What the first screen says, three ways: the first reward evaluation at full
lambda on a blurred $\hat x_0$ is what costs. `S60` removes it and doubles the
gain; `A05` resamples less often there and gains a little; `L2` softens it and
loses the rest. A lambda that grows with the denoising progress is the
synthesis, and the FK formalism allows it: any sequence $G_t$ is admissible as
long as the product along the lineage reaches $\lambda_T\, r(x_0)$.

**Schedules**, in progress $p = 1 - t/T$ over the paper's five steps
$t \in \{80, 60, 40, 20, 0\}$, $\lambda_T = 10$ so the target is unchanged.
`fk_steer` takes `lam_schedule`, a list over loop indices built in the script
from `--fk-lam-schedule`; `S60` is the degenerate schedule (0, 10, 10, 10, 10).

| tag | flags | $\lambda_t$ | at the five steps | predicted ESS at t = 80 |
|---|---|---|---|---|
| `T1` | `--fk-lam-schedule linear` | $10\,p$ | 2, 4, 6, 8, 10 | about 2.9, from `L2` |
| `T2` | `--fk-lam-schedule quad` | $10\,p^2$ | 0.4, 1.6, 3.6, 6.4, 10 | close to 4 |

**What counts as a result.** `T1` or `T2` above `S60` on the paired ImageReward
difference says the smooth ramp beats the hard cut; below it says the first step
is worth dropping outright. Either way the winner goes to 100 prompts x 3 seeds
and enters the table as a stated deviation from the paper's constant lambda.

## Where the ramp's deficit is paid: terminal correction or tempering (20/09)

Two FK models share the schedules above and the same product along the lineage.
They differ in **where** the weight that a rising $\lambda_t$ leaves unpaid is
collected, and they are not the same experiment. Both are in `smc/fk.py` behind
`lam_placement="terminal" | "tempering"`, `--fk-lam-placement` in the script,
`lam_placement` in every record.

**(a) Terminal correction.** With the MAX potential and running max $m_i$ at the
$i$-th scheduled step,

$$\log G_i = \lambda_i\,(m_i - m_{i-1}) \quad (i < T), \qquad
\log G_T = \lambda_T\, r(x_0) - \text{acc}, \quad \text{acc} = \sum_{i<T} \lambda_i (m_i - m_{i-1}).$$

`acc` is a per-particle integral along the lineage: indexed by the ancestors at
each resampling like `gate`, never reset, unlike `logW`. With constant lambda
`acc` $= \lambda\, m_{T-1}$ and the old formula is the special case. The deficit
$\sum_{i<T} (\lambda_T - \lambda_i)(m_i - m_{i-1})$ is paid in one piece at the
terminal step, and that step's weight is never used: `resample_last` is `False`
and the script picks the particle by `argmax` ImageReward. The images of a `T1`
run are therefore **exactly** those of a run with per-step $\lambda_i$ and no
correction at all. Under (a) the ramp is, operationally, a weaker steering: the
final cloud is tilted by $\sum_i \lambda_i \Delta m_i \le \lambda_T m_{T-1}$.

**(b) Tempering.** The gate carries the tempered previous max:

$$\log G_i = \lambda_i\, m_i - \lambda_{i-1}\, m_{i-1} \quad (i < T), \qquad
\log G_T = \lambda_T\, r(x_0) - \lambda_{T-1}\, m_{T-1},$$

the textbook sequence $G_t = \pi_t / \pi_{t-1}$, $\pi_t \propto p(x)\, e^{\lambda_t m_t}$
(Chopin & Papaspiliopoulos, ch. 17). Written as
$\lambda_i \Delta m_i + (\lambda_i - \lambda_{i-1})\, m_{i-1}$ it is (a) plus a
catch-up term at every scheduled step, so the deficit is paid where resampling
can still act, and the terminal weight stays small and local as in the constant
case. $\lambda_{i-1}$ is the lambda of the previous *scheduled* step, a scalar,
not resampled.

**Verified (20/09, 21:30)**, on the dummy model, 24 tests green: the lineage sum
equals $\lambda_T\, r(x_0)$ under both placements for the three potentials; with
a constant `lam_schedule` tempering produces the same `logG` tensor as the
constant path, for the three potentials; at $\lambda = 0$ tempering is neutral.
On the linear ramp, k = 16, the terminal $\log G$ spreads over 7.2..13.5 under
(a) and 3.9..4.7 under (b), which resamples once more on the way: the deficit
did change place.

**What the two placements predict.** At the first scheduled step both give
$\lambda_1 m_1$: the ESS at t = 80 and the first resampling are identical. They
diverge from t = 60 on, where (b) resamples harder, so (b) sits between (a) and
the constant-lambda `fk4` in intermediate weight spread. If (a) beats `fk4`, (b)
is worth its run; if (a) loses to `fk4`, the ramp is not the lever and (b) loses
too. `T1` against `T1t` on the same $x_T$ is the placement effect alone.

**Lineage collapse does not depend on lambda at threshold 1.0.** The smoke run
of `T1` on prompt 0 gave ESS 2.07 at t = 80 (median 1.18 for `fk4`) and still a
single surviving $x_T$ among the four finals: at threshold 1.0 the cloud is
resampled at every scheduled step whatever the ESS, and four systematic draws on
k = 4 end on one root, ramp or not. The ramp can protect diversity only if the
threshold lets it skip the steps where it kept the ESS up; hence the cross with
`A05`.

**Two record fields for the collapse, since `ir_max` cannot see it.**
`n_lineages`: distinct $x_T$ among the k finals, from a backward walk over
`ancestors`; 1 is the figure-6 failure, four images from one noise. `div_pix`:
mean pairwise RMSE of the k finals at 64 x 64 in [0, 1], a pixel proxy (neither
LPIPS nor CLIP is in the venv) that separates "the same image four times" from
"four images", no more; it applies to `bon4` too. `compare_sd_variants.py`
prints both; `plot_fig6_sd_grid.py --rows k1 bon4 fk4 T1 ...` writes both under
each strip. A variant can win on these and tie on `ir_max`: that is a claim
about collapse, not about the table's number, and the write-up must say which.

**Runs, 20/09 evening**, first 20 prompts, seed 2024, one JSON per tag in
`results/sd_variants/`, in one queue on the T4:

| launcher | tags | placement | log |
|---|---|---|---|
| `run_sd_lambda_t.sh` (20:47) | `T1`, `T2`, `T1A05`, `T2A05` | terminal | `ddpm/sd_lambda_t.log` |
| queued | `bon4`, `fk4` regenerated with the two fields, `results/sd_baseline_div20.json` | none | `ddpm/sd_ref_div.log` |
| `run_sd_tempering.sh`, queued | `T1t`, `T2t`, `T1tA05`, `T2tA05` | tempering | `ddpm/sd_tempering.log` |

The tempering launcher runs `tests/test_fk.py` first and refuses to start if it
is red. Each variant is a fresh process that imports `smc/fk.py` at start, so
the file must not be left broken while the queue runs.

## Code rules, from the review of the tempering change (20/09)

The change was correct and doubled `smc/fk.py` (120 to 218 lines). What it
taught, kept as rules:

- **One formula, one body.** The terminal mode's intermediate potentials are the
  tempering formula with $\lambda_{\text{prev}} = \lambda$; `potentials` computes
  (current, previous) per potential once and `logG = lam * current - lam_prev * previous`,
  terminal being `lam_prev = lam` plus the `acc` override on the last step. A
  mode that duplicates the branches is wrong even when every branch is right.
- **Nothing that cannot run.** No default for an argument the only caller always
  passes; no `if/else` with identical branches; no mode-conditional indexing when
  the unconditional one is harmless (`acc[idx]` on zeros).
- **A comment states the non-obvious**, once: `acc` is an integral and is never
  reset, `logW` is a weight and is. "Accumulator for terminal mode" paraphrases
  the name and goes.
- **Docstrings carry the equations, not a sentence saying what the name says.**
  Signature style follows the file, not a formatter.
- **A property test is never deleted when a mode is added**: the constant-lambda
  telescoping over the three potentials and the $\lambda = 0$ neutrality stay
  alongside the two-placement test; the constant-equivalence test is
  parametrised over the three potentials since the property holds for all;
  no dead parameter in a test (`lam` when `lam_schedule` is given).
- **A branch no test reaches is either tested or removed** (`difference` and
  `sum` under tempering; merging the bodies makes the question vanish).
- **A refactor is gated by the same green.** All 24 before, all 24 after, and the
  terminal path's `logG` byte-identical, since it is what the running screen
  imports.

## What the screen says (21/09)

The results of the eight ramps, paired against the regenerated reference, are
runs 13 to 15 of `docs/results.md`; the fork they leave (the `ir_max` claim
against the collapse claim, and which variant goes to 100 prompts x 3 seeds) is
the last entry of `docs/chronology.md`. This file stays the protocol: what was
planned, what was launched, and the predictions written before the runs, so the
two can be compared.

## Pre-registration of nights 1 and 2 (21/09, written before the runs)

Named here before the GPU is booked, so the reading of each run is fixed in
advance and the blog post quotes a prediction rather than a rationalisation.
The design is the 21/09 editorial plan (not in the repository); the launchers are
`scripts/run_sd_night1.sh` and `scripts/run_sd_night2.sh`.

**What the 300 runs of run 11 already say, and what it forbids saying.**
`scripts/analyze_sd_collapse.py` reads the ESS at the first scheduled step
against the paired gain `fk4` - `bon4`, run by run: Spearman +0.018 on 300
runs, and the three ESS terciles give median gains of +0.060, +0.042 and +0.083,
which is not an order. The median ESS is 1.18, 60 % of runs are below 1.5 and
the maximum is 4.00. Within one setting, the run that collapses early is not the
run that loses. Two things follow. The mechanism claim is about what changing
the schedule does to the average, which is what runs 12 and the confirmatory
run below measure, and not about a within-setting correlation. And the post
cannot write that a low ESS predicts a bad run, which is the sentence the
diagnostics of run 11 invite. The restriction of range is real, since two thirds
of the runs sit between 1.00 and 1.85, and so is the variance that `bon4`
contributes to the difference; neither turns +0.018 into evidence for the
sentence.

**Night 1, the confirmatory run, `S60` at 100 prompts x 3 seeds.** Read as the
paired difference `S60` - `fk4` on `ir_max`, seeds averaged per prompt first,
then the standard error over the 100 prompts, which was 0.0259 for
`fk4` - `bon4` on the same design. The screen gave +0.094 +/- 0.080 at 20
prompts, 1.2 standard errors.

- Above +0.052, two standard errors: the schedule deviation is kept, the FK row
  of the table becomes `S60` and the post states the deviation from the paper's
  interval in the table's caption.
- Between 0 and +0.052: reported as not settled. The table keeps the paper's
  schedule and `S60` is given with its interval as an observation the run could
  not separate from zero.
- At or below 0: the screen's +0.094 was the 20-prompt subset leaning toward it,
  which is the failure mode the screening design was written to expect. The post
  says so, the table carries the paper's setting alone, and what is left is the
  mechanism and the diversity result, which is enough for the claim the post
  makes.

The `S60` - `bon4` difference is read in all three cases, since it is the
comparison the paper's table makes. HPS is predicted inside +/- 0.01 of `fk4`
whatever happens on ImageReward, as it has been on all fifteen variants.

**Night 1, `S80`, one selection at t = 80 then four continuations.** If `S80`
holds `fk4`'s `ir_max` within one standard error, the gain at k = 4 is bought by
a single early selection and the four later resampling steps carry nothing,
which would make the paper's five-point schedule mostly ornamental at this k. If
`S80` falls clearly below `fk4`, the later steps do work. Predicted: below
`fk4`, because `S40` at three steps already gave back part of `S60`'s gain.

**Night 1, `D10`, ten scheduled steps.** Predicted: no gain over `fk4`, inside
one standard error. The screen said three times that the collapse follows the
first evaluation and not the clock, and a denser running max adds evaluations
after the first one. It is run because it is the last costed variant of the
screen's list never launched, and because the post cannot write that the
suspect list is closed while it is open.

**Night 2, `T2tA05` at 100 prompts x 3 seeds.** The screen gave, against the
regenerated `fk4`, `div_pix` +0.090 +/- 0.018, 2.05 lineages out of 4, 38 % of
the diversity gap to `bon4` closed, and `ir_max` -0.034 +/- 0.081. At 100
prompts the standard error on `div_pix` should fall to about 0.008.

- The diversity result survives scale if `div_pix` - `fk4` stays above +0.05 and
  the lineages stay above 1.5. It is then the post's second result, with its own
  figure, and it is stated as a claim about the particle cloud that `ir_max`
  cannot see.
- `ir_max` is predicted inside +/- 0.03 of `fk4`, one screen standard error. If
  `T2tA05` turns out to lose more than 0.05 of ImageReward at 100 prompts, the
  trade is no longer free and the post reports the price instead of the result.
- `div_clip`, if the field is in place, is read next to `div_pix`. The two are
  expected to agree in sign. If they disagree, the pixel proxy is what is
  reported, with the disagreement, and the perceptual claim is dropped.

**What is not pre-registered.** The wrong-root rate, the fraction of runs where
the root FK keeps is not the one `bon4` would have chosen, has no prediction
attached: it is descriptive, it has never been measured, and `root_slots` does
not exist in any file written before tonight. The correspondence it rests on,
slot $i$ of `bon4` and slot $i$ of `fk4` starting from the same $x_T$, is the
one thing to check before the number is quoted; the wrapper reproducing the
pipeline bit for bit at $\lambda = 0$ (run 11) is the reason to expect it to
hold, not a proof that it does.

## Pre-registration of the reference arms and the collapse night (22/09, 19h35, written before the runs)

Configuration table and sources: `docs/reference_config.md`. Everything below runs from
`collapse_lab/nuit2.sh` in this order: `m_latents`, `R1`, `R0`, `B1`, `stat0`, `multi`,
`vae`, `idx`, `thr05`, `rise`. About nine hours on the T4. Readout:
`collapse_lab/r_solutions.py`, `f_probe.py`, `ref/parse_authors.py`. Each prediction is
confronted one by one afterwards (held / missed / under tension), as in constat 10.

**The latents test (`collapse_lab/m_latents.py`, constat 11).** Two prompts, indices 0
and 1, `seed_effective` 2024000 and 2024001. Pipeline latents captured by
`callback_on_step_end` at loop indices 0, 1, 10, 50, 99 against `initial_state` then
`step`, same generator. Predicted: $x_T$ identical at the bit (same `randn_tensor` shape
and order); after index 0, max |diff| under 5e-3 and per-slot correlation above 0.999
(fp16 rounding of `eps`, batched against expanded text encoding); the correlation falls
under 0.9 before index 50 and under 0.5 by index 99 for at least one slot. Reading if
held: $x_T$ is shared and the trajectory is numerically chaotic at $\eta = 1$; slot $j$ of
`bon4` is not the continuation of root $j$, and no slot pairing measures "what the root
becomes". Constats 3 and 5 bis are reformulated, not dropped: the root does not determine
the image. If instead all four correlations stay above 0.99 at index 99, the mismatch of
constat 11 has another cause and constats 3, 5 and 5 bis are dropped.

**`R0`, the released code under the paper's configuration** (`collapse_lab/ref/run_authors.py
--config paper`, 100 prompts, seed 42, one pass). Predicted `ir_max` mean **0.77**, interval
[0.72, 0.82]. Reason: of the four implementation differences, the only one measured here,
the floor on the max statistic, costs -0.09 against `ctl` (arm `floor`, n = 40); the
multinomial resampler adds variance with no known sign; the VAE and the one-index shift are
predicted neutral. Against `bon4` at seed 2024 (same prompts, not the same $x_T$):
`R0 - bon4` in [-0.05, +0.05]. The paper's 0.898 is not predicted to appear. The terminal
adaptive resampling is predicted to return fewer than four distinct images in 20 to 50 % of
runs. Decision rule, fixed now: the gap is **closed** if `R1 - bon4` (paired, 100 prompts)
is at least +0.12; **bounded** otherwise. Predicted outcome: bounded, and issue 3 of the
plan: the released code does not return the published number on this material.

**`R1`, `smc/fk.py` with the four implementation choices** (`probe.py --arms R1`: floor,
statistic form, multinomial at every scheduled step, guide decoded by the pipeline VAE,
indices {20, 40, 60, 80, 99}). Paired to `ctl`: `ir_max` **-0.08 +/- 0.05**, lineages 1.0
to 1.3, t = 80 inert in about 90 % of runs. `R1 - R0` (unpaired means, same prompts):
within +/- 0.03. If |`R1 - R0`| exceeds 0.06, the two codes differ in something the table
does not list, and the bisection below says where.

**The bisection, one choice at a time, 40 prompts, paired to `ctl`.**

- `stat0` (floor + statistic form): `ir_max` -0.09 +/- 0.05, the `floor` arm again; the
  form itself changes nothing (`fk4_stat - fk4` = -0.004 +/- 0.024). Lineages about 1.7.
- `multi` (multinomial at every scheduled step): `ir_max` -0.02 +/- 0.04; `n_resamplings`
  4.0 on every run; lineages 1.0 to 1.1; about 1.8 ancestors after t = 80, as `ctl`, because
  the weights are peaked enough that the comb and the multinomial draw agree.
- `vae` (guide decoded by the pipeline VAE): |`ir_max` difference| under 0.03; the kept
  root (`root_slots`) equals `ctl`'s in at least 70 % of prompts.
- `idx` (indices {20, 40, 60, 80, 99}): |`ir_max` difference| under 0.03; no prediction on
  root agreement, one index moves the reward noise the first step reads.

**`B1`, the image grid.** Six prompts chosen by rule from `out/probe.json` at n = 40, by
`collapse_lab/q_image_grid.py --choose`: the two prompts ranked 20th and 21st of 40 on
`ctl`'s `ir_max`; the two with the largest `ctl - bon4` on `ir_max`; among the prompts where
`floor2` keeps four lineages, the two with the highest `floor2` `ir_max`. Ties by
`prompt_id`. Arms `lam0`, `ctl`, `floor2` and `R1`, rerun with `--save-images` on those six
prompts (same $x_T$, so the same trajectories up to the fp16 drift the latents test
measures). No prediction: the grid is shown, not scored.

**`thr05`** (floor, resampling only when ESS < k/2, 40 prompts). `n_resamplings` 1.2 to 1.8;
lineages 2.4 to 2.8; `ir_max` -0.10 +/- 0.06 against `ctl`; `ir` mean between `floor` and
`floor2`; terminal ESS (`ess_at_schedule[-1]`) under 1.5 in more than half the runs, the
selection being deferred to the last step. If `n_coalescence.py` is ready before this arm
starts, its prediction is added here as a dated line; otherwise these numbers stand.

*Added 22/09, 19h55, from `collapse_lab/n_coalescence.py`, about six hours before the arm
starts.* The coalescence model, applied to the recorded weights of the `floor` arm (40
prompts) with the rule "resample only if ESS < k/2", predicts for `thr05`: **1.84 roots**,
**61 % of runs on a single root**, **1.12 resamplings** per run. This replaces the guess
above (2.4 to 2.8 roots) as the prediction to confront. The same model reproduces the
observed lineages of the seven existing arms within 0.05 (`ctl` 1.07 against 1.05, `floor`
1.73 against 1.73, `fadapt` 2.16 against 2.12, `floor2` 2.97 against 3.00) and the
single-root fractions within three points. The mechanism it implies: with the floor, the
early steps are inert and the accumulated weights stay flat, so `thr05` rarely resamples
before t = 40; when it does, the accumulated weights are peaked (two or three steps of
`exp(10 r)` summed) and one comb pass takes most lineages at once. Fewer resamplings, but
each one harder. If `thr05` lands near 1.8 and not near 2.6, "resample less" is not the
lever the plan expected, and the weights' peakedness at the moment of resampling is.

**`rise`** (floor, bisected lambda with `lam_max = 100`, 20 prompts). Lineages 2.0 to 2.3
(as `fadapt`); at t = 20, the bisected lambda exceeds 10 in the runs where it bites;
`ir_max` -0.05 +/- 0.07 against `ctl`, not above it.

**Cut order if the night overruns:** `rise`, then `idx`, then `vae`. Never cut: the latents
test, `R1`, `R0`, `B1`.

*Added 22/09, about 21h50, after `R0` at 22 prompts and `R1` at 100, before the arm runs.* `R0`
returned 0.554 against 0.770 for `bon4` on the same prompts (-0.216 +/- 0.061), 0.206 +/-
0.072 under `R1`. The diagnostic (`collapse_lab/ref/diag_authors.py`, two prompts) shows
their scorer equal to the official ImageReward to the third decimal and their guide working
as written: floored rewards, flat weights, multinomial drift, then selection. What separates
`R0` from `R1` is therefore not the scorer and not the potential; the remaining difference
is the noise stream (seed 42 through the global RNG against `seed_effective = 2024000 + i`
through a generator). **`R0g`**: their pipeline with `generator=torch.Generator(seed_effective)`
(`run_authors.py --generator`), 40 prompts, so that their code runs from the same $x_T$ and
the same DDIM noise as `bon4`, `ctl` and `R1`; their multinomial draws stay on the global RNG.
Predictions: if the two codes are equivalent, `R0g - R1` within +/- 0.05 paired on 40
prompts and `R0g - bon4` within +/- 0.05; if `R0g - R1` stays below -0.12, their pipeline
differs from `smc/` in something the reference table does not list, and the next step is a
step-by-step comparison of the two trajectories on one prompt under FK (the latents test
without FK already gives correlation above 0.98 at the last step, so the base sampler is not
the suspect).

*Added 22/09, about 22h05, before the latents rerun.* The latents test held $x_T$ at the bit and the
trajectories at correlation 0.984 to 0.9999 per slot at the last step: the chaotic-divergence
prediction is **missed**. Yet the `lam0` records and `bon4` disagree slot by slot (correlation
0.64 across 80 slots, 0.89 once sorted within prompt), and on prompt 0 the largest ImageReward
gap (0.05 against 0.77) sits on the slot with the lowest latent correlation (0.984).
`m_latents.py` now decodes both final latents and scores them, next to the recorded `bon4`
value. Predictions: (a) the pipeline's finals today score within 0.05 of the recorded `bon4`
on every slot (the 20/09 file is reproducible); (b) this repo's finals differ from the
pipeline's by up to 0.7 on the least-correlated slot, the gaps ordering as 1 minus the
correlation: ImageReward is that sensitive to an fp16-level divergence. If (a) fails, the
`bon4` file is the odd one out and constats 3, 5, 5 bis are to be recomputed against `lam0`,
which is the paired free baseline by construction. If (b) fails (all gaps under 0.1), the
`lam0` arm departs from the bare model somewhere in `fk_steer` at lambda 0, to be found.

*Added 23/09, about 03h05, before the run.* Outcome of the rerun: (a) **held**, the pipeline today
returns the recorded `bon4` rewards to the third decimal on both prompts; (b) **missed in its
strong form**: the bare model's finals differ from the pipeline's by 0.17 to 0.34, not 0.7,
and they also differ from the `lam0` records of the probe (prompt 0: 1.11 / -0.29 / 0.69 /
0.87 today against 1.01 / 0.77 / 1.10 / 0.68 recorded). The image redo (`nuit2e.sh`) reruns
`lam0` on that prompt with the same seed and settles whether the probe path is deterministic.
`R0g` at 14 of 40 prompts sits 0.30 +/- 0.19 under `R1`: the equivalence prediction is
failing, and the noise stream is cleared as the cause. **`authors_free_g`**: their pipeline
with `fkd_args=None` and our generator, five prompts (`run_authors.py --no-smc --generator`).
Prediction: if their pipeline is the diffusers pipeline it copies, the four rewards equal the
recorded `bon4` to the third decimal on all five prompts, and the difference between `R0g`
and `R1` lies inside their FK loop; if they differ by more than 0.05 on any slot, their base
sampler is not ours and the reference comparison has to be read against their own free
baseline, not against `bon4`.

*Added 23/09, about 03h35, before the run.* `R0g` at 40 prompts: `ir_max` 0.807, `R0g - R1`
-0.147 +/- 0.092, `R0g - bon4` -0.039 +/- 0.073, and **`R0g - R0` (same prompts, their
seeding against our generator) +0.310 +/- 0.077**. The equivalence prediction is missed on
`R1` and nearly met on `bon4`; the 0.31 between the two seedings of the same code is what
needs explaining, since best-of-4 varies by under 0.02 across seeds. **`authors_free`**: their
pipeline without FK under their seeding path (`torch.manual_seed(42000 + i)`, `generator=None`),
40 prompts. Prediction: if the path is sound, the per-particle mean of its four free samples is
within 0.05 of `bon4`'s per-particle mean on the same prompts (0.288) and its best-of-4 within
0.08 of `bon4`'s (0.846); if it sits 0.2 or more lower, the global-RNG path of their pipeline
is defective on this setup, `R0`'s 0.554 is an artefact of that path, and `R0g` is the
reference number to report (their code at 0.04 under best-of-4, 0.15 under `R1`).

*Added 23/09, about 04h05, before the run.* `authors_free_g` on five prompts: their pipeline without
FK, with our generator, did not return `bon4`'s rewards (slot gaps 0.33 to 2.69). Question left
distributional: `authors_free_g` extended to 40 prompts (`nuit2h.sh`). Prediction: per-particle
mean within 0.05 of `bon4`'s (0.288), best-of-4 within 0.08 of `bon4`'s (0.846).

*Correction, 23/09, about 06h20.* The generator runs above (`R0g`, `authors_free_g`) were seeded
`42000 + i`, the driver's default, not `2024000 + i`: the five-prompt "does not return `bon4`"
verdict compared two seeds. Rerun with `--seed 2024`, their pipeline without FK returns `bon4`'s
four rewards **to the fourth decimal on all five prompts** (20 slots at 0.0000): their base
sampler is bit-compatible with the diffusers pipeline it copies, and the seeding path is sound.
`authors_free` and `authors_free_g` at seed 42 are bit-identical to each other (a CUDA generator
and the global CUDA RNG seeded alike give the same stream), and both match `bon4` in law (0.824
against 0.846 on the best of four, 0.295 against 0.288 per particle): the distributional
prediction held. `R0g - authors_free_g` (-0.018 +/- 0.074) is a noise-paired comparison at seed
42; `R0g - R1` and `R0g - bon4` above are prompt-paired only. **`R0g24`**: their FK, paper
configuration, generator at `2024000 + i`, 40 prompts, so that their code shares $x_T$ and the
DDIM stream with `bon4`, `ctl` and `R1`. Prediction: `R0g24 - bon4` within +/- 0.06 and
`R0g24 - R1` within +/- 0.08, paired; fewer than four distinct final images in 5 to 15 % of runs.
The 0.31 between `R0` (their seeding, shared stream) and `R0g` (generator stream) at seed 42
stays unexplained by this run.

*Outcome, 23/09 about 06h30.* `R0g24`: `ir_max` 0.720, `- bon4` -0.126 +/- 0.070, `- R1`
-0.233 +/- 0.085, duplicates in 10 % of runs. Both intervals missed on the low side; the
duplicate rate held. Their loop steers less well than `smc/` with the same four choices on
identical $x_T$ and DDIM noise (0.56 against 0.82 per particle).

## Pre-registration of session C and the two traces (23/09, about 07h30, written before the runs)

**The trace of the two loops on one prompt.** `ref/diag_authors.py --i 0 --seed 2024
--generator` records, at loop indices 20, 40, 60, 80, 99, the raw rewards their guide sees on
the same $x_T$ as `R1`'s record for prompt `005695-0057`, whose `r_at_schedule` row 0 holds
`smc/`'s rewards at index 20 on the same four particles before any resampling. Prediction:
the four rewards at index 20 agree to 0.05 (same particles, same Tweedie estimate, same
pipeline VAE, same ImageReward), so the two loops see the same first step and diverge only
through the resampling draw (global RNG for theirs, the generator for ours); the 0.23 between
`R0g24` and `R1` is then variance of the root choice plus their terminal duplication, not a
difference of guide. If the index-20 rewards differ by more than 0.2 on any particle, the
guide's input differs (decode or postprocess) and that is the difference to name.

*Outcome, 23/09 about 07h45 (`out/diag_authors_0_s2024.json`).* Their loop on our $x_T$ at
index 20 sees -0.21 / -2.28 / -0.77 / -1.13, all negative: floored, flat, drift, as in `R1`,
whose recorded first-step rewards are the clamped zeros (so the value comparison is not
possible on `R1`; the behaviour prediction holds). On the same four particles one index
earlier, this repo's guide sees -1.78 / -2.27 / -1.37 / -1.15 with the pipeline VAE (`vae`
arm) and -0.56 / -2.26 / -1.46 / -1.09 with sd-vae-ft-mse (`ctl_b1`): particle 0 moves by 1.2
with the decoder and by 1.6 between decoder and one denoising step. The 0.05 agreement
predicted on values is **missed**, and the reason is the finding: at t = 80 the guide's reward
is decoder noise at the unit level, which is why the `vae` arm keeps a different root than
`ctl` in 70 % of prompts. The 0.23 between `R0g24` and `R1` is not a difference of guide at
the first step; it is left to the resampling draws and to their terminal duplication, and it
stays open as variance to be measured, not as a code defect to be found.

**Session C, one process** (`probe.py --arms ctl lam0 floor2 --limit 100 --redo --out
out/probe_C.json`, `HF_HOME` set, about 5 h). Predictions, paired by $x_T$ inside the
session: `floor2 - ctl` on `ir_max` in [-0.14, -0.02]; `floor2` 2.8 to 3.1 roots, 5 to 10 %
single-root; `lam0` four roots, `ir_max` within 0.05 of `bon4`'s mean over the 100 prompts
and slot-correlated with `bon4` under 0.7 (a different session); `ctl` at 1.0 to 1.1 roots
and `ir_max` within 0.05 of `sd_ref_fields100.json`'s mean, slot-uncorrelated with it beyond
0.7. On the recomputed constats: the Kendall tau between the first-step ranking and the final
`lam0` reward stays under 0.15 (constat 3 reformulated, "the root does not determine the
image"); the root `ctl` keeps scores under `lam0` between the median and the best of the
four (constat 5 bis, B between M and A), with A - B in [0.25, 0.50].

*Outcome of session C, 23/09 about 12h30 (`collapse_lab/out/probe_C.json`,
`t_sessionC.py`).* `lam0` returns `bon4`'s slot rewards at correlation 1.00 (the "under 0.7"
prediction missed, in the useful direction: the free path reproduces across sessions, the
resampling path does not, `ctl` at 0.70 and 30 % same root against the 21/09 file).
`floor2 - ctl` paired by $x_T$: -0.012 [-0.082, +0.060] (predicted [-0.14, -0.02], missed on
the shallow side), 3.03 roots and 4 % single-root (held). Constat 3: Kendall tau +0.137 +/-
0.050, top-1 35 % (held, under 0.15). Constat 5 bis: A - B = +0.313 +/- 0.042 (held), B - M =
+0.233 +/- 0.047, C - B = +0.333 +/- 0.038, C - A = +0.021 +/- 0.038.

**`R0` at seed 2024 under their own seeding** (`run_authors.py --config paper --seed 2024
--limit 40`, global RNG, 40 min). Prediction: `ir_max` between `R0` (0.55) and `R0g24` (0.72),
i.e. in [0.55, 0.75]; if it lands within 0.05 of `R0g24`, the 0.31 was specific to seed 42's
stream and is said in one sentence; if it stays near 0.55, the shared-stream path itself
costs 0.2 and stays open.

*Outcome, 23/09 about 13h15.* **0.949**, `- bon4` +0.103 +/- 0.056 on the same $x_T$ (under
their seeding path at seed 2024 the global CUDA stream draws the same $x_T$ as our
generator). The prediction is **missed on the high side**, and the sign of the
"seeding path" effect flips between seeds (-0.31 at 42, +0.23 at 2024): there is no
path effect, there is a spread. On the same 40 prompts the four runs of their FK give
-0.35, -0.04, -0.13, +0.10 against `bon4`; pooled over 220 run-prompts, **-0.110 +/-
0.036**; the standard deviation of `ir_max` across their four runs of one prompt has
median 0.25. This repository's reference on the same 40 prompts in two sessions:
+0.11 and +0.08. The reference reading becomes: the released code's mean is under
best-of-4, and its run-to-run spread on 40 prompts is wider than the effect the paper
reports; one run in four reaches the paper's number within a standard error.

## Pre-registration of session D (23/09, 18h47 UTC, written before the run, launched at 18h47:29)

**Why a session D.** Session C saved no image, and the figures that show images (F1, F5, the
demo) must show images whose rewards are those of their own record. `probe.py` now saves,
with `--save-images`, the four finals and the guide's decoded Tweedie estimates at the five
scheduled steps, and writes `session_id` and the device name in every record. The pod came
back at 18:02 on an NVIDIA A2; the GPU of every earlier session is unrecorded (the docs say
T4). The machine is the one the service allocates; the run goes ahead on it and the record
says which it is.

**The run** (`collapse_lab/nuitD.sh`): `probe.py --arms ctl lam0 floor2 --limit 100
--save-images --out out/session_D/probe_D.json`, one process, prompt-major, `HF_HOME` set.

**Predictions.**

- Speed: 80 to 95 s per run (the sessions that returned the 21/09 numbers ran at 87-90 s,
  every session since the 22/09 evening at 55-60 s). Under 65 s would put this pod in session
  C's speed group.
- `lam0` (free path): slot rewards equal `lam0` of session C at correlation 0.99 or more, mean
  absolute difference of `ir_max` under 0.02 over the 100 prompts.
- `ctl` (resampling path), the test of the machine: if the runs take 80 to 95 s, `ctl` returns
  the 21/09 reference (`sd_ref_fields100.json`, `fk4`, seed 2024) to 0.001 on at least 95 of
  the 100 prompts, and session C's `ctl` only at slot correlation near 0.7 with about 30 % of
  roots in common, as C against 21/09. If the runs take 55 to 65 s, `ctl` equals session C's to
  0.0001 on all 100. Anything else (neither file reproduced) means a third group.
- Whatever the group, on the 100 prompts: `ctl` mean `ir_max` within 0.08 of session C's
  0.799, 1.0 to 1.15 roots; `floor2` 2.8 to 3.2 roots; `floor2 - ctl` on `ir_max`, paired by
  x_T, in [-0.10, +0.08].
- Controls on the thumbnails: `ctl` and `lam0` share their four t = 80 Tweedie estimates (same
  x_T, same noise, no resampling before), so their t = 80 thumbnails agree to 1/255 on every
  prompt. `floor2` too.
- The selection rule (`docs/visual_selection.md`) run on session D: if `ctl` changes group,
  at least 3 of the 8 gains change against the selection read on session C; if it stays in C's
  group, none changes.

*Addendum, 23/09 19h03, before any HPS of session D is computed.* `collapse_lab/nuitD_hps.sh`
scores the 1200 finals of session D with HPS v2.1 once the session ends
(`collapse_lab/w_hps_finals.py`). Prediction: at the image ImageReward selects, HPS of `ctl`
minus HPS of best-of-4 (`lam0`'s best slot) within +/- 0.01 on average over the 100 prompts,
as on 21/09 (+0.0017 +/- 0.0014 between `fk4` and `bon4`).

*Outcome of session D, 24/09 03h05 UTC (`collapse_lab/y_sessionD.py`, `v_ess_check.py`,
`select_visual_prompts.py`).* 300 of 300 runs in one process on the NVIDIA A2, 88.3 s per run
(predicted 80 to 95: held). `ctl` returns the 21/09 reference slot by slot on 100 of 100 prompts
and session A on 40 of 40, and keeps session C's roots in 30 % of prompts (held). `lam0` equals
session A on its 17 prompts and correlates with session C's `lam0` at 0.62 only, mean |`ir_max`
gap| 0.303 (predicted 0.99 and 0.02: **missed**; the free path does not reproduce across GPU
models either). On the 100 prompts: `ctl` `ir_max` 0.826, 1.04 roots (held); `floor2` 2.96 roots
(held); `floor2 - ctl` -0.043 +/- 0.037 (predicted [-0.10, +0.08]: held); the t = 80 thumbnails
of the three arms agree to 0/255 (held); the ESS rebuilt from `logG` matches the record to 1.3e-4.
FK against best-of-4 of the same process: +0.043 +/- 0.038 (session C: +0.021 +/- 0.038). HPS
at the image ImageReward selects, `ctl` - best-of-4: -0.003 +/- 0.002 (predicted within +/-
0.01: held). The selection rule changes 5 of its 8 gains against the rule read on session C
(predicted at least 3: held).

## Pre-registration of session E (25/09, 18h38 UTC, before any run of it)

The pod runs on an NVIDIA T4 (15 GB). Two questions, one launcher (`collapse_lab/nuitE.sh`), nothing
added to the frozen records: the outputs go to `collapse_lab/out/session_E/` and
`results/sd_authors_R0_100.json`.

1. *Which machine was the faster card.* The author states that the runs used an A2 and a T4; the
   records before 23/09 18:47 name no device. `ctl` and `lam0` of `probe.py` on the benchmark's
   first two prompts (`005695-0057`, `005784-0093`), seed 2024. Prediction: equal to session C
   (`probe_C.json`, the faster card) on all eight slots of each arm within 1e-4, and 52 to 65 s per
   run. If they equal session D (the A2) instead, or neither, the faster card was not this T4 and
   the post keeps "a faster card".
2. *The released code on all 100 prompts.* Its seed-2024 runs covered the benchmark's first 40
   prompts only, where this repository's filter with the released code's choices (`R1`) reads
   +0.108 +/- 0.057 against best-of-4 and -0.090 +/- 0.055 on the other 60. `run_authors.py
   --config paper --seed 2024` completes both seedings (global seed, then `--generator`) to the
   100 prompts, the 80 records of 23/09 copied in first so that only positions 40 to 99 run.
   Predictions, if the machine check of 1 holds (otherwise the 60 new prompts pair by prompt only):
   - global seed: minus `R1` within +/- 0.05 on the 60 new prompts (on the first 40: -0.004 +/-
     0.012); minus best-of-4 on all 100 within 0.05 of `R1`'s -0.011, so the released code does not
     reach the +0.12 of the gap test either;
   - generator: if its gap of -0.229 to the global seed on the first 40 belongs to the seeding, the
     60 new prompts give a negative gap again (generator under global seed by 0.1 or more); if it is
     the draw of two random streams, a gap within +/- 0.1 of zero. No sign is assumed.

## Pre-registration of session F (25/09, 18h47 UTC, before any run of it)

The released repository has a commit of 15/06/2025, `699c929` "address max potential bug", which
changes the MAX potential of `fkd_class.py`: before it, the weight is `exp(lambda max(r_t, r_prev))`
with `r_prev` the raw reward of the previous scheduled step (floored at 0 only at the first), and the
last step weighs `exp(lambda r(x_0))` over the product, as the paper's text writes; after it, the
carried reward is the running maximum, floored at 0, including at the last step. Table 1 is older
than the fix (arXiv v1, January 2025). Between the fix's parent (`6726324`) and the version run so far
(`9413005`) the text-to-image code differs in this one change and an unused potential.

Session F runs `6726324` (a separate checkout, `/home/onyxia/work/fkd_ref/prefix_6726324`) under the
paper's configuration, seed 2024, global seed, on the 100 prompts, on the T4, after session E
(`collapse_lab/nuitF.sh`, into `results/sd_authors_prefix.json`, sampler `authors_paper_prefix`).
Paired by x_T with best-of-4 (`sd_baseline.json`, seed 2024) and with session E's post-fix run.

- Decision, the gap test of 22/09 applied to it: if pre-fix minus best-of-4 on the best image reads
  +0.12 or more on the 100 prompts, the code as it stood before the fix reproduces the published gain,
  and the fix is the candidate for the gap; under +0.12, it does not account for it.
- Prediction: pre-fix minus post-fix within +/- 0.10 on the best image (the two weigh the same first
  step, floor included, and differ in which reward they carry after it); pre-fix on one root in 80 %
  of runs or more.
