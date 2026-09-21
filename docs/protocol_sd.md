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
The design is `docs/paper_plan.md`; the launchers are
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
