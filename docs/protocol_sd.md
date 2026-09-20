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

Not on the list: a tempered $\lambda_t$ growing with the denoising progress. It
is the natural fix for a guide that is noise early and signal late, but it
changes the potential, which is `smc/`, and it is not in the paper.

## A time-dependent lambda (20/09, written before running)

What the screen says, three ways: the first reward evaluation at full lambda on a
blurred $\hat x_0$ is what costs. `S60` removes it and doubles the gain; `A05`
resamples less often there and gains a little; `L2` softens it and loses the
rest. A lambda that grows with the denoising progress is the synthesis, and the
FK formalism allows it: any sequence $G_t$ is admissible as long as the product
reaches the target.

**The constraint.** With the MAX potential the intermediate terms are
$\lambda_t (m_t - m_{t-1})$ and they no longer telescope against
$\lambda\, m_{T-1}$ once $\lambda_t$ varies. The terminal correction must cancel
the product accumulated **along the lineage**, not $\lambda \cdot$`gate`: a
second per-particle tensor carried and resampled exactly like `gate`, and at
the last step $\log G = \lambda_T\, r(x_0) - \text{acc}$. The test that has to
hold is `test_telescoping_with_active_non_collapsed_resampling` in
`tests/test_fk.py`, run with a non-constant lambda. This is `smc/fk.py` and is
written by hand.

**Schedules to screen**, in progress $p = 1 - t/T$ over the paper's five steps
$t \in \{80, 60, 40, 20, 0\}$, $\lambda_T = 10$ so the target is unchanged:

| tag | $\lambda_t$ | at the five steps | predicted ESS at t = 80 |
|---|---|---|---|
| `T1` | $10\,p$ | 2, 4, 6, 8, 10 | about 2.9, from `L2` |
| `T2` | $10\,p^2$ | 0.4, 1.6, 3.6, 6.4, 10 | close to 4 |

`S60` is the degenerate schedule (0, 10, 10, 10, 10). Same screening design as
above: the first 20 prompts, seed 2024, one JSON per tag, paired against the
900-record file. About 21 minutes each.

**What counts as a result.** `T1` or `T2` above `S60` on the paired ImageReward
difference says the smooth ramp beats the hard cut; below it says the first step
is worth dropping outright. Either way the winner goes to 100 prompts x 3 seeds
and enters the table as a stated deviation from the paper's constant lambda.
