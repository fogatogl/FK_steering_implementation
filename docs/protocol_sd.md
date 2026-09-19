# SD v1.5 reduced protocol (19/09)

Written before the run, as the plan requires, and amended only to record what was
actually launched. The target is the SD v1.5 row of table 1 of the FK Steering paper:
ImageReward 0.187 / 0.737 / 0.898 and HPS 0.245 / 0.265 / 0.263 for k = 1,
best-of-4 and FK at lambda = 10, k = 4.

## Prompts

40 of the 100 prompts of the ImageReward benchmark
(`data/imagereward-benchmark-prompts.json`, taken verbatim from the THUDM repository,
which is where the list lives; it is not on the Hub). The subset is
`data/prompts_subset_40.json`, drawn at seed 2026 by `scripts/sample_prompts.py`, which
sorts the pool by `id` first so the upstream file order does not decide the draw. Both
files are versioned: the subset is reproducible from the file, not only from the
procedure.

40 and not 100 is a compute choice, not a methodological one, and the table has to say
so: at 66.5 s per prompt for two configurations, 100 prompts would be close to two
hours per seed and three configurations would be three.

## Sampler

SD v1.5 fp16 (`stable-diffusion-v1-5/stable-diffusion-v1-5`, the `runwayml` repo is
404 since 2024), DDIM with `eta = 1`, 100 steps, classifier-free guidance 7.5,
512 px, no attention slicing (peak VRAM measured at 2.16 GiB out of 15).

## Configurations

| name | what | status |
|---|---|---|
| `k1` | one sample per prompt | measured |
| `bon4` | four samples, the best one by ImageReward | measured |
| FK | lambda = 10, k = 4, MAX potential, fixed schedule `[0, 20, 40, 60, 80]` | waiting on the wrapper |

The first two do not go through `fk_steer` at all: best-of-N here is generate four,
score, take the max, which is why they could be measured before the SD wrapper existed.
The FK row plugs into the same harness as a third sampler.

## Seeds

**One seed.** The base seed is 2024 and the generator is seeded at
`2024 * 1000 + prompt index`, so each prompt gets its own `x_T` and the 40 draws are
not correlated through a shared initial noise. Three seeds was the ambition and one
seed is what the calendar allows; the table says so rather than implying an average
over repetitions. The base seed and the effective seed both go into every record, so
adding seeds later extends the file instead of replacing it.

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

40 prompts rather than 100, one seed rather than three, and the VAE is
`stabilityai/sd-vae-ft-mse` where the paper does not say which decoder it used. The
current run does not keep the images: the image grid of C7 regenerates the one or two
prompts it needs, which is cheap because the seeds are recorded.
