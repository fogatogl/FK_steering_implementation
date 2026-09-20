# Measurements, and what they cost

One section per run that produced a file in `results/`. Every number below was
recomputed from the JSON for this document; the figure scripts print the same
aggregates, so the two can be compared. Wall times are the T4 of the Onyxia
service, 15 GiB.

## SD v1.5, the two baseline rows (20/09)

`results/sd_baseline.json` held 600 records when the night ended: the 100 prompts
of the ImageReward benchmark x {`k1`, `bon4`} x seeds {2024, 2025, 2026},
complete, no gap. The protocol is `docs/protocol_sd.md`; the figure is
`figures/fig4_sd_ir_hps.png`.

Averaged over the three seeds of a prompt first, then over the 100 prompts,
which is what the error bar is taken on:

| | ImageReward | HPS v2.1 | paper (IR / HPS) |
|---|---|---|---|
| `k1` | +0.2368 ± 0.0816 | 0.2453 ± 0.0032 | 0.187 / 0.245 |
| `bon4` | +0.7577 ± 0.0692 | 0.2576 ± 0.0032 | 0.737 / 0.265 |

Measured in error bars, the four gaps to the paper are 0.61 (`k1` IR), 0.30
(`bon4` IR), 0.09 (`k1` HPS) and -2.31 (`bon4` HPS). Only the last is outside
its bar, and it is the one to name in the write-up: `bon4` scores 0.0074 below
the paper on the judge nobody optimises, while matching it on the reward that
guides.

The error bars are standard errors of the 100-prompt mean. The underlying spread
across prompts is much wider: 0.82 (`k1`) and 0.69 (`bon4`) of standard
deviation. Against that, the seed-to-seed movement of the mean is small, 0.013
for `k1` and 0.038 for `bon4`. The benchmark has only 100 prompts, so spending
the night on three seeds of all of them was the available choice; had more
prompts existed, they would have bought more than the second and third seed did.

Cost, from the `seconds` field, which times the sampling only and not the
ImageReward and HPS scoring that follows it:

| | UNet rows | s per run | total |
|---|---|---|---|
| `k1` | 200 | 15.5 | 1.29 h |
| `bon4` | 800 | 56.1 | 4.68 h |

5.97 h of sampling for the 600 runs. `bon4` costs 3.6x `k1` and not 4x: the four
particles cross the UNet in one batched forward, so the ratio is exact on rows
and favourable on time. `k1` came out at 15.5 s against the 13.7-14.4 s measured
on an idle GPU before the pilot. The job's start time was not recorded, so the
gap is unexplained rather than attributed.

The night produced no `fk4` row. That sampler is the only one that imports
`smc`, the SD venv had no editable install of the project, and it would have died
on `import smc` at the first prompt. The two baselines never import it and ran
the whole night without noticing. Fixed in `scripts/setup_sd_env.sh`; the FK row
is the run described at the end of this file.

## FK vs best-of-k in k, CIFAR (19/09)

Three files, all at a budget matched on network calls (k*T, counted, not
assumed), classifier reward `log p(class 3)`, three seeds. Each file holds the
FK grid plus the `best_of_n` rows it is compared against:

- `results/sweep_k_classifier.json`, 48 runs = 36 FK (`difference`, k in
  {2, 4, 8, 16}, lambda in {0.5, 1, 2}) + 12 best-of-k;
- `results/sweep_k_max_classifier.json`, 36 runs = 27 FK (`max`, k in {4, 8, 16},
  same lambdas) + 9 best-of-k;
- `results/sweep_k_hub_classifier.json`, 18 runs = 12 FK (the CelebA-HQ 256 px
  model, `difference`, k in {4, 16}, lambda in {1, 2}) + 6 best-of-k.

`figures/fig5_sweep_k.png` carries the first two. What it shows, on `r_max`, the
quantity both methods can be compared on: at k = 4 FK is ahead by a wide margin
(`difference` lambda = 1 at -0.068 against -2.37 for best-of-4), and by k = 16
best-of-k has caught up (-0.020 against -0.004 for `max` lambda = 1, inside one
standard deviation).

On `r_sample`, the particle actually drawn from the final weights, the best FK
configuration is above best-of-k at every k, by 3.55 nats at k = 2, 5.64 at
k = 4, 1.64 at k = 8 and 8.08 at k = 16. That compares a reweighted draw to a
uniform one, so it is a statement about the weights, not about search.

Three seeds on an unbounded log-probability is not much: a single bad draw moves
a mean by several nats, and the non-monotone grey line is that, not a trend. At
k = 2 no resampling can ever fire, because the rule is ESS < k/2 and ESS is at
least 1: those points are importance weighting with no selection, which is why
the three lambdas give one identical number there, equal to best-of-2's `r_max`.

The 256 px file is two points wide and gets no figure. At k = 4 FK and best-of-4
are within their error bars of each other; at k = 16 FK's `r_sample` is 2.6 nats
above.

The CIFAR runs are T = 1000 DDPM, so k*1000 network calls each. Wall time was not
recorded in these files, and the JSON has no timing key, so there is no cost
figure to quote. The 256 px runs are DDIM 50 steps on a much larger UNet; their
particles are the 136 MB `samples/sweep_k_hub_classifier.pt`.

## What the FK row on SD will cost (20/09, estimated before launching)

Three prompts at seed 2024 measure `fk4` at **61.0 s of sampling per run**
(57.7, 61.4, 63.8), against 56.1 s for `bon4`. The budget matches exactly: 800
UNet rows, 100 calls of batch 8, the same counter `bon4` reads. The extra 9 % is
the twenty VAE decodes and twenty ImageReward scorings the guide adds, five
scheduled steps on four particles, which decision 5 reports without folding into
the matching.

Both numbers time the sampling only, so add the ImageReward and HPS scoring of
the final four images, the same tax the baselines paid.

| | runs | sampling | with scoring, roughly |
|---|---|---|---|
| 100 prompts, seed 2024 | 100 | 1.7 h | ~1.9 h |
| 100 prompts, three seeds | 300 | 5.1 h | ~5.6 h |

Launched 20/09 at 07:18 on the three seeds, prompt-major as the protocol
requires, so an instance that dies mid-run leaves complete prompts. The
alternative was one seed for a third of the time. On `k1` the seed-to-seed
movement is a sixth of the prompt-to-prompt standard error, which would have made
one seed nearly free; on `bon4` it is half of it, and `bon4` is the configuration
FK is budget-matched to. The extra 3.7 h buys comparability with the two rows
already measured, which average three seeds per prompt before aggregating.

The pilot also says the guide and the judge agree on which particle wins: over
the three prompts the argmax of `ir_guide` (sd-vae-ft-mse) and of `ir` (the
pipeline VAE) is the same particle each time, though the scores differ by up to
0.27. On this sample of three the choice of decoder moves the number, not the
ranking.
