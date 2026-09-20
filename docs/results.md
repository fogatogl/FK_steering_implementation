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

## The FK row on SD (20/09)

Launched at 07:18 on the three seeds, prompt-major, finished at 12:36:
`results/sd_baseline.json` now holds 900 records, 100 prompts x {`k1`, `bon4`,
`fk4`} x 3 seeds, no gap. lambda = 10, MAX potential, fixed schedule
`[0, 20, 40, 60, 80]`, k = 4.

| | ImageReward | HPS v2.1 | paper (IR / HPS) |
|---|---|---|---|
| `k1` | +0.2368 ± 0.0816 | 0.2453 ± 0.0032 | 0.187 / 0.245 |
| `bon4` | +0.7577 ± 0.0692 | 0.2576 ± 0.0032 | 0.737 / 0.265 |
| `fk4` | +0.8196 ± 0.0690 | 0.2592 ± 0.0030 | 0.898 / 0.263 |

The budget matched exactly: 800 UNet rows for `bon4` and 800 for `fk4`, measured
by the forward hook, one single value across all 600 runs of the two.

The three samplers are measured on the same prompts under the same seeds, so the
comparison that matters is paired, not a read of two overlapping error bars:

| | ImageReward | prompts won | HPS | prompts won |
|---|---|---|---|---|
| `fk4` - `bon4` | +0.0619 ± 0.0259 (2.4 se) | 69/100 | +0.0015 ± 0.0011 (1.4 se) | 50/100 |
| `fk4` - `k1` | +0.5828 ± 0.0433 (13.5 se) | 97/100 | +0.0139 ± 0.0016 (8.7 se) | 82/100 |
| `bon4` - `k1` | +0.5209 ± 0.0375 (13.9 se) | 98/100 | +0.0124 ± 0.0014 (8.7 se) | 84/100 |

FK beats best-of-4 at equal UNet budget on the reward that guides, by 2.4
standard errors and on 69 prompts out of 100. It does not beat it on the judge:
+0.0015 at 1.4 standard errors, 50 prompts each way, which is what the paper's
own HPS column says too (0.263 for FK against 0.265 for best-of-4, FK slightly
below).

The size of the gain is where the reproduction falls short. The paper has FK
0.161 above best-of-4 on ImageReward; this run has 0.062, about 40 % of it. Both
baselines land on the paper, so the discrepancy is in the FK row alone. The
three candidates, in the order I would test them: the running max is taken over
a five-point grid rather than the full hundred steps (decision 1), the decoder
is `sd-vae-ft-mse` where the paper does not say (decision 6), and the schedule
is fixed at threshold 1.0 rather than adaptive (decision 2).

Diagnostics, over the 300 FK runs: 3.75 resamplings per run out of 5 scheduled
steps, median ESS 2.18 out of k = 4, minimum 1.00, and 273 of 300 runs dip below
an ESS of 1.5 at least once. The collapse the scales predicted is there, and the
method still gains: at k = 4 there is not much room between "all particles
survive" and "one ancestor".

Cost: 62.5 s of sampling per run against 56.1 s for `bon4`, 5.21 h for the 300
runs, 5 h 18 min of wall clock. The estimate written before launching was 5.1 h
of sampling and ~5.6 h wall.

## What the numbers say so far (20/09)

Three blocks have run: CIFAR-10 with a classifier reward, CelebA-HQ 256 px with
an attribute reward, SD v1.5 with ImageReward. Read together:

**The implementation is right.** The product constraint
$\prod_t G_t = \exp(\lambda r(x_0))$ holds on the three potentials
(`tests/test_fk.py`), the SD wrapper reproduces the diffusers pipeline bit for
bit at lambda = 0, and the UNet budget of FK and best-of-4 is the same measured
number, 800 rows, on all 600 runs. The two baselines land on the paper's table
within 0.3 to 0.6 standard errors on ImageReward. Whatever is missing is not a
bug in the weights, the resampler or the potential.

**FK beats best-of-N at equal budget, and by less than the paper.** On SD,
+0.062 ImageReward over best-of-4, 2.4 standard errors on 100 paired prompts,
where the paper has +0.161. On the CIFAR k sweep the same shape: FK is far ahead
at k = 4 and best-of-k has caught up by k = 16. The steering pays where the
budget is small; with enough independent samples, taking the max is enough.

**The gain does not reach the judge.** HPS moves by +0.0015 between best-of-4
and FK, 1.4 standard errors, 50 prompts each way. The paper's own table has FK
slightly *below* best-of-4 on HPS (0.263 against 0.265). Steering optimises the
reward it is given, and ImageReward and HPS agree on the trend but not on the
ranking (Pearson 0.58 on `results/c2_ir_hps.json`). An interviewer will ask
whether +0.06 of ImageReward is worth anything; the honest answer is that the
metric that was not optimised did not move.

**Where the missing 60 % most likely went: the particles collapse before the
image exists.** The FK diagnostics point one way:

- At the first scheduled step, t = 80 in the paper's convention, only twenty of
  the hundred denoising steps have run and the Tweedie estimate $\hat x_0$ is a
  blur. The median ESS there is **1.18 out of 4**. With the fixed schedule
  (threshold 1.0) the resample fires immediately, so from step 20 onwards three
  of the four particles are copies of one ancestor chosen on noise. At the
  terminal step the median ESS is 2.78: the weights are well spread once the
  image is sharp, which is when they no longer have much to select from.
- The per-prompt gain has a heavy left tail: median +0.08, quartiles -0.02 and
  +0.18, minimum **-1.30**. On `001369-0084` ("greek or roman sculpture in
  marble of a philosopher") best-of-4 draws a +0.92 particle at seed 2024; FK,
  on the same four $x_T$, collapses to ESS 1.002 at the first step and ends with
  four particles between -0.71 and -1.07, below `k1`. Same story on the three
  seeds of that prompt. The particle that would have won was killed at step 20.
- The decoder is not the story. `sd-vae-ft-mse` (guide) and the pipeline VAE
  (judge) put the same particle first in 225 of 300 runs, and the median score
  gap on the same particle is 0.034. Decision 6 costs at most the 75 runs where
  the ranking flips on scores that are already within noise.
- Final particles are not clones: only 9 of 300 runs end with two identical
  ImageReward values. The per-slot generator does its job; the damage is done
  by the selection, not by a failure to diverge.

lambda = 10 on a reward whose useful range spans about two units means a 0.2
gap in early ImageReward is a weight ratio of $e^2$. That is fine at the
terminal step and destructive at step 20.

**What this rules out.** The adaptive ESS < k/2 rule alone would not fix it:
1.18 is already below 2, so it fires at the same step. Replacing the decoder
would move a handful of rankings inside the noise. Neither is worth a night on
its own.

**What it leaves open, in the order I would test.** Whether the first scheduled
step is the problem (drop it, or start at t = 60), whether the weight sharpness
is (lambda 2 to 20), whether the five-point running max is (score every ten
steps), and whether the paper's number needs k > 4. All four are one CLI flag
away and are costed in `docs/protocol_sd.md`.

## The FK screen on SD (20/09, 13:21 to 16:57)

Seven variants of `fk4`, the first 20 prompts, seed 2024, one JSON each in
`results/sd_variants/`, 3.44 h of sampling. Every run shares its $x_T$ with the
`k1`, `bon4` and `fk4` records of the same prompt and seed in
`results/sd_baseline.json`, so all differences below are paired. On these 20
prompts the reference `fk4` - `bon4` is +0.093 ± 0.064 (13/20), against +0.062
on the 100: the subset leans towards FK, so the columns are read against each
other and not against the table above. `python scripts/compare_sd_variants.py`
prints this table from the JSON.

| variant | what changes | IR vs best-of-N | won | IR vs `fk4` | won | ESS, first step | resamplings | s/run |
|---|---|---|---|---|---|---|---|---|
| `S60` | schedule `[0, 20, 40, 60]` | **+0.187 ± 0.049** | 16/20 | +0.094 ± 0.080 | 14/20 | 1.03 | 2.85 | 60 |
| `K8` | k = 8, against best-of-8 | +0.146 ± 0.091 | 13/20 | +0.182 ± 0.064 | 15/20 | 1.33 | 3.85 | 133 |
| `S40` | schedule `[0, 20, 40]` | +0.133 ± 0.043 | 16/20 | +0.040 ± 0.059 | 10/20 | 1.05 | 2.00 | 59 |
| `A05` | resample when ESS < k/2 | +0.124 ± 0.062 | 13/20 | +0.031 ± 0.035 | 8/20 | 1.21 | 2.25 | 62 |
| `fk4` | reference, the paper's setting | +0.093 ± 0.064 | 13/20 | | | 1.18 | 3.75 | 62 |
| `L5` | lambda = 5 | +0.091 ± 0.076 | 12/20 | -0.002 ± 0.037 | 6/20 | 1.71 | 3.70 | 62 |
| `L2` | lambda = 2 | +0.051 ± 0.081 | 12/20 | -0.042 ± 0.062 | 8/20 | 2.88 | 3.90 | 62 |
| `L20` | lambda = 20 | +0.019 ± 0.090 | 14/20 | -0.074 ± 0.073 | 6/20 | 1.02 | 3.60 | 62 |

HPS moves on none of them: every paired difference against `fk4` is inside
±0.009, most inside ±0.002. Whatever the variants do, they do it on the reward
that guides.

**The schedule is the lever, lambda is not.** Dropping the t = 80 step doubles
the gain over best-of-4 on these prompts; dropping t = 60 as well gives back
part of it. The three lambdas all lose to lambda = 10 on the paired difference,
in both directions. A lower lambda does relax the collapse (ESS 2.88 at the
first step for lambda = 2) and loses the selection with it; a higher one changes
nothing about the collapse and adds noise.

**The collapse follows the first evaluation, not the clock.** The ESS column is
read at whichever step is scheduled first: 1.18 at t = 80, 1.03 at t = 60 for
`S60`, 1.05 at t = 40 for `S40`. At the first evaluation the weights are
$\exp(\lambda\, r)$ over particles that were uniform a step earlier, and with
lambda = 10 one of the four takes almost everything whatever the step. What
`S60` buys is not fewer collapses but a collapse on a sharper $\hat x_0$, so
the single surviving ancestor is a better one.

**Adaptive resampling helps a little, for a different reason.** `A05` collapses
at t = 80 like the reference (1.21) but resamples 2.25 times instead of 3.75 and
comes out +0.031 ahead, 0.9 standard errors. The prediction in
`docs/protocol_sd.md` that it would not help was too strong; the mechanism is
letting the weights carry across a step rather than avoiding the collapse.

**k = 8 keeps the edge.** `fk8` beats best-of-8 by +0.146 at twice the budget,
about what `fk4` does against best-of-4 on the same prompts. On the CIFAR sweep
the edge had closed by k = 16; on SD at k = 8 it has not. Against `fk4` itself,
`fk8` is +0.182 ahead, 15/20, so more particles help FK as much as they help
the baseline. At 133 s per run it is the one variant whose cost the table would
have to carry.

None of this is settled at 20 prompts: `S60` over `fk4` is 1.2 standard errors,
`K8` over `fk8` about the same. The screen ranks. What the ranking says, three
times over, is that the first evaluation at full lambda on a blurred image is
what costs, which is the case for a lambda that grows with the denoising: the
protocol for it is written, and it waits on `smc/fk.py`.
