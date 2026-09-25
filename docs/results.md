# Every run, what it tested, what it cost, what it gave

One block per run that left a file in `results/`, in the order they were run.
Each block has the same five fields, so a run can be looked up without reading
the others. Every number was recomputed from the JSON for this document; the
figure scripts and `scripts/compare_sd_variants.py` print the same aggregates.
Wall times are on the T4 of the Onyxia service, 15 GiB, and come from the
`seconds` field of the records or from the job logs in `/home/onyxia/work/ddpm/`
when the JSON has no timing key. The story that joins the runs is in
`docs/chronology.md`; the reason behind each choice is in `docs/decisions.md`.

## Index

| # | date | run | files | cost | one line |
|---|---|---|---|---|---|
| 1 | 16/09 | free samples, red reward | `free_samples_seed12345.json` | 256 k UNet calls | the toy reward is not degenerate on the free model |
| 2 | 16/09 | best-of-N, red reward | `bestofn.json` | 31 k calls x 3 seeds | reward grows with N, noisily |
| 3 | 17/09 | lambda sweep, three potentials, red | `sweep_lambda.json` | 108 FK runs, 16 k calls each | `sum` overshoots, `difference` collapses at lambda 4, `max` barely moves |
| 4 | 17/09 | class-3 FID on CIFAR | `fid.json` (5 rows) | 3 x 2048 images | base 80.2, fine-tuned 51.4, reproduces the notebook |
| 5 | 17/09 | lambda sweep, classifier reward | `sweep_lambda_classifier.json` | 15 FK runs | log p from -8.3 to -0.5 at lambda 1, ESS from 16 to 2 |
| 6 | 18/09 | CelebA-HQ 256, red and glasses | `sweep_lambda_hub_red.json`, `sweep_lambda_hub_classifier.json` | 25 + 42 min | at 256 px lambda 2 already collapses to one face |
| 7 | 18/09 | FID on CelebA-HQ 256 | `fid.json` (6 rows) | 7 h 35 min | lambda 1 halves the distance to the target |
| 8 | 19/09 | FK against best-of-k in k, CIFAR | `sweep_k_classifier.json`, `sweep_k_max_classifier.json`, `sweep_k_hub_classifier.json` | 102 runs | FK far ahead at k = 4, caught up by k = 16 |
| 9 | 19/09 | ImageReward against HPS on SD | `c2_ir_hps.json` | 5 min | the two judges agree on the trend, not the ranking |
| 10 | 19-20/09 | SD v1.5 baselines `k1`, `bon4` | `sd_baseline.json` | 5.97 h | both rows land on the paper |
| 11 | 20/09 | SD v1.5 FK row `fk4` | `sd_baseline.json` | 5.21 h | +0.062 over best-of-4 where the paper has +0.161 |
| 12 | 20/09 | FK screen on SD, seven variants | `sd_variants/{S60,S40,L2,L5,L20,A05,K8}.json` | 3 h 36 min | the schedule is the lever, lambda is not |
| 13 | 20/09 | time-dependent lambda, terminal placement | `sd_variants/{T1,T2,T1A05,T2A05}.json` | 1 h 24 min | no ramp beats `fk4` on ImageReward |
| 14 | 20/09 | reference `bon4` / `fk4` with the collapse fields | `sd_baseline_div20.json` | 39 min | `fk4` ends on one lineage in 20 of 20 prompts |
| 15 | 20-21/09 | time-dependent lambda, tempering placement | `sd_variants/{T1t,T2t,T1tA05,T2tA05}.json` | 1 h 24 min | tempering buys back the lineages, not the reward |
| 16 | 21/09 | the last two screen variants, `S80` and `D10` | `sd_variants/{S80,D10}.json` | 19 + 33 min | both predictions held; neither end of the schedule buys reward |
| 17 | 21/09 | `S60` confirmatory, 100 prompts x 3 seeds | `sd_s60_full.json` | 7 h 28 min | +0.100 over best-of-4, not settled against `fk4`, root at chance |
| 18 | 21-23/09 | the collapse lab and the reference night: fifteen arms, the released code, the coalescence model | `sd_ref_fields100.json`, `collapse_lab/out/probe.json`, `sd_authors_R0.json` | about 14 h | the paper's configuration is this repo's; nothing reaches +0.161; lineages are predicted from the weights alone |

Not in the table: `smoke_hub.json` (2 runs, the Hub model's first FK run, 17/09),
kept as the check that the wrapper worked before the sweeps.

---

## 1. Free samples of the CIFAR DDPM, red reward (16/09)

**Files.** `results/free_samples_seed12345.json`, `samples/free_seed12345.pt`,
`figures/fig0_free_reward.png`. Script `experiments/run_free_samples.py`.

**Tested.** Whether the toy reward, red channel minus the mean of the two
others, has a usable distribution on the model it will steer. A degenerate
reward would make every later curve meaningless.

**Cost.** 256 samples at T = 1000, 256 000 UNet calls. Raw weights
(`ckpt["model"]`), see run 3 for why that matters.

**Result.** Mean 0.69, standard deviation 1.05, range -3.4 to 4.7. Not
constant, not saturated, a right tail to push into.

**What it changed.** Nothing yet: this is figure 0, the histogram the plan asked
for before any steering.

## 2. Best-of-N, red reward (16/09)

**Files.** `results/bestofn.json`, `samples/bestofn_best.pt`,
`figures/fig1_best_of_n.png`. Script `experiments/run_best_of_n.py`.

**Tested.** The baseline every later comparison is made against: draw N, score,
keep the max. N in {1, 2, 4, 8, 16}, seeds 2024-2026.

**Cost.** N x 1000 UNet calls per run, 31 000 per seed, no wall time recorded.

**Result.** `r_best` 1.59, 1.35, 2.58, 2.16, 3.16 for N = 1 to 16 (means of 3
seeds, standard deviations 0.15 to 1.39). The trend is up and the noise at three
seeds is large enough to make N = 2 and N = 8 dip.

**What it changed.** Fixed the compute unit for the whole project: the number of
UNet calls, counted, `n_model_calls` in every record since.

## 3. Lambda sweep, three potentials, two resamplers, red reward (17/09)

**Files.** `results/sweep_lambda.json` (111 records), `samples/sweep_lambda.pt`,
`figures/fig2_sweep_lambda.png`, `figures/fig3_samples_by_reward.png`.
Script `scripts/run_sweep_lambda.py`.

**Tested.** `fk_steer` at k = 16, T = 1000, potentials `difference`, `max`,
`sum`, lambda in {0, 0.5, 1, 2, 4, 8}, multinomial and systematic resampling,
adaptive rule ESS < k/2, three seeds, plus best-of-16. Whether the three
potentials share a target and differ in dynamics, as the paper says.

**Cost.** 108 FK runs + 3 best-of-16, 16 000 UNet calls each. No timing key in
this file. Raw weights: this sweep ran on `ckpt["model"]` while the FID anchors
used the EMA weights, a different model (max parameter gap 0.018), found on
17/09 and logged in `LEARNING.md`. Not rerun: the red reward was about to be
dropped.

**Result** (`r_sample`, the particle drawn from the final weights, mean of the
systematic runs):

| lambda | `difference` | `max` | `sum` | `ess_min` (diff / max / sum) | resamplings |
|---|---|---|---|---|---|
| 0 | 0.66 | 0.66 | 0.66 | 16 / 16 / 16 | 0 |
| 1 | 1.31 | 0.99 | 10.90 | 6.9 / 7.0 / 2.8 | 4 / 0.3 / 62 |
| 2 | 5.79 | 1.85 | 11.10 | 3.5 / 3.3 / 2.3 | 25 / 1 / 64 |
| 4 | 9.72 | 1.93 | 11.16 | 1.4 / 1.3 / 1.4 | 74 / 1.7 / 71 |
| 8 | 10.45 | 2.61 | 11.08 | 1.04 / 1.03 / 1.04 | 79 / 10 / 78 |

Best-of-16 gives `r_max` 2.18. At lambda = 0 the three potentials and both
resamplers return the identical run, which is the lambda = 0 neutrality property.
Multinomial and systematic agree within seed noise everywhere.

**What it changed.** `sum` pushes the reward past its own bound from lambda 0.5,
the images leave [-1, 1], and at lambda 8 the samples are flat red squares. The
bound is 10.19, the largest value the score can take on pixels inside [-1, 1];
at lambda 8 the `sum` pixels span [-1.26, 1.30]. The judge B, a calibrated
ResNet-18, still gives those squares a mean p(cat) of 0.54 over the three
systematic seeds, against 0.27 for the free model: a reward that can be gamed
and a metric that follows it. (Recount 21/09: the 0.43 written here first was
the `difference` seed-2024 cell, not `sum`; `sum` reads 0.66, 0.37, 0.60.) This is what sent the
project to a classifier reward (run 5) and to a guide / judge split.

## 4. Class-3 FID on CIFAR-10 (17/09)

**Files.** `results/fid.json`, first five rows. Scripts `scripts/run_fid.py`,
`scripts/run_fid_all.sh`.

**Tested.** Whether the repo's FID pipeline reproduces the notebook's anchors
(base 79.2, fine-tuned on class 3 50.7, EMA weights), so that a fine-tuned DDPM
is available as the reference point "what retraining buys".

**Cost.** 2048 generated images per lot at T = 1000, three lots, Inception
statistics on 5000 class-3 and 10 000 test images. Wall time not recorded.

**Result.** Base 80.2, fine-tuned 51.4 against class 3; base 31.6 against the
full test set. The floor: 8.8 for a 2048 class-3 subset against class 3, 35.5
for the 1000 class-3 test images (the small-sample bias of FID at n = 1000).

**What it changed.** The reference point quoted in the README: fine-tuning the
whole model on the cat class buys 29 FID points. Also settled the decision to
compute FID on one particle per run, and to move to DDIM for the cost.

## 5. Lambda sweep, classifier reward (17/09)

**Files.** `results/sweep_lambda_classifier.json` (18 records),
`samples/sweep_lambda_classifier.pt`,
`figures/fig2_sweep_lambda_classifier.png`, `figures/fig3_samples_by_reward.png`.
Guide A `classifier_small_seed0.pt` (small VGG, 1.15 M parameters, 30 epochs,
89.4 %), judge B `classifier_resnet18_seed1.pt` (11.2 M, 90 epochs, 93.2 %,
temperature 1.82).

**Tested.** `r(x) = log p_A(cat | clamp(x))`, `difference` potential, k = 16,
lambda in {0, 0.5, 1, 2, 4}, systematic, three seeds, EMA weights. With
`difference` the product telescopes to `p(x) p_A(cat | x)^lambda`, so lambda = 1
is the Bayes posterior under A.

**Cost.** 15 FK runs + 3 best-of-16 at 16 000 calls. No timing key.

**Result.**

| lambda | `r_sample` | `r_max` | `ess_min` | resamplings |
|---|---|---|---|---|
| 0 | -8.29 | -0.02 | 16 | 0 |
| 0.5 | -1.53 | -0.16 | 4.4 | 27 |
| 1 | -0.48 | -0.06 | 2.1 | 118 |
| 2 | -0.64 | -0.004 | 1.2 | 192 |
| 4 | -0.16 | -0.01 | 1.05 | 252 |

Judged by B on the sixteen final particles of seed 2024: 10 of 16 are cats at
lambda = 1 against 4 of 16 for the free model, on images that stay plausible.
Over the three seeds (recount 21/09, argmax): 11 of 48 free, 15 at lambda = 1,
32 at lambda = 2, 37 at lambda = 4, the lambda = 1 column being 10, 0 and 5 per
seed. The mean pairwise pixel distance between the sixteen finals falls from
0.339 free to 0.231 at lambda = 1 and 0.183 at lambda = 4: the count rises by
spending the cloud, which is the collapse of run 11 at another scale.

**What it changed.** The classifier reward became the result and the red reward
the counter-example. The ESS column, 16 to 1.05, is the first appearance of the
collapse that runs the rest of the project.

## 6. CelebA-HQ 256, red and glasses rewards (18/09, 04:50 to 05:57)

**Files.** `results/sweep_lambda_hub_red.json` (21), `results/sweep_lambda_hub_classifier.json`
(18), `samples/sweep_lambda_hub_{red,classifier}.pt` (264 and 226 MB),
`figures/grid_sweep_lambda_hub_{red,classifier}.png`,
`figures/fig3b_hub_samples_by_reward.png`. Log `chain_hub.log`.
Guide `classifier_eyeglasses64_small_seed0.pt` (98.4 %, recall 95.9 %), judge
`classifier_eyeglasses64_resnet18_seed1.pt` (99.0 %, recall 97.3 %), trained at
64 px with class weights since positives are 4.9 %.

**Tested.** The same `smc/` code on `google/ddpm-ema-celebahq-256` behind the
`CifarDDPM` interface (`smc/pretrained.py`), DDIM 50 steps, eta = 1, fp16,
k = 16, `difference`, systematic, three seeds. Red reward lambda in
{0, 0.5, 1, 2, 4, 8}; glasses reward `log p_A(Eyeglasses)` lambda in
{0, 0.5, 1, 2, 4}. The question: does the damage lambda does, invisible at
32 px, show on a 256 px face.

**Cost.** 800 UNet calls per run (16 x 50). Red sweep 25 min for 21 runs,
classifier sweep 42 min for 18 runs, from the chain log.

**Result.** Red: `r_sample` 0.93 free, 3.24 at lambda 1, 6.87 at lambda 8, `ess_min`
16 to 1.003. Glasses: `r_sample` -4.95 free, -2.39 at lambda 1, -0.76 at lambda 4;
`ess_min` 2.2 at lambda 1, 1.19 at lambda 2, 1.002 at lambda 4. On the grid of
seed 2024 judged by B: lambda = 1 puts glasses on 7 of 16 faces that stay clean;
lambda = 2 gives sixteen copies of one pink blurred face, A says glasses, B says
none. Red degrades the same way: plausible at lambda 2, a red wash at 8.
Over the three seeds (recount 21/09, argmax of B): 2 of 48 free, 18 at
lambda = 0.5, 21 at lambda = 1, then 0 at lambda = 2 and 2 at lambda = 4. The
lambda = 1 column is 7, 0 and 14 per seed, so k = 16 on three seeds ranks this
reward and does not settle it; the zero at lambda = 2 is the collapse, measured
rather than described.

**What it changed.** The collapse became visible to the eye: from lambda 2 every
particle descends from one ancestor chosen on Tweedie estimates at large t where
A is fooled. "Lambda must stay small on the 256 px model" entered
`docs/decisions.md`.

## 7. FID on CelebA-HQ 256, two references (18/09, 06:32 to 14:07)

**Files.** `results/fid.json`, last six rows. Log `chain_fid_hub.log`, chain
`chain_fid_hub.sh`.

**Tested.** Whether the eye's verdict on sixteen images holds on two thousand.
Three lots of 2048 images: free model, FK glasses lambda 1, FK glasses lambda 2
(k = 16, `difference`, DDIM 50, eta 1). Two references: the 1468 faces with
glasses (distance to the target) and the 30 000 faces (what the guidance costs).

**Cost.** 7 h 35 min of wall clock: references 13 min, free lot 2 h 33, each FK
lot 2 h 24 (128 runs of 16 particles at 66-71 s per run). The lots are the
sixteen final particles of 128 runs, duplication included, not one particle per
run as decided for CIFAR: one per run would have been 2048 runs, two and a half
days.

**Result.**

| lot | vs glasses | vs all faces |
|---|---|---|
| free | 123.8 | 43.6 |
| FK lambda 1 | 65.1 | 52.6 |
| FK lambda 2 | 73.7 | 67.3 |

**What it changed.** At lambda 1 the distance to the target is halved for nine
points of global FID, without retraining, comparable to what fine-tuning bought
on CIFAR (run 4). At lambda 2 both columns get worse. Since the lots include
duplicates these are pessimistic bounds, and the final weights are kept so the
one-per-run figure stays computable. This closed the CelebA block.

## 8. FK against best-of-k in k, matched budget (19/09)

**Files.** `results/sweep_k_classifier.json` (48), `results/sweep_k_max_classifier.json`
(36), `results/sweep_k_hub_classifier.json` (18), the `.pt` next to each,
`figures/fig5_sweep_k.png`.

**Tested.** Whether FK's edge over best-of-k survives a bigger k. Classifier
reward, three seeds, budget matched on k x T UNet calls, counted. CIFAR
`difference` k in {2, 4, 8, 16} lambda in {0.5, 1, 2}; CIFAR `max` k in
{4, 8, 16} same lambdas; CelebA-HQ 256 `difference` k in {4, 16} lambda in
{1, 2}. Each file holds the best-of-k rows it is compared against. The ESS trace
per step was logged for the first time here.

**Cost.** CIFAR runs are k x 1000 calls; no timing key. The 256 px runs are DDIM
50 on the large UNet; their particles are the 136 MB `.pt`.

**Result**, on `r_max`, the quantity both methods can be compared on: at k = 4 FK
is ahead by a wide margin (`difference` lambda 1 at -0.068 against -2.37 for
best-of-4); by k = 16 best-of-k has caught up (-0.020 against -0.004 for `max`
lambda 1, inside one standard deviation). On `r_sample` the best FK
configuration is above best-of-k at every k, by 3.55 nats at k = 2, 5.64 at
k = 4, 1.64 at k = 8 and 8.08 at k = 16: that compares a reweighted draw to a
uniform one, so it is a statement about the weights, not about search.

Three seeds on an unbounded log-probability is not much: one bad draw moves a
mean by several nats, and the non-monotone grey line of figure 5 is that. At
k = 2 no resampling can fire (ESS < k/2 with ESS at least 1): those points are
importance weighting with no selection, and the three lambdas give one identical
number equal to best-of-2's `r_max`. The 256 px file is two points wide: within
error bars at k = 4, FK 2.6 nats above on `r_sample` at k = 16.

**What it changed.** The shape "FK pays where the budget is small" that the SD
block would find again. And the question of what happens at k > 4 on SD (run 12,
`K8`).

## 9. ImageReward against HPS v2.1 on SD v1.5 (19/09)

**Files.** `results/c2_ir_hps.json` (20 records).

**Tested.** Before optimising ImageReward, whether the judge that will not be
optimised (HPS) agrees with it. 20 of the 100 benchmark prompts drawn at seed
1234, one image each (SD v1.5 fp16, DDIM 100 steps, eta 1, CFG 7.5, 512 px),
scored matched and with prompts shifted by one (crossed).

**Cost.** 301 s of generation.

**Result.** Pearson 0.58, Spearman 0.59 between the two on matched pairs.
Matched beats crossed 20/20 on both judges: ImageReward 0.33 against -2.11, HPS
0.257 against 0.137. ImageReward spans 0.17 to 1.87 on one prompt's images, HPS
0.286 to 0.343.

**What it changed.** Two things written before the FK row ran: the two judges
agree on the trend and not on the ranking, so a gain on one need not reach the
other; and at lambda = 10 a 1.7-point ImageReward gap is a weight ratio of
e^17, so the ESS collapse was already in the scales.

## 10. SD v1.5, the two baseline rows (19/09 night to 20/09)

**Files.** `results/sd_baseline.json`, the `k1` and `bon4` records (600 of the
900). `figures/fig4_sd_ir_hps.png`. Protocol `docs/protocol_sd.md`.
Script `scripts/run_sd_baseline.py`, separate venv (`scripts/setup_sd_env.sh`).

**Tested.** The first two columns of the paper's SD row: one sample, and
best-of-4 by ImageReward, on the 100 prompts of the ImageReward benchmark,
seeds 2024-2026, HPS read at the ImageReward argmax.

**Cost**, from the `seconds` field, sampling only:

| | UNet rows | s per run | total |
|---|---|---|---|
| `k1` | 200 | 15.5 | 1.29 h |
| `bon4` | 800 | 56.1 | 4.68 h |

5.97 h for 600 runs. `bon4` costs 3.6x `k1` and not 4x: the four particles
cross the UNet in one batched forward. `k1` came out at 15.5 s against the
13.7-14.4 s measured on an idle GPU before the pilot; the job's start time was
not recorded, so the gap is unexplained rather than attributed.

**Result**, averaged over seeds then prompts, error bar the standard error of
the 100-prompt mean:

| | ImageReward | HPS v2.1 | paper (IR / HPS) |
|---|---|---|---|
| `k1` | +0.2368 +/- 0.0816 | 0.2453 +/- 0.0032 | 0.187 / 0.245 |
| `bon4` | +0.7577 +/- 0.0692 | 0.2576 +/- 0.0032 | 0.737 / 0.265 |

The four gaps to the paper in error bars: 0.61, 0.30, 0.09 and -2.31. Only the
last is outside its bar: `bon4` scores 0.0074 below the paper on HPS while
matching it on ImageReward. Spread across prompts 0.82 (`k1`) and 0.69
(`bon4`); seed-to-seed movement of the mean 0.013 and 0.038. The benchmark has
only 100 prompts, so three seeds of all of them was the available choice.

**What it changed.** The night produced no `fk4` row: that sampler is the only
one that imports `smc`, the SD venv had no editable install of the project,
and it would have died on its first prompt. Fixed in `setup_sd_env.sh`, logged
in `LEARNING.md`, and the FK row became its own run.

## 11. The FK row on SD (20/09, 07:18 to 12:36)

**Files.** `results/sd_baseline.json`, the 300 `fk4` records. Same figure and
protocol as run 10.

**Tested.** The paper's setting: lambda = 10, MAX potential, k = 4, fixed
schedule `[0, 20, 40, 60, 80]` (threshold 1.0), guide decoding with
`sd-vae-ft-mse`. Three seeds, prompt-major.

**Cost.** 62.5 s of sampling per run against 56.1 for `bon4`, 5.21 h for the
300 runs, 5 h 18 min wall. Estimate written before launch: 5.1 h and ~5.6 h.
Budget matched exactly: 800 UNet rows for `bon4` and 800 for `fk4`, one value
across all 600 runs of the two, measured by the forward hook.

**Result.**

| | ImageReward | HPS v2.1 | paper (IR / HPS) |
|---|---|---|---|
| `fk4` | +0.8196 +/- 0.0690 | 0.2592 +/- 0.0030 | 0.898 / 0.263 |

Paired on the same prompts and seeds:

| | ImageReward | prompts won | HPS | prompts won |
|---|---|---|---|---|
| `fk4` - `bon4` | +0.0619 +/- 0.0259 (2.4 se) | 69/100 | +0.0015 +/- 0.0011 (1.4 se) | 50/100 |
| `fk4` - `k1` | +0.5828 +/- 0.0433 (13.5 se) | 97/100 | +0.0139 +/- 0.0016 (8.7 se) | 82/100 |
| `bon4` - `k1` | +0.5209 +/- 0.0375 (13.9 se) | 98/100 | +0.0124 +/- 0.0014 (8.7 se) | 84/100 |

FK beats best-of-4 at equal budget on the reward that guides, by 2.4 standard
errors and on 69 prompts of 100. It does not beat it on the judge, which is
also what the paper's HPS column says (0.263 against 0.265).

The size of the gain is where the reproduction falls short: the paper has FK
0.161 above best-of-4, this run has 0.062, about 40 % of it. Both baselines land
on the paper, so the discrepancy is in the FK row alone.

Diagnostics over the 300 runs: 3.75 resamplings out of 5 scheduled steps,
median ESS 2.18 out of 4, minimum 1.00, 273 of 300 runs below an ESS of 1.5 at
least once. At the first scheduled step (t = 80, twenty of the hundred steps
run) the median ESS is 1.18; at the terminal step 2.78. Per-prompt gain over
`bon4`: median +0.08, quartiles -0.02 and +0.18, minimum -1.30. On `001369-0084`
best-of-4 draws a +0.92 particle; FK on the same four x_T collapses to ESS 1.002
at the first step and ends with four particles between -0.71 and -1.07, on all
three seeds. The decoder is not the story: guide and judge decoders put the same
particle first in 225 of 300 runs, median score gap 0.034 on the same particle.
Final particles are not clones: no run ends with two bit-identical
ImageReward values, and 9 of 300 have two that agree to three decimals
(21/09, recount at full precision).

**What it changed.** The diagnosis "the particles collapse before the image
exists" and the four variants it costed, which became run 12. The reading of
these numbers as a whole, and what they ruled out, is in `docs/chronology.md`
under 20/09.

## 12. The FK screen on SD, seven variants (20/09, 13:21 to 16:57)

**Files.** `results/sd_variants/{S60,S40,L2,L5,L20,A05,K8}.json`, one per
variant, 20 records each (40 for `K8`, which holds `bon8` too).
`scripts/run_sd_variants.sh`, log `sd_variants.log`. Read with
`python scripts/compare_sd_variants.py`.

**Tested.** Seven single-knob changes to `fk4`, on the first 20 prompts at seed
2024, so every run shares its x_T with the `k1`, `bon4`, `fk4` records of
run 11 and the comparison is paired. A separate file per variant is required:
the resume key is `(prompt_id, sampler, seed)` and a variant in the main file
would be skipped as done. On these 20 prompts the reference `fk4` - `bon4` is
+0.093 +/- 0.064 (13/20), against +0.062 on the 100: the subset leans toward FK,
so columns are read against each other and not against run 11.

**Cost.** 3 h 36 min of wall clock, 3.44 h of sampling: about 21 min per
variant at 59-62 s per run, 84 min for `K8` at 133 s per run.

**Result.**

| variant | what changes | IR vs best-of-N | won | IR vs `fk4` | won | ESS, first step | resamplings | s/run |
|---|---|---|---|---|---|---|---|---|
| `S60` | schedule `[0, 20, 40, 60]` | **+0.187 +/- 0.049** | 16/20 | +0.094 +/- 0.080 | 14/20 | 1.03 | 2.85 | 60 |
| `K8` | k = 8, against best-of-8 | +0.146 +/- 0.091 | 13/20 | +0.182 +/- 0.064 | 15/20 | 1.33 | 3.85 | 133 |
| `S40` | schedule `[0, 20, 40]` | +0.133 +/- 0.043 | 16/20 | +0.040 +/- 0.059 | 10/20 | 1.05 | 2.00 | 59 |
| `A05` | resample when ESS < k/2 | +0.124 +/- 0.062 | 13/20 | +0.031 +/- 0.035 | 8/20 | 1.21 | 2.25 | 62 |
| `fk4` | reference, the paper's setting | +0.093 +/- 0.064 | 13/20 | | | 1.18 | 3.75 | 62 |
| `L5` | lambda = 5 | +0.091 +/- 0.076 | 12/20 | -0.002 +/- 0.037 | 6/20 | 1.71 | 3.70 | 62 |
| `L2` | lambda = 2 | +0.051 +/- 0.081 | 12/20 | -0.042 +/- 0.062 | 8/20 | 2.88 | 3.90 | 62 |
| `L20` | lambda = 20 | +0.019 +/- 0.090 | 14/20 | -0.074 +/- 0.073 | 6/20 | 1.02 | 3.60 | 62 |

HPS moves on none of them: every paired difference against `fk4` is inside
+/- 0.009, most inside +/- 0.002.

**The schedule is the lever, lambda is not.** Dropping the t = 80 step doubles
the gain over best-of-4 on these prompts; dropping t = 60 as well gives part of
it back. The three lambdas all lose to lambda = 10 on the paired difference, in
both directions: a lower lambda relaxes the collapse (ESS 2.88 at the first step
for `L2`) and loses the selection with it; a higher one changes nothing about
the collapse and adds noise.

**The collapse follows the first evaluation, not the clock.** The ESS column is
read at whichever step is scheduled first: 1.18 at t = 80, 1.03 at t = 60
(`S60`), 1.05 at t = 40 (`S40`). At the first evaluation the weights are
exp(lambda r) over particles that were uniform a step earlier, and with
lambda = 10 one of the four takes almost everything whatever the step. What
`S60` buys is not fewer collapses but a collapse on a sharper x0-hat, so the
single surviving ancestor is a better one.

**Adaptive resampling helps a little, for a different reason.** `A05` collapses
at t = 80 like the reference (1.21) but resamples 2.25 times instead of 3.75 and
comes out +0.031 ahead, 0.9 standard errors. The protocol's prediction that it
would not help was too strong; the mechanism is letting the weights carry
across a step rather than avoiding the collapse.

**k = 8 keeps the edge.** `fk8` beats best-of-8 by +0.146 at twice the budget,
about what `fk4` does against best-of-4 on the same prompts; on CIFAR (run 8)
the edge had closed by k = 16, on SD at k = 8 it has not. At 133 s per run it
is the one variant whose cost the table would have to carry.

None of this is settled at 20 prompts: `S60` over `fk4` is 1.2 standard errors,
`K8` over `fk8` about the same. The screen ranks.

**What it changed.** Three times over, the ranking says the first evaluation at
full lambda on a blurred image is what costs. That is the case for a lambda
that grows with the denoising, which is a change to the potential, hence to
`smc/fk.py`: runs 13 to 15.

## 13. Time-dependent lambda, terminal placement (20/09, 20:47 to 22:11)

**Files.** `results/sd_variants/{T1,T2,T1A05,T2A05}.json`, 20 records each.
`scripts/run_sd_lambda_t.sh`, log `sd_lambda_t.log`. Design in
`docs/protocol_sd.md`, "A time-dependent lambda" and "Where the ramp's deficit
is paid".

**Tested.** `fk_steer` with `lam_schedule`, a lambda per scheduled step with
lambda_T = 10 so the target exp(10 r(x0)) is unchanged. `T1` linear, lambda
2, 4, 6, 8, 10 at the paper's five steps; `T2` quadratic, 0.4, 1.6, 3.6, 6.4, 10.
Under the terminal placement the deficit a rising lambda leaves unpaid is
collected in one piece at the last step, whose weight nothing reads
(`resample_last` is off and the script picks by argmax ImageReward), so the
images of `T1` are exactly those of a run with per-step lambda_i and no
correction: operationally a weaker steering. `T1A05` and `T2A05` cross the ramp
with the 0.5 threshold, because the smoke run had shown that at threshold 1.0
four systematic draws on k = 4 end on one root whatever the ramp does (ESS 2.07
at t = 80 on prompt 0 and still one lineage).

**Cost.** 1 h 24 min of wall clock for the four, 1.32 h of sampling, 59-60 s per
run, slightly under `fk4`'s 62.

**Result.** With run 15, below: the eight ramps are read together.

## 14. Reference `bon4` and `fk4` with the collapse fields (20/09, 22:11 to 22:50)

**Files.** `results/sd_baseline_div20.json`, 40 records. Queue
`/home/onyxia/work/ddpm/queue_ref_div.sh`, log `sd_ref_div.log`.

**Tested.** `ir_max` cannot tell four copies of one image from four images, so
two record fields were added and the reference regenerated on the 20 prompts at
seed 2024: `n_lineages`, distinct x_T among the k finals from a backward walk
over the ancestor indices; `div_pix`, mean pairwise RMSE of the k finals at
64 x 64 in [0, 1], a pixel proxy since neither LPIPS nor CLIP is in the venv.
Same seeds, so same x_T: the regenerated `fk4` reproduces the original `ir_max`
exactly (max absolute difference 0.0 on the 20 prompts).

**Cost.** 39 min for 40 runs.

**Result.** `fk4` ends on **1.00 lineage out of 4 on all twenty prompts**,
`div_pix` 0.0910 against best-of-4's 0.3269. The four images of an `fk4` run are
one image, which is what figure 6 had been showing. The diversity gap
`bon4` - `fk4` is 0.236 of `div_pix`; the "gap closed" column below is a
variant's paired `div_pix` gain over `fk4` divided by that.

## 15. Time-dependent lambda, tempering placement (20/09 22:50 to 21/09 00:15)

**Files.** `results/sd_variants/{T1t,T2t,T1tA05,T2tA05}.json`, 20 records each.
`scripts/run_sd_tempering.sh` (runs `tests/test_fk.py` first and refuses to
start if red), queue `queue_tempering.sh`, log `sd_tempering.log`.
`figures/fig6_sd_collapse.png`.

**Tested.** The same two ramps under `lam_placement="tempering"`: the gate
carries the tempered previous max, `log G_i = lambda_i m_i - lambda_{i-1} m_{i-1}`,
the textbook sequence G_t = pi_t / pi_{t-1} with pi_t proportional to
p(x) exp(lambda_t m_t) (Chopin and Papaspiliopoulos, ch. 17). Written out it is
the terminal placement plus a catch-up term at every scheduled step, so the
deficit is paid where resampling can still act and the terminal weight stays
small. `T1` against `T1t` on the same x_T is the placement effect alone.
Verified on the dummy model before launch, 24 tests green: lineage sum equal to
lambda_T r(x0) under both placements for the three potentials, constant
`lam_schedule` under tempering identical to the constant path, lambda = 0
neutral.

**Cost.** 1 h 24 min of wall clock, 1.32 h of sampling, 59-60 s per run. The
whole evening queue (runs 13 to 15) held the T4 from 20:47 to 00:15, 3 h 28 min.

**Result**, the eight ramps paired against the regenerated `fk4` of run 14:

| tag | placement | threshold | IR - fk4 | won | div_pix - fk4 | lineages | gap to bon4 closed | ESS at t = 80 | resamplings |
|---|---|---|---|---|---|---|---|---|---|
| `T1` | terminal | 1.0 | +0.034 +/- 0.053 | 11/20 | +0.007 +/- 0.010 | 1.10 | 3 % | 2.88 | 3.85 |
| `T1t` | tempering | 1.0 | +0.012 +/- 0.048 | 11/20 | +0.036 +/- 0.014 | 1.35 | 15 % | 2.88 | 4.00 |
| `T1A05` | terminal | 0.5 | +0.033 +/- 0.058 | 12/20 | +0.008 +/- 0.012 | 1.25 | 3 % | 2.88 | 1.95 |
| `T1tA05` | tempering | 0.5 | +0.002 +/- 0.049 | 9/20 | +0.045 +/- 0.016 | 1.55 | 19 % | 2.88 | 1.55 |
| `T2` | terminal | 1.0 | -0.100 +/- 0.092 | 10/20 | +0.048 +/- 0.018 | 1.55 | 20 % | 3.92 | 4.00 |
| `T2t` | tempering | 1.0 | +0.030 +/- 0.054 | 13/20 | +0.050 +/- 0.019 | 1.50 | 21 % | 3.92 | 4.00 |
| `T2A05` | terminal | 0.5 | -0.023 +/- 0.070 | 9/20 | +0.081 +/- 0.017 | 1.95 | 34 % | 3.92 | 1.05 |
| `T2tA05` | tempering | 0.5 | -0.034 +/- 0.081 | 9/20 | +0.090 +/- 0.018 | 2.05 | 38 % | 3.92 | 1.00 |

Against best-of-4 on the same prompts, every ramp except `T2` stays ahead on
ImageReward (+0.06 to +0.13); `T2` is at -0.007. HPS against `fk4` is inside
+/- 0.008 on all eight.

**The ESS prediction held.** The protocol predicted about 2.9 at t = 80 for the
linear ramp (from `L2`) and close to 4 for the quadratic one: measured 2.88 and
3.92, identical under both placements as the equations say. Over the five
steps the median ESS of `T2t` reads 3.92, 2.76, 2.99, 2.84, 2.27 where `fk4`
reads 1.21, 1.39, 2.34, 2.96, 2.59: the ramp keeps the cloud alive through the
early steps and the two meet at the end.

**Two findings, and they are not the same claim.**

On `ir_max`, nothing beats `fk4`: all eight differences sit inside one standard
error of zero. A time-dependent lambda is not the lever for the reward, which is
what run 12 had already said of a constant one. `S60` at +0.094 stays the only
candidate for the table.

On the collapse the improvement is real and measured. `T2tA05` doubles the
lineages and closes 38 % of the diversity gap to best-of-4, at 5 standard errors
from zero. The tempering placement is what buys it: at equal ramp and threshold
it wins on `div_pix` in all four pairs (+0.007 to +0.036, +0.008 to +0.045,
+0.048 to +0.050, +0.081 to +0.090). On `ir_max` it pulls the variant back
toward `fk4`, which costs where the terminal placement was ahead and rescues
`T2` (-0.100 to +0.030).

**The threshold does most of the lineage work.** With the quadratic ramp and
threshold 0.5 the cloud is resampled exactly once per run (`T2A05` 1.05,
`T2tA05` 1.00) and never at t = 80 (ESS below 2 in 0 of 20 runs there, against
14 of 20 for `A05` at constant lambda). The ramp keeps the first ESS above the
threshold; the threshold turns that into a skipped resampling; a skipped
resampling is a lineage kept.

**The prediction written before the run was half wrong.** Tempering kept
`T2A05`'s diversity and added to it, 34 % to 38 %, as expected. It did not lift
`ir_max` back above zero the way it did for `T2`: -0.023 to -0.034, a move
smaller than its own error bar. The rescue does not reproduce once the 0.5
threshold is in place, which says the two mechanisms, the catch-up term and the
skipped resamplings, overlap rather than add.

Figure: `figures/fig6_sd_collapse.png`, rows `bon4` / `fk4` / `T2A05` /
`T2tA05`, on a win (`007086-0024`, div +0.26, 4 lineages out of 4), a median
(`007045-0013`, +0.08) and a loss (`007086-0087`, -0.04, still one lineage).

**What it changed.** The block ends on a fork that has to be named before the
next night on the GPU: the claim is either `ir_max`, where `S60` is ahead and no
ramp is, or the collapse, where `T2tA05` is. They do not point at the same
variant. Whichever goes to 100 prompts x 3 seeds (5.3 h, paired standard error
about 0.026) enters the table as a stated deviation from the paper's constant
lambda. If diversity becomes the claim, `div_pix` needs a perceptual metric
behind it, LPIPS or CLIP distance, and neither is in the sd venv today.

## 16. The last two screen variants, `S80` and `D10` (21/09, 07:40 to 09:17)

**Files.** `results/sd_variants/S80.json` and `results/sd_variants/D10.json`,
20 records each. `scripts/run_sd_night1.sh`, logs `sd_night1.log` then
`sd_chain.log`. Both carry the three fields added on 21/09: `root_slots`,
`n_lineages` and `div_clip`.

**Tested.** The two variants of the screen that had been costed and never
launched. `S80` schedules one selection at t = 80 and then four free
continuations, which asks whether the gain at k = 4 is bought by a single early
pick. `D10` schedules ten steps, t = 0 to 90, which asks whether a denser running
max helps. Both at lambda 10, k = 4, MAX, threshold 1.0, the same 20 prompts at
seed 2024 as `sd_baseline.json`, so both are paired on shared x_T.

**Cost.** `S80` 19 min of sampling at 57.6 s per run, `D10` 33 min at 98.6 s, the
ten decodes showing up as +58 % on the clock. `S80` was cut in half by the service
suspension at 08:00 UTC and finished after the chain restarted at 08:40, resuming
from the records already written: the first time the tmp-plus-rename and the
resume key were exercised by a real interruption.

**Result**, paired against the `fk4` of run 11 on the same prompts:

| tag | ir_max - `fk4` | won | ir_max - `bon4` | ESS at its first step | resamplings | lineages | div_pix |
|---|---|---|---|---|---|---|---|
| `S80` | -0.0274 +/- 0.0607 | 6/20 | +0.0653 +/- 0.0756 | 1.48 at t = 80 | 1.00 | **1.85** | 0.2330 |
| `D10` | +0.0279 +/- 0.0877 | 11/20 | +0.1206 +/- 0.0727 | 2.10 at t = 90 | 8.15 | **1.00** | 0.0749 |

HPS against `fk4` is inside +/- 0.007 for both.

**Both pre-registered predictions held.** The protocol predicted `S80` below
`fk4` and `D10` inside one standard error of it; measured -0.027 and +0.028.

**The two bracket the trade.** `S80` resamples once and keeps 1.85 lineages out
of 4, the most diverse FK arm measured so far, `div_pix` +0.142 +/- 0.014 above
`fk4` on 19 of 20 prompts, and it is the only arm that loses to `fk4` on win rate
as well as on the mean, 6 of 20. `D10` resamples 8.15 times, ends on one lineage
in 20 of 20 runs with the lowest `div_pix` of any arm, and buys nothing for its
58 % of extra clock. Selection and diversity trade against each other along the
schedule, and neither end of the range buys reward.

**What it changed.** It closed the screen: every variant that was costed has now
been run. It also made the case that the schedule is not where the reward is,
since the densest and the sparsest schedules both land inside a standard error of
`fk4`.

## 17. `S60` at 100 prompts x 3 seeds, the confirmatory (21/09, 09:17 to 16:47)

**Files.** `results/sd_s60_full.json`, 300 records. `scripts/run_sd_night1.sh`,
log `sd_chain.log`. Kept outside `results/sd_variants/` on purpose:
`compare_sd_variants.py` globs that directory and would read 300 records against
a 20-record reference.

**Tested.** The variant the screen ranked first, at the scale that settles it.
`S60` drops the t = 80 step and keeps t = 20, 40, 60 and the terminal one, at the
paper's lambda 10, k = 4, MAX, threshold 1.0. Same 100 prompts and three seeds as
run 11, so the comparison against `fk4` and `bon4` is paired throughout, seeds
averaged within a prompt before anything else.

**Cost.** 7 h 28 min of sampling, 87.4 s per run against the screen's 60.4, and
7 h 30 min of wall clock. The pre-registration had budgeted 5.0 h.

**Result.**

| against | ir_max | prompts won | HPS |
|---|---|---|---|
| `k1` | +0.6215 +/- 0.0476 (13.1 se) | 96/100 | |
| `bon4` | **+0.1005 +/- 0.0311 (3.2 se)** | 64/100 | +0.0032 +/- 0.0015 |
| `fk4` | **+0.0386 +/- 0.0351 (1.1 se)** | 57/100 | +0.0017 +/- 0.0014 |

Absolute `ir_max` 0.8583 against the paper's 0.898, so the row is still 0.040
short. ESS medians at t = 60, 40, 20, 0 read 1.13, 2.07, 2.73, 2.87, and 2.84
resamplings of the 3 available.

**The pre-registration says not settled.** The bands written before the run were:
above +0.052, two standard errors, the deviation is kept and the FK row becomes
`S60`; between 0 and +0.052, reported as not settled; at or below 0, the table
carries the paper's setting alone. It lands at +0.0386. The screen's
+0.094 +/- 0.080 at 20 prompts shrank to +0.039 +/- 0.035 at five times the
prompts, which is what a 1.2-standard-error result does when it is given power.

**It does not repair the collapse either.** 258 of 300 runs end on one lineage,
mean 1.140 of 4, `div_pix` 0.1040 against best-of-4's 0.3269 and `div_clip`
0.0140. Dropping the earliest evaluation moves the reward a little and the
ancestry almost not at all.

**The surviving root is no better than random.** This is the first file with
`root_slots` at scale, so the wrong-root rate can be read for the first time:
the particle best-of-4 would have chosen is absent from FK's surviving roots in
**72.7 % of 300 runs, against a chance level of 72 %** given how few roots
survive. FK is not keeping the root that wins, and it is not avoiding it either.
The caveat the protocol attached to this number still stands: it rests on slot i
of `bon4` and slot i of `S60` starting from the same x_T, which the lambda = 0
bit-for-bit check makes plausible and does not prove.

**What it changed.** The reward claim for the write-up is now `S60` against
best-of-4 at +0.1005 +/- 0.0311, which is the strongest paired result in the
project, rather than `S60` against `fk4`, which the confirmatory declines to
settle. And the wrong-root rate turns the collapse from a diversity cost into a
statement about the selection itself.

## 18. The collapse lab and the reference night (21/09 to 23/09)

**Files.** `results/sd_ref_fields100.json` (the FK row regenerated on 21/09 with the
collapse fields, 100 prompts, seed 2024), `collapse_lab/out/probe.json` (fifteen arms
of `collapse_lab/probe.py`, seed 2024, paired to the reference by prompt; `ctl` is the
reference itself, identical to `sd_ref_fields100.json` on all 40 of its prompts),
`results/sd_authors_R0.json` (the released code, driven from its own clone by
`collapse_lab/ref/run_authors.py`), `results/paper_table1_sd15.json` (the paper's
row, with its source). Readouts: `collapse_lab/r_solutions.py`, `n_coalescence.py`,
`ref/parse_authors.py`; the full account with the predictions written before each run
is `collapse_lab/ASSESSMENT.md` and the last sections of `docs/protocol_sd.md`.

**Cost.** About 14 h of T4 over two nights and a day, at 58 to 62 s per run.

**Result, the collapse at the paper's setting.** On the 100 prompts of
`sd_ref_fields100.json`, `n_lineages` is 1 in **96 runs of 100** and 2 in the
other 4; mean ESS at the five scheduled steps 1.24, 1.57, 1.97, 2.94, 2.61 out
of 4. Reweighting best-of-4's four free draws by exp(10 ir) gives a median ESS of
1.23: at lambda = 10 the target itself puts about 90 % of its mass on one of four
candidates.

**Result, the corrections** (40 prompts unless stated, paired to `ctl` by prompt,
`ctl` = 1.06 roots, 95 % single-root, `ir_max` 0.958 on these prompts, `ir` mean
of the four 0.824):

| arm | what | single-root | roots | div_pix | `ir_max` - `ctl` | `ir` mean - `ctl` |
|---|---|---|---|---|---|---|
| `late` | schedule without t = 80 (`sd_s60_full`, n = 300) | 86 % | 1.14 | 0.104 | +0.039 +/- 0.033 | |
| `adapt` | lambda bisected to ESS = k/2 | 60 % | 1.40 | 0.153 | -0.099 +/- 0.044 | -0.171 |
| `floor` | the released code's floor at 0 on the max statistic | 68 % | 1.73 | 0.150 | -0.091 +/- 0.055 | -0.153 |
| `lam2` | lambda = 2 | 35 % | 1.82 | 0.205 | -0.052 +/- 0.103 | -0.138 |
| `fadapt` | floor + bisected lambda | 30 % | 2.12 | 0.200 | -0.106 +/- 0.058 | -0.214 |
| `floor2` | floor + lambda = 2 | **5 %** | **3.00** | **0.286** | -0.083 +/- 0.093 | -0.334 |
| `thr05` | floor, resample only if ESS < k/2 | 62 % | 2.00 | 0.172 | -0.058 +/- 0.100 | -0.181 |
| `rise` | floor, bisected lambda, cap 100 (n = 20) | 35 % | 1.95 | 0.170 | +0.007 +/- 0.109 | -0.135 |
| `lam0` | free (n = 17) | 0 % | 4.00 | 0.338 | -0.070 +/- 0.082 | -0.557 |

`floor2`'s row is the n = 40 readout of 22/09; a rerun on 23/09 overwrote six of its
records with values from another session (see reproducibility below), so the file
now holds 34 for that arm. Pooled over the seven corrections, `ir_max` - `ctl` is
-0.069 [-0.176, +0.041] (bootstrap over prompts); over the five measured before the
incident, -0.106 [-0.193, -0.019]. Against `bon4` on the same prompts, `ctl` is at
+0.112 +/- 0.091 and every correction between -0.01 and +0.06.

**Result, the coalescence model** (`n_coalescence.py`, output `collapse_lab/out/n_coalescence.txt`).
Replaying the recorded per-step weights through the systematic comb integrated over
its offset (or through the multinomial draw at every scheduled step) predicts the mean
number of final roots of all fifteen arms within 0.1, and within 0.05 for eleven of
them (`ctl` 1.08 against 1.06, `floor` 1.73 against 1.73, `fadapt` 2.16 against 2.12,
`floor2` 2.94 against 2.94, `R1` 1.25 against 1.19; the largest miss is `stat0`, 1.84
against 1.75, then `late` and `thr05` at 0.07) and the single-root fraction within
three points except `R1` (48 % predicted, 81 % observed) and `thr05` (54 against 62); per-run correlation 0.87 to 0.99 on
the arms with spread. Under flat weights the comb is the identity and the multinomial
is not: four flat multinomial passes leave 1.58 roots of 4. The pre-registered
prediction for `thr05` (1.84 roots, 61 % single-root, 1.12 resamplings) came out
2.00, 62 %, 0.97.

**Result, the reference.** The paper's stated SD configuration (max potential,
schedule [0, 20, 40, 60, 80], lambda 10, k 4, DDIM eta 1, 100 steps, guidance 7.5,
ImageReward on the Tweedie estimate) is this repository's. The released script's
defaults (`diff`, 5-30-5) are a different configuration that the paper's appendix
scores lower. The released code differs in four unstated implementation choices:
the max statistic floored at 0, multinomial resampling at every scheduled step
including flat weights, adaptive resampling of the terminal population, and the
pipeline VAE for the guide's decode. On the 100 prompts:

| row | code | seeding | n | `ir_max` | `ir_max` - `bon4` (paired by prompt) |
|---|---|---|---|---|---|
| paper, Table 1 | | | | 0.898 | +0.161 |
| `ctl` | `smc/` | generator 2024 | 100 | 0.826 | +0.056 +/- 0.052 |
| `R1` | `smc/` with the four choices | generator 2024 | 100 | 0.756 | -0.011 +/- 0.041 |
| `R0g24` | released code | generator 2024 (same x_T) | 40 | 0.720 | -0.126 +/- 0.070 |
| `R0` | released code | its own, seed 42 | 100 | 0.554 | -0.216 +/- 0.061 |
| `R0g` | released code | generator, seed 42 | 40 | 0.807 | -0.039 +/- 0.073 |
| `R0` at seed 2024 | released code | its own path, seed 2024 (same x_T as ours) | 40 | 0.949 | **+0.103 +/- 0.056** |
| the four runs pooled | released code | | 220 | 0.702 | **-0.110 +/- 0.036** |

On the same 40 prompts the four runs of their filter give -0.35, -0.04, -0.13 and
+0.10 against `bon4` (standard errors 0.06 to 0.09); two of them share x_T and the
DDIM noise and differ only by the stream of the multinomial draw (-0.13 and +0.10).
The standard deviation of `ir_max` across their four runs of one prompt has median
0.25. The two free baselines under the two seedings are bit-identical (0.824 both,
against 0.846 for `bon4` on those prompts). This repository's `ctl` on the same 40
prompts in two sessions: +0.112 +/- 0.091 and +0.075 +/- 0.050. On the same four particles at the first scheduled step, the guide's
reward for one particle reads -1.78 with the pipeline VAE, -0.56 with sd-vae-ft-mse
and -0.21 in the released code one index later (`out/diag_authors_0_s2024.json`,
arms `vae` and `ctl_b1`): the decoder alone moves a first-step reward by 1.2.

The released code without FK, given the same generator, returns `bon4`'s four
rewards to the fourth decimal on 20 slots; its ImageReward scorer equals the
official one to the third decimal. On our x_T its FK loop landed at -0.13 against
`bon4` in the run `R0g24` and at +0.10 in the run at seed 2024 under its own seeding
(`R1`, the same four choices in `smc/`, at -0.01); the difference is not in the
guide's first step, where both see four negative rewards and draw
flat, and a reading of the code (23/09) found no re-seeding and no bias between
the two seeding paths. Without a generator the multinomial advances the global
stream and changes the later DDIM noise; with one it does not; the two are two
random runs of the same filter. Each of the four choices alone costs 0.05 to 0.06 in `smc/` (`stat0`
-0.050, `multi` -0.061, `vae` +0.009, `idx` -0.051, 40 prompts each); the guide's VAE
changes which root survives in 70 % of prompts while leaving the reward unchanged.
*Superseded on 23/09 evening:* those four costs and the 70 % pair fast-card arms with the A2
reference. Against the reference of the same machine they read -0.013, -0.024, +0.045 and -0.014,
and the guide's VAE keeps the reference's roots in 78 % of prompts (`collapse_lab/r_solutions.py`,
`p_two_rewards.py`).

**Result, three scales** (`scripts/fig_three_scales.py`, `figures/fig8_three_scales.png`).
Minimum ESS over the run divided by k, and pixel diversity of the finals relative to
the free model, against lambda: CIFAR 32 px (k = 16) 1.00 to 0.066 and 1.00 to 0.52
from lambda 0 to 4; CelebA 256 px (k = 16) 1.00 to 0.063 and 1.00 to 0.11; SD 512 px
(k = 4) 1.00 to 0.30 and 1.00 to 0.26 from lambda 0 to 10.

**Reproducibility.** The diffusers pipeline with a seeded generator returns the
`bon4` rewards of 20/09 to the third decimal on 23/09. The `smc.models` path returns
identical numbers within one session and different roots and rewards across sessions
(the reference of 21/09 and the probe of the 22/09 morning agree to 0.0000; the
22-23/09 night disagrees with both, `ir_max` off by up to 1.6 on a prompt), with
identical code and weights. A test on 23/09 afternoon (`collapse_lab/nuit4.sh`,
`u_determinism.py`, not pre-registered) ran `ctl` on two prompts in three separate
processes, one of them under `cudnn.benchmark = False` and
`use_deterministic_algorithms(True)`, and once more with the 22/09-morning version of
`probe.py`: all four return the same rewards to the fourth decimal, equal to session C
and to the 22/09 night, across a pod restart. Ruled out: the process, the flags, the
cache path, the pod, the `sd` venv (no install since 20/09), `smc/` (unchanged since
21/09 17:27) and the probe's rewrite. What separates the two groups is the machine's
execution path: the sessions that return the 21/09 numbers ran at 87 to 90 s per run,
every session since the 22/09 evening at 55 to 60 s, same code; the first machine is
gone and the cause cannot be pushed further. x_T is shared at the bit
between the pipeline and the wrapper, and the trajectories correlate at 0.984 or
more at the last step; the ImageReward of a slot still differs by up to 0.34
between the two paths (`out/latents_0.json`, `out/latents_1.json`). Slot pairing
between files from different sessions is therefore invalid and the wrong-root rate
of run 17 is not measurable that way.

*Settled on 23/09 evening (session D, `collapse_lab/y_sessionD.py`).* The pod came back at
18:02 on an NVIDIA A2, and `probe.py` records the device since 18:47. Session D on the A2
returns the 21/09 reference and the 22/09-morning arms of `probe.json` slot by slot, `lam0`
included, and differs from session C on `ctl`, `lam0` and `floor2` alike. The "machine
execution path" above is the GPU model: the 87-90 s group is the A2, the 55-60 s group a
faster card whose model was not recorded. The free path does not reproduce across the two
either; the correlation of 1.00 between session C's `lam0` and `bon4` is a comparison inside
the fast group. Every slot or root pairing across groups is invalid, and eight arms of the
probe (`R1`, `idx`, `late`, `multi`, `rise`, `stat0`, `thr05`, `vae`, fast card) had been
paired with the A2 reference; `collapse_lab/r_solutions.py` and `p_two_rewards.py` now pair
each arm with the `ctl` of its own group (`collapse_lab/commun.py`, `references()`).

**Session C (23/09, `collapse_lab/out/probe_C.json`).** `ctl`, `lam0` and `floor2` at
100 prompts in one process. At the first scheduled step (t = 80) the four guide rewards
of a `ctl` run span 0.905 on average (median 0.777; `r_at_schedule[0]`), which at
lambda = 10 is 9 nats between the largest and the smallest weight; the median ESS there
is 1.18, then 1.63, 2.21, 3.29 and 2.71 at t = 60, 40, 20, 0, with 3.69 resamplings of 5
per run. `ctl` ends on one root in 93 runs and two in 7; `floor2` on 1, 2, 3, 4 roots in
4, 24, 37 and 35 runs (`div_pix` 0.300 against 0.109). `lam0` returns `bon4`'s four rewards slot by slot at
correlation 1.00 (`ir_max` +0.009 +/- 0.006): the free path of the wrapper reproduces
across sessions, and only the resampling path does not (`ctl` against the 21/09 file:
correlation 0.70, same root in 30 % of prompts, `ir_max` -0.026 +/- 0.058). Paired by
x_T inside the session: `floor2 - ctl` on `ir_max` **-0.012 [-0.082, +0.060]**, on the
mean of the four -0.250 +/- 0.045; `floor2` 3.03 roots, 4 % single-root, `div_pix`
0.300; `ctl` 1.07 roots, 93 %; `ctl - bon4` +0.030 +/- 0.037 in this session. On this
pairing, with `lam0[j]` the free continuation of root j: Kendall tau between the
first-step ranking and the free outcome **+0.137 +/- 0.050**, top-1 35 % against 25 %;
best free root A 0.779, mean root M 0.233, the root `ctl` keeps read free B 0.466, what
`ctl` makes of it C 0.799, so **B - M +0.233 +/- 0.047**, **A - B +0.313 +/- 0.042**,
**C - B +0.333 +/- 0.038**, C - A +0.021 +/- 0.038.

**Session D (23/09 18:47 to 24/09 03:05, `collapse_lab/out/session_D/probe_D.json`).** `ctl`,
`lam0` and `floor2` at 100 prompts in one process on the NVIDIA A2, 88.3 s per run, the device and
`session_id` in every record, the four finals and the guide's decoded Tweedie estimates at the five
scheduled steps saved for every prompt and arm (`images/`), and HPS v2.1 of the 1200 finals
(`hps_finals.json`, `collapse_lab/w_hps_finals.py`). `ctl` equals the 21/09 reference slot by slot on
100 of 100 prompts and session A's arms on all their prompts; nothing equals session C. On the A2:
`ctl` 0.826, 1.04 roots, one root in 96 runs; `floor2` 2.96 roots, `floor2 - ctl` -0.043 +/- 0.037
on `ir_max` and -0.261 +/- 0.043 on the mean of the four; FK against the best of the four free runs
of the same process +0.043 +/- 0.038 (session C: +0.021 +/- 0.038); `ctl` D - C on the same x_T
+0.026 +/- 0.058, same roots in 30 % of prompts; HPS `ctl` - best-of-4 -0.003 +/- 0.002. The t = 80
thumbnails of the three arms are equal (0/255 on 100 prompts), the ESS rebuilt from `logG` matches
the record to 1.3e-4. The selection rule, read on session D, keeps 46 complete prompts of 46 and
changes 5 of its 8 gains against session C (`data/visual_selection.json`). Readout:
`collapse_lab/y_sessionD.py`; predictions and outcome: `docs/protocol_sd.md`.

**What it changed.** The gap to the paper is bounded, not closed, and it is not in
the implementation. The collapse has a mechanism with a predictive model behind it.
The one correction that keeps lineages does so by changing the target and returns
best-of-4's reward; paired by x_T its price on the best reward is 0.01, on the mean
of the four 0.25. Constats 3 and 5 bis hold on the same-session pairing in a weaker
form: a little information at the first step, a kept root a third of the way from the
median to the best, 0.31 of root lost and 0.33 returned by the steering.
