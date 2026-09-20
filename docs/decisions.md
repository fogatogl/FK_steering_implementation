# Design decisions

## `right=False` in `resample_systematic`

In `torch.searchsorted(cumw, points, right=False)`:
- The function uses half-open intervals $[c_{i-1}, c_i)$.
- For a sampling point $p$ with $p = c_i$, the returned index is $i$ (not $i+1$).
- This matches the convention where the first particle is selected when $p \in [0, w_0]$, and keeps a point landing exactly on an upper edge from spilling into the next slot or out of bounds when $p = 1.0$.

## `t_idx` in `sample_trajectory`

In `sample_trajectory` the snapshot labelled `t` is the state *before* step `t`.

## Tests against the `particles` oracle

- **Determinism and RNG**: `particles.resampling` takes no explicit generator and relies on the NumPy runtime. The oracle tests do not look for bit-for-bit equality of indices but for matching empirical laws.
- **Multinomial vs systematic precision**:
  - For the multinomial, the standard sampling error is $O(k^{-1/2})$, hence a statistical tolerance of `atol=0.02`.
  - For the systematic resampler, the comb guarantees that the number of copies of particle $i$ lies between $\lfloor k w_i \rfloor$ and $\lceil k w_i \rceil$. The error on each frequency is therefore strictly bounded by $1/k$, which allows a strict threshold of `2.0 / k`.

## One target for the three potentials

The three potentials do not differ in their terminal target but in the temporal dynamics of the guidance:

- `difference`: guides particles through local increments of the predicted reward.
- `max`: keeps the guidance on the peak reward seen along the Tweedie path, then compensates the gap with the true $x_0$ at the final step.
- `sum`: accumulates the whole signal along the path before the final correction.

Without the corrective term at $t=0$, the product would be $\exp(\lambda \max_t r_t)$ or $\exp(\lambda \sum_t r_t)$: the final marginal in $x_0$ would then be biased by the intermediate Tweedie estimates $\hat{x}_0(x_t)$, notoriously blurry and inaccurate at large $t$.

**Decision**: keep $\prod_{t=0}^{T-1} G_t = \exp(\lambda\, r(x_0))$ for all three potentials, via the corrective term at $t=0$.

## EMA weights everywhere (17/09)

The notebook FID (base 79.2 / fine-tuned 50.7) was measured on the EMA weights; the lambda sweep and figures 0–2 ran on the raw weights (`ckpt["model"]`), a different model (max parameter gap 0.018). Sweeps and samples switch to EMA, the DDPM standard and the only choice comparable to the FID anchors. `results/sweep_lambda.json` and `results/bestofn.json` remain raw-weight results; new JSON files carry an `"ema"` key.

## Classifier reward: log p, clamped (17/09)

`r(x) = log p_A(y = c | clamp(x))`, with A the small VGG of `smc.classifier`. Log-probability rather than a normalised score: with the `difference` potential the product telescopes to $p(x)\,p_A(y=c \mid x)^\lambda$, so $\lambda = 1$ is the Bayes posterior under A, and the model fine-tuned on class 3 is a direct comparison point. The clamp lives in the reward: `predict_x0` clamps, `state["x"]` at $t=0$ did not, and the classifier must never see values outside its training support.

Guide with A (89.4 %), judge with B (ResNet-18, 93.2 %, temperature 1.82 fitted on the test set). Two architectures, not two seeds: twins share their blind spots.

## $p_B$ does not judge out-of-distribution images

Red reward, $\lambda = 8$: sixteen flat red squares, mean $p_B(\text{cat}) = 0.43$. A classifier returns a number on anything. $p_B$ only means something on plausible images; FID is the complement, and neither suffices alone.

## FID on FK samples: one particle per run

At $\lambda \geq 1$ the sixteen final particles descend from two or three ancestors (`ess_min` ≈ 1, 100–270 resamplings). FID on all $k$ would measure duplication. Use the particle drawn from the final weights, one per run. 16 s per sample at $T = 1000$, hence DDIM.

## Noise-conditioned classifier: deferred

Clean classifier on the Tweedie $\hat x_0$ first, as in the paper. On pure noise A gives $\log p(\text{cat})$ between −3 and −6, not $\log(1/10)$: the early potentials are noise. Revisit once the timesteps at which resampling fires are recorded.

## A pretrained 256 px model from the Hub (18/09)

CIFAR at 32 px hides the damage lambda does to an image. `google/ddpm-ema-celebahq-256` has the CIFAR schedule (linear beta, $T = 1000$), so `smc/` runs unchanged: `smc/pretrained.py` wraps the diffusers UNet behind `CifarDDPM`, fp16 on the T4. DDIM 50 steps, $\eta = 1$: a $T = 1000$ run at 256 px would take minutes per particle set.

## Attribute reward on CelebA-HQ: Eyeglasses

Binary classifier on one attribute, trained at 64 px (the classifier resizes a 256 px sample itself). Positives are 4.9 %: class-weighted cross-entropy, recall printed with the accuracy. A small VGG 98.4 % / recall 95.9 %, B ResNet-18 99.0 % / 97.3 %. Same guide/judge split as on CIFAR.

## On the 256 px model lambda must stay small

Glasses reward, $\lambda = 1$: 7 of 16 particles wear glasses for B, on clean faces. $\lambda = 2$: sixteen copies of one pink blurred face, A says glasses, B says none. `ess_min` is 1 from $\lambda = 2$ on, every particle descends from one ancestor, and that ancestor was chosen on Tweedie estimates at large $t$ where A is fooled. Same failure as the red squares on CIFAR, visible to the eye this time. Red reward degrades the same way: plausible at $\lambda = 2$, a red wash at $\lambda = 8$.

## FID on CelebA-HQ 256: all sixteen particles, two references (18/09)

2048 images per lot, DDIM 50 steps, $\eta = 1$. Two references: the 1468 faces with glasses (distance to the target) and the 30000 faces (what the guidance costs). Free model 123.8 / 43.6, FK glasses $\lambda = 1$ 65.1 / 52.6, $\lambda = 2$ 73.7 / 67.3. Half the distance to the target for nine points of global FID at $\lambda = 1$; at $\lambda = 2$ both get worse, the collapse seen on sixteen images holds on two thousand.

The lots are the sixteen final particles of 128 runs, not one particle per run as decided above: one per run would be 2048 runs, two and a half days at 256 px. The FID therefore includes duplication and reads as a pessimistic bound. `runs/*.pt` keeps the final weights, so the one-per-run figure stays computable on 128 images.

## The six SD decisions (18/09)

The block that reproduces the SD v1.5 row of the paper's table 1 at reduced scale.
These six were settled before any SD code was written; this section is what the
"implementation choices" part of the write-up is built from.

**1. The reward is evaluated at the five resampling steps only.** This is not a
liberty taken with the algorithm: it is the paper's *interval resampling*, where
$G_t = 1$ off schedule and the schedule always contains the terminal step. Five VAE
decodes and five reward evaluations per particle, twenty in total at $k = 4$, not four
hundred. What remains a choice is that the running max is then a max over a five-point
grid, so it underestimates the max over a hundred. That changes which particles get
selected, not the target: $G_0$ closes the product either way. If the pilot shows the
VAE decode is cheap relative to the UNet, one step in ten is the fallback.

**2. Fixed schedule for the reproduction table, adaptive as a measured variant.**
The table is reproduced on the paper's `[0, 20, 40, 60, 80]`. Resampling when
$\mathrm{ESS} < k/2$, the rule used on CIFAR and CelebA, is kept as a variant run at
the end if time allows. The gap between the two is itself a result, which is why the
schedule mode goes into every output record rather than being remembered.

**3. The schedule is converted once, at entry.** It enters as a list of loop indices
and is written verbatim into the output JSON. Converting it inside the loop is how an
off-by-one survives a night of GPU.

**4. The paper's MAX potential, corrective term at $t = 0$ included**, target
$\exp(\lambda\, r(x_0))$. This is what `smc/fk.py` already computes, so there is no
deviation to report here. What it does require is that the terminal step be identifiable
on any schedule: see the next section.

**5. The compute budget is matched on UNet calls.** It is the dominant cost and the
portable one: a reader with another GPU can check the ratio. FK at $k = 4$ and
best-of-4 must land on the same count, counted at runtime and logged, not derived on
paper. The twenty VAE decodes and twenty ImageReward calls FK adds, against four and
four for best-of-N, are reported but not folded into the matching.

**6. `stabilityai/sd-vae-ft-mse` as the decoder.** Same latent space, fine-tuned
decoder, drop-in. Not TAESD: TAESD buys speed that only matters under dense scoring,
and at five evaluations there is nothing to buy. The paper does not say which decoder
it used, so this is a deviation and is flagged as one in the write-up.

## Identifying the terminal step (19/09)

Decision 4 looked like a fork and is not one. The paper defines the MAX and SUM
potentials with the corrective term built in: $G_t = \exp(\lambda \max_{s \geq t}
r_\phi(\hat x_s))$ for $t \geq 1$, and $G_0 = \exp(\lambda\, r(x_0))
(\prod_{t=1}^{T} G_t)^{-1}$, that is a $G_0$ defined to cancel the running product.
The section "One target for the three potentials" above therefore reproduces the paper
rather than extending it, and `smc/fk.py` and `tests/test_fk.py` are right as they
stand. An earlier reading of mine claimed the paper had no such correction; it was
wrong, and the plan's A1 has been corrected accordingly.

What the correction does require is a reliable notion of "terminal step", and that part
is a real defect. `fk_steer` identifies it by `t == 0`. This holds for the CIFAR
schedule by construction (`DDIMScheduler` builds `tau` with `linspace(0, T-1, steps)`,
so the list always ends exactly on 0), but SD v1.5 ships `steps_offset = 1` and its
diffusers timesteps at 100 steps end on 1: `[991, 981, ..., 21, 11, 1]`. Wired as is,
the corrective branch never fires on SD, the target silently becomes
$\exp(\lambda \max_s r(\hat x_s))$, and nothing raises. The numbers would look
plausible and would not be the paper's measurement.

**Decision**: the terminal step stops being inferred from a magic value. `fk_steer`
walks `model.timesteps` by position and the terminal step is the last one, whatever
its value; the resampling schedule (decision 3, converted once into loop indices at
entry) must contain it, and `fk_steer` refuses to run otherwise, because that step
carries the corrective term. A model cannot reintroduce the bug: no flag to forget, no
value to match. The property to hold, on CIFAR and on SD alike: the corrective branch
fires exactly once per run, observed rather than read off the source.

## What the T4 measured before the pilot (19/09)

SD v1.5, fp16, CFG 7.5, DDIM $\eta = 1$, 100 steps, 512 px, batch 1: **13.7-14.4 s per
image**, peak VRAM **2.16 GiB** out of 15. That is 5.7x the paper's 2.4 s, consistent
with an A100 baseline. Attention slicing is therefore off by default: at 2.16 GiB it
buys nothing and costs time.

$\eta = 1$ does inject sampling noise, which had to be verified before anything else:
two samples sharing $x_T$ and differing only in the generator give decorrelated final
latents (relative difference 1.74), while at $\eta = 0$ they are equal bit for bit.
Had $\eta$ been silently zero, resampling would have cloned particles that could never
diverge, and the whole block would have been dead without saying so.

`ImageReward.load` ignores `HF_HOME`: it passes `local_dir` to `hf_hub_download`, which
short-circuits the hub cache and drops 1.7 GB in `~/.cache/ImageReward`, on the
ephemeral overlay. It takes `download_root`, and the weights live in
`/home/onyxia/work/ir_cache`.

ImageReward and HPS are on scales with nothing in common: on thirteen images of one
prompt, ImageReward spans 0.173 to 1.867 while HPS spans 0.2856 to 0.3429. At
$\lambda = 10$ a 1.7 point gap is a weight ratio of $e^{17}$, so the ESS collapse is
not something the pilot has to wait for; it is already in the scales. This only
concerns the reward that guides: HPS judges and never enters a potential.

## The three choices C3 left open (19/09)

**The VAE lives in the reward, not in the model.** `fk_steer` calls
`reward(model.predict_x0(state))` without knowing what domain `predict_x0` returns.
On SD it returns a latent and ImageReward needs pixels, so "decode" belongs to the
reward object (`smc/rewards_sd.py`: decode, then score) and not to the model or the
potential. This is what makes the C3 criterion hold literally: not one line of
`fk.py` changed for SD.

**Cache $\hat\varepsilon$ rather than the scheduler's `pred_original_sample`.**
`DDIMScheduler.step` already returns $\hat x_0$; storing it in the state would have
made `predict_x0` a mere accessor. Keeping the explicit Tweedie computation on the
cached $(\hat\varepsilon, x_t, t)$ preserves the symmetry with `CifarDDPM` and a
`predict_x0` that has content. The cost is a possible stale cache; it is acceptable
because `step` rewrites `eps`, `x_t` and `t` as one block, and `fk_steer` calls
`predict_x0` right after `step`.

**One generator per batch: the noise follows the slot, not the lineage.** A single
`torch.Generator` for the whole batch draws one $(k, 4, 64, 64)$ tensor per step, so
two clones produced by resampling receive two different slices and diverge on the
next step. Had the generator followed the particle, clones would have shared their
noise stream and stayed identical, which empties resampling of its meaning. The
consequence for the image grids: "same seed" means "same $x_T$ for slot $i$", not
"same lineage".

## C3 closed: the wrapper reproduces the pipeline bit for bit (19/09)

`fk_steer(lambda=0, k=1, schedule=[99])` on `StableDiffusion` against
`pipe(..., output_type="latent")`, same prompt, same seed, CFG 7.5, DDIM eta = 1, 100
steps: `torch.equal` is true, max absolute difference 0.0 on latents of scale 3.6. The
diffusers timesteps run `[991, 981, ..., 11, 1]`, so the terminal step is the last
element of the list and never `t == 0`; no resampling fired and every `logG` was zero.
Not one line of `fk.py` was changed for SD.
