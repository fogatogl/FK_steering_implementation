# Four particles, one image: reproducing FK Steering on Stable Diffusion, and what its gain over best-of-N buys

## Context of the project

This is a solo project, from August 2026 to its submission on 7 October; its Sequential Monte Carlo
(SMC) part started on 16 September. It ran on a shared GPU service that switched between two GPU
models (section 9). I reimplemented FK Steering from its equations and ran its Stable Diffusion
experiment; most of the time went into why my numbers differ from the paper's. I wrote the SMC core
with the tests of its mathematics, and the analysis of the collapse; an AI assistant wrote the launch
and figure scripts and the documentation, and drafted this post, which I reread and corrected
(section 10).

## TL;DR

FK Steering generates several images at once and, along the way, copies the promising ones over the
others. On Stable Diffusion v1.5 it beats generating four images and keeping the best (best-of-4) by
+0.062 ± 0.026 ImageReward over 100 prompts at three seeds. The paper reports +0.161, and best-of-4 itself
matches it (0.758 against 0.737).

FK's four images nearly always descend from one starting noise, in 93 to 96 runs of 100, and the first
copying step alone leaves one in about half the runs. The one variant that keeps three noises of four
also changes the target, and the average of its four images falls 0.250 ± 0.045 below FK's.

On the same noises at three seeds, the authors' released code gains -0.003 ± 0.025 over best-of-4, and
its version from before a June 2025 fix +0.074 ± 0.023, level with this repository's FK there. Both
stay under the bar of +0.12 I set before these runs; the earlier one's interval ends at +0.120.

## 1. What inference-time steering does to an image

A text-to-image model starts from random noise and removes it step by step; the same prompt with
another starting noise gives another image. Inference-time steering spends extra computation during
generation to get an image that scores higher on a reward. Here the reward is ImageReward
[@xu2023imagereward], a learned score of human preference that runs from about -2 to +2 on these
prompts (A.1).

Take one prompt and four starting noises: the free sampler turns each into an image, and best-of-4
keeps the one ImageReward scores highest.

![F1. A free sample, best-of-4 and the best of FK's four particles from the same four noises, with ImageReward and HPS v2.1 under each image. Three prompts where FK beats best-of-4 on ImageReward, two picked by a written rule and one by eye, labels paraphrased (A.3). On HPS best-of-4 scores higher on these three, FK on the rule's replaced pick.](../figures/f1_hero_grid.png)

These three prompts are favourable cases (A.3). In the run they
come from (A2, seed 2024), FK beats best-of-4 on 48 of the 100 prompts, by +0.043 ± 0.038 on average
(a screen, A.7).

## 2. How FK Steering works

A diffusion model [@ho2020ddpm] denoises a Gaussian $x_T$ into a draw from $p(x)$. FK Steering
[@singhal2025fk] draws instead from

$$\pi(x) \propto p(x)\, e^{\lambda r(x)},$$

the model tilted toward a reward $r$; $\lambda$ sets how hard it tilts, and nothing is retrained. It
runs $k$ particles, here four images in progress, and at a few scheduled steps weights and resamples
them: a high-weight particle is copied, a low-weight one dropped. This is a Feynman-Kac particle system,
a form of SMC [@delmoral2004fk; @chopin2020smc].

The reward is defined on clean images, and at step $t$ there is no clean image. So it is read on the
Tweedie estimate $\hat x_0(x_t)$, the *guide*: the model's own guess of where the trajectory ends. At
$t = 80$ of 100 that guess is a blur.

To target $\pi$, the weights along a particle's chain of ancestors must multiply to $e^{\lambda
r(x_0)}$. The paper gives three weightings, called potentials, that do. I use MAX, which weights a
particle by the best reward its ancestors have reached, $G_t = \exp(\lambda\, \max_{s \ge t} r(\hat
x_0(x_s)))$. At each scheduled step the paper multiplies a particle's weight by $G_t$. I multiply it by
$G_t$ divided by its value at the scheduled step before, so that a lineage's weights multiply to $G_t$.

In both forms the last step closes the product on $e^{\lambda r(x_0)}$, so both target $\pi$. They
differ on the way: the paper's tilts step $t$ by the product of every $G$ so far, mine by $G_t$ alone. On 20
prompts the paper's form reads +0.015 ± 0.009 above mine (a screen, A.7).

A final image's **lineage** is its chain of ancestors, found by walking the ancestor indices backwards
[@jacob2015path], and its root is the initial noise $x_T$ that chain starts from. Four particles start from four roots, and each resampling
can drop some (F2).

![F2. A schematic: four particles run from noise (left) to image (right) through five scheduled steps (dotted); at the first, one is dropped and another copied, so two finals carry the same colour, the root. Dot size is the normalised weight.](../figures/f2_algorithm.png)

The paper's appendix already reports that steering costs diversity. On SD v1.5, with another potential
and on GenEval prompts, the CLIP diversity of the four finals falls from 0.312 for the base model to 0.104 at
$\lambda = 10$ (A.7). Particle systems are also known to lose their ancestral lines [@jacob2015path;
@chopin2020smc]. What I add is measured on FK itself. Its four finals share one root in more than nine runs of ten, and
the first resampling alone decides it in about half. Neither version of the released code reaches a bar
set before its runs.

## 3. The reproduction

I reproduce the ImageReward and HPS columns of the paper's Table 1 on Stable Diffusion v1.5
[@rombach2022ldm], with the paper's configuration (A.4). That is the 100 prompts of the ImageReward
benchmark, four particles, $\lambda = 10$, MAX, scheduled steps at $t = 80, 60, 40, 20$ and 0, and DDIM
[@song2021ddim] with 100 steps. Two choices depart from the paper's text. I use MAX as increments (section 2). I resample with the
comb, a systematic resampler that copies each particle once under equal weights, where the paper's Algorithm 1 draws at random (section 7). Each prompt runs at three
seeds.

An FK run and the best-of-4 run of the same prompt and seed start from the same four noises, so they pair prompt by prompt. HPS v2.1 [@wu2023hpsv2], a second preference score that
nothing here steers toward, is read as a check.

Conventions: ± is a standard error over prompts, [a, b] a 95 % interval (A.1), and n = 100 prompts
unless stated. A *test* compares a result with a band I wrote down before the run; any other comparison
is a *screen*. A test is *held* or *missed* by where its point estimate falls, as in A.5 and A.7, and also *open* when its interval crosses the band's edge. A.5 lists the
76 predictions, 30 held and 30 missed, and A.7 the numbers behind the main text.

These runs ran on an NVIDIA T4, as their run times and session C show (A.7); the project also used an
NVIDIA A2. Runs are compared only
within one machine and torch build, since the same noise gives a different image on each (section 9).

| | ImageReward, best of $k$ | HPS v2.1 at the image ImageReward picks (pre-registered) | HPS v2.1, best of $k$ | ImageReward, mean of $k$ | paper (IR / HPS) |
|---|---|---|---|---|---|
| one sample | 0.237 ± 0.082 | 0.245 ± 0.003 | 0.245 ± 0.003 | 0.237 ± 0.082 | 0.187 / 0.245 |
| best-of-4 | 0.758 ± 0.069 | 0.258 ± 0.003 | 0.266 ± 0.003 | 0.207 ± 0.078 | 0.737 / 0.265 |
| FK, $k = 4$ | 0.820 ± 0.069 | 0.259 ± 0.003 | 0.265 ± 0.003 | 0.687 ± 0.072 | 0.898 / 0.263 |

I set the paper's values as targets without a tolerance, so every comparison with the paper is a
screen. The paper does not state its seeds, so I measure the distance to it in units of seed noise. One unit is the gap that seed noise alone would typically put between my three-seed mean and the paper's value (A.7).
One sample and best-of-4 land within one unit of the paper (F3).

HPS at the image ImageReward picks, as pre-registered, lands 0.007 and 0.004 under the paper, where a
mean moves by 0.001 from seed to seed. Read as the released evaluation reads it, on the best of the
four, it lands within 0.002 (A.4).

Prompt by prompt, FK gains +0.062 ± 0.026 and wins on 69
prompts of 100, against the paper's +0.161. That is 2.2 to 3.1 units under, depending on how many seeds Table 1 averages,
or 1.7 to 2.6 if its two rows did not share their noises (A.7). The comparison is at matched
compute: both send 800 sample rows per run through the UNet, and FK's extra decoding makes its median
run 62.5 s against 54.7 s, the time of best-of-4.57 (A.7).

![F3. ImageReward of the best image (left) and best HPS v2.1 of the k images (right), 100 prompts times three seeds, the paper's value as a dark tick. The error bars are the seed-to-seed spread of a one-seed mean.](../figures/f3_reproduction.png)

## 4. Four particles, one image

Traced back in 100-prompt reruns (A.2), FK's four finals at the paper's setting share a single root in
93 runs of 100 on the T4 and 96 on the A2. With the released code's resampler alone, a random draw at every
scheduled step (`multi`, T4), all 40 runs of 40 do (screens, A.7). Accordingly, the mean
pixel distance of the four images, `div_pix` (A.1), is on the T4 a third of the free sampler's, 0.109 ± 0.006 against
0.355 ± 0.005 (F4 and F5: an A2 run).

![F4. Ancestry of F5's particles (A2), the free sampler (left) and FK (right): the top row is the noises $x_T$, each row below the particles after that step's resampling, edges go to the parent (width: its weight, colour: the root), crosses are particles the next resampling drops. FK keeps one root from the first step on.](../figures/f4_ancestry.png)

The usual metric cannot see this: best-of-$k$ scores four near-copies of one good image the same as
four different images whose best is that image. The mean of the four, the table's mean-of-k column, is high too: FK's four finals average 0.481 ±
0.030 more than best-of-4's four free draws (three seeds, a screen).

![F5. The four finals of one prompt under the free sampler and FK, from the same four noises, each framed in the colour of its root; prompt chosen by eye (A.3).](../figures/f5_root_grid.png)

## 5. Why the particles collapse

The effective sample size, $\mathrm{ESS} = 1 / \sum_i w_i^2$ over
the $k$ normalised weights [@kong1994ess], counts how many particles effectively carry the weight: 4
when the weights are equal, 1 when one particle carries them all. At FK's first scheduled step its
median on the T4 is 1.18 of 4: one particle takes almost all the weight.

That first step is extreme because the four rewards at $t = 80$ spread by 0.905 on average, and
$\lambda = 10$ exponentiates the differences. The first resampling alone leaves a single root in 46 runs of 100.

Each later pass of the comb drops the particles whose weight falls well
below 1/k, and at $\lambda = 10$ a few passes leave every survivor with one ancestor. Removing the first step does not prevent the collapse: FK still ends on one root in 84 runs of
100 (A2, seed 2024, a screen). Nor does adaptive tempering [@chopin2020smc], which lowers $\lambda$ on the fly to keep the ESS at 2 or
above. It ends on 1.40 ± 0.08 roots over 40 runs, under the 1.5 to 2 I
predicted in a note committed with its first run (missed, open; A.5).

Replaying each run's recorded weights through the resampler [@jacob2015path], without a GPU, returns
the observed root counts within 0.09 on fourteen arms and four reruns (F6), a consistency check. Its
one forecast, from the floor's weights under `thr05`'s rule (resample only when ESS $< k/2$), gave
1.84 roots, where 40 runs returned 2.00 ± 0.22 (a screen, A.5).

![F6. Mean final roots replayed from the recorded weights and the resampler (x) against observed (y), one point per variant, a square per 100-prompt rerun in its variant's colour, standard errors over prompts.](../figures/f6_coalescence.png)

## 6. Four draws against the target, and the variants that keep roots

### Four draws against the target

An exact sampler of $\pi$ would draw new roots and keep four, but four draws from the model sit far from $\pi$. Reweighting best-of-4's four free draws by $e^{10\, r}$, as the target does, leaves a median
ESS of 1.23 of 4 (a screen): at $\lambda = 10$ one of four model draws takes most of the
target's weight.

### The floor with λ = 2, and its price

Of the variants meant to keep several roots (A.2, F7), the floor comes from the released code. It floors the reward at 0, the final one included, so a step where all
four rewards are negative leaves the weights equal, and the target is flat wherever the reward is
negative. Alone, it idles the first step in 90 % of runs. The collapse then resumes: 1.73 ± 0.19 roots over 40 runs, where I predicted 2 to 3 (a test: missed,
open).

With a milder tilt as well, $\lambda = 2$, the floor keeps 3.03 ± 0.09 roots of 4 in the 100-prompt T4
rerun, inside the 2.8 to 3.1 I predicted (a test: held, open). Only 4 % of its runs end on one root, under
the 5 to 10 % predicted (missed, open). Both changes alter the target, and in 17 prompts it never
resamples at all.

![F7. Each variant of A.2 minus FK on the same machine, on the best image's ImageReward (left) and the mean of the four (right), sorted by roots kept; normal 95 % intervals over 17 to 100 prompts. In colour: floor + λ = 2 (its 100-prompt T4 rerun), adaptive λ and no step at t = 80. Only adaptive λ's left interval excludes zero, about what fourteen intervals give by chance.](../figures/f7_two_rewards.png)

The price shows on the mean of the four, which falls by 0.250 ± 0.045 against FK (a screen); the fall
mixes three changes, the roots kept, the milder tilt and the floor's flat target. On the best image,
100 prompts cannot tell the two apart: -0.012 [-0.082, +0.060], above the -0.14 to -0.02 I predicted (a test:
missed, open).

### Copying earlier in the run

The paper's appendix, on SDXL with adaptive resampling at $\lambda = 2$ and $k = 8$, reads that
"diversity can be increased by increasing the number of sampling steps from 100 to 200". It adds that "even if the samples $x_0$ share the same particle as parent, there is diversity in the final samples". Its
200-step schedule also copies earlier, at steps 180 to 120 of 200 where Table 1 copies at 80 to 20 of
100. I separated the two on the A2 at Table 1's $\lambda = 10$ and $k = 4$, on the `div_pix` of runs that end on one root.

On a first 40 prompts, 200 steps at Table 1's positions (`st200`) move it by +0.003 ± 0.008 over the 35
prompts where both end on one root. That is within the ± 0.03 I predicted (a test: held). The appendix's
positions at 100 steps (`pos100`, steps 90 to 60) raise it by +0.050 ± 0.005 (34 of 36 won, a screen).
They also soften the first resampling: its median ESS reads 2.94 against 1.19 at Table 1's positions, where I predicted 1.0 to 1.6
(missed).

On 100 prompts the full schedule, `d200`, raises it by +0.059 [+0.050, +0.067] over the 88 where both
end on one root (a test: held). On the 60 prompts the screen had not seen it reads +0.056 ± 0.006
(held, open). That bears out the paper's second sentence at Table 1's $\lambda$ and $k$. Its runs still
end on one root in 94 of 100, and its mean of the four falls 0.091 ± 0.047 below Table 1's schedule (a
test: held, open).

### Where FK's gain comes from

In the 100-prompt T4 rerun of section 4, the free sampler started from FK's four noises, so its image
$j$ shows what root $j$ becomes unsteered (one draw each), which splits FK's result into steps:

| | ImageReward | step |
|---|---|---|
| a root drawn at random, left unsteered | 0.233 | |
| the root FK keeps, left unsteered | 0.466 | +0.233 ± 0.047 |
| the mean of FK's four images, grown from that root | 0.651 | +0.185 ± 0.036 |
| FK's best image | 0.799 | +0.148 ± 0.011 |
| the best of the four roots, left unsteered | 0.779 | |

FK keeps a better root than chance, but 0.313 ± 0.042 under the best of the four single draws, inside the 0.25 to 0.50 I predicted (a test: held, open). As that best is read on one noisy draw per root, 0.313
overstates what the choice costs.

FK's first choice is made on the rewards at $t = 80$, which rank the
final images poorly (Kendall τ +0.137 ± 0.050, under the 0.15 predicted; a test: held, open). The rise of
the kept root's four images and the pick of the best of them win the cost back, and FK's best image
ends level with the best root (+0.021 ± 0.038, a screen). The split was measured at one seed, the one where FK gains least (A.7).

## 7. The released code and the published gain

I ran the authors' code [@fkd_code] through this repository's
launcher under the paper's configuration (A.4). It differs from this repository in six choices:

- it weights each step by $G_t$ itself, as the text does;
- it draws at random, as the paper's Algorithm 1 does;
- it floors the running maximum at 0, which the text does not state;
- it resamples the final population when ESS $< k/2$, even with adaptive resampling off;
- it decodes the guide with the pipeline's decoder;
- it sets its first four steps one index later.

`R1` is this repository's code with all these choices but the final resampling. On the same noises the
released code reads +0.011 ± 0.009 above it (seed 2024, a screen), and `R1` lands -0.040 ± 0.043 from
this repository's FK (a test: held, open). A fix to the released code in June 2025, after the paper,
changed its MAX potential. Before it, MAX took the larger of each reward and the previous step's raw
reward; after it, the larger of the reward and the floored running maximum, last step included (A.4). On the same noises and
three seeds, best-of-4 reads 0.770 (each difference a test, A.5):

| | gain over best-of-4 |
|---|---|
| this repository's FK | +0.080 ± 0.023 |
| released code | -0.003 ± 0.025 |
| released code before its fix | +0.074 ± 0.023 |
| the paper | +0.161 |

FK reads +0.080 here, against +0.062 in section 3: the two share seed 2024's T4 runs, and the other
two seeds ran on the A2 (a test: held, open). Before these runs I fixed the bar that would close the
gap: a gain of +0.12 or more over best-of-4, about three quarters of the paper's +0.161. Neither released version reaches it, though the earlier
one's interval ends at +0.120.

Against this repository's FK, the released code sits 0.083 ± 0.026
under: -0.030 ± 0.044 at the T4 seed, -0.109 ± 0.045 and -0.110 ± 0.041 at the A2 seeds. Its earlier version is level (-0.006
± 0.020). I predicted a gap within ± 0.06 and the earlier version above (both tests: missed,
open).

Seeded as the released launcher seeds, with the versions it pins, the version before the fix gains
-0.016 ± 0.035 over its own best-of-4 (0.798; A2, seeds 42 to 44, 45 of 100 won). That run draws other noises and pairs by prompt only, not with the table's +0.074. It stays under the
bar (a test: held) and under the band of ± 0.08 around +0.074 I predicted (missed, open; A.4). On 28 September I asked the authors which command, seeds and settings produced
Table 1 (A.4); they have not replied yet.

## 8. The same shape at three scales, and the judge

On two smaller image models, with a classifier as reward and sixteen particles (A.6), the ESS falls toward one particle too (screens, F8). On SD the independent judge is HPS. It sees no gain of FK over best-of-4 (-0.003 ± 0.002 on the A2 at one seed; a test: held), as the paper's
Table 1 shows too (FK 0.263, best-of-4 0.265).

![F8. Minimum ESS over a run divided by k, against λ (symmetric-log axis): CIFAR-10 and CelebA-HQ 256 (k = 16, three seeds) and SD v1.5 (k = 4, 17 prompts, A2). The dashed line, 1/k, is one particle carrying all the weight: CelebA reaches it, SD ends at 1.2 of 4.](../figures/f8_three_scales.png)

## 9. Limitations and open questions

The variants ran at one seed, on 17 to 100 prompts; only the main table, the variant without the step
at $t = 80$ and section 7's released-code rows have three seeds (A.2).

Within one machine and torch build a rerun returns the same numbers; across the two machines
the images differ, and FK keeps the same roots in only 30 % of prompts ([22, 40] %, A.7). FK is compared here with best-of-4 alone,
not with other particle methods for diffusion [@wu2024practical]. More particles on SD appear only in one 20-prompt screen at $k = 8$, where FK leads best-of-8 by +0.146 ± 0.091 (13 of 20
won, `results/sd_variants/K8.json`).

Two questions stay open: why the released code's gain moves with its seeding, down to -0.216 at seed
42, and which potential the paper's Table 1 used. The appendix table that repeats Table 1's FK values
follows "Here we use the difference potential", though here the two potentials differ by -0.004 ± 0.024
on the best image (n = 20, A.4).

## 10. Reproducibility, and how this was made

The mathematics is tested as properties: along a surviving lineage the potentials multiply to
$e^{\lambda r(x_0)}$, and at $\lambda = 0$ the system is the free model. The resamplers are tested
against the `particles` library as an oracle (`tests/test_resampling_oracle.py`). Every SD number measured in the main text is printed by a script
that `scripts/post_numbers.py` runs, and the paper's values come from its source (A.7). 

I wrote the core, that is the weights, the resamplers, the three potentials, the Feynman-Kac loop and
the model wrappers, every test of a mathematical property, and the analysis of the collapse. An AI
assistant (Claude, from Anthropic) wrote the scripts that launch the experiments, the figure scripts
and the documentation, and drafted this post, which I corrected; the working environment blocked it
from editing the core.

## Appendix

### A.1 Glossary

**ImageReward (IR)** [@xu2023imagereward]. A learned scorer of text-to-image alignment and quality,
trained on human preference rankings; higher is better, typical range -2 to +2 on this benchmark.

**HPS v2.1** [@wu2023hpsv2]. A second human-preference scorer trained on a different dataset, never
optimised here. Section 3 reads it as the released evaluation (`fks_utils.do_eval`) does, the best HPS
of the $k$ images, and gives the other reading; section 8 reads it at the image ImageReward selects, as
a judge.

**Best of $k$, mean of $k$** (`ir_max`, `ir` mean). The best and the mean ImageReward over the $k$ final
images of a run. On a collapsed population the best scores one image; the mean scores the population.

**ESS** [@kong1994ess; @chopin2020smc]. Effective sample size of $k$ normalised weights, $1/\sum w_i^2$;
$k$ under flat weights, 1 when one particle carries all the mass.

**Lineage, root** [@jacob2015path]. The initial noise $x_T$ a final image descends from, recovered by
walking the ancestor indices backwards; `n_lineages` counts distinct roots among the $k$ finals.

**Systematic resampling, comb.** $k$ evenly spaced points shifted by one uniform draw over the
cumulative weights; each point copies the particle it falls on. Under flat weights it copies every
particle once.

**`div_pix`.** Mean pairwise RMSE between the $k$ finals downsampled to 64 × 64; separates four copies
from four images and nothing more.

**UNet rows.** The compute unit: sample rows crossing the UNet, counted by a forward hook; FK and
best-of-4 both read 800 per run, one sample 200.

**FID** [@heusel2017fid]. Fréchet distance between Inception features of two image sets; CIFAR and
CelebA only.

**Paired difference.** Computed per prompt (and per seed) on runs that share their $x_T$, then averaged;
the standard error is over prompts, the seeds of a prompt averaged first. Runs at another seed (the
seed-42 runs of A.4) are paired by prompt only. A slot or root comparison is made only between runs of one
machine.

**Screen, test.** A comparison is a test when a prediction written before the run names it with a
tolerance, and a screen otherwise; either can leave its question unsettled at its n. A.5 lists every
prediction, the headline's among them, which had no tolerance.

**Intervals.** [a, b] is a 95 % interval: a bootstrap over prompts for a difference, normal in F7,
Wilson for a proportion (those in A.7).

### A.2 All variants

The runs, the machine they ran on and the file that holds them. The probe (`collapse_lab/probe.py`)
runs the variants with every weight recorded.

| runs | machine | what | prompts | file |
|---|---|---|---|---|
| section 3's table | T4 | one sample, best-of-4, FK, three seeds | 100 | `results/sd_baseline.json` |
| the FK reference; `late` | A2 | FK at the paper's setting, seed 2024; no step at t = 80, three seeds | 100 | `results/sd_ref_fields100.json`, `sd_s60_full.json` |
| session A of the probe | A2 | `ctl`, `adapt`, `fadapt`, `floor`, `lam0`, half of `lam2` and `floor2` | 17 to 40 | `collapse_lab/out/probe.json` |
| the other probe arms | T4 | `late`, `stat0`, `multi`, `vae`, `idx`, `R1`, `thr05`, `rise`, the other halves | 20 to 100 | `collapse_lab/out/probe.json` |
| session C | T4 | `ctl`, `lam0`, `floor2` in one process | 100 | `collapse_lab/out/probe_C.json` |
| session D | A2 | the same three, every image and Tweedie estimate saved | 100 | `collapse_lab/out/session_D/probe_D.json` |
| session E | T4 | `ctl` and `lam0` on two prompts, the machine check; the released code at seed 2024 completed to the 100 prompts, both seedings | 2, 100 | `collapse_lab/out/session_E/t4_check.json`, `results/sd_authors_R0_100.json` |
| session F | T4 | the released code at its commit before the fix of the MAX potential, seed 2024, global seed | 100 | `results/sd_authors_prefix.json` |
| session G | A2 | best-of-4, FK, the released code and its commit before the fix, seeds 2025 and 2026, global seed | 100 | `results/sd_seeds_bon4.json`, `sd_seeds_fk4.json`, `sd_seeds_authors.json` |
| session H | A2 | the paper's step advice, `ctl`, `lam0`, `free200`, `d200`, `st200` and `pos100` at seed 2024; the released code before its fix under its launcher's seeding, FK and the same pipeline without it, seeds 42 to 44, pinned versions | 40, 100 | `collapse_lab/out/session_H/probe_H.json`, `results/sd_authors_once.json`; H1-bis, the same four arms on prompts 41 to 100: `collapse_lab/out/session_H/probe_H1bis.json` |

Each variant is paired by prompt, seed 2024, with FK at the paper's setting run on the same machine:
session C on the T4 (equal to the FK row of section 3 at that seed to 5e-5), session A on the A2
(`collapse_lab/r_solutions.py`, `p_two_rewards.py`); `lam2` and `floor2`, which ran half on each
machine, pair each half with its own reference. Mean ± standard error over prompts.

| variant | what | machine | n | single-root | roots | `div_pix` | best of 4, minus reference | mean of 4, minus reference |
|---|---|---|---|---|---|---|---|---|
| `late` | no step at t = 80 | A2 | 100 | 84 % | 1.16 | 0.107 | +0.041 ± 0.037 | +0.063 ± 0.037 |
| `adapt` | λ lowered by bisection to keep ESS ≥ k/2 before the last step, capped at 10 | A2 | 40 | 60 % | 1.40 | 0.153 | -0.099 ± 0.044 | -0.171 ± 0.056 |
| `floor` | floor at 0 (released code) | A2 | 40 | 68 % | 1.73 | 0.150 | -0.091 ± 0.055 | -0.153 ± 0.075 |
| `lam2` | λ = 2 | both | 40 | 35 % | 1.82 | 0.205 | -0.039 ± 0.079 | -0.113 ± 0.084 |
| `fadapt` | floor + bisected λ | A2 | 40 | 30 % | 2.12 | 0.200 | -0.106 ± 0.058 | -0.214 ± 0.078 |
| `floor2` | floor + λ = 2 | both | 34 | 6 % | 2.94 | 0.283 | -0.036 ± 0.058 | -0.285 ± 0.084 |
| `thr05` | floor, resample only if ESS < k/2 | T4 | 40 | 62 % | 2.00 | 0.172 | -0.021 ± 0.053 | -0.118 ± 0.072 |
| `rise` | floor, bisected λ, cap 100 | T4 | 20 | 35 % | 1.95 | 0.170 | +0.053 ± 0.075 | -0.058 ± 0.113 |
| `lam0` | free sampler | A2 | 17 | 0 % | 4.00 | 0.338 | -0.070 ± 0.082 | -0.557 ± 0.133 |
| `stat0` | floor, statistic form | T4 | 40 | 62 % | 1.75 | 0.155 | -0.013 ± 0.054 | -0.083 ± 0.074 |
| `multi` | multinomial at every scheduled step | T4 | 40 | 100 % | 1.00 | 0.074 | -0.024 ± 0.041 | +0.014 ± 0.037 |
| `vae` | guide decoded by the pipeline VAE | T4 | 40 | 95 % | 1.05 | 0.087 | +0.045 ± 0.059 | +0.067 ± 0.063 |
| `idx` | indices {20, 40, 60, 80, 99} | T4 | 40 | 100 % | 1.00 | 0.088 | -0.014 ± 0.054 | -0.003 ± 0.056 |
| `R1` | the four above together: five choices, the floor included | T4 | 100 | 81 % | 1.19 | 0.105 | -0.040 ± 0.043 | -0.038 ± 0.048 |

**`late`.** The row reads `sd_s60_full.json` at seed 2024 against the FK reference, both on the A2; over
its three seeds against section 3's row it reads +0.0386 ± 0.0351, paired by prompt across two
machines. F7 draws this row, and floor + λ = 2 from session C (100 prompts, T4, -0.012 ± 0.038)
where the table's row is the probe's 34 prompts.

**`thr05`.** It resamples once per run and ends on 2.00 roots: when the threshold fires, the
accumulated weights are peaked and one pass takes almost everything.

**Roots kept.** `vae` keeps the same set of roots as its reference in 78 % of prompts, `multi` and
`idx` in 70 %.

**`lam0` at 17 and `floor2` at 34 prompts.** An interrupted rerun of six prompts replaced their
session-A records of FK, the free sampler and floor + λ = 2; FK's six were restored from the reference
run without their weights, and the other two arms lost them.

**Section 3's tail.** Its worst prompt over three seeds is at -1.30 against best-of-4. Per seed, FK
gains +0.030 ± 0.037, +0.089 ± 0.034 and +0.068 ± 0.050 over best-of-4. The mean of the four for its
three rows (0.237, 0.207, 0.687) is printed by `scripts/make_table_sd.py`.

Sessions C and D, one process each, 100 prompts:

| | session C, T4 | session D, A2 |
|---|---|---|
| `ctl` after the first resampling: roots, single-root | 1.80, 46 % | 1.68, 53 % |
| `ctl`: roots, single-root, `div_pix` | 1.07, 93 %, 0.109 | 1.04, 96 %, 0.094 |
| `floor2`: roots, single-root, `div_pix` | 3.03, 4 %, 0.300 | 2.96, 4 %, 0.293 |
| `floor2` - `ctl`, best of 4 | -0.012 [-0.082, +0.060] | -0.043 ± 0.037 |
| `floor2` - `ctl`, mean of 4 | -0.250 ± 0.045 | -0.261 ± 0.043 |
| `floor2` and `ctl` - best-of-4 of the same process, best of 4 | +0.009 ± 0.012, +0.021 ± 0.038 | +0.000 ± 0.018, +0.043 ± 0.038 |
| best-of-4 of the process - section 3's best-of-4 at seed 2024 (0.770) | +0.009 ± 0.006 (0.779) | |
| `floor2` never resamples, the free sampler's four images | 17 prompts | 20 prompts |
| `lam0`: roots, `div_pix` | 4, 0.355 | 4, 0.354 |

### A.3 The prompts shown as images

Eligibility was written from the prompt text alone, after the rewards of the first runs on the same
prompts had been read: a prompt is eligible when it names something a reader can check by eye, no
named person, artist, brand or franchise, and no human portrait as its main subject
(`data/visual_pool.json`: 46 eligible in six categories, object, animal, scene, food, vehicle, figure;
54 excluded with a reason code). A rule fixed before session D ran (`docs/visual_selection.md`) keeps
eight eligible prompts where FK beats best-of-4, the first by FK's margin over the free sample with at
most two per category, plus the prompt closest to the median and a typical loss. F1 shows the best
gains of three categories: figure and animal, as the rule picks them, and scene, which I chose by eye
on 26/09 after seeing the 46 candidates, in place of the rule's food (kept as `F1_first_rule`). On HPS
at the image ImageReward picks, best-of-4 scores above FK on the three prompts shown and below it on
that food prompt, `007171-0104` (0.2388 against 0.2529, `collapse_lab/out/session_D/hps_finals.json`):
the swap made F1's HPS reading one-sided. The rule reads session D, the one session that saved every image
(`data/visual_selection.json`, 46 complete eligible prompts of 46). Five of the eight gains differ
from the selection the same rule returned on session C, on the other machine
(`data/visual_selection_pre_amendment.json`), as predicted.

F5, and with it F4, was chosen after the images were seen. On the first rule's pick, the retriever
below, the free sampler's and floor + λ = 2's images are nearly the same; a second rule (25/09) took,
among the eligible prompts where FK ends on one root and floor + λ = 2 keeps four, the one where floor
+ λ = 2's mean ImageReward rises most above the free sampler's (`004971-0071`, a sports car). On 26/09
both figures dropped floor + λ = 2 to show the collapse alone, and I chose the prompt by eye, for
images that follow it, among the 43 eligible prompts where FK ends on one root
(`docs/visual_selection.md`). On it FK keeps the best of the four roots, and its best image reads 0.09
under best-of-4.

`free` is slot 0 of the free sampler, `bo4` the best of its four slots, `fk` the best of FK's four,
all on the same noises.

| role | id | prompt | fk - free | fk - bo4 | roots FK / floor2 |
|---|---|---|---|---|---|
| gain | 001227-0027 | footage of an astronaut in a tropical beach | +3.22 | +0.03 | 1 / 2 |
| gain | 010361-0095 | retro sci fi art of an astronaut next to jupiter in a car | +1.81 | +0.24 | 1 / 2 |
| gain | 010525-0074 | a nova scotia duck tolling retriever with white chest and pink nose... | +1.59 | +0.29 | 1 / 4 |
| gain | 000304-0153 | chicken | +1.41 | +0.29 | 1 / 3 |
| gain | 007171-0104 | a bag of frozen breaded scampi with maerl spilling out | +1.31 | +0.73 | 1 / 4 |
| gain | 008265-0060 | a coin bag item from a videogame ui | +1.30 | +1.27 | 1 / 4 |
| gain | 007187-0044 | an underwater rollercoaster, cinematic, dramatic, - | +1.24 | +0.31 | 1 / 2 |
| gain | 007654-0033 | wood logs in the shape of a square | +1.21 | +0.66 | 1 / 4 |
| median | 008614-0134 | a checoslovaquian wolfdog, black and white painting | +0.57 | -0.03 | 1 / 3 |
| loss | 009606-0136 | logo for a coffee chain | +0.28 | -0.18 | 1 / 1 |
| worst loss | 002870-0080 | a modern style living room in a big mansion next to a giant window... | -0.68 | -0.81 | 1 / 3 |
| F5 | 007171-0040 | a foggy forest with cherry blossom leaves on the ground, liminal, quiet | +0.23 | -0.09 | 1 / 3 |

The prompt that started the project, "funny peanut butter" (`010856-0009`), is not selected; chosen
by hand, it reads fk - free -0.03 and fk - bo4 -0.36 in session D, with one root for FK and for
floor + λ = 2.

### A.4 The reference configuration

| item | paper (text) | released code (defaults, or `launch.sh`) | this repository |
|---|---|---|---|
| model, sampler | SD v1.5, DDIM η = 1, T = 100, CFG 7.5 | SD v1.5 at `--model_idx` 4 to 7, fp16 (default SD 2.1; `launch.sh` names SDXL, but with no `--model_idx` the override sets SD 2.1 and k = 2); DDIM η = 1, T = 100, CFG 7.5 | SD v1.5, DDIM η = 1, T = 100, CFG 7.5, fp16 |
| reward | ImageReward on $\hat x_0$ | decoded by the pipeline VAE | decoded by `sd-vae-ft-mse` |
| λ, k | 10, 4 | `--lmbda 10`; k set by `--model_idx % 4` (4 at index 6) | 10, 4 |
| schedule | {0, 20, 40, 60, 80}, 0 terminal | default 5-30-5; `launch.sh` {20, 40, 60, 80, 99} | indices {19, 39, 59, 79, 99} |
| potential | MAX, as the statistic $\exp(\lambda \max_{s \ge t} r)$ | default DIFFERENCE; `launch.sh` MAX | MAX, increment form (the statistic 0.015 ± 0.009 above it, n = 20) |
| max statistic | not floored | floored at 0, carried through resampling | no floor |
| terminal step | closes the product on $r(x_0)$ | closes it on the floored running maximum, then resamples if ESS < k/2 | closes it on $r(x_0)$, never resampled |
| resampler | multinomial at every step (Algorithm 1) | multinomial at every scheduled step | systematic comb, only if ESS < k |
| adaptive resampling | appendix C.3: "If ESS$_t < k/2$, then we skip the resampling step. This encourages particle diversity.", the reverse of the usual rule [@chopin2020smc] | off in these runs; when on, it resamples if ESS < k/2, the usual rule (`fkd_class.py`); the terminal step above still resamples if ESS < k/2 | not used |
| HPS in the table | "the highest reward particle"; the appendix's best-of-4 rows carry the best HPS of the $k$ | the best of the $k$ images (`fks_utils.do_eval`) | the best of the $k$ (section 3); at the image ImageReward picks, best-of-4 0.258 and FK 0.259 against the paper's 0.265 and 0.263 (seed spread 0.001) |
| prompts | ImageReward benchmark | default `geneval_metadata.jsonl`, `launch.sh` names none; the ImageReward file is `benchmark_ir.json`, 100 | byte-identical ids, order, text to `benchmark_ir.json` |
| seeds | not stated | `manual_seed` once per pass, 42, 43, 44 | one generator per prompt; once per pass for the launcher's seeding below |

**The runs of section 7.** They pass the paper's configuration to the released code (SD v1.5, λ = 10,
k = 4, MAX, 20-80-20) through this repository's launch script, `collapse_lab/ref/run_authors.py`,
under diffusers 0.31, which reseeds per prompt and replaces the unused LLM grader with a stub. They cover the 100 prompts,
except the seed-42 run through a generator, which covers the first 40. Seeds 2025 and 2026 ran on the
A2 (session G), with best-of-4 and FK rerun there so that each seed pairs by noise on one machine: the
A2's best-of-4 agrees with the T4's on none of the 200 prompt-seed pairs. On an issue of the released repository (#14, December 2025), the one reply from the authors' side
sets the first resampling at step 20, as `launch.sh` and these runs do. Sources cell by
cell, with file and line numbers of the paper source and the released repository:
`docs/reference_config.md`.

**The two codes agree.** Without its particle filter the released code returns best-of-4's four rewards
to the fourth decimal (5 prompts); with it, at seed 2024 and on best-of-4's noises, it lands +0.011 ±
0.009 from `R1` on the 100 prompts (`parse_authors.txt`), and +0.021 ± 0.013 on the 60 prompts the
test of A.5 covers.

**Seedings.** Under "global seed" this repository's launch script resets the global seed per prompt,
where the released launcher seeds once per pass; under "generator" the initial and DDIM noises come
from a generator passed to the pipeline. The two seedings of one seed share the noise up to the first
resampling only. At seed 2024 both start from best-of-4's noises; the seed-42 runs start from others
and are paired by prompt only. The 80 records of 23/09 name no device; their 59 to 60 s per run place
them on the T4, with section 3's runs.

| released code | n | ImageReward | against best-of-4 |
|---|---|---|---|
| seed 2024, global seed | 100 | 0.770 | 0.000 ± 0.041 |
| seed 2024, generator | 100 | 0.675 | -0.095 ± 0.046 |
| seed 42, global seed | 100 | 0.554 | -0.216 ± 0.061 |
| seed 42, generator | 40 | 0.807 | -0.039 ± 0.073 |
| before the fix, seed 2024, global seed | 100 | 0.846 | +0.077 ± 0.039 |
| seed 2025, global seed, A2 | 100 | 0.775 | +0.010 ± 0.040 |
| seed 2026, global seed, A2 | 100 | 0.756 | -0.018 ± 0.048 |
| before the fix, seed 2025, A2 | 100 | 0.864 | +0.098 ± 0.033 |
| before the fix, seed 2026, A2 | 100 | 0.823 | +0.049 ± 0.046 |

On the first 40 prompts, where best-of-4 reads 0.846, the four runs of the fixed code read 0.497,
0.949, 0.807 and 0.720, a standard deviation of 0.19 where the spread within a prompt predicts 0.07 for
exchangeable runs. Against FK, the fixed code's seed means, -0.030 on the T4 and -0.109 and -0.110 on the A2, leave seed
and machine unseparated; before the fix they read +0.047, -0.021 and -0.044. On the 100 prompts the three global-seed runs at seeds 2024 to 2026 read 0.770,
0.775 and 0.756, and their gains over best-of-4 spread by 0.014 where one seed's standard error is
0.043. At seed 2024 the two seedings differ by +0.229 ± 0.085 on those 40 and +0.006 ±
0.067 on the other 60, which the prediction of A.5 reads as chance between runs. At seed 42, on the
first 40 only, they differ by -0.310 ± 0.077, and no replication tests that gap.

**Flat weights.** Under flat weights the comb is the identity and the multinomial draw of the released
code is not: four flat multinomial passes at $k = 4$ leave 1.58 roots by chance alone.

**The fix of the MAX potential.** The released repository's commit `699c929` ("address max potential
bug", June 2025) changes the MAX potential of `fkd_class.py`. Before it, the weight is
$\exp(\lambda \max(r_t, r_{\text{prev}}))$ with $r_{\text{prev}}$ the raw reward of the previous
scheduled step and, before the first, `reward_min_value` (0 by default, and the launcher passes no
other), so both versions floor the first step; the last step closes the product on $r(x_0)$; after it, the carried reward is the
floored running maximum, last step included. The version run in section 7 (`9413005`) is after the
fix, and Table 1 predates it (arXiv v1, January 2025). Run at the fix's parent `6726324` (sessions F
and G, global seed, 100 prompts, three seeds), the code gains +0.074 ± 0.023 over best-of-4 and +0.077
± 0.020 over the fixed version on the same noises, each prompt averaged over its seeds; at seed 2024 it
returns fewer than four distinct images in 21 % of runs, against 13 % after the fix.

**The launcher's seeding.** Session H runs the version before the fix as the released launcher seeds
it, once per pass at 42, 43 and 44 (`run_authors.py --seed-once`), on the A2, in a venv with the versions
the released repository pins (torch 2.4.0, transformers 4.38.2, diffusers at `af28ae2d`, ImageReward at
`2ca71bac`), against the same pipeline without its particle filter. In that venv session D's saved
finals rescore to the rewards recorded on 23/09, and best-of-4's slots differ from the current venv's by
up to 0.088 on 3 prompts, as between two machines. The two passes of a seed share their noise on the first prompt
only, so their difference pairs by prompt and not by noise. Best-of-4 reads 0.857, 0.738 and 0.799 at the
three seeds and FK 0.791, 0.793 and 0.762; FK minus best-of-4 reads -0.066 ± 0.057, +0.055 ± 0.058 and
-0.037 ± 0.051, and on the three passes, each prompt averaged, -0.016 ± 0.035 (45 won), 5.0 standard
errors under the paper's +0.161. That baseline, 0.798 over the three passes, is not the best-of-4 of
section 7's table, so this reading does not pair with the version's +0.074 there. FK's four finals read
`div_pix` 0.064 against 0.358 for best-of-4, and 56 of its 300 runs return fewer than four distinct
images.

**The potential of Table 1.** The paper's text gives MAX for Table 1. Its appendix table of ImageReward
and HPS by $\lambda$ and schedule follows the sentence "Here we use the difference potential"; its FK
rows at $\lambda = 10$ and 20-80-20 on SD v1.4, v2.1 and SDXL carry Table 1's FK ImageReward to the
third decimal (0.927, 1.006, 1.298), and its base rows Table 1's best-of-4 (0.800, 0.888, 1.236); its
HPS maxima differ (0.259, 0.266, 0.297 against 0.263, 0.268, 0.302). SD v1.5 is not in that table; section 2's CLIP values come from the appendix's GenEval table (A.7). In this repository the two potentials differ by -0.004 ± 0.024 on the best
image (max minus difference, n = 20).

### A.5 Predictions written before the runs, and their outcome

Every prediction with a number, each written before the run it predicts: the dated sections of
`docs/protocol_sd.md`, sections 7 and 12 of `collapse_lab/FINDINGS.md` (section 12 was first
committed with the first run of each arm quoted), and `collapse_lab/ASSESSMENT.md` (written with the
first screens of `adapt`, `floor`, `fadapt`, `floor2`, `lam2` and `lam0` already on disk). Outcomes
are read against the reference of the same machine. A verdict compares the point estimate with the band written with the prediction; where the interval crosses the band's edge the verdict is the point's, and the question stays open at that n. Seven lines, grouped last as checks and not counted,
restate a quantity that records on disk already fixed when the line was written: the rewards at
t = 80, which every arm shares on one machine, FK's reference run, or a run finished before the line. Of the other 76, 30 held (one of
them in part unmeasured), 30 missed and 6 held in part; a two-branch prediction came out on the side of
chance, the replay's forecast, which set no numeric tolerance, came out near its value, the four gap tests left the gap open, the rules of one screen and of its test held, `late`'s
gain over FK stayed unsettled, and the headline had no tolerance.

#### The reproduction and the judge

| written | prediction | outcome | verdict |
|---|---|---|---|
| 19/09 | the SD v1.5 row of Table 1 as the target | best-of-4 +0.021 from the paper, one sample +0.050, FK -0.078; FK - best-of-4 +0.062 ± 0.026 against +0.161 | no tolerance written |
| 21/09 | `S80` (one step, t = 80) clearly below FK; `D10` (ten steps) no gain over FK (20 prompts) | -0.027 ± 0.061; +0.028 ± 0.088 | in part |
| 21/09 | `late` - FK at 100 prompts × 3 seeds: above +0.052 kept, between 0 and +0.052 not settled | +0.0386 ± 0.0351 | not settled |
| 21/09 | HPS within ± 0.01 of FK whatever ImageReward does | `late` - FK, +0.0017 ± 0.0014 | held |
| 22/09 | latents test: the free wrapper and best-of-4 diverge slot by slot (chaotic at η = 1) | correlation 0.984 or more at the last step | missed |
| 23/09 | HPS, `ctl` - best-of-4 at the image ImageReward selects, within ± 0.01 | -0.003 ± 0.002 | held |

#### The collapse and the variants

| written | prediction | outcome | verdict |
|---|---|---|---|
| 21/09 | `floor` ends on 2 to 3 roots | 1.73 | missed |
| 21/09 | `lam2`: 1.3 to 1.6 roots at the end | 1.82 | missed |
| 21/09 | `lam0`: four roots, its four rewards equal to best-of-4's slot by slot | four roots; median slot gap 1.27 (another machine) | in part |
| 21/09 | no variant moves the best image by more than one standard error from the reference | `adapt` -0.099 ± 0.044, `fadapt` -0.106 ± 0.058, `floor` -0.091 ± 0.055 (and `late`, 1.1 standard errors, among the checks); with seven or more comparisons at 20 to 40 prompts, a likely miss without any effect | missed |
| 22/09 | `adapt`: bisected λ near 4 at t = 80, 2.5 to 2.9 roots after it, 1.5 to 2 at the end ("written before the analysis", committed with the first run quoted) | λ median 3.53; 2.45 and 1.40 | missed |
| 22/09 | `fadapt`: at least 3 roots after t = 60 | 3.35 | held |
| 22/09 | `floor2` at 40 prompts: under 10 % single-root, 2.6 to 2.9 roots, best image -0.10 ± 0.06 from FK | 6 %; 2.94; -0.036 ± 0.058 (n = 34) | in part |
| 22/09 | `lam2` at 40 prompts: 30 to 45 % single-root, 1.7 to 2.0 roots | 35 %, 1.82 | held |
| 22/09 | `late` 1.4 to 1.8 roots, 40 to 70 % single-root | 1.00 at 20 prompts (the 100-prompt run, 1.16 and 84 %, finished on 21/09) | missed |
| 22/09 | no variant at λ = 10 under 25 % single-root | lowest 30 % (`fadapt`) | held |
| 22/09 | `stat0` - reference -0.09 ± 0.05 | -0.013 ± 0.054 | missed |
| 22/09 | `multi` - reference -0.02 ± 0.04 | -0.024 ± 0.041 | held |
| 22/09 | `rise` 2.0 to 2.3 roots, - reference -0.05 ± 0.07 | 1.95 roots, +0.053 ± 0.075 | missed |
| 22/09 | `idx` - reference under 0.03 in absolute value | -0.014 ± 0.054 on the same machine (-0.051 as first read across machines) | held |
| 22/09 | `vae` - reference under 0.03 in absolute value | +0.045 ± 0.059 on the same machine | missed |
| 22/09 | `vae` keeps the reference's roots in at least 70 % of prompts | 78 % on the same machine (30 % as first read across machines) | held |
| 22/09 | `thr05`, a first guess: 2.4 to 2.8 roots, 1.2 to 1.8 resamplings | 2.00, 0.97 | missed |
| 22/09 | `thr05` by the replay of section 5, replacing the guess: 1.84 roots, 61 % single-root, 1.12 resamplings; criterion near 1.8 and not near 2.6 (dated 22/09 19h55 in `docs/protocol_sd.md`; the commit that first carries it also carries the run, so git does not date it) | 2.00 ± 0.22, 62 %, 0.97; 0.7 standard errors off | near its value, no numeric tolerance |
| 23/09 | the two loops see the same first-step rewards within 0.05 | the decoder moves one reward by 1.2 | missed |

#### Sessions C and D

| written | prediction | outcome | verdict |
|---|---|---|---|
| 23/09 | session C: `floor2` - reference in [-0.14, -0.02] on the best of 4 | -0.012 [-0.082, +0.060] | missed |
| 22/09 | `floor2` at 100 prompts: fewer than a quarter of runs on a single root (the criterion fixed after its 20-prompt screen, `collapse_lab/ASSESSMENT.md`) | 4 % | held |
| 23/09 | session C: `floor2` 2.8 to 3.1 roots, 5 to 10 % single-root | 3.03, 4 % | in part |
| 23/09 | session C: Kendall τ under 0.15; A - B in [0.25, 0.50] | +0.137 ± 0.050; +0.313 ± 0.042 | held |
| 23/09 | session D: `floor2` - `ctl` in [-0.10, +0.08]; 2.8 to 3.2 roots | -0.043 ± 0.037, 2.96 | held |
| 23/09 | the selection read on session D changes at least 3 of the 8 gains | 5 | held |

#### The released code

| written | prediction | outcome | verdict |
|---|---|---|---|
| 22/09 | released code under the paper's configuration: 0.77 [0.72, 0.82] | 0.554 at its seed (n = 100) | missed |
| 22/09 | released code - best-of-4 in [-0.05, +0.05] | -0.216 ± 0.061 (n = 100) | missed |
| 22/09 | `R1` - reference -0.08 ± 0.05, 1.0 to 1.3 roots | -0.040 ± 0.043 on the same machine (-0.067 ± 0.055 as first read against the A2 reference), 1.19 roots | held |
| 22/09 | `R1`: t = 80 inert in about 90 % of runs | 81 % | missed |
| 22/09 | gap closed only if `R1` - best-of-4 ≥ +0.12 | -0.011 ± 0.041 | not closed |
| 22/09 | `R1` - released code within ± 0.03 (above 0.06, the codes differ in an unlisted choice) | +0.206 at seed 42 | missed |
| 22/09 | released code returns fewer than four distinct images in 20 to 50 % of runs | 8 % | missed |
| 22/09 | released code through a generator - `R1` within ± 0.05 | -0.147 | missed |
| 23/09 | released code at seed 2024 through a generator: - best-of-4 within ± 0.06, - `R1` within ± 0.08 | -0.126 and -0.233 | missed |
| 23/09 | released code at seed 2024 under its own seeding in [0.55, 0.75] | 0.949 | missed |
| 25/09 | session E, global seed: - `R1` within ± 0.05 on the 60 new prompts; - best-of-4 on the 100 within 0.05 of `R1`'s -0.011 | +0.021 ± 0.013; -0.000 ± 0.041 | held |
| 25/09 | session E, generator - global seed on the 60 new prompts: under -0.1 if the first 40's gap belongs to the seeding, within ± 0.1 if it is the draw of two streams | -0.006 ± 0.067 | two streams |
| 25/09 | session F, the rule: the released code before its fix - best-of-4 ≥ +0.12 on the 100 prompts closes the gap | +0.077 ± 0.039 | not closed |
| 25/09 | session F: before - after the fix within ± 0.10 on the best image; one root in 80 % of runs or more | +0.077 ± 0.029; the released code records no ancestry | held, in part unmeasured |
| 26/09 | session G, at seeds 2025 and 2026: released code - best-of-4 within ± 0.10; before - after the fix positive | +0.010, -0.018; +0.088, +0.067 | held |
| 26/09 | session G, three seeds: the standard error of released code - best-of-4 in [0.022, 0.032]; after the fix within ± 0.06 of zero, before it between 0 and +0.12 | 0.025; -0.003 ± 0.025; +0.074 ± 0.023 | held |
| 26/09 | session G, the rule of 22/09 on three seeds: closed if either version gains +0.12 or more over best-of-4 | -0.003 and +0.074 | not closed |
| 26/09 | session G: the paper's +0.161 more than four standard errors above both versions | 6.5 and 3.8 | missed |
| 26/09 | session G: FK - best-of-4 on the A2 in [-0.05, +0.15] at each seed; on three seeds within 0.03 of the T4's +0.062 | +0.119, +0.092; +0.080 | held |
| 26/09 | session G, three seeds: released code - FK within ± 0.06 of zero; before the fix - FK positive | -0.083 ± 0.026; -0.006 ± 0.020 | missed |

#### Session H

| written | prediction | outcome | verdict |
|---|---|---|---|
| 29/09 | H0: `ctl` on two prompts equal to session D on the 8 slots within 1e-3 (the same A2) | 1 of 8: the pod's torch build moved from cu130 to cu132 | missed |
| 29/09 | H3: the pinned versions rescore session D's finals like the current ones within 1e-3; the same DDIM scheduler | 0.009, the pinned ones returning the recorded rewards exactly; the same scheduler | in part |
| 29/09 | H3: best-of-4's slots differ between the two venvs by more than 1e-3 | up to 0.088 | held |
| 29/09 | H1: 165 to 185 s per 200-step run, 85 to 92 s for `pos100` | 165 s, 87 s | held |
| 29/09 | H1, the controls rerun in session: `ctl` on one root in 34 runs of 40 or more, `div_pix` within ± 0.03 of 0.083; `lam0` within ± 0.03 of 0.344; `ctl` - `lam0` within 0.055 of +0.127 | 37, 0.082; 0.344; +0.088 ± 0.053 | held |
| 29/09 | H1: `d200`, `st200` and `pos100` each on one root in 32 runs of 40 or more | 39, 38, 39 | held |
| 29/09 | H1: median first-step ESS of `d200` and `pos100` between 1.0 and 1.6 | 2.56, 2.94 | missed |
| 29/09 | H1, one-root `div_pix`: `d200` 0.05 or more above `ctl`; `pos100` within ± 0.03 of `d200`; `st200` within ± 0.03 of `ctl` | +0.063 ± 0.005; 0.013 apart; +0.003 ± 0.008 | held |
| 29/09 | H1: `free200`'s `div_pix` within ± 0.03 of `lam0`'s | 0.350 against 0.344 | held |
| 29/09 | H1: `d200` - `free200` on the best image in [-0.05, +0.10] and below `ctl` - `lam0`; `d200`'s mean of the four below `ctl`'s | +0.033 ± 0.070 against +0.088; 0.721 against 0.803 | held |
| 29/09 | H1, the rule of the screen: `d200` at `ctl` + 0.05 or more with its interval over `ctl` above 0, and its gain no more than one standard error under `ctl`'s | 0.148 against 0.132; [+0.053, +0.073]; -0.056 ± 0.076 | holds (a screen) |
| 29/09 | H1-bis, the same rule on 100 prompts (the test the screen announced) | 0.151 against 0.141; [+0.050, +0.067]; -0.024 ± 0.055 | holds |
| 29/09 | H1-bis, H1's predictions on 100 prompts: `d200` on one root in 80 % of runs or more; `free200` within ± 0.03 of `lam0`; `d200` - `free200` in [-0.05, +0.10] and below `ctl` - `lam0`; `d200`'s mean of the four below `ctl`'s | 94 %; 0.363 against 0.354; +0.005 ± 0.044 against +0.029; 0.579 against 0.670 | held |
| 29/09 | H1-bis: median first-step ESS of `d200` between 1.0 and 1.6 | 1.91 | missed |
| 29/09 | H1-bis, the controls: `ctl` on one root in 85 % of runs or more, one-root `div_pix` within ± 0.03 of 0.083; `lam0` within ± 0.03 of 0.344; `ctl` - `lam0` within 0.055 of +0.127 | 93 %, 0.091; 0.354; +0.029 ± 0.037 | in part |
| 29/09 | H2: 86 to 92 s per FK run, 75 to 90 s per best-of-4 run | 83.5 s, 74.4 s | missed |
| 29/09 | H2: best-of-4's three-pass mean in [0.71, 0.83]; FK's three pass means within 0.10 of one another | 0.798; 0.762 to 0.793 | held |
| 29/09 | H2: FK - best-of-4 on three passes within ± 0.08 of +0.074 | -0.016 ± 0.035 | missed |
| 29/09 | H2, the rule of 22/09: closed at +0.12 or more | -0.016 | not closed |

#### The machines

| written | prediction | outcome | verdict |
|---|---|---|---|
| 21/09 | FK in the probe equal on the best image to the FK row of section 3 at seed 2024 | equal to the A2 reference instead (the two ran on different machines) | missed |
| 23/09 | free sampler of session C slot-correlated with best-of-4 under 0.7 | 1.00 (same machine) | missed |
| 23/09 | session D, the A2: 80 to 95 s per run, `ctl` equal to the A2 reference on at least 95 of 100 prompts | 88.3 s, 100 of 100 | held |
| 23/09 | session D's `lam0` slot-correlated with session C's at 0.99 or more | 0.62: the free path does not cross machines | missed |
| 25/09 | session E, on a T4: `ctl` and `lam0` equal to session C on every slot within 1e-4, 52 to 65 s per run | 16 of 16 slots equal, 56 to 59 s (two prompts) | held |
| 26/09 | session G, on the A2: best-of-4 against the T4's at the same seeds, `ir_max` correlated 0.4 to 0.8 over 200 pairs, the mean difference within two standard errors at each seed; 80 to 100 s per released-code run, 75 to 90 s per best-of-4 run | 0.79; +0.050 ± 0.042, -0.014 ± 0.056; 88 s, 79 s | held |

#### Checks against recorded data

| written | line | already fixed by | outcome |
|---|---|---|---|
| 21/09 | FK: about 1.8 roots after t = 80 and 1.05 at the end | the comb on the t = 80 rewards of 20 recorded runs (1.835); the FK reference, 1.05 on these 40 prompts | 1.74 and 1.06 (n = 40) |
| 21/09 | `floor`: first step inert in about 90 % of runs, about 3.8 roots after it | the same rewards (90 %, 3.849) | 90 %, 3.83 |
| 21/09 | `lam2`: about 2.9 roots after t = 80 | the same rewards (2.898) | 3.00 |
| 21/09 | `floor2`: about 4.0 roots after t = 80 | the same rewards (3.970) | 3.97 (n = 34) |
| 22/09 | `fadapt`: first step inert in about 90 % of runs | the floor's 90 % on the same rewards | 90 % (n = 40) |
| 23/09 | session D: t = 80 thumbnails of the three arms equal to 1/255 | the t = 80 rewards, equal across arms on one machine | 0/255 |
| 22/09 | `late`'s best image within one standard error of FK | the 100-prompt run of 21/09 (`sd_s60_full.json`) | +0.041 ± 0.037, 1.1 standard errors |

### A.6 CIFAR-10 and CelebA-HQ

CIFAR-10, k = 16, T = 1000, three seeds, classifier reward $\log p_A(\text{cat} \mid x)$, DIFFERENCE
potential (`results/sweep_lambda_classifier.json`). The counts by the judge $B$, here and for
CelebA-HQ, are a recount by hand of 21/09 (`docs/results.md`), not printed by a script:

| λ | log-probability of the drawn particle | minimum ESS | cats by $B$ (of 48) | pixel distance |
|---|---|---|---|---|
| 0 | -8.29 | 16 | 11 | 0.339 |
| 0.5 | -1.53 | 4.4 | | |
| 1 | -0.48 | 2.1 | 15 | 0.231 |
| 2 | -0.64 | 1.2 | 32 | |
| 4 | -0.16 | 1.05 | 37 | 0.183 |

CelebA-HQ 256, k = 16, DDIM 50, three seeds, glasses reward (`results/sweep_lambda_hub_classifier.json`,
`results/fid.json`); FID of 2048 finals against the 1468 faces with glasses and against the 30 000
faces:

| λ | glasses by $B$ (of 48) | minimum ESS | FID, faces with glasses | FID, all faces |
|---|---|---|---|---|
| 0 | 2 | 16 | 123.8 | 43.6 |
| 0.5 | 18 | 4.2 | | |
| 1 | 21 (7, 0 and 14 per seed) | 2.2 | 65.1 | 52.6 |
| 2 | 0 | 1.18 | 73.7 | 67.3 |
| 4 | 2 | 1.002 | | |

At λ = 2 the sixteen finals of seed 2024 are copies of one blurred face that the guide $A$ scores as
wearing glasses and the judge $B$ does not. The reference point for FID is the CIFAR DDPM fine-tuned on
the cat class: 51.4 against 80.2 for the base model.

### A.7 The numbers behind the main text, section by section

The main text keeps the numbers that carry its argument. The others are here, with their intervals, sample sizes and verdicts; every one is printed by a script under `results/post_numbers/` (`scripts/post_numbers.py`), except the CIFAR and CelebA judge counts of A.6 and a few values kept in `docs/results.md` or session H's check files: section 3's 5e-5 (a check recorded in `collapse_lab/commun.py`), `late`'s +0.0386 ± 0.0351, the latents correlation of 0.984, session H's run times and H3's checks. Two outputs there, `j_decomposition.txt` and `i_root100.txt`, hold an earlier reading of section 6's split that paired FK's reference run on the A2 with the free draws of the T4; `t_sessionC.txt` recomputes it inside one process, and the post uses that one.

**Section 1.** Over the 46 eligible prompts of the run F1 comes from (session D, A2, seed 2024), FK's median margin over best-of-4 is -0.024 (`select_visual_prompts.txt`); over the run's 100 prompts its mean margin is +0.043 ± 0.038, 48 won (`y_sessionD.txt`).

**Section 2.** The paper's statistic form of MAX reads 0.015 ± 0.009 above this repository's increments on the best image (20 prompts, a screen, `d_forms.txt`). The paper's appendix (20-80-20 schedule, difference potential) gives a CLIP diversity of 0.104 at λ = 10 and 0.225 at λ = 2 against 0.312 for the base model on SD v1.5, over GenEval prompts (`sections/appendix_experiments.tex`, table `tab:geneval_diversity`, 0.1038, 0.2252 and 0.3115), and on SD v1.4 a mean ImageReward of 0.811 for the four particles against 0.927 for the best.

**Section 3.** The 900 runs are placed on the T4 by their run times and by session C, which returns section 3's FK at seed 2024 to 5e-5 (A.2, A.5). Pooled over prompts, the seed-to-seed spread of a one-seed, 100-prompt mean is 0.063 for one sample, 0.034 for best-of-4, 0.040 for FK and 0.039 for FK's gain over best-of-4. The unit is the standard deviation of my three-seed mean minus the paper's value: its variance adds mine, the spread squared over three, and the paper's, the spread squared if Table 1 is one seed or over three if it averages three, hence the spread times $\sqrt{4/3}$ or $\sqrt{2/3}$; one unit is the typical distance seed noise alone produces, 0.045 for FK's gain if Table 1 is one seed and 0.031 if it averages three. In that unit one sample and best-of-4 sit within one of the paper, FK 1.7 under if Table 1 is one seed and 2.4 if it averages three. FK's paired gain, +0.062 ± 0.026 (69 of 100 won, bootstrap [+0.009, +0.110]), is 2.2 units under the paper's +0.161 if Table 1 is one seed and 3.1 if it averages three, as the released launcher's seeds 42 to 44 suggest; if the paper's two rows did not share their noises, 1.7 and 2.6 (`make_table_sd.txt`). HPS read at the image ImageReward picks, as pre-registered: best-of-4 0.258 and FK 0.259 against the paper's 0.265 and 0.263, 0.007 and 0.004 under, several times their seed spread (A.4). FK and best-of-4 send the same 800 sample rows per run through the UNet (A.1); FK decodes and scores each particle five times, and its median run takes 62.5 s against best-of-4's 54.7 s: at equal time the baseline would be best-of-4.57. Best-of-4 reads 0.758 over section 3's three seeds, 0.770 over section 7's (two of them on the A2) and 0.770 at seed 2024 alone, where the free sampler run inside FK's code, the baseline of section 6, reads 0.779, +0.009 ± 0.006 above it on the same noises (A.2).

**Section 4.** Single-root runs at the paper's setting: 93 of 100 on the T4 ([86, 97] %) and 96 of 100 on the A2 ([90, 98] %); `multi`, the multinomial draw at every scheduled step, 40 of 40 (T4). `div_pix` on the T4: 0.109 ± 0.006 for FK against 0.355 ± 0.005 for the free sampler. Mean of the four over section 3's 300 runs: FK minus best-of-4 +0.481 ± 0.030 paired (a screen).

**Section 5.** Median ESS over 100 runs of FK (T4): 1.18 of 4 at the first scheduled step, 1.63 to 3.29 at the next four. At $t = 80$ the four rewards span 0.905 on average, 9 nats between the largest and the smallest weight at λ = 10. After the first resampling, 1.80 roots on average and one root in 46 runs of 100 (T4). Adaptive λ (bisected ≤ 10 to keep ESS ≥ 2 before the last step): 1.40 ± 0.08 roots, one root in 60 % of runs ([45, 74] %, n = 40), against 1.5 to 2 predicted (missed). The replay forecast for the floor with resampling only under ESS < k/2 (`thr05`): 1.84 roots, observed 2.00 ± 0.22 (n = 40); the criterion, near 1.8 and not near 2.6, the value first guessed, set no numeric tolerance (a screen).

**Section 6.** Target ESS of best-of-4's four free draws reweighted by $e^{10 r}$: median 1.23 of 4 (100 prompts). The floor alone: first step inert in 90 % of runs ([77, 96] %, n = 40), 1.73 ± 0.19 roots at the end against 2 to 3 predicted (missed, open), 68 % single-root; λ = 2 alone 35 % single-root (A.2). Floor + λ = 2 on 100 prompts (T4, session C): 3.03 ± 0.09 roots, 4 % single-root ([2, 10] %), under the quarter fixed as its criterion (held) and one point under the 5 to 10 % predicted (in part); `div_pix` 0.300 ± 0.008 against FK's 0.109; in 17 prompts it never resamples, half of its four-root runs. Its price against FK: mean of the four -0.250 ± 0.045 (a screen); best image -0.012 [-0.082, +0.060], shallower than the -0.14 to -0.02 predicted (missed); against the free sampler's best image of the same run +0.009 ± 0.012, where FK reads +0.021 ± 0.038 (A2 rerun in A.2). The split: the root FK keeps beats a random one by +0.233 ± 0.047; the $t = 80$ rewards rank the four free outcomes with a Kendall τ of +0.137 ± 0.050 (held) and rank the best root first in 35 % of prompts ([26, 45] %) against 25 % by chance; the root choice costs 0.313 ± 0.042 against the best root (held), and what follows returns 0.333 ± 0.038, 0.185 ± 0.036 as the rise of the mean of FK's four above their root and 0.148 ± 0.011 as the best-of-four read-out; FK's best image ends +0.021 ± 0.038 above the best root. The free sampler's best image of that run reads 0.779 where section 3's best-of-4 at seed 2024 reads 0.770 (A.2). The split was measured at seed 2024, where FK gains least over best-of-4 (+0.030; +0.089 and +0.068 at the other seeds). The copying steps moved earlier (session H, A2, seed 2024, the first 40 prompts, a screen, `z_sessionH.txt`): single-root runs 37, 39, 38 and 39 of 40 for `ctl`, `d200` (200 steps, the appendix's schedule), `st200` (200 steps, Table 1's) and `pos100` (100 steps, the appendix's position); `div_pix` in those runs 0.082, 0.148, 0.090 and 0.135, paired against `ctl` +0.063 ± 0.005 for `d200` (35 of 36 won), +0.003 ± 0.008 for `st200` ([-0.012, +0.019], 17 of 35) and +0.050 ± 0.005 for `pos100` (34 of 36); the free samplers 0.344 at 100 steps and 0.350 at 200. Median first-step ESS 1.19, 2.56, 1.11 and 2.94. On the best image `d200` gains +0.033 ± 0.070 over the free sampler at 200 steps (22 of 40 won) and `ctl` +0.088 ± 0.053 over it at 100 (25 won), a difference of -0.056 ± 0.076. `pos100` moves all four copying steps, not only the last, and its first step reads a blurrier guide (first-step ESS 2.94 against 1.19, a missed prediction). On 100 prompts, H1 with H1-bis, the pre-registered test: `ctl` and `d200` on one root in 93 and 94 of 100; one-root `div_pix` 0.091 and 0.151, `d200` minus `ctl` +0.059 ± 0.004 (83 of 88 won) [+0.050, +0.067]; the free samplers 0.354 and 0.363; median first-step ESS 1.25 and 1.91 (predicted 1.0 to 1.6 for `d200`: missed); on the best image `d200` - `free200` +0.005 ± 0.044 (50 won) and `ctl` - `lam0` +0.029 ± 0.037 (49 won; predicted within 0.055 of +0.127: missed), their difference -0.024 ± 0.055; the mean of the four 0.579 against 0.670. On the 60 new prompts alone: one-root `div_pix` 0.153 against `ctl` + 0.05 = 0.146, `d200` minus `ctl` +0.056 ± 0.006 (48 of 52 won), lower bound +0.043, gain difference -0.003 ± 0.076: the rule holds.

**Section 7.** `R1` against FK at seed 2024 (0.799): -0.040 ± 0.043 (held). Against FK on three seeds the released code reads -0.083 ± 0.026, predicted within ± 0.06 (missed), and its version before the fix -0.006 ± 0.020, predicted above FK (missed); at seed 2024 alone, the T4 seed where `R1` ran, the released code reads -0.030 ± 0.044 against FK (41 won). Mean ImageReward on the three seeds: best-of-4 0.770, FK 0.850, released code 0.767, before its fix 0.844. Under the launcher's seeding (session H, A2, pinned versions, A.4): FK minus best-of-4 -0.066, +0.055 and -0.037 at seeds 42 to 44, -0.016 ± 0.035 on the three passes (45 of 100 won), predicted within ± 0.08 of +0.074 (missed) and under +0.12 (held).

**Section 8.** CIFAR-10: 11 cats in 48 free finals and 37 at λ = 4, as the minimum ESS falls from 16 to 1.05; CelebA-HQ: glasses on 2 of 48 free faces, 21 at λ = 1 and none at λ = 2 (A.6). On SD, HPS at the image ImageReward selects moves by -0.003 ± 0.002 between FK and best-of-4 (n = 100, A2, held).

**Sections 7 and 9.** On the first 40 prompts four released-code runs (two seeds, two seedings) spread by 0.19 where 0.07 is expected; at seed 42 it reads -0.216 ± 0.061 against best-of-4, paired by prompt only (A.4). Within one machine and one torch build a rerun returns its records particle by particle (100 of 100 prompts on either machine); across machines best-of-4's four rewards match on none of 200 prompt-seed pairs, and FK keeps the same roots in 30 % of prompts ([22, 40] %), its mean best image moving by +0.026 ± 0.058 (n = 100, a screen).
