# Why the four final images descend from a single x_T

## In short

*Updated 28/09 to the figures of the post (`docs/paper.md`); the findings below keep their
dated text. The +0.056 that findings 5 bis, 9, 13, 16 and 20 quote sets FK on the A2 against
best-of-4 on the T4, so it is not paired by noise (amendment of finding 16).*

**The problem.** FK with k = 4 returns four images from the same x_T in 93 runs out of
100 on the T4 and 96 on the A2; pixel diversity 0.109 against 0.355 for four free draws (T4).

**The cause.** Two distinct degeneracies, which the ESS does not separate (finding 15).
*The weights*: at each scheduled step the weight is `exp(lambda * r_phi)` up to a
shared factor, and at lambda = 10 the first step (t = 80) already has a range of 9 nats
on a guide reward that is all negative and carries no information (median ESS 1.18 out of 4
on the T4, 1.24 on the A2). The authors' code floors the `max` statistic at 0 and makes this
step inert; `smc/fk.py` does not, the only departure that makes this repository more
aggressive than the reference (finding 6; finding 16 lists the others). *The paths*:
resampling four particles four times makes the ancestor tree coalesce to one root even at
moderate weights. `adapt` holds the ESS at 2.0 and still ends at 1.40 lineages.

**The cost.** On `ir_max`, FK gains +0.062 +/- 0.026 over best-of-4 (100 prompts x 3 seeds,
T4, paired by x_T, 69 won), against +0.161 in the paper. Its four finals are near-copies of
one image: their mean `ir`, 0.687, sits far above that of best-of-4's four free draws, 0.207,
and their pixel diversity falls to a third (`results/post_numbers/make_table_sd.txt`).

**The leads** (findings 14, 17, 18, 20). All of them bring `ir_max` back to the level of
best-of-4, within one or two standard errors of noise; they differ by the diversity
kept and by the mean `ir` of the four (-0.15 to -0.56). Measured: floor + lambda = 2
(3.0 lineages, -0.01 on `ir_max` paired by x_T at n = 100) > floor + bisected lambda
(2.12) > ESS threshold < k/2 ~ floor alone ~ lambda = 2 alone (1.7-2.0) > bisected lambda
alone (1.40); schedule without t = 80: 1.14. The number of final roots is predicted from
the weights and the resampler alone, to within 0.09 over eighteen steered arms (finding 17,
`results/post_numbers/n_coalescence.txt`). At
lambda = 10 the target itself carries only 1.2 to 1.5 particles out of 4 (finding 15):
nothing "resolves" the collapse at this lambda, and the only correction that goes under
25 % of one-root runs changes the target. Dead: centring the reward, changing the
potential, resampling less.

**The reference** (finding 16). The paper's configuration is the one of this repository.
On the same 100 prompts and noises, on three seeds (2024 on the T4, 2025 and 2026 on the
A2), the released code gains -0.003 +/- 0.025 over best-of-4 and its commit before the MAX
fix `699c929` +0.074 +/- 0.023, where this repository's FK gains +0.080 +/- 0.023 on the same
runs. With the released code's choices, all but the final resampling (`R1`), this
repository's filter lands 0.011 +/- 0.009 from the released code at seed 2024. No version
reaches the paper's +0.161. Gap bounded, not closed.

---

## The problem, and the most likely cause

**The problem.** With k = 4, the four images that FK returns at the end descend from a
single root x_T in 96 runs out of 100. Pixel diversity falls to 0.09 against 0.33 for
four free draws. This is no longer a four-particle sampler: it is a sequential greedy
search on a root chosen early.

**The most likely cause.** The first scheduled step (t = 80) decides everything, and it
is the step where the guide reward knows nothing. The weight there is
`exp(lambda * r_phi)` up to a shared factor: a ranking on the **level** of the reward,
not on its increment, because the term the potential subtracts is common to the
particles and vanishes at normalisation. At lambda = 10, a range of 0.90 becomes 9 nats,
the ESS falls to 1.24 out of 4, and a single pass of the comb leaves only 1.8 ancestors.
Yet the ranking at this step does not predict final quality (Kendall tau = +0.067 +/- 0.071).
Four steps like that one and only one lineage is left.

**What makes it worse, and is a departure from the reference code.** The authors
initialise the statistic of the `max` potential at `reward_min_value = 0.0`; here it
starts from minus infinity. Since ImageReward is negative for all four particles in 90 %
of runs at t = 80, their step is **inert** where ours spends 9 nats on noise. It is the
only place where this repository is more aggressive than its reference, and it falls
exactly on the step that kills the lineage. The counterfactual (finding 10) confirms it:
with the floor, 3.9 ancestors survive at t = 80 instead of 1.8.

**What the correction does not promise.** Keeping lineages costs about 0.10 of `ir_max`
against `ctl`, and this price is **flat**: it is the same for the arm that keeps 0.35
lineage more and for the one that keeps 2.95 more (finding 13). Leaving the collapse
brings `ir_max` back to the level of best-of-4, after which diversity costs nothing more
on `ir_max`; it is the **mean** `ir` of the four images that pays, and that one follows
what is bought. The best measured compromise is the floor **with lambda = 2** (`floor2`):
2.85 lineages instead of 1.05, pixel diversity 0.263 against 0.092, for the same -0.10 as
the other arms (n = 20). Adaptive lambda alone repairs almost nothing: an ESS target at
k/2 makes it resample at every step and divides the lineages more slowly instead of
keeping them. The leads, measured and unmeasured, are in finding 14.

---

State as of 22/09 after the full night. Findings 1 to 6 and 8 are closed and replayable
without a GPU from `collapse_lab/*.py`. Finding 7 carries the predictions written before
the data, finding 10 what the probe says about them on 10 prompts, finding 11 a control
that fails and casts doubt on findings 3, 5 and 5 bis, finding 12 the two correction arms
on 20 prompts, finding 13 the seven arms at the final n (40 or 20 prompts) and the three
readings of finding 12 that it overturns, finding 14 the ways to repair, finding 15 an
independent rereading: weights and genealogy are two degeneracies.

## The causal chain, in one sentence

At each scheduled step, a particle's weight is `exp(lambda * r_phi(x_t))` up to a
**shared** factor, hence a ranking on the **level** of the guide reward; at
`lambda = 10` the range of this level is 9 nats at the first step, the ESS falls to 1.3
out of 4, and the systematic comb gives 3 or 4 slots out of 4 to a single particle. Four
rounds of that and only one root is left. The step that decides is the one where the
guide reward predicts nothing.

## 1. It is not a bug in `weights.py` or in `resampling.py`

The ESS recomputed from `r_at_schedule[0]` matches `ess_at_schedule[0]` to
**4.3e-05** at the median and 4.8e-04 at worst, over the 20 runs that carry the field,
and the same figure on `fk4_diff`, which is the same set of prompts
(`collapse_lab/a_step0_ess.py`). The collapse is exact arithmetic.

## 2. A single resampling is enough to kill half the lineages

The comb of `smc/resampling.py` gives particle j a number of slots equal to `floor` or
`ceil` of `k * w_j`. Integrating over `u ~ U(0, 1/k)` from the weights of step t = 80:
**1.835 distinct ancestors expected**, and P(a single lineage from this step on) = 0.46
(`collapse_lab/b_comb.py`).

Independent control: `S80.json`, whose schedule `[0, 80]` contains a single resampling,
observes **1.850** lineages. Prediction and measurement coincide on a file that was not
used to compute it.

## 2 bis. The reference figure, on 100 prompts

`sd_ref_fields100.json`, the exact configuration of the FK row:

| step | median ESS (out of k=4) | Q1 - Q3 | share of runs under 1.5 |
|---|---|---|---|
| t = 80 | **1.24** | 1.01 - 2.00 | **58 %** |
| t = 60 | 1.57 | 1.10 - 2.25 | 46 % |
| t = 40 | 1.97 | 1.19 - 3.16 | 29 % |
| t = 20 | 2.94 | 1.82 - 3.73 | 17 % |
| t = 0 | 2.61 | 1.84 - 3.22 | 13 % |

`n_lineages` is 1 in **96 runs out of 100** and 2 in the other 4; mean `n_resamplings`
3.78 out of 4 possible. An ESS of 1.24 over 4 particles means that one particle alone
carries close to 90 % of the weight.

## 3. The selection pressure is highest where the signal is zero

| step | mean level of r_phi | median range | nats at lambda=10 | median ESS |
|---|---|---|---|---|
| t = 80 | -1.62 | 0.90 | 9.0 | 1.34 |
| t = 60 | -0.33 | 0.77 | 7.7 | 1.48 |
| t = 40 | +0.32 | 0.52 | 5.2 | 1.92 |
| t = 20 | +0.68 | 0.32 | 3.2 | 2.58 |
| t = 0  | +0.84 | 0.27 | 2.7 | 2.39 |

And the information goes the other way. Slot j of `fk4` and slot j of `bon4` share
x_T, and row 0 is read before any resampling: `bon4["ir"][j]` is therefore what this
root becomes if it is left alone. **This sentence is the hypothesis that finding 11
defeats**: the `lam0` arm does not reproduce `bon4` slot by slot. What follows, and
finding 5 bis, depend on it.

- Kendall tau between the ranking at t = 80 and the final ranking: **+0.067 +/- 0.071**
- the best root at t = 80 is the best final one in **3 runs out of 20** (chance 25 %)
- `ir(best root) - ir(chosen root)` = **+0.627**, against **+0.568** for a root drawn
  at random: paired difference **+0.059 +/- 0.132**, hence **indistinguishable from
  chance**. The sign is unfavourable, the standard error covers it.

n = 20 and not 40: `fk4_stat` and `fk4_diff` carry the same prompts at the same x_T,
and finding 4 shows that their row 0 is identical, with zero difference. Stacking them
would divide the standard error by the square root of 2 without adding an observation
(`collapse_lab/commun.py`).

## 4. The subtraction of the potential protects against nothing

`_potential_terms` subtracts `prev` (the `gate`, the running max, the previous reward,
depending on the potential). But `prev` is **shared between slots** as soon as they
descend from the same ancestor, and it is 0 at the first step by construction (empty
`gate` for `max`, zero for `difference` and `sum`). A shared term vanishes at
normalisation: the weight is `exp(lambda * r_t)`.

Direct measurement on a probe run, t = 60: `logG - 10 * r_phi` is 3.5707 for all four
slots, exactly constant. The "ratchet" only starts to flatten at t = 40 and t = 20,
when the record becomes hard to beat, well after the lineage has died.

A checkable consequence, and checked: **the choice of potential cannot change
anything**. `fk4_stat` (max, statistic form) and `fk4_diff` (difference) share prompts
and seeds; their `r_phi` are identical **with zero difference** at t = 80 and t = 60,
they keep the same root in 19 prompts out of 20, and the paired difference on `ir_max`
is **-0.0043 +/- 0.0240** (`collapse_lab/d_forms.py`). This is the answer to the
question left open by `docs/max_potential.md`: `difference` did not close the gap
because the step that decides the lineage does not look at the potential.

## 5. The 18 variants already on disk say the same thing

`collapse_lab/e_variants.py`: `div_pix` stays at the floor of 0.09 (against 0.33 for
`bon4`) for the dense schedule, the ramps, the thresholds, the two forms and the three
potentials. The only variants that raise the diversity are those that **weaken the
first step** (`T2`, lambda_1 = 0.4: 1.55 lineages, div_pix 0.139) or that have only one
step (`S80`: 1.85 lineages, div_pix 0.233).

## 5 bis. It is not best-of-n: it is worse than best-of-n on the root

With an ESS of 1.3 at each scheduled step, the degenerate method is a **sequential
greedy search**: it draws 4 roots, keeps one early, clones it, the clones diverge under
the DDIM noise at eta = 1, it keeps one at the next step, and so on. The four final
images are four siblings separated since about t = 20.

The difference from best-of-4 can be put in figures. On the 100 prompts of
`sd_ref_fields100.json`, paired slot by slot with `bon4`
(`collapse_lab/j_decomposition.py`):

| | ImageReward |
|---|---|
| A best of the 4 roots, run freely | +0.7698 |
| M average root, free | +0.2232 |
| B **the root that fk keeps**, run freely | +0.3592 |
| C **what fk actually gets from it** | +0.8257 |

- **B - M = +0.136 +/- 0.057**: the root kept is worth more than a random draw. It
  recovers about **a quarter** of the available gap (bootstrap CI95 over the prompts
  [+0.023, +0.246], mean rank 1.220 out of 4 against 1.500 at random, CI95
  [1.020, 1.430], `collapse_lab/i_root100.py`).
- **B - A = -0.411 +/- 0.060**: the collapse costs 0.41 of root quality relative to
  what best-of-4 chooses, which takes the best one by construction.
- **C - B = +0.467 +/- 0.073**: steering along the trajectory, at a fixed root, wins
  back more than that.
- **C - A = +0.056 +/- 0.052**, and the decomposition closes exactly.

So: `fk4` **starts worse than best-of-4** and catches up by steering. The published gain
is not "FK chooses a better root", it is "FK chooses worse and steers better". The
collapse is the **price** of steering, not its means.

**Correction to an earlier reading.** `docs/max_potential.md` finding 4 concludes that
"the root it keeps is no better than chance", from a binary rate (does the argmax of
bon4 survive) measured on `S60`. The binary test has little power and `S60` has no step
at t = 80. On the reference file with 100 prompts, the rank and the cost in reward
reject chance at 2.5 standard errors. What remains true is that **the t = 80 step
alone** brings nothing (finding 3): what `root_slots` measures is the composition of
all the resamplings, and in the 54 % of runs where two roots survive t = 80, it is
t = 60 that decides, with a guide reward that is already less noisy.

## 6. The authors' released code has a safeguard that `smc/fk.py` lacks

`fkd_class.py` from <https://github.com/zacharyhorvitz/Fk-Diffusion-Steering>:

```python
self.population_rs = torch.ones(self.num_particles, device=...) * reward_min_value  # 0.0
...
if self.potential_type == PotentialType.MAX:
    rs_candidates = torch.max(rs_candidates, self.population_rs)
    w = torch.exp(self.lmbda * rs_candidates)
```

The statistic of the `max` potential is **floored at 0**. `smc/fk.py` applies this floor
to `prev` (`torch.where(torch.isneginf(gate), zeros, gate)`) but **not to `curr`**
(`curr = torch.maximum(gate, r_t)` with `gate` at -inf). Yet ImageReward is negative
for all particles in **90 %** of runs at t = 80 and **40 %** at t = 60. In the authors'
code these steps give `max(r, 0) = 0` everywhere, hence uniform weights and **no
selection**; here they give 9 nats of spread on noise.

The floor protects only `max`. For `diff`, `population_rs` is 0 at the first step and
the weight is `exp(lambda * (r_t - 0))`: a ranking on the level, too. The released
evaluation configuration (`lmbda=10`, `resample_frequency=5`, `resample_t_start=5`,
`resample_t_end=30`, `potential_type="diff"`, `adaptive_resampling` **disabled** by
default) places its first resampling at loop index 5 out of 100, that is t of about 950,
even noisier than the t = 80 here. **The collapse is therefore not specific to this
repository: it is in the regime of the released code too.** Their qualitative
configuration, for its part, is `lmbda=2.0`, `adaptive_resampling=True`,
`resample_frequency=20`, `t_start=20`, `t_end=80`.

The other departures all go in the direction of a repository **gentler** than the
reference, not harsher: the systematic comb of `smc/resampling.py` has a lower variance
than the authors' multinomial, and at threshold 1.0 `should_resample` tests `ESS < k`
in the strict sense, so it does not resample at exactly uniform weights where their
non-adaptive loop draws anyway. The missing floor is the only place where this
repository is more aggressive than the reference.

## 7. GPU counterfactual: predictions written before the data

Five arms, same prompts, same x_T, `collapse_lab/probe.py`: `ctl` (the reference fk4),
`floor` (the authors' floor, obtained by wrapping the reward passed to `fk_steer`,
without touching `smc/`), `lam2`, `floor2`, `lam0` (pairing control). `ctl` and `floor`
run on 40 prompts, the others on 20. Analysis: `collapse_lab/f_probe.py`.

The comb, applied at the first step to the weights actually recorded
(`collapse_lab/b_comb.py`), gives the number of ancestors expected just after t = 80:

| regime | median ESS | ancestors after t=80 | inert step |
|---|---|---|---|
| lam=10 (`ctl`) | 1.34 | **1.835** | 0 % |
| lam=10 + floor (`floor`) | 4.00 | **3.849** | **90 %** |
| lam=2 (`lam2`) | 2.72 | **2.898** | 0 % |
| lam=2 + floor (`floor2`) | 4.00 | **3.970** | 90 % |

**Predictions, written on 21/09 before the analysis.**

1. `ctl`: 1.8 lineages after t = 80, then a decrease down to 1.05 at the end;
   reproduces the `ir_max` of the `fk4` of `sd_baseline.json` up to rounding error.
2. `floor`: t = 80 inert in about 90 % of runs, hence about 3.8 lineages after this
   step. The floor **does not remove** the collapse, it **delays** it until the first
   step where a particle goes above 0: the trial run shows a prompt where only one of
   the four is positive at t = 80, logG = [3.57, 0, 0, 0], ESS 1.17, immediate
   collapse. Final prediction: between 2 and 3 lineages, clearly above 1, clearly
   below 4.
3. `lam2`: about 2.9 lineages after t = 80, about 1.3 to 1.6 at the end.
4. `floor2`: the most protective, about 4.0 after t = 80.
5. `lam0`: 4 lineages everywhere, and the four `ir` equal slot by slot to those of
   `bon4` (fp16 drift expected, not bitwise equality).
6. **`ir_max` does not move.** No arm should depart from `ctl` by more than one
   standard error (about 0.05 to 0.07 here). This is the most exposed prediction, and
   it follows from finding 3: if the root is chosen at random, losing three of them
   costs nothing on average. The collapse and the 0.078 gap on the reward are two
   distinct questions, and this file deals only with the first.


**The night of the 21st to the 22nd was killed with the terminal after 20 runs out of
80**; the probe was relaunched detached on the morning of the 22nd, with `lam0` first.
The partial analysis is in finding 10, the `lam0` control in finding 11.

## 8. What cannot work, and why (`collapse_lab/h_invariances.py`)

The weight is `exp(lambda * r)` normalised. **Any transformation of r that adds the
same thing to all particles vanishes at normalisation.** On the first step of a real
run, ESS = 1.1761; centring the reward, shifting it by +100, subtracting a shared
running max from it: **1.1761 every time, to the bit**. Centring the guide reward is
therefore a dead lead, and it is also the reason why the subtraction of `prev` in
`_potential_terms` protects against nothing (finding 4).

What moves the ESS is what compresses the **gaps** between particles: lowering lambda
(2.70 at lambda = 2), or creating ties, which is what the floor does (4.0000, the step
becomes inert).

The lambda that would hold the ESS at k/2, computed by `smc.fk.bisect_lambda` on the
recorded r_phi:

| step | lambda for ESS = 2 (median) | Q1 - Q3 |
|---|---|---|
| t = 80 | **3.91** | 2.55 - 15.49 |
| t = 60 | 4.24 | 3.00 - 7.02 |
| t = 40 | 8.53 | 5.01 - 11.19 |
| t = 20 | 11.93 | 8.15 - 20.07 |
| t = 0 | 15.22 | 9.73 - 62.39 |

The run uses 10.0 everywhere: **2.5 times too strong at the step where the reward
predicts nothing, and too weak at the steps where it predicts**. The correct profile
rises along the denoising, which is the opposite of the ramps of runs 13 to 15, which
lowered lambda at the start without raising it at the end.

## 9. What this implies, without writing the code

Three things to do, from the cheapest to the most expensive, each in a file that
belongs to the author.

**The floor.** `_potential_terms`, `max` branch of `smc/fk.py`: the floor at 0 is
applied to `prev` and not to `curr`. The released code initialises its statistic at
`reward_min_value = 0.0`, not at minus infinity, and reads `max(r_t, statistic)`. It is
the only place where this repository is more aggressive than its reference, and it
falls exactly on the step that decides the lineage. Make it a parameter rather than a
constant: the right floor depends on the reward, and 0 for ImageReward is in no way
universal.

**The adaptive lambda.** `fk_steer` has had `adaptive_lam` and `bisect_lambda` since
21/09, and `scripts/run_sd_baseline.py` does not expose them. The table above says what
it would do: about 4 at the first step instead of 10, about 15 at the last. The
trade-off is still to be decided: it changes the intermediate target but not
`exp(lambda * r(x_0))`, since `acc` carries the correction and the terminal step keeps
lambda.

**And the question that remains open.** Repairing the collapse could bring more than
`docs/max_potential.md` led one to expect (subject to finding 11, which casts doubt on
the pairing this figure comes from). Finding 5 bis puts at **0.411 +/- 0.060** what the
collapse costs at the level of the root: a sampler that kept four distinct lineages and
steered all of them would have this 0.411 within reach, where the published gain is
only 0.056. But nothing says it would take it: steering, for its part, benefits from
the concentration, and the Spearman of +0.015 between the ESS of the first step and the
gain (`docs/max_potential.md`) says that the most collapsed runs do not lose more than
the others. Diversity and reward remain two questions; this file deals with the first
and bounds the second.

The measured ranking of these leads, and those that remain to be tested: finding 14.

## 10. The GPU counterfactual: what the probe returned

The night of the 21st was killed with the terminal after 20 runs out of 80 (`ctl` and
`floor` on the first ten prompts, the other three arms never launched). Relaunched
detached on the 22nd. On these ten prompts, paired:

| arm | t=80 | t=60 | t=40 | t=20 | t=0 | resamplings | div_pix | ir_max |
|---|---|---|---|---|---|---|---|---|
| `ctl` | 1.80 | 1.40 | 1.20 | 1.10 | 1.10 | 3.90 | 0.105 | +1.066 |
| `floor` | 3.90 | 2.80 | 2.10 | 1.80 | 1.80 | 2.30 | 0.167 | +0.895 |
| `lam0` | 4.00 | 4.00 | 4.00 | 4.00 | 4.00 | 0.00 | 0.338 | +0.961 |

Share of steps where the weights come out uniform, hence with no selection at all:
`floor` 90 % at t = 80, 40 % at t = 60, 30 % at t = 40. `ctl`: 0 % everywhere except
10 % at t = 40.

Against the predictions of finding 7, written before:

1. **Held.** `ctl`: 1.80 ancestors after t = 80 against 1.835 predicted by the comb, and
   1.10 at the end.
2. **Held on the mechanism, missed on the final figure.** `floor`: t = 80 inert in 90 %
   of runs (predicted 90 %), 3.90 ancestors (predicted 3.849). But the collapse resumes
   faster than announced: 1.80 at the end, below the 2-3 range. The floor **delays** the
   collapse by one to two steps, it does not remove it.
6. **Under strain.** `floor - ctl` on `ir_max` is **-0.171** on average, where the
   prediction wanted less than one standard error. But the median is -0.025 and three
   prompts carry the whole mean, two of them where `floor` never resampled, that is,
   where it *is* best-of-4. At n = 10 the direction is contrary to the prediction and
   compatible with finding 5 bis (steering pays, diversity does not); nothing is
   settled.

The pilot check 0a of `f_probe.py` compared `ctl` to the `fk4` of `sd_baseline.json`,
written on 20/09, hence **before** commit 5025180: it showed a false gap of +0.090.
Against `sd_ref_fields100.json`, which is the FK row of the current code, `ctl` is
identical on the ten prompts, ESS at t = 80 included, paired difference +0.0000. The
pilot is clean; the reference file was not.

## 11. The pairing control fails, and it takes findings 3, 5 and 5 bis with it

`lam0` (lambda = 0, no resampling) was supposed to return the four free draws of
`bon4`, slot by slot, up to fp16 drift. On ten prompts:

- maximum gap per prompt over the four `ir`: **0.76** at the median, 2.97 at worst;
- within-prompt correlation between `lam0[j]` and `bon4[j]` over the 40 slots:
  **-0.107** (control with the prompts permuted: +0.059);
- the argmax coincides in 3 prompts out of 10, that is, chance;
- but the marginals match: mean `ir` 0.416 against 0.468, mean `ir_max` 0.961 against
  0.949, `div_pix` 0.338 against 0.33.

So `lam0` is a correct free sampler **in distribution**, and the identity of the slot
does not survive from one sampler to the other. Reading the code, `initial_state` draws
the same `randn_tensor((4, 4, 64, 64))` as the pipeline's `prepare_latents` and both go
through the same `scheduler.step(..., generator)`: x_T should be shared to the bit and
only the fp16 rounding of `eps` differs (CLIP encoded in a batch of 4 against an
expanded encoding). Two readings remain open, and they have the same consequence:

- either the noise stream diverges despite everything, and `bon4[j]` is not the
  continuation of root j;
- or x_T is indeed shared, and at eta = 1 over 100 steps the root explains almost none
  of the variance of the final reward, which is what the correlation of -0.107 says
  directly.

In both cases, the sentence "`bon4["ir"][j]` is what root j becomes if it is left
alone" is not measurable this way. **Finding 3 (tau = +0.067), finding 5 and the
A / M / B / C decomposition of finding 5 bis rest on it and have to be redone.** What
does not move: findings 1, 2, 2 bis, 4, 6 and 8, which never read `bon4`.

The deciding test is at the level of the latents, not of the reward: one prompt,
capture the pipeline's latents at steps 0 and 1 through `callback_on_step_end`, and
compare them to `initial_state` followed by one `step`. If they coincide, it is the
second reading.

*Addition of 23/09.* The test was done (`m_latents.py`): x_T coincides to the bit and
the trajectories stay correlated at 0.98 or more up to the last step; the third reading
is the right one, the path with resampling is not replayable from one session to
another (finding 19). Session C replays `ctl` and `lam0` in a single process and
recomputes findings 3 and 5 bis on this pairing (finding 20): they hold, more weakly.

## 12. The two corrections, evaluated outside `smc/`

Neither arm requires modifying `smc/`: the floor goes through the reward
(`torch.clamp(r, min=0)`, equivalent to the authors' potential to 9e-7, finding 6 and
`g_equivalence.py`), the adaptive lambda through `adaptive_lam` / `ess_target` /
`lam_max`, which `fk_steer` has carried since 21/09.

| arm | lambda | floor | what it isolates |
|---|---|---|---|
| `adapt` | 10 as a cap | no | lambda bisected to ESS = k/2 at non-terminal steps |
| `fadapt` | 10 as a cap | yes | both |

Tuning caveat: `bisect_lambda` returns the cap as soon as the ESS there is already
above the target, and the terminal step keeps lambda. With `lam_max = 10`, lambda_t can
therefore only **go down**: these two arms test "weaker early", not the rising profile
that the table of finding 8 calls for (about 4 at t = 80, about 15 at t = 0). "Stronger
late" would require `lam_max = 100`, and that is a third arm.

**Predictions, written before the analysis.** `adapt`: median lambda around 4 at
t = 80, ESS pinned at 2.0 at each non-inert step, hence about 2.5 to 2.9 ancestors after
t = 80. But since the target ESS is under k, it resamples at **every** step, and the
lineages keep halving: 1.5 to 2 at the end, above `ctl`, close to `floor`. `fadapt`:
t = 80 inert in about 90 % of runs (flat `base`, `bisect_lambda` returns its default,
zero logG), then bisection at the first step where a reward goes above zero instead of
the immediate collapse; the most protective, at least 3 ancestors after t = 60. On the
reward, both should stay **under** `ctl`, for the reason of finding 5 bis.

### What the two arms returned, on 20 prompts

Paired against `ctl`, which is `sd_ref_fields100.json` slot by slot (finding 10):

| arm | n | lineages | div_pix | ir_max | against `bon4` |
|---|---|---|---|---|---|
| `adapt` | 20 | **+0.20 +/- 0.09** | +0.041 +/- 0.014 | **-0.028 +/- 0.033** | +0.111 +/- 0.094 |
| `fadapt` | 20 | **+1.20 +/- 0.26** | +0.117 +/- 0.032 | -0.103 +/- 0.077 | +0.036 +/- 0.093 |
| `floor` | 11 | +0.91 +/- 0.44 | +0.082 +/- 0.054 | -0.198 +/- 0.079 | -0.015 +/- 0.149 |
| `lam0` | 10 | +2.90 +/- 0.10 | +0.241 +/- 0.025 | -0.105 +/- 0.104 | +0.012 +/- 0.168 |

`ctl` on these 20 prompts: 1.05 lineages, div_pix 0.089, ir_max +0.961.

**Adaptive lambda alone repairs almost nothing, and that is the surprise.** It does what
it is asked: median lambda **3.91** at t = 80, exactly the table of finding 8, and the
ESS pinned at 2.0 at t = 80 and t = 60. It still ends at 1.25 lineages, below the
predicted 1.5-2 range. The reason is structural: an ESS target at k/2 is **under** the
resampling threshold 1.0, so the arm resamples at every scheduled step (3.95 out of 4)
and each time gives two slots out of four to a single particle. Holding the ESS at k/2
does not keep the lineages, it divides them more slowly.

**What works is making the step inert, not softening it.** `floor` and `fadapt` leave
t = 80 inert in 90 % of runs and end at 2.0 and 2.25 lineages. And `fadapt` dominates
`floor` alone on both axes at this n: more diversity (+1.20 against +0.91) for half the
cost in reward (-0.103 against -0.198). Reading: once the first steps are neutralised by
the floor, the bisection prevents the following steps from collapsing what is left. The
standard errors overlap, 20 prompts against 11: this is a direction, not a result.

**The ranking on reward confirms finding 5 bis again.** Every arm that keeps lineages
pays for them, and the price follows what it buys: `lam0` (no steering, 4 lineages)
-0.105, `fadapt` -0.103, `floor` -0.198. `adapt` is the only one that is almost free,
and it is also the one that buys nothing. Prediction 6 of finding 7 ("`ir_max` does not
move") is now false in the sense that **repairing the collapse costs reward**.

A loose thread: with adaptive lambda, `logG - lambda_t * r_phi` stops being constant
across slots from t = 60 on (0 % of runs, against 50 % for `ctl`). The probable
explanation is the ratchet of `max` starting to bite once lambda_t is small enough for
no particle to beat the record (the potential would finally do something), but this is
not checked and it is not what the arm was testing.

**At 40 prompts, three of the readings above no longer hold**: `adapt` is not free,
`fadapt` does not halve the cost of `floor`, and the price does not follow what it buys
on `ir_max`. The -0.20 of `floor` at n = 10 and 11 was noise. See finding 13; the text
above is kept as written at 20 prompts.

## 13. The full night: seven arms, 40 or 20 prompts

`nuit.sh` ran to the end (`NUIT TERMINEE` in `out/nuit.log`): `ctl`, `floor`, `adapt`,
`fadapt` on 40 prompts, `lam0`, `lam2`, `floor2` on 20. Analysis:
`collapse_lab/f_probe.py`. Controls first:

- `ctl` against the `fk4` of `sd_ref_fields100.json`, 40 prompts: paired difference
  **0.0000** at worst. The pilot is the reference FK row.
- ESS recomputed from `logG` against recorded ESS, seven arms x 5 steps: 1.2e-04 at
  worst. Exact arithmetic everywhere, including under adaptive lambda.
- `lam0` against `bon4` slot by slot, 20 prompts: median max gap **0.89**, 2.96 at
  worst. Finding 11 holds at double n; slot-by-slot pairing stays broken.

### Where the lineages die

Mean number of distinct x_T roots after each scheduled step:

| arm | n | t=80 | t=60 | t=40 | t=20 | t=0 | resamplings | div_pix |
|---|---|---|---|---|---|---|---|---|
| `ctl` | 40 | 1.70 | 1.20 | 1.07 | 1.05 | 1.05 | 3.75 | 0.092 |
| `adapt` | 40 | 2.45 | 1.75 | 1.55 | 1.40 | 1.40 | 3.88 | 0.153 |
| `floor` | 40 | 3.83 | 3.08 | 2.20 | 1.73 | 1.73 | 2.00 | 0.151 |
| `lam2` | 20 | 3.00 | 2.15 | 1.85 | 1.85 | 1.85 | 4.00 | 0.215 |
| `fadapt` | 40 | 3.90 | 3.35 | 2.67 | 2.12 | 2.12 | 2.02 | 0.200 |
| `floor2` | 20 | 4.00 | 3.50 | 3.10 | 2.85 | 2.85 | 2.15 | 0.263 |
| `lam0` | 20 | 4.00 | 4.00 | 4.00 | 4.00 | 4.00 | 0.00 | 0.335 |

Share of inert steps (uniform weights): 90 % at t = 80 for the three arms with a floor,
then 50 % at t = 60, about 30 % at t = 40, 15 to 22 % after that. Without a floor: 0 to
12 %.

Against the predictions still open:

- finding 7, prediction 3 (`lam2`): 3.00 after t = 80 against about 2.9 predicted,
  **held**; 1.85 at the end against 1.3-1.6, **missed on the high side**.
- finding 7, prediction 4 (`floor2`): 4.00 after t = 80, **held**. 2.85 at the end.
- finding 12, `adapt`: 2.45 after t = 80 against 2.5-2.9, 1.40 at the end against 1.5-2,
  **both just below**. Median lambda at t = 80: 3.53 (3.91 at 20 prompts).
- finding 12, `fadapt`: at least 3 ancestors after t = 60 predicted, 3.35 obtained,
  **held**.

### The reward, paired against `ctl`

`ctl` on its 40 prompts: 1.05 lineages, `ir_max` +0.958, mean `ir` of the four +0.824.

| arm | n | lineages | div_pix | ir_max | mean ir |
|---|---|---|---|---|---|
| `adapt` | 40 | +0.35 +/- 0.08 | +0.061 +/- 0.012 | -0.099 +/- 0.044 | -0.171 +/- 0.056 |
| `floor` | 40 | +0.68 +/- 0.19 | +0.059 +/- 0.021 | -0.091 +/- 0.055 | -0.153 +/- 0.075 |
| `lam2` | 20 | +0.80 +/- 0.16 | +0.126 +/- 0.020 | -0.132 +/- 0.120 | -0.198 +/- 0.125 |
| `fadapt` | 40 | +1.08 +/- 0.17 | +0.108 +/- 0.020 | -0.106 +/- 0.058 | -0.214 +/- 0.078 |
| `floor2` | 20 | +1.80 +/- 0.24 | +0.175 +/- 0.024 | -0.099 +/- 0.076 | -0.302 +/- 0.111 |
| `lam0` | 20 | +2.95 +/- 0.05 | +0.247 +/- 0.015 | -0.098 +/- 0.072 | -0.532 +/- 0.115 |

Between correction arms, paired:

| comparison | n | lineages | div_pix | ir_max | mean ir |
|---|---|---|---|---|---|
| `fadapt - floor` | 40 | +0.40 +/- 0.08 | +0.050 +/- 0.010 | -0.015 +/- 0.021 | -0.061 +/- 0.033 |
| `floor2 - fadapt` | 20 | +0.60 +/- 0.15 | +0.058 +/- 0.014 | +0.004 +/- 0.029 | -0.098 +/- 0.039 |
| `floor2 - lam2` | 20 | +1.00 +/- 0.26 | +0.048 +/- 0.023 | +0.033 +/- 0.116 | |

**1. The price in `ir_max` is flat.** Six arms that keep from 0.35 to 2.95 more
lineages all lose between 0.09 and 0.13, indistinguishable from each other. And that is
the gap from `ctl` to best-of-4: `ctl - bon4` = +0.112 +/- 0.091 on these 40 prompts,
while each correction arm falls between +0.006 and +0.040 of `bon4`. As soon as the
selection is loosened, `ir_max` comes back to the level of best-of-4; beyond that,
keeping more lineages costs nothing more on `ir_max`. The choice is not a slider, it is
binary: bet everything on one lineage and take about +0.1 over best-of-4 (not
significant at n = 40, +0.056 +/- 0.052 at n = 100 in finding 5 bis), or keep diversity
at the level of best-of-4.

**2. The price in mean `ir`, for its part, follows what is bought.** It goes from -0.15
(`floor`) to -0.53 (`lam0`) in the order of the lineages kept. It is mechanical: under
collapse the four images are four clones of the best one, so mean `ir` is almost
`ir_max`; four different images have a lower mean. `ir_max` on a collapsed population
measures only one image. The two figures have to be reported together.

**3. `floor2` dominates.** At equal `ir_max` (+0.004 +/- 0.029 against `fadapt`), it
keeps +0.60 lineage and +0.058 of pixel diversity more. The floor is what makes t = 80
inert, lambda = 2 is what prevents t = 60 and t = 40 from redoing the collapse that the
floor only delayed (`floor2 - lam2`: +1.00 lineage at equal reward). Caveat: n = 20
against 40.

**4. The three readings of finding 12 that fall.** `adapt` costs -0.099 +/- 0.044, not
-0.028: it is not free, it is dominated (fewer lineages than `floor` for the same
price). `fadapt` does not halve the cost of `floor` (-0.015 +/- 0.021 between them);
what it buys in addition is 0.40 lineage, at 5 standard errors. And the announced
mechanism holds in part: the bisection of `fadapt` bites in only 5 % of runs at t = 80
(the floor makes the step inert first), but in 28 %, 48 % and 35 % at t = 60, 40 and 20,
with a median lambda around 4 when it bites. The median lambda of 10.00 shown by
`f_probe.py` for `fadapt` is that of the steps where it does not bite.

**5. The information of the first step, at 40 prompts**: Kendall tau +0.117 +/- 0.055,
top-1 30 % against 25 %. The figure rises from +0.067, but it reads `bon4[j]` as the
continuation of root j, and the `lam0` control says that this is not measurable that
way. It stays under finding 11 and serves as an argument for nothing until the test on
the latents is done.

## 14. How to repair: what is measured, what is not

All the leads below are for the author to write, in the files that finding 9 names.
What follows gives the trade-offs, not the choice.

### Measured, ranked by what the data say

At indistinguishable `ir_max` (about -0.10 against `ctl`, finding 13), from the most
protective to the least protective:

1. **floor + lambda = 2** (`floor2`): 2.85 lineages, div_pix 0.263 out of 0.335
   possible. Two changes: the floor in the `max` branch of `_potential_terms`
   (`smc/fk.py`), and lambda in the configuration. Trade-off: lambda = 2 is the
   authors' qualitative configuration, not the evaluation one (lambda = 10); the
   published gain of FK over best-of-n is measured at 10.
2. **floor + bisected lambda** (`fadapt`): 2.12 lineages. More expensive in plumbing
   (`adaptive_lam`, `ess_target`, `lam_max` to expose in `scripts/run_sd_baseline.py`)
   for fewer lineages than `floor2`.
3. **floor alone** (`floor`) or **lambda = 2 alone** (`lam2`): 1.73 and 1.85 lineages.
   The floor delays the collapse by one to two steps, lambda = 2 slows it; neither is
   enough.
4. **bisected lambda alone** (`adapt`): 1.40 lineages. To be set aside as it stands, for
   the structural reason of finding 12 (ESS target under the threshold, resampling at
   every step).

Two shared caveats. The floor at 0 is specific to ImageReward, whose first step is
negative for all four particles in 90 % of runs: another reward needs another floor,
hence a parameter and not a constant. And no lead returns a better `ir_max` than `ctl`:
they return diversity at the cost of `ctl`'s advantage over best-of-4.

### Not measured, to be tested

- **Start the schedule later.** The cheapest lever, and already in the data of
  finding 5: `S80` (a single resampling) ends at 1.85 lineages, `T2` (weakened first
  step) at 1.55. Removing t = 80 from the schedule amounts to making this step inert
  without touching the potential, for any reward. The authors' qualitative
  configuration starts at `t_start = 20` out of 100. Trade-off: fewer selection steps,
  hence less steering, and the floor stays necessary at the following steps if the
  reward is still negative there.
- **A resampling threshold under the ESS target.** `adapt` fails because the target
  (ESS = k/2) is under the threshold (ESS < k), so every step resamples. Standard
  adaptive resampling (Chopin and Papaspiliopoulos, chapter 10) resamples only under
  ESS < k/2. With a target above the threshold, the bisected steps would no longer
  resample at all and the weights would accumulate in `logW` until the terminal step.
  Trade-off: it is more lineages by construction, but the selection is then postponed
  rather than softened, and at lambda = 10 at the terminal step it can collapse in a
  single blow.
- **The rising profile of lambda** that the table of finding 8 calls for (about 4 at
  t = 80, about 15 at t = 0). `adapt` and `fadapt` can only go down under
  `lam_max = 10`; "stronger late" requires `lam_max = 100`. Missing arm.
- **Report mean `ir` next to `ir_max`.** Not a repair, a measurement: `ir_max` on a
  collapsed population is the reward of a single image, and hides what steering does
  (finding 13, point 2).

- **More particles than images returned.** The coalescence of the genealogy is a matter
  of k and of the number of resamplings (finding 15): k = 16 particles, and return the
  four best from distinct roots. Trade-off: four times the GPU cost, and it is no
  longer the configuration compared to best-of-4 at equal budget.

### What cannot work

Centring, shifting or normalising the reward by a quantity shared between particles, and
changing the potential (`max`, `difference`, `sum`): invariant at the first step
(findings 4 and 8), ESS identical to the bit. And, more broadly, any setting that acts
on the **weights** without changing the number of resamplings: it slows the
coalescence, it does not stop it (finding 15, `adapt`).

### What remains to be settled before writing anything

The test on the latents of finding 11 (one prompt, `callback_on_step_end` at steps 0
and 1 against `initial_state` followed by one `step`). It changes none of the leads
above, which never read `bon4` slot by slot, but it decides whether findings 3, 5 and
5 bis survive.

## 15. Independent rereading: degeneracy of the weights, degeneracy of the paths

Everything above measures the collapse with the ESS, then counts the lineages. These
are two different quantities, and the file let them blur together.

**What the ESS measures.** `collapse_lab/l_target_ess.py` takes the four images of
`bon4` (four roots run freely), reweights them by `exp(lambda * ir)` and reads the ESS,
on 100 prompts:

| lambda | median ESS out of 4 | Q1 - Q3 | share < 1.5 |
|---|---|---|---|
| 0.5 | 3.85 | 3.71 - 3.93 | 0 % |
| 1 | 3.51 | 3.16 - 3.74 | 0 % |
| 2 | 2.83 | 2.28 - 3.26 | 6 % |
| 5 | 1.89 | 1.25 - 2.45 | 37 % |
| **10** | **1.23** | **1.02 - 1.90** | **59 %** |

This is importance sampling of the target p(x0) exp(lambda r) with the prior as
proposal, in other words reweighted best-of-4, the estimator **without** steering. At
lambda = 10 it is 1.23: four free draws represent the target as badly as the first step
of `fk4` (ESS 1.24, Q1 - Q3 1.01 - 2.00, 58 % under 1.5, finding 2 bis; the two
distributions are the same). This is **why FK needs intermediate steps** at this
lambda: to move the particles toward the target before the terminal weight. And
steering does it: the ESS of `exp(10 * r)` over the four final images of each arm is 2.1
to 2.9 for all the steered arms (`floor` 2.89, `floor2` 2.47, `lam2` 2.07), against 1.22
for `lam0`. An exact sampler of the target, for its part, would have an ESS of 4: the
ESS of 1.23 is not a ceiling of the target, it is the distance from the prior to the
target.

The observation that remains: the median range of the free final ir is 1.04, that of
r_phi at t = 80 is 0.90. From the first step on, the guide reward has the range of the
final reward, without having its information (finding 3).

**What the ESS does not measure.** The number of distinct roots is a property of the
**genealogy**, not of the weights. Resampling k particles R times makes the ancestor
tree coalesce in O(k) generations whatever the moderation of the weights (Jacob, Murray and
Rubenthaler 2015 on path storage; Chopin and Papaspiliopoulos, chapters on resampling,
numbers to be checked). With k = 4 and R = 4, coalescence is almost certain. `adapt` is
the clean demonstration of it: ESS held at 2.0 at each step, 3.88 resamplings,
**1.40 lineages**. And over the seven arms, the final count of lineages follows the
number of resamplings before lambda:

| arm | lambda | resamplings | lineages |
|---|---|---|---|
| `lam2` | 2 | 4.00 | 1.85 |
| `adapt` | 3.5-10 | 3.88 | 1.40 |
| `ctl` | 10 | 3.75 | 1.05 |
| `floor2` | 2 | 2.15 | 2.85 |
| `fadapt` | 4-10 | 2.02 | 2.12 |
| `floor` | 10 | 2.00 | 1.73 |
| `lam0` | 0 | 0 | 4.00 |

Per prompt, the reweighted ESS of `bon4` does not predict the number of lineages
(correlation -0.02 for `ctl`, -0.39 for `floor2`): same signal.

**What this changes in the reading.**

- The floor works because it makes steps **inert**, hence removes resamplings
  (3.75 -> 2.00), not because it softens the weights. Lambda = 2 alone resamples at
  every step and gains only 0.8 lineage from it. The two together: fewer resamplings,
  and those that remain are at moderate weights.
- No setting that acts on the weights alone (potential, centring, bisected lambda with a
  target under the threshold) can stop the coalescence: it slows it. What stops it is
  fewer resamplings (shorter schedule, threshold under the ESS target, inert steps) or
  more particles than images returned (finding 14).
- The flat price of -0.10 on `ir_max` (finding 13) reads again as follows: the
  intermediate steps at lambda = 10 are what carries `ctl`'s advantage over best-of-4;
  any arm that neutralises or softens some of them falls to the level of plain
  importance sampling.

What this rereading does not call into question: the first step is the wrong place to
choose (finding 3, subject to the caveat of finding 11), and the missing floor is a
departure from the reference (finding 6).


## 16. The reference: the gap to the paper is bounded, not closed

Source: `docs/reference_config.md`, `results/sd_authors_R0.json`, `collapse_lab/ref/`,
analysis `ref/parse_authors.py`; the timestamped detail in `collapse_lab/ASSESSMENT.md`,
section C of the final state.

The paper's configuration (max, [0, 20, 40, 60, 80], lambda 10, k 4, DDIM eta 1, 100
steps, CFG 7.5, ImageReward on the Tweedie estimate) is the one of this repository; the
defaults of the released script (`diff`, 5-30-5) are another configuration, which the
paper's appendix scores lower. The released code differs by four unwritten
implementation choices: `max` statistic floored at 0, multinomial at every scheduled
step (flat weights included), adaptive resampling of the terminal population, the
pipeline's VAE for decoding the guide.

On the 100 prompts of the benchmark, SD v1.5, against paired `bon4`:

| reading | code | n | `ir_max` | against `bon4` |
|---|---|---|---|---|
| table 1 of the paper | | | 0.898 | +0.161 |
| `ctl`, this repository | `smc/` | 100 | 0.826 | +0.056 +/- 0.052 |
| `R1`, `smc/` with their four choices | `smc/` | 100 | 0.756 | -0.011 +/- 0.041 |
| their code, four runs pooled | theirs | 220 | 0.702 | -0.110 +/- 0.036 |
| the same four runs on their 40 common prompts | theirs | 40 x 4 | | -0.35, -0.04, -0.13, +0.10 |

Their pipeline without FK, at equal generator, returns the four rewards of `bon4` to the
fourth decimal; their ImageReward scorer matches the official one to the third. The two
implementations therefore start from the same images and score them the same way. With
the filter, their mean is below best-of-4 and their four runs differ from each other by
more than the paper's effect: two of them share x_T and DDIM noise and differ only by
the stream of the multinomial draw, and they return -0.13 and +0.10 (standard deviation
per prompt between their runs: median 0.25; this repository's `ctl` moves by 0.04
between two sessions on the same prompts). One run in four reaches the +0.16 within one
standard error; no mean reaches it. Reading of the code (23/09, `fkd_class.py`,
`fkd_pipeline_sd.py`): no reseeding, no bias of one seed path over the other; without a
generator the multinomial advances the global stream and changes the DDIM noise of the
following steps, with a generator it does not touch it; the terminal weight divides by
the float32 product of the intermediate weights, which can overflow. None of this
explains 0.23 between two runs at equal noise (three standard errors): **the spread of
their filter is measured, not explained.** Pre-registered rule: gap **bounded**, not
closed; it is not in the implementation, and part of it is in the variance of a
four-particle filter that keeps one root.

*Amended 28/09: the text above is the reading of 23/09, and the runs of 25 to 27/09 replace
its figures.*

- The `ctl` row sets FK at seed 2024 on the A2 (`sd_ref_fields100.json`) against the T4's
  `bon4`, so it is not paired by noise (finding 19). On the T4, FK gains +0.030 +/- 0.037
  over best-of-4 at seed 2024 and +0.062 +/- 0.026 on three seeds.
- `R1` takes every choice of the released code but the resampling of the terminal
  population: the floor, the statistic form, the multinomial at every scheduled step, the
  pipeline's VAE and the indices {20, 40, 60, 80, 99} (`probe.py`).
- Completed to the 100 prompts, the run that read +0.10 on the first 40 reads -0.000 +/-
  0.041 against `bon4`, and the two seedings at seed 2024 differ by +0.006 +/- 0.067 on the
  other 60 prompts, which the post reads as chance between runs. "One run in four reaches the
  +0.16" does not hold on 100 prompts: no released-code run gains more than +0.098 there
  (the commit before the fix, seed 2025).
- On three seeds the released code gains -0.003 +/- 0.025 over best-of-4, its commit before
  `699c929` +0.074 +/- 0.023, and this repository's FK +0.080 +/- 0.023 on the same runs. The
  released code's three seed means spread by 0.014 where one seed's standard error is 0.043;
  the spread wider than chance remains on the first 40 prompts only (0.19 against 0.07).

Sources: `docs/results.md` blocks 19 and 20, `results/post_numbers/parse_authors.txt` and
`z_sessionG.txt`; the post, section 7 and A.4.

## 17. The coalescence can be read from the weights alone

`n_coalescence.py`. Replaying the systematic comb (integrated over its offset `u`) or
the multinomial on the weights recorded at each scheduled step predicts the mean number
of final roots of fifteen arms to **within 0.07** (`ctl` 1.08 against 1.06, `floor` 1.73
against 1.73, `fadapt` 2.16 against 2.12, `floor2` 2.94 against 2.94, `R1` 1.25 against
1.19) and the share of one-root runs to within three points; per-run correlation 0.87
to 0.99 on the arms that have some range. Steering does not enter the prediction.
`adapt` holds the ESS at 2.0 at each step and ends at 1.4 roots: the ESS measures the
degeneracy of the weights, not that of the paths (finding 15). At flat weights the comb
is the identity and the multinomial is not: four flat passes leave 1.58 roots out of 4,
and the released code loses roots before any information. Pre-registered prediction on
`thr05`, written by this model before the measurement: 1.84 roots, 61 % with one root,
1.12 resamplings; measured: 2.00, 62 %, 0.97. The plan said 2.4 to 2.8. Figure:
`out/fig_coalescence.png`.

## 18. Resampling less does not keep the lineages

`thr05` (floor, resample only if ESS < k/2): 0.97 resamplings per run, 2.0 roots, 62 %
with one root (predicted by finding 17, see above). When the threshold finally
triggers, the accumulated weights are peaked and a single pass takes almost everything.
`ir_max` -0.06 +/- 0.10 against `ctl`. `late` (schedule without t = 80) repairs nothing
either: 86 % with one root at n = 300, 1.14 roots, for +0.04 +/- 0.03 on `ir_max`. This
last figure says something else: removing the t = 80 step costs nothing on the score,
so this step carries no information useful to the selection, without needing `bon4` to
say so.

## 19. Replayability: the free path is replayable, the path with resampling is not

The diffusers pipeline with a seeded generator returns on 23/09 the rewards of `bon4`
(20/09) to the third decimal. The `smc.models.StableDiffusion` path: deterministic
within a session; across sessions, 21/09 and the morning of 22/09 return the same
figures to 0.0000, and the evening of 22/09 other roots and other `ir_max` (up to 1.6
apart on one prompt), same weights, same code, same seed. Session C (finding 20)
separates the two cases: `lam0` (lambda = 0, no reward read) returns `bon4` slot by slot
at correlation 1.00, `ctl` returns the `ctl` of 21/09 at 0.70 with the same root in
30 % of prompts. The free path is therefore replayable from one session to another; the
one that reads the guide's reward (VAE ft-mse, ImageReward, BERT in fp16) is not, and
1e-3 on a reward is enough to move a tooth of the four-slot comb.

*Test of the afternoon of 23/09 (`nuit4.sh`, `u_determinism.py`, not pre-registered).*
`ctl` on prompts 0 and 1 in three separate processes: two without changing anything,
one with `cudnn.benchmark = False` and `use_deterministic_algorithms(True)`; then the
version of `probe.py` from the morning of 22/09 (the one of session A) on the same
prompts. All four return the **same four rewards** to the fourth decimal, equal to
those of session C in the morning and of session B on the evening of 22/09 (six common
prompts, `ctl_b1` = `ctl_C` to 0.0000), across a restart of the pod between session C
and the test. Ruled out: the process, the deterministic flags, the cache path
(`~/.cache` for B, `work/hf_cache` for C and the test, same revisions), the pod, the
`sd` venv (no installation since 20/09), `smc/` (unchanged since 21/09 17h27, before all
sessions), the rewrite of `probe.py`. What remains: the sessions that return the
reference of 21/09 (the reference itself, session A on the morning of 22/09) ran at
**87 to 90 s per run**; all those since the evening of 22/09 run at **55 to 60 s**, same
code, same pipeline. The two groups differ by the machine's execution path (fp16
kernels chosen, hardware), not by anything in the repository; the machine of the first
group no longer exists and the point cannot be pushed further. Consequence unchanged:
any slot-by-slot pairing between files from different sessions is invalid; per-prompt
means between FK arms stay usable; within the group since the evening of 22/09,
slot-by-slot pairing holds.

## 20. Session C: findings 3 and 5 bis hold, reworded

`out/probe_C.json`, `t_sessionC.py`: `ctl`, `lam0`, `floor2` at 100 prompts in **a
single process**, the only valid slot-by-slot pairing (finding 19). The predictions to
set against them are in `docs/protocol_sd.md`, "Pre-registration of session C".

- **Finding 3.** Kendall tau between the ranking of `r_phi(t = 80)` in `ctl` and the
  free `ir` of the same root (`lam0`): **+0.137 +/- 0.050**, top-1 in 35 % of prompts
  against 25 % at random (n = 100). The first step carries a little information, not
  none; the +0.067 of finding 3 was read on an invalid pairing.
- **Finding 5 bis.** A best free root 0.779; M average root 0.233; B the root that `ctl`
  keeps, read free, 0.466; C what `ctl` gets from it 0.799. B - M = +0.233 +/- 0.047
  (the root kept is worth a third of the way to the best one, mean rank 2.04 out of 4);
  **A - B = +0.313 +/- 0.042** (the collapse still costs 0.31 of root); C - B = +0.333
  +/- 0.038 (steering gives back a little more); C - A = +0.021 +/- 0.038 (the balance
  over best-of-4).
- **`floor2` paired by x_T**: `ir_max` **-0.012 [-0.082, +0.060]** against `ctl`
  (predicted [-0.14, -0.02], missed on the high side), mean `ir` of the four
  -0.250 +/- 0.045, 3.03 roots, 4 % with one root. The price of the lineages on
  `ir_max` is of the order of `ctl`'s advantage over best-of-4 (+0.030 +/- 0.037 in this
  session, +0.056 on 21/09), no more; the price on the mean of the four remains.

What this changes in finding 13: "flat price of -0.10" was overstated. At n = 40 each CI
covers zero, the mean of the seven arms is -0.07 [-0.18, +0.04], and `floor2` paired by
x_T at n = 100 is -0.01. The tenable sentence: **no configuration beats best-of-4 by
more than the noise on `ir_max`, while diversity varies by a factor of three and the
mean `ir` from -0.15 to -0.56.** The large effects are on the lineages and the mean.
