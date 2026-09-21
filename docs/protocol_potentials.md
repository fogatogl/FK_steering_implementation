# Protocol: the paper's potentials against the code's

Two mismatches between FK Steering and `smc/` came out of reading the paper's
LaTeX source against the code on 21/09. One is immaterial. The other looked
material, `potential_form` was written to settle it, and the check said it is
inert wherever the reproduction actually runs. This file carries both, the
measurements behind the verdict, and the plan as it stood before the check.

Source throughout: Singhal et al., *A General Framework for Inference-time
Scaling and Steering of Diffusion Models*, ICML 2025, arXiv 2501.06848, section
3.3 and the appendix section "Adaptive Resampling". Nothing here has been checked
against the authors' released code, which still costs no GPU and is still worth
doing.

## Mismatch 1: the adaptive resampling rule reads backwards

The paper defines $\mathrm{ESS}_t = 1/\sum_i (\widehat{G}_t^i)^2$ and then writes
that if $\mathrm{ESS}_t < k/2$ the resampling step is skipped, "This encourages
particle diversity". `should_resample` in `smc/weights.py` does the opposite and
standard thing: it resamples when the ESS has fallen below the threshold, since a
low ESS is what says the weights have degenerated. That is also the rule in the
two SMC references the paper cites for this passage, Naesseth, Lindsten and Schön
(2019) and Chopin and Papaspiliopoulos (2020).

**Impact: none on the reproduced row.** The paper's table 1 protocol is
algorithm 1, which resamples at every scheduled step, and `fk4` runs at threshold
1.0, which does the same. The adaptive rule never enters the row being
reproduced. It touches `A05`, the four `*A05` ramp variants, and the CIFAR and
CelebA sweeps, all of which run at threshold 0.5.

**Decision: no code change.** The code holds the standard rule and the paper's
sentence contradicts its own references. Recorded in `docs/architecture.md`.

## Mismatch 2: `max` and `sum` are written as a statistic, computed as its increment

The paper defines, for $t \geq 1$:

$$G_t = \exp\Big(\lambda \max_{s=t}^{T} r_\phi(x_s)\Big), \qquad
G_t = \exp\Big(\lambda \sum_{s=t}^{T} r_\phi(x_s)\Big),$$

each closed by $G_0 = \exp(\lambda r(x_0))\big(\prod_{t=1}^{T} G_t\big)^{-1}$.
`potentials()` in `smc/fk.py` builds the same statistic into `curr` and then
subtracts the previous one, so what reaches the weights is
$\log G_t = \lambda(S_t - S_{t-1})$: the increment of the running maximum, and
for `sum` simply $\lambda\, r_\phi(x_t)$. `difference` coincides under both
readings, because there the statistic already is an increment.

Both families satisfy the product constraint, so both target
$\exp(\lambda r(x_0))$ and the telescoping test passes under either. With
`resample_last` off and the final particle picked by argmax ImageReward, the
terminal factor is never read in either. The two forms can differ only in the
selection weights at the intermediate resampling steps, $\exp(\lambda S_t)$
against $\exp(\lambda \Delta S_t)$, and the next section is about how much of a
difference that is.

`smc/fk.py` carries both since 21/09, under `potential_form`, `"increment"` for
the historical behaviour and `"statistic"` for the paper's. Every record says
which one produced it.

## What the check found (21/09, CPU, no GPU)

Measured while the T4 was busy with the S60 confirmatory run.

**The suite is green**, 42 passed and 1 skipped for missing weights. It was
written before `potential_form`, so that says the increment path is intact, not
that the statistic path is right.

**The invariant holds under the statistic form.** Accumulating `logG` along each
surviving lineage the way `tests/test_fk.py` does, the lineage total matches
$\lambda\, r(x_0)$ to 1.5e-05 over 400 toy runs at $\lambda = 10$, $k = 4$. The
terminal `acc` branch closes the product as intended.

**The two forms are algebraically identical on a one-step epoch.** Right after a
resampling every particle carries the same running maximum $S_{a-1}$, so
$\lambda S_t$ and $\lambda(S_t - S_{a-1})$ differ by a constant, and
normalisation removes it. Checked directly: same weights, same ESS to 1e-6. The
two forms can diverge only over an epoch spanning two or more scheduled steps,
that is, only after a resampling has been skipped.

**Which almost never changes an `fk4` image.** Over the 300 records of
`results/sd_baseline.json`:

| | runs |
|---|---|
| one-step epochs throughout, provably identical under both forms | 236 (78.7 %) |
| the only skipped resampling is at $t = 20$, inert for the images | 30 (10.0 %) |
| a skipped resampling at $t$ = 80, 60 or 40, so the images could differ | 34 (11.3 %) |

A skip at $t = 20$ cannot change anything: no resampling follows it, and the
particle is picked by argmax ImageReward.

**And the reward does not move.** Four toy regimes, $k$ particles on a random
walk whose intermediate reward is the walk seen through shrinking noise: the
`fk4` geometry at threshold 1.0 and at 0.5, a CIFAR-like geometry with every step
scheduled, and a saturating regime built to reproduce run 3, where `max`
resamples 3.3 times in 200 steps against `difference`'s 173. In all four the gain
over best-of-$k$ differs between the forms by less than one standard error:
+0.3548 against +0.3491, +0.3380 against +0.3275, +0.5062 against +0.5341,
+0.4836 against +0.4940. The statistic form does resample more often, 2.17 to
2.95 and 3.34 to 4.20, and that is all it does.

**Verdict: the statistic form is not a fix.** `max` under-resamples because the
running maximum saturates and is then shared by every clone, which drives its
across-particle spread to zero. Both forms are functions of that same saturated
statistic, so neither restores the spread. The 0.078 on the FK row is unexplained
again.

The change is worth keeping on fidelity grounds. The code now offers the paper's
definition, and "the two differ by a constant wherever the schedule resamples at
every step" is a stronger thing to write in the report than "we chose a different
member of the family". It is not a fix.

**Nothing queued needs changing.** `S60` at 100 x 3 and the `fk4` reference both
run at threshold 1.0, where the forms coincide on 79 % of runs outright and on
most of the rest by an inert margin. Night 2's `T2tA05` is a tempering ramp, and
`potential_form="statistic"` refuses to combine with `lam_placement="tempering"`
by construction, so it sits on another axis entirely.

## What the runs said before the check, and what survives

**`max` almost never resamples.** Run 3 of `docs/results.md`, CIFAR red reward,
$T = 1000$ steps, adaptive threshold: `max` resamples 0.3, 1, 1.7 and 10 times at
$\lambda$ = 1, 2, 4, 8, against 4, 25, 74 and 79 for `difference`. The reward
follows, 0.66 to 2.61 across the whole sweep for `max` against 0.66 to 10.45 for
`difference`. The paper reports `max` as its best potential on prompt fidelity,
for every model it tests. Here it is the weakest of the three. **Still open**, and
no longer attributable to the potential form.

**The uniform-weight signature.** Over the 300 `fk4` records: mean
`n_resamplings` 3.753 of a possible 4, 74 resamplings skipped, and in 293 of 300
runs the skips equal the steps where the weights came out exactly uniform, which
happens at $t$ = 80, 60, 40, 20 in 4, 10, 23 and 44 runs. Median ESS across the
five scheduled steps reads 1.18, 1.54, 2.20, 3.11, 2.78.

This was read at the time as a signature of the increment form. **That reading
was wrong.** Weights come out uniform when no particle beats the running maximum
its clones all share, and under the statistic form `curr` is then the shared
maximum for every particle, so the weights are uniform there too. On four
particles sharing a maximum of 1.5 that none of them beats, the increment form
gives `logG` 0.0 four times and the statistic form gives 15.0 four times, ESS
4.00 under both. The condition is the same under both forms, and the measurement
says nothing about which form is in use.

**The reproduction gap sits on the FK row alone.** `bon4` reads 0.7577 against
the paper's 0.737, `fk4` reads 0.8196 against 0.898: the baseline reproduces and
the method row is short by 0.078. The observation stands. The attribution written
here first, that the potential form weakens exactly that row, does not.

## The plan as it stood before the check

Kept because the repository keeps its predictions next to what happened. None of
it is live guidance now.

**Implementing the paper's form** was two sites. `smc/fk.py:37-44`, the `max` and
`sum` branches, where the weight becomes $\lambda$ times the statistic itself
with nothing subtracted and the `prev` term stops existing; and
`smc/fk.py:125`, `acc=(acc if lam_schedule is not None else None)`, where the
terminal step has to take the `lam * r_t - acc` branch because $G_0$ is the only
thing that closes the product. Two consequences: `acc` accumulates
$\sum_t \lambda S_t$ so the terminal `logG` is large and negative, inert while
`resample_last` is off and the pick is argmax; and the `-inf` initialisation of
`gate` for `max` loses the `prev` it was there to define, so the first scheduled
step needs a decision of its own. All of this is what `potential_form` now does.

**The run that was to settle it.** Twenty prompts, seed 2024, `fk4` otherwise
unchanged, paired against the existing `fk4` on shared $x_T$ as for `S80` and
`D10`, about 20 minutes on the T4. Four predictions were written before the
check, and the check answered three of them:

1. `n_resamplings` goes to 4.00. **Wrong.** Uniform weights occur under both
   forms, so most skips survive the switch.
2. The ESS falls at every scheduled step and the profile flattens. **Wrong**, and
   for the same reason: on a one-step epoch the two forms give the same ESS
   exactly.
3. `n_lineages` stays at 1.00 and `div_pix` near its floor of 0.0910, so the
   diversity findings of runs 13 to 15 are not at stake. **Held**, in simulation.
4. `ir_max` rises, and closing a useful part of the 0.078 would mean rebuilding
   the table's FK row on the paper's form. **Not on the real model yet**, but the
   simulation puts the difference inside one standard error in all four regimes,
   and only 11 % of runs can differ at all.

The run is still worth its twenty minutes as a check on the real model rather
than a proxy, with the prediction now inverted: `ir_max` should move by less than
its own standard error. At 20 prompts the paired standard error runs about 0.05
to 0.08 on this design, so it ranks rather than settles; settling is 100 prompts
at three seeds, about 5 h, where it falls to roughly 0.026.

**What it would have cost had it landed.** The 100 x 3 confirmatory is 5 h and
the fifteen-variant screen another 5 h or so at 20 minutes a variant. The heavier
item was never GPU: the $\lambda_t$ work of runs 13 to 15 is built on the
increment form, `docs/protocol_sd.md` derives it as tempering at constant
$\lambda$, and the paper's form has intermediate targets
$\pi_t \propto p(x)\exp(\lambda \sum_{s \geq t} S_s)$, so that section would have
needed re-deriving rather than re-running. Runs using `difference`, which is runs
5, 6 and most of 8, were unaffected throughout.
