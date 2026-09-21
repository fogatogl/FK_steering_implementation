# Protocol: the paper's potentials against the code's

Two mismatches between FK Steering and `smc/` came out of reading the paper's
LaTeX source against the code on 21/09. One is immaterial and is recorded here
only so it stops being rediscovered. The other is not, and this file says what it
costs, what the fix is, and what run settles it.

Source throughout: Singhal et al., *A General Framework for Inference-time
Scaling and Steering of Diffusion Models*, ICML 2025, arXiv 2501.06848, section
3.3 and the appendix section "Adaptive Resampling". Nothing here has been checked
against the authors' released code, which is the first thing to do and costs no
GPU.

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
terminal factor is never read in either. **The two forms differ only in the
selection weights at the intermediate resampling steps**: $\exp(\lambda S_t)$
under the paper's, $\exp(\lambda \Delta S_t)$ under the code's.

### What the runs already say

**`max` almost never resamples.** Run 3 of `docs/results.md`, CIFAR red reward,
$T = 1000$ steps, adaptive threshold: `max` resamples 0.3, 1, 1.7 and 10 times at
$\lambda$ = 1, 2, 4, 8, against 4, 25, 74 and 79 for `difference`. The reward
follows, 0.66 to 2.61 across the whole sweep for `max` against 0.66 to 10.45 for
`difference`.

**That is the opposite of the paper's finding.** The paper reports `max` as its
best potential on prompt fidelity, for every model it tests. Here `max` is the
weakest of the three.

**The same signature is in the SD runs.** Over the 300 `fk4` records of
`results/sd_baseline.json`, measured 21/09:

| | value |
|---|---|
| mean `n_resamplings` | 3.753 of a possible 4 |
| resamplings skipped | 74 |
| runs where skips equal the steps with exactly uniform weights | 293 of 300 |
| steps with $\mathrm{ESS} = k$, at $t$ = 80 / 60 / 40 / 20 | 4 / 10 / 23 / 44 of 300 |
| median ESS at $t$ = 80 / 60 / 40 / 20 / 0 | 1.18 / 1.54 / 2.20 / 3.11 / 2.78 |

Exactly uniform weights mean no particle improved its running maximum at that
step, so every $\Delta S_t$ was zero. The paper's form cannot produce uniform
weights except by coincidence.

**The reproduction gap sits on the FK row alone.** `bon4` reads 0.7577 against
the paper's 0.737, `fk4` reads 0.8196 against 0.898. The baseline reproduces and
the method row is short by 0.078, and this mismatch weakens exactly that row.

## Before touching anything

Read the authors' released implementation. If it computes the increment, the code
matches the implementation, only the paper's prose is loose, and nothing needs to
be re-run. If it computes $\exp(\lambda S_t)$, the deviation is real. This is free
and it decides the whole question.

## Implementing the paper's form

Two sites, and the second is the one that is easy to miss.

**`smc/fk.py:37-44`, the `max` and `sum` branches.** At non-terminal steps the
weight has to be $\lambda$ times the statistic itself, with nothing subtracted.
The `prev` term stops existing for these two potentials. `difference` is
untouched.

**`smc/fk.py:125`, `acc=(acc if lam_schedule is not None else None)`.** Under the
paper's form the terminal step must take the `lam * r_t - acc` branch, because
$G_0$ is the only thing that closes the product back onto
$\exp(\lambda r(x_0))$. That branch already exists; it was written for the
$\lambda_t$ ramps and is currently handed `acc` only when a ramp is present. It
becomes the always-branch.

**Two consequences to know before running.** `acc` now accumulates
$\sum_t \lambda S_t$ rather than $\lambda S_{T-1}$, so the terminal `logG` is
large and negative. It stays inert while `resample_last` is off and the pick is
argmax, and it would matter the day either changes. And the `-inf` initialisation
of `gate` for `max` exists to make `prev` well defined at the first scheduled
step; with no `prev` in the non-terminal branch, the first step needs a decision
of its own.

**The design decision is not mine.** The existing `lam_placement` axis already
means "ratio of successive tempered targets", and with constant $\lambda$ its two
values compute the same thing, which `tests/test_fk.py` asserts. The paper's form
is not a third placement; it is a different definition of the potential. So it
wants an axis of its own, and then the question is whether it replaces `max` or
sits beside it. Replacing invalidates the comparability of the 300 `fk4` records
and the fifteen-variant screen, which all share their $x_T$; keeping both costs
one more field in every record. Either way every record has to say which form
produced it, or the JSON stops being readable a week from now.

## The guards

- The telescoping test is the guard, not a formality. If the terminal branch is
  not routed through `acc`, the product stops equalling $\exp(\lambda r(x_0))$
  and the test has to go red. Green without touching line 125 means the test is
  not checking what it is supposed to check.
- $\lambda = 0$ neutrality must survive unchanged.
- The observable that says the change took: `logG` at a scheduled step is
  all-zeros today whenever no particle improved. It should be spread at every
  scheduled step afterwards, and `n_resamplings` should read 4 of 4 rather than
  3.75.

## The run that settles it, written before it is launched

Twenty prompts, seed 2024, `fk4` otherwise unchanged: $\lambda = 10$, $k = 4$,
schedule $\{0, 20, 40, 60, 80\}$, threshold 1.0. Same prompts and seed as
`results/sd_baseline.json`, so the $x_T$ are shared and the comparison against
the existing `fk4` is paired, as for `S80` and `D10`. One tag in the screen's
convention, one JSON under `results/sd_variants/`. About 20 minutes on the T4.

Read on it: the paired difference in `ir_max`, `n_lineages`, `div_pix`, the ESS
at each of the five scheduled steps, and `n_resamplings`.

**Predictions.** The first three follow from the mechanism and are close to
arithmetic; the fourth is the question the run exists to answer.

1. `n_resamplings` goes to 4.00, because uniform weights stop occurring.
2. The ESS falls at every scheduled step, and the rise from 1.18 at $t = 80$ to
   3.11 at $t = 20$ flattens: under $\exp(\lambda S_t)$ with $\lambda = 10$ the
   weights are sharp wherever the particles differ at all.
3. `n_lineages` stays at 1.00 and `div_pix` near its floor of 0.0910. The
   collapse is already total under the current form, so this run cannot make it
   worse, and the diversity findings of runs 13 to 15 are not at stake.
4. `ir_max` rises. If it closes a useful part of the 0.078, the deviation is the
   explanation for the reproduction gap and the table's FK row has to be rebuilt
   on the paper's form. If it moves by less than its own standard error, the
   potential form is not the explanation and the search moves elsewhere.

At 20 prompts the paired standard error on `ir_max` has run about 0.05 to 0.08 on
this design, so this run ranks, it does not settle. Settling is 100 prompts at
three seeds, about 5 h, where the paired standard error falls to roughly 0.026.

## What it costs if it lands

The 100 x 3 confirmatory is 5 h. The fifteen-variant screen was measured under the
current form and would have to be read again, another 5 h or so at 20 minutes a
variant. The heavier item is not GPU: the $\lambda_t$ work of runs 13 to 15 is
built on the increment form, and `docs/protocol_sd.md` derives it as tempering at
constant $\lambda$. The paper's form has intermediate targets
$\pi_t \propto p(x)\exp(\lambda \sum_{s \geq t} S_s)$, so that section needs
re-deriving and not merely re-running. Runs using `difference`, which is runs 5,
6 and most of 8, are unaffected.
