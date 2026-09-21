# The `max` potential: why it underperforms here

The FK row of this reproduction reaches ImageReward **0.8196** where the paper
reports **0.898**, while best-of-4 reproduces, 0.7577 against 0.737. `max` is also
the weakest of the three potentials on CIFAR, where the paper reports it as its
best. This file collects everything measured about that, what was ruled out, and
what is still open. Every constat below is replayed by
`scripts/verify_max_findings.py`, which recomputes it from `results/*.json` and
from `potentials()` itself; a constat that stops replaying is a constat to
rewrite.

Name the referent first, because it is not best-of-N. At equal U-Net rows `fk4`
**beats** `bon4` by +0.0619 +/- 0.0259, 2.4 standard errors, 69 of 100 prompts.
The shortfall is against the paper, and against what `difference` does on CIFAR.

## What `max` computes

At the first scheduled step `gate` is empty, so `prev` is 0 and

    logG = lambda * r_phi(x_t)

This is the only step of the run where particles are ranked on the **level** of
the reward. Every later step gives

    logG = lambda * max(0, r_phi(x_t) - m_{t-1})

a ratchet that pays only for a particle beating its own record. Once the running
maximum saturates, every clone carries the same `m`, the across-particle spread
goes to zero, and the weights come out uniform.

## Six measurements

**1. The ratchet is exactly that.** Calling `potentials()` directly: on an empty
gate it returns `lambda * r`; on a gate already above the incoming reward it
returns zeros for every particle, ESS 4.00 of 4.

**2. A skipped resampling is exactly a step with uniform weights.** Over the 300
`fk4` runs, `4 - (steps with ESS = k) == n_resamplings` in **299 of 300**. Mean
`n_resamplings` 3.753 of 4. Uniform weights occur at t = 80, 60, 40, 20 in 4, 10,
23 and 44 runs. The tolerance matters: an ESS of 3.9997 is not uniform and does
resample, so the accounting only closes under exact equality.

**3. Where `max` never resamples it gives back best-of-k.** CIFAR, classifier
reward, run 8: at lambda = 0.5 it resamples 0.00 times at k = 4, 8 and 16, and
its `r_max` returns best-of-k's own value to the precision of the stored float,
the largest gap over the eleven matching runs being 1.4e-06. Seed-averaged, the
cells are equal at the four decimals the table prints: -2.3716 at k = 4, -0.1080
at k = 8, -0.0200 at k = 16. The potential is not weak in those cells, it is
absent. Run 3 says the same on the red reward, where `max` returns best-of-16's
own 2.184 at every lambda up to 4, while `difference` moves 0.66 to 10.45.

**4. The root it keeps is no better than chance.** Run 17 measured the wrong-root
rate for the first time, on the only file with `root_slots` at scale: the
particle best-of-4 would have chosen is absent from the surviving roots in
**72.7 % of 300 runs, against a chance level of 72 %** given how few roots
survive. `S80`, whose only selection is the early one, reads 50 % against 54 %.

**5. The survivors are near-copies.** Mean spread of ImageReward across the k
final particles is **0.281 for `fk4` against 1.177 for `bon4`**. `div_pix` 0.0910
against 0.3269, and `fk4` ends on one lineage in 20 of 20 prompts (run 14).

**6. The first-step ESS is monotone in lambda**, which `exp(lambda * spread)`
predicts and nothing else does: 2.88, 1.71, 1.18, 1.02 at lambda = 2, 5, 10, 20.

## What was ruled out

**The potential form.** The paper writes `max` as the running maximum itself and
`smc/fk.py` computes its increment. On a one-step epoch the two differ by a
constant that normalisation removes, 236 of 300 runs are provably identical, and
the reward is unchanged across four simulated regimes. `docs/protocol_potentials.md`.

**The schedule.** Every truncation has now been run. `S60` +0.0386 +/- 0.0351
against `fk4` at 100 prompts x 3 seeds, inside its own pre-registered "not
settled" band; `S40` +0.040 +/- 0.059; `S80` -0.027 +/- 0.061, losing 6 of 20;
`D10` +0.028 +/- 0.088 for 58 % more clock. The densest and the sparsest
schedules both land inside a standard error of `fk4`.

**"The early pick is made on noise."** This is the explanation the evidence
refuses. `spearman(first-step ESS, gain over bon4)` is **+0.015** over 300 runs,
and the most collapsed tercile gains the population mean. Three arms have
weakened or isolated that pick: `T1` at lambda_1 = 2 gives +0.034 +/- 0.053, `T2`
at lambda_1 = 0.4, which is almost no early selection at all, is the **worst** arm
at -0.100 +/- 0.092, and `S80`, which keeps only the early pick, loses. Making the
first evaluation gentler does not buy reward.

**Lowering lambda.** It fixes the degeneracy and costs reward: `L2` -0.042,
`L20` -0.074, with `L5` at -0.002 the only cell that is free. Tilt strength and
weight degeneracy are coupled through one scalar.

**The ramps and the tempering placement.** All eight are inside one standard error
of zero on `ir_max` (runs 13 and 15). They buy lineages, not reward.

## What the paper says, and one discrepancy

The paper reports `max` as its best potential on prompt fidelity for every model
it tests, and says in the same breath that it costs diversity because "resampling
at intermediate steps with the max potential favors higher scoring particles more
so than the difference potential".

**But its appendix says the diversity study uses `difference`**
(`sections/appendix_experiments.tex:21`), and that table's ImageReward maxima are
bit-identical to table 1's FK column, which the main text attributes to `max`
(`sections/experiments_new.tex:35`): **0.927** for SD v1.4, **1.006** for v2.1,
**1.298** for SDXL. Three exact matches, and the GenEval side of the same table
matches the `difference` row of the potential ablation for three of four models.
One of those two sentences is wrong and the source does not say which.

Two further facts bound the search. The paper never pairs `max` with a dense
resampling schedule: every `max` run is five interval points over 100 steps, and
its own T = 1000 run uses `difference`. And its adaptive ESS rule is stated only
for the qualitative SDXL appendix, not for any table.

**`difference` has never been run on SD here.** `sample_fk` passes the literal
`"max"` and no flag changes it, so all twenty SD result files carry
`potential: "max"`.

## What cannot be answered yet

Whether the early signal is *wrong*, as opposed to merely sharp, is not decidable
from any file on disk: no run records `r(predict_x0)` at the scheduled steps.
`ir_guide` is not it, being the terminal reward through the ft-mse decoder, which
is why it correlates 0.997 with the final ImageReward. Until the per-step reward
is recorded, the correlation between `r_phi(t)` and `r(x_0)` cannot be computed,
and that correlation is the paper's own `fig:reward-corr`, a figure whose values
appear nowhere in its source.

## Replaying this

    python scripts/verify_max_findings.py

Seven constats, each recomputed from the files or from `potentials()`. It takes a
few seconds and needs no GPU.
