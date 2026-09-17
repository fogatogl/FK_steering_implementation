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

The notebook FID (base 79.2 / fine-tuned 50.7) was measured on the EMA weights; the lambda sweep and figures 0–2 ran on the raw weights (`ckpt["model"]`), a different model (max parameter gap 0.018). Sweeps and samples switch to EMA — the DDPM standard and the only choice comparable to the FID anchors. `results/sweep_lambda.json` and `results/bestofn.json` remain raw-weight results; new JSON files carry an `"ema"` key.
