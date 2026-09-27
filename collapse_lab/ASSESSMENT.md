# Do the corrections resolve the collapse? Protocol, predictions, verdict

*Block proposed for `collapse_lab/README.md` (the author inserts it if they want to):*

    python collapse_lab/r_solutions.py      # the three criteria of "resolved", per arm, bootstrap CI
    python collapse_lab/n_coalescence.py    # the lineages predicted from the weights alone, against the observed ones
    python collapse_lab/p_two_rewards.py    # F4: ir_max and mean ir paired against ctl (ddpm venv)
    python collapse_lab/o_ancestry_fig.py --pid <id> --arms lam0 ctl floor2 R1   # F1, runs of 22/09 and after
    python collapse_lab/q_image_grid.py     # F2 from out/images/ (--choose: the six prompts by rule)
    python collapse_lab/ref/parse_authors.py  # R0: the authors' code against bon4, ctl, table 1
    python scripts/fig_three_scales.py      # F5: CIFAR / CelebA / SD (ddpm venv, samples/*.pt)
    # GPU: collapse_lab/nuit2.sh (R1, R0, B1, bisection, thr05, rise); m_latents.py (finding 11)
    # The authors' code: /home/onyxia/work/fkd_ref/ (clone, outside the repository), sd venv + google.genai shim

A file separate from `FINDINGS.md`, which stays the author's. Here: the criteria fixed before the
data, the timestamped predictions, then what the data returned. Readout:
`python collapse_lab/r_solutions.py` (no GPU). Runs: `collapse_lab/nuit2.sh`.

## Final state (23/09, 07h). Everything after this section is the chronological log.

**In short.** At k = 4 and lambda = 10, no correction returns four lineages: the target
itself carries only 1.2 to 1.5 of the 4 (finding 15), and the number of final roots is
predicted from the weights and the resampler alone, to within 0.05 across fifteen arms, without
knowing anything about the steering. The only correction that goes under 25 % of one-root runs
changes the target (`floor2`, lambda 2: 6 %, 2.9 roots) and returns exactly the diversity that
this target carries. Every correction that keeps lineages at lambda = 10 returns `ir_max` at the
level of best-of-4: FK's gain over best-of-4 is the price of the concentration, and making it
reversible costs that gain. On the reference side, the paper's configuration is the one of this
repository, and the +0.161 of table 1 is produced neither by this repository (+0.056, +0.030 in
two sessions) nor on average by the released code (-0.11 +/- 0.04 over four runs, 220
run-prompts), whose four means at 40 prompts range from -0.35 to +0.10 depending on the random
stream at equal x_T: one run in four reaches the paper's figure, and the spread of their filter
exceeds the effect it reports.

### A. The corrections, paired with `ctl` by prompt (session A, seed 2024)

| arm | n | 1 root | roots | weighted ESS / ceiling | div_pix | `ir_max` - `ctl` | mean `ir` - `ctl` | `ir_max` - `bon4` |
|---|---|---|---|---|---|---|---|---|
| `ctl` (reference) | 40 | 95 % | 1.06 | 1.01 / 1.51 | 0.092 | | | +0.112 +/- 0.091 |
| `late` (without t = 80) | 20; 300 | 100 %; 86 % | 1.00; 1.14 | 1.00 / 1.43 | 0.082 | +0.048 +/- 0.104; **+0.039 +/- 0.033** | +0.081 | +0.187 |
| `adapt` (bisected lambda) | 40 | 60 % | 1.40 | 1.10 / 1.51 | 0.153 | **-0.099 +/- 0.044** | -0.171 | +0.013 |
| `floor` (floor at 0) | 40 | 68 % | 1.73 | 1.64 / 1.51 | 0.150 | -0.091 +/- 0.055 | -0.153 | +0.021 |
| `lam2` | 40 | 35 % | 1.82 | 1.62 / 2.72 | 0.205 | -0.052 +/- 0.103 | -0.138 | +0.060 |
| `fadapt` (floor + bisected) | 40 | 30 % | 2.12 | 1.72 / 1.51 | 0.200 | -0.106 +/- 0.058 | -0.214 | +0.006 |
| **`floor2`** (floor + lambda 2) | 40* | **5 %** | **3.00** | 2.73 / 2.72 | **0.286** | -0.083 +/- 0.093 | -0.334 | +0.029 |
| `thr05` (floor, ESS < k/2) | 40 | 62 % | 2.00 | 1.66 / 1.51 | 0.172 | -0.058 +/- 0.100 | -0.181 | +0.054 |
| `rise` (floor, bisected, lam_max 100) | 20 | 35 % | 1.95 | 1.63 / 1.43 | 0.170 | +0.007 +/- 0.109 | -0.135 | +0.146 |
| `lam0` (free) | 17 | 0 % | 4.00 | 4.00 / 4.00 | 0.338 | -0.070 +/- 0.082 | **-0.557** | +0.034 |

\* `floor2`: figures at n = 40 read on 22/09 at 22h, before the incident that lost six records;
at n = 34 the current output of `r_solutions.py` is biased (+0.009 on `ir_max`). Mean of the
seven corrections against `ctl` on `ir_max`: -0.069 [-0.176, +0.041]; on the first five,
before the incident: -0.106 [-0.193, -0.019]. The price in mean `ir` of the four follows the
roots kept, from -0.14 to -0.56.

### B. The implementation choices of the released code, one by one in `smc/` (40 prompts, paired with `ctl`)

| arm | choice | roots | `ir_max` - `ctl` | root kept = `ctl` |
|---|---|---|---|---|
| `stat0` | floor at 0 + statistical form | 1.75 | -0.050 +/- 0.096 | 25 % |
| `multi` | multinomial at every scheduled step | 1.00 | -0.061 +/- 0.081 | 38 % |
| `vae` | guide decoded by the pipeline's VAE | 1.05 | +0.009 +/- 0.083 | **30 %** |
| `idx` | indices {20, 40, 60, 80, 99} | 1.00 | -0.051 +/- 0.085 | 22 % |
| `R1` | the four together (100 prompts) | 1.19 | -0.067 +/- 0.055 | |

No single choice is worth more than 0.06; the four together are worth `ctl`'s lead over
best-of-4 (`R1 - bon4` = -0.011 +/- 0.041). The guide's VAE, neutral on the reward, changes the
root kept in 70 % of the prompts: the selection hangs on the rounding of a decoder.

### C. The reference: the released code, table 1 configuration, SD v1.5

| run | code | seed | n | `ir_max` | against `bon4` | pairing |
|---|---|---|---|---|---|---|
| table 1 of the paper | | | | 0.898 | **+0.161** | |
| `ctl` | `smc/` | 2024 | 100 | 0.826 | +0.056 +/- 0.052 | x_T |
| `R1` | `smc/` + their 4 choices | 2024 | 100 | 0.756 | -0.011 +/- 0.041 | x_T |
| `R0g24` | their code | 2024, generator | 40 | 0.720 | **-0.126 +/- 0.070** | x_T and DDIM noise |
| `R0g` | their code | 42, generator | 40 | 0.807 | -0.039 +/- 0.073 (prompt); **-0.018 +/- 0.074** against their free sampler | noise (seed 42) |
| `R0` | their code | 42, global RNG | 100 | 0.554 | -0.216 +/- 0.061 (prompt); **-0.328 +/- 0.083** against their free sampler | prompt |
| `R0` seed 2024 | their code | 2024, global RNG (= our x_T) | 40 | **0.949** | **+0.103 +/- 0.056** | x_T |
| their pipeline without FK | their code | 2024, generator | 5 | | **0.0000** gap over 20 slots | bit |
| **the four runs of their FK, grouped** | their code | | 220 | 0.702 | **-0.110 +/- 0.036** | |

On the same 40 prompts, the four runs of their FK return against `bon4`: **-0.35, -0.04,
-0.13, +0.10** (+/- 0.06 to 0.09 each). Two of them share x_T and DDIM noise and differ only
by the stream of the multinomial draw (-0.13 and +0.10); the standard deviation of `ir_max`
between the four runs of one prompt has a median of 0.25. This repository, on the same 40
prompts and two sessions: +0.11 and +0.08. The paper's +0.161 is within the range of what
their code returns from one stream to another, and 0.25 above its mean.

Their base sampler is bit-compatible with the diffusers pipeline; their ImageReward scorer
returns the same values as the official one to the third decimal; their FK loop does what it
writes (floor, flat weights, multinomial drift, late selection); with identical choices and noise
it returned 0.56 per particle against 0.82 for `smc/` in one run (`R0g24`) and +0.10 over
best-of-4 in another (seed 2024, global RNG): this is not "steers less well", it is a spread
between runs that the reading of the code (23/09, no reseeding, no bias between seed paths)
does not reduce. Pre-registered rule: gap to the paper **bounded**, not closed.

*Trace of the two loops on prompt 0, same x_T (23/09, 07h45).* At the first scheduled step
both guides see four negative rewards, apply the floor, draw flat: same behaviour. On the same
four particles, the guide's reward is -1.78 with the pipeline's VAE (`vae`), -0.56 with
sd-vae-ft-mse (`ctl_b1`), -0.21 in their code one step later: **the decoder alone moves the
reward of a particle by 1.2 at the first step**. This is the most direct version of finding 3:
at t = 80 the guide reads decoder noise, and the `vae` arm keeps a different root from `ctl` in
70 % of the prompts. The 0.23 between their loop and `smc/` is not in the guide of the first
step; it remains in the draws and their terminal duplication, a variance to measure rather than
a defect to find.

### D. The mechanism, in three lines

1. *Weights.* At each scheduled step the weight is `exp(lambda r_phi)` up to a shared factor
   (findings 4, 8); at lambda = 10 the range of the first step is 9 nats on noise.
2. *Paths.* The number of final roots is a function of the recorded weights and of the resampler
   (`n_coalescence.py`, fifteen arms to within 0.05, per-run correlation 0.87 to 0.99); the ESS
   of one step does not measure it (`adapt`: ESS 2.0, 1.4 roots). With flat weights the comb is
   the identity, the multinomial is not (1.58 roots out of 4 after four flat steps).
3. *Target.* At lambda = 10, four free draws reweighted by `exp(10 ir)` have an ESS of 1.2 to
   1.5 (finding 15): keeping more roots means returning particles the target crushes
   (`floor`, `fadapt`, `thr05`, `rise`: weighted ESS above the ceiling).

### E. What changed in the reading of the findings

- Finding 11, third reading: x_T is shared to the bit and the trajectories stay correlated at
  0.98 or more; but the `smc.models` path does not replay from one session to another (the
  diffusers pipeline does) and the rounding gap changes the root kept. Two cases to tell apart.
  The free path (`lam0`) returned `bon4` score for score in session C and not in session
  B: at lambda = 0 no reward is read, so neither the model cache (`HF_HOME` differed
  between the two sessions, same revisions) nor the scorer can explain it; same code,
  same weights, same seed, another process, cause **not identified**. The path with
  resampling (`ctl`) has a candidate: the guide's reward stack (VAE ft-mse,
  ImageReward, BERT tokenizer) loaded from another cache path, where 1e-3 on one
  reward is enough to move one tooth of the comb; not tested in isolation. Slot-by-slot pairing
  between files from different days measures nothing. **Session C (H) recomputes them on a
  valid pairing**: tau +0.14, A - B +0.31, B - M +0.23. Findings 3 and 5 bis hold,
  reworded: a little information at the first step, a root kept a little better than
  chance, 0.31 of root lost, 0.33 returned by the steering.
- Finding 14: `thr05` (resampling less) returns 2.0 roots, not 2.4-2.8; `late` repairs
  nothing; `rise` does not beat `ctl`. The "threshold under the target" lead is importance sampling
  (settled without GPU). The ranking of the corrections holds: `floor2` > `fadapt` > `thr05` ~
  `floor` ~ `lam2` > `adapt`.

*Session C, partial reading at 33 prompts (23/09, 09h).* `lam0` of session C returns the
rewards of `bon4` (20/09) **slot by slot at correlation 1.00** (`ir_max` +0.011 +/- 0.009);
`ctl` of session C does not return the `ctl` of 21/09 (per-slot correlation 0.65, same root
kept in 32 % of the prompts). The free path of `smc.models` therefore replays from one session
to another; it is the path **with resampling** that does not, and the suspect moves to the
guide's rewards (VAE decoding and ImageReward in fp16, kernels chosen at run time), where a
gap of 1e-3 is enough to tip a four-slot comb. In session C, `ctl`, `lam0` and `floor2` share
the process: the slot-by-slot pairing holds there, and findings 3 and 5 bis are recomputed
(`t_sessionC.py`); figures at 100 prompts in the next section when the session is finished.

### H. Session C: `ctl`, `lam0`, `floor2` at 100 prompts, a single process (23/09, 12h30)

`out/probe_C.json`, readout `t_sessionC.py`. Predictions from `protocol_sd.md` ("Pre-registration
of session C") alongside.

| measure | value | prediction | outcome |
|---|---|---|---|
| `lam0_C` against `bon4` (20/09), per slot | correlation **1.00**, `ir_max` +0.009 +/- 0.006 | correlation under 0.7 | **missed**, in the right direction: the free path replays |
| `ctl_C` against `ctl` of 21/09, per slot | correlation 0.70, same root 30 %, `ir_max` -0.026 +/- 0.058 | mean within 0.05, decorrelated slots | held |
| `floor2 - ctl`, `ir_max`, x_T paired | **-0.012 [-0.082, +0.060]** | [-0.14, -0.02] | missed from above: the price is smaller than predicted |
| `floor2 - ctl`, mean `ir` of the 4 | -0.250 +/- 0.045 | | |
| `floor2` roots; one root | **3.03; 4 %** | 2.8-3.1; 5-10 % | held |
| `ctl` roots; one root | 1.07; 93 % | 1.0-1.1 | held |
| `ctl - bon4` in this session | +0.030 +/- 0.037 | | (+0.056 in the session of 21/09) |
| finding 3, Kendall tau r_phi(t = 80) against free ir of the same root | **+0.137 +/- 0.050**; top-1 35 % | under 0.15 | held, narrowly |
| finding 5 bis, A - B | **+0.313 +/- 0.042** | [0.25, 0.50] | held |

The decomposition, on the only valid slot-by-slot pairing (`lam0` = the free continuation of
each root, same process): A best free root **0.779**, M mean root
**0.233**, B the root that `ctl` keeps, read free **0.466**, C what `ctl` draws from it **0.799**.
B - M = +0.233 +/- 0.047: the root kept is worth more than a random draw, by a third of the
way to the best (mean rank 2.04 out of 4). A - B = +0.313: the collapse still costs
0.31 of root. C - B = +0.333: the steering returns a little more than that. C - A = +0.021: the
balance over best-of-4 is what the steering adds minus what the early choice loses, and it
is small. Finding 3 is reworded: the first step carries **a little** information (tau
0.14, top-1 35 % against 25 %), not none; finding 5 bis holds in its figures, A - B
0.31 against 0.41 in the cross-session reading of 22/09.

What session C changes in the reading: (1) the free path of `smc.models` replays from one
session to another to the bit of the score, it is the path with resampling that does not,
and the suspect is the guide's reward (VAE + ImageReward in fp16), where 1e-3 is enough to
tip a comb; (2) `floor2` costs less than predicted when it is paired by x_T
(-0.01 instead of -0.08 to -0.10 under pairing by prompt): the price of the lineages on `ir_max`
is of the order of `ctl`'s lead over best-of-4 (0.03 to 0.06), no more; (3) the price on
mean `ir` remains (-0.25).

### F. Draft findings for `FINDINGS.md` (for the author to rework)

**16. The reference.** The paper's configuration (max, [0, 20, 40, 60, 80], lambda 10, k 4, DDIM
eta 1, 100 steps, CFG 7.5) is the one of this repository; the released script's defaults (`diff`, 5-30-5)
are another configuration, which the paper's appendix notes further down. The released code under the
paper's configuration returns, on the 100 benchmark prompts and SD v1.5, 0.554 (its seed) and
0.720 (our x_T, 40 prompts), against 0.770 for best-of-4 and 0.826 for this repository; `smc/` with
their four implementation choices returns 0.756. The +0.161 appears in no reading; the gap
is bounded, it is not in the implementation. Source: `docs/reference_config.md`,
`results/sd_authors_R0.json`, `collapse_lab/ref/`.

**17. Coalescence reads on the weights.** Replaying the comb (or the multinomial) on the weights
recorded at each step, integrating over `u`, predicts the number of final roots of fifteen
arms to within 0.05 and the share of one-root runs to within three points (`n_coalescence.py`). The
steering does not enter the prediction. `adapt` holds the ESS at 2 and loses 2.6 roots: the ESS is
the degeneracy of the weights, not that of the paths. With flat weights the authors' multinomial
loses roots through the draw alone (4 -> 1.58 in four steps).

**18. Resampling less does not keep the lineages.** `thr05` (floor, ESS < k/2):
0.97 resampling per run, 2.0 roots, 62 % with one root, predicted at 1.84 / 61 % / 1.12
by finding 17 before the measurement. When the threshold fires, the accumulated weights are peaked and
a single pass takes almost everything. `ir_max` -0.06 +/- 0.10 against `ctl`.

**19. Replayability.** The diffusers pipeline today returns the rewards of `bon4` (20/09) to the
third decimal; the `smc.models.StableDiffusion` path, same weights and same code, returns on
21/09 and on the morning of 22/09 the same figures, and on the evening of 22/09 other roots and other
`ir_max` (up to 1.6 apart on one prompt). Deterministic within a session, not between sessions;
cause not identified. A consequence for every slot-by-slot pairing.

### G. What to do next, in order

**GPU, one night (session C, ~5 h, T4).** `ctl`, `lam0`, `floor2` at 100 prompts in a single
process (`probe.py --arms ctl lam0 floor2 --limit 100 --redo --out out/probe_C.json`, with
`HF_HOME=/home/onyxia/work/hf_cache`). What it returns: a valid x_T pairing for
findings 3, 5, 5 bis; the leading correction (`floor2`) at n = 100 on the same x_T as its
reference, without the six lost prompts; and a `ctl` from the session of the arms `R1`, bisection,
`thr05`, `rise`, which today are paired with `ctl` by prompt only. Prediction to
write beforehand: `floor2 - ctl` on `ir_max` in [-0.14, -0.02], roots 2.8 to 3.1, 5 to 10 % with
one root; `lam0`: four roots, `ir_max` within 0.05 of `bon4` on average and not
paired slot by slot with it.

**GPU, ten minutes, before the night.** The 0.23 between their loop and `smc/` with identical choices
and noise (`R1 - R0g24`): one prompt, trace the two loops side by side at the first scheduled
step (raw rewards, weights, drawn indices) with the same x_T; `ref/diag_authors.py` already does
half of it. If the rewards of step 20 coincide and only the draw differs, the difference
is the RNG of the multinomial (global in their code, generator here) and it is variance; if the
rewards differ, it is the decoding of the guide and it is one more reproduction fact.

**GPU, forty minutes, optional.** `R0` at seed 2024 under their seed path (global
RNG), 40 prompts: if the result joins `R0g24` (0.72) and not `R0` (0.55), the 0.31 between
the two paths was specific to seed 42 and can be said in one sentence.

**CPU / writing.** (1) Insert findings 16-19 into `FINDINGS.md` and the proposed block into
`collapse_lab/README.md`. (2) Section 2 of the post from `reference_config.md` and table C
above: "bounded", with the three readings and their pairing. (3) Extended F0
(`scripts/plot_fig4_sd.py` with `R0`, `R0g24`, `R1`). (4) F3 is drawn
(`out/fig_coalescence.png`, `s_coalescence_fig.py`); F1, F2, F4, F5 exist. (5) A5 (target
ESS against lambda) and A6 (invariance table) remain to be put in shape, can be cut.
(6) Machine: `HF_HOME` in the jobs' shell, delete `~/.cache/huggingface` (2.4 GB of
duplicates), take the token out of the git remote. (7) `smc/`: the floor as a `reward_floor` parameter
after the submission, not before.

**What not to do.** Trying to "resolve" the collapse at lambda = 10 and k = 4: the target
forbids it, and the arms that get there return particles it crushes. The post says it
as a fact about the target, not as a failure of the corrections.

---

## What is tested

Finding 14 of `FINDINGS.md` ranks five measured corrections (`floor2` > `fadapt` >
`floor`, `lam2` > `adapt`) and leaves three unmeasured. The ranking rests on
20 prompts for `floor2` and `lam2`, 40 for the others. And finding 15 changes the question:
at lambda = 10 the target `p(x0) exp(lambda r)` restricted to four free draws has a median
ESS of 1.23. Four particles cannot carry diversity at this lambda, whatever
the kernel does. "Resolved" must therefore be read layer by layer.

## Criteria, fixed on 22/09 at 18h30, before the first run of `nuit2.sh`

Everything is paired by prompt, on the intersection of the `prompt_id` values, against `ctl` (the
reference FK line, identical to `sd_ref_fields100.json`) and against `bon4`.

1. **The mechanism (layer 2).** The collapse no longer happens at the non-informative step.
   - share of the prompts that end on **a single root** x_T; `ctl`: 95 %. Proposed
     success: **under 25 %**.
   - lineages after t = 80 and t = 60, aligned on the value of t (the `late` arm has no
     t = 80).
   - **weighted root ESS** at the end: final weights reconstructed (at threshold 1.0,
     `logW` restarts from zero at each step where ESS < k), summed by root, `1 / sum W_r^2`.
     This is the diversity the sampler weighs, not the one it returns.
2. **The ceiling of the target (layer 1).** Weighted root ESS against the ceiling of
   `l_target_ess.py` at the arm's lambda: **2.83** at lambda = 2, **1.23** at lambda = 10.
   An arm at lambda = 10 that returns more roots than 1.23 returns particles its
   target crushes. The repository's evaluation (`ir_max`, `div_pix` on the four images) ignores
   the final weights: a legitimate choice for "four candidates", not a proof that the
   sampler is correct for its target. The two readings side by side.
3. **The price.** `ir_max`, mean `ir`, weighted `ir`, paired against `ctl` and against
   `bon4`, bootstrap CI over the prompts. The question: does the correction cost all
   of FK's lead over best-of-4 (+0.11 +/- 0.09 at n = 40)?

## Settled without GPU (written before, checked by reading `smc/fk.py`)

- **Resampling threshold under the ESS target** (finding 14, lead 2). If the
  bisected target is above the threshold, no scheduled step resamples. `logW` is
  never reset to zero and equals `acc`; at the terminal step the branch `last and acc is not
  None` of `_potential_terms` returns `lam * r_0 - acc`; the final weights are
  `exp(lam * r_0)` over four free trajectories. This is importance sampling on
  `bon4`, exactly what `l_target_ess.py` measures: ESS 1.23 at lambda = 10. The arm
  "resolves" the collapse by no longer doing SMC. Settled, not to run.
- **`lam_max = 100`** (lead 3). The table of finding 8 puts the bisected lambda above
  10 at the single step t = 20 (11.9 at the median). The arm would differ from `adapt` at one step out of
  four, the one where the collapse is already done. Discarded.
- **`S60` is not a control for `late`.** The file is dated 20/09 13:42, before
  5025180; against the current reference it is worth +0.048 +/- 0.104 on 20 prompts. `late`
  goes through the probe.

## Predictions, written on 22/09 at 18h30, before the data

What `probe.json` already says, at 20 prompts, and which serves as the base:

| arm | n | 1 root | lineages | weighted root ESS | div_pix | ir_max | mean ir |
|---|---|---|---|---|---|---|---|
| `ctl` | 40 | 95 % | 1.05 | 1.01 | 0.092 | +0.958 | +0.824 |
| `adapt` | 40 | 60 % | 1.40 | 1.10 | 0.153 | +0.859 | +0.653 |
| `floor` | 40 | 68 % | 1.73 | 1.64 | 0.150 | +0.867 | +0.671 |
| `lam2` | 20 | 35 % | 1.85 | 1.67 | 0.215 | +0.829 | +0.621 |
| `fadapt` | 40 | 30 % | 2.12 | 1.72 | 0.200 | +0.852 | +0.610 |
| `floor2` | 20 | 5 % | 2.85 | 2.59 | 0.263 | +0.862 | +0.517 |
| `lam0` | 20 | 0 % | 4.00 | 4.00 | 0.335 | +0.863 | +0.287 |

1. `floor2` at 40 prompts: under 10 % of prompts with one root, 2.6 to 2.9 lineages, weighted
   root ESS between 2.4 and 2.8, that is at the ceiling of its target (2.83). `ir_max`
   stays at -0.10 +/- 0.06 from `ctl` and within 0.05 of `bon4`.
2. `lam2` at 40 prompts: 30 to 45 % with one root, 1.7 to 2.0 lineages. `floor2 - lam2`
   keeps about +1.0 lineage at equal `ir_max`.
3. `late`: between `ctl` and `floor`. Removing t = 80 amounts to making this step inert in
   100 % of the runs instead of 90 %, but without a floor at t = 60 (where the reward is still
   negative everywhere in 40 % of the runs) the collapse resumes one step earlier than for
   `floor`: **1.4 to 1.8 lineages, 40 to 70 % with one root**, `S80` (1.85 with a single step)
   as upper bound. Most exposed prediction: `ir_max` **indistinguishable from `ctl`**
   (less than one standard error), because t = 80 carries no information (finding 3) and
   the steering keeps its three other steps. If it holds, `late` dominates `floor2` on
   the price and the ranking of finding 14 changes; if `late` also pays -0.10, the flat
   price of finding 13 is confirmed on an arm that touches neither lambda nor potential.
4. Expected reading as a whole: no arm at lambda = 10 goes under 25 % of prompts
   with one root, because the target forbids it; the only one that gets there changes the target
   (lambda = 2). "Resolving the collapse" at lambda = 10 with k = 4 is not a
   sampler problem.

## Partial verdict, 22/09 19h45: `floor2` and `lam2` at 40 prompts, `late` at 20 and at 300

Output of `r_solutions.py` on `probe.json` (40 `ctl`, `floor`, `adapt`, `fadapt`, `floor2`,
`lam2`; 20 `lam0`, `late`). Controls: `ctl` identical to `sd_ref_fields100.json` (0.0000),
ESS recomputed to 1.2e-4, final weights reconstructed to 1.0e-4.

| arm | n | 1 root | lineages | weighted ESS | target ceiling | div_pix | ir_max - ctl | mean ir - ctl |
|---|---|---|---|---|---|---|---|---|
| `ctl` | 40 | 95 % | 1.05 | 1.01 | 1.51 | 0.092 | | |
| `late` | 20 | 100 % | 1.00 | 1.00 | 1.40 | 0.087 | +0.03 [-0.24, +0.31] | |
| `adapt` | 40 | 60 % | 1.40 | 1.10 | 1.51 | 0.153 | -0.10 [-0.19, -0.02] | -0.17 |
| `floor` | 40 | 68 % | 1.73 | 1.64 | 1.51 | 0.150 | -0.09 [-0.20, +0.01] | -0.15 |
| `lam2` | 40 | 35 % | 1.82 | 1.62 | 2.72 | 0.205 | -0.05 [-0.19, +0.08] | |
| `fadapt` | 40 | 30 % | 2.12 | 1.72 | 1.51 | 0.200 | -0.11 [-0.22, +0.01] | -0.21 |
| `floor2` | 40 | **5 %** | 3.00 | 2.73 | 2.72 | 0.286 | -0.08 [-0.23, +0.06] | -0.32 |
| `lam0` | 20 | 0 % | 4.00 | 4.00 | 4.00 | 0.335 | -0.10 [-0.23, +0.04] | -0.53 |

Predictions of 22/09 18h30, set against the data:

1. `floor2` at 40: **held**. 5 % with one root (predicted under 10 %), 3.00 lineages (predicted
   2.6-2.9, just above), weighted ESS 2.73 for a ceiling of 2.72: the arm returns
   exactly the diversity its target carries (ratio 1.01). `ir_max` -0.08 against `ctl`
   (predicted -0.10 +/- 0.06), +0.03 against `bon4` (predicted under 0.05).
2. `lam2` at 40: **held**. 35 % with one root (predicted 30-45), 1.82 lineages (predicted 1.7-2.0).
   `floor2 - lam2`: +1.18 lineage [+0.85, +1.48] at equal `ir_max` (-0.03 [-0.17, +0.12]).
3. `late`: **missed on the lineages, held on the price**. 1.00 lineage at 20 prompts and 1.14 at
   300 (`sd_s60_full.json`), 86-100 % with one root, where 1.4-1.8 were predicted: removing
   t = 80 only postpones the first resampling to t = 60, where the median ESS is
   1.4 and the comb kills as many. `ir_max` +0.03 against `ctl` at 20 prompts, +0.04 +/- 0.03 at
   300: the only arm that does not pay, and it does one reward evaluation fewer (but
   87 s per run that night against 62, `results.md` block 17: the saving is in evaluations,
   not in measured wall-clock time).
4. Reading as a whole: **held**. No arm at lambda = 10 goes under 25 % of prompts with
   one root; the only one that does changes the target (lambda = 2).

The flat price of finding 13, in one figure: mean of `ir_max` over the six
correction arms against `ctl`, n = 40, **-0.078 [-0.177, +0.024]**; without `late`, -0.106 [-0.193,
-0.019]. The price is real for any arm that returns lineages; `late` returns none and does not
pay.

## Draft finding: coalescence reads on the weights alone (22/09, 20h, `n_coalescence.py`)

For each run, the genealogy is replayed from the weights recorded at each scheduled step,
without knowing anything about the steering: the distribution of the vector of roots propagates step by step,
exactly, under the systematic comb integrated over `u` (the distinct assignments are finite
in number) and under the authors' multinomial at each step (4^4 assignments). Controls:
first step alone of `ctl` on the 20 prompts of `fk4_stat` = **1.835**, the figure of
`b_comb.py`; `lam0` = 4.000.

| arm | n | observed | predicted (comb) | predicted (multinomial at each step) | P(1 root) obs / predicted | resamp. |
|---|---|---|---|---|---|---|
| `ctl` | 40 | 1.05 | 1.07 | 1.03 | 95 % / 93 % | 3.75 |
| `late` | 20 | 1.00 | 1.07 | 1.06 | 100 % / 93 % | 2.85 |
| `adapt` | 40 | 1.40 | 1.41 | 1.16 | 60 % / 60 % | 3.88 |
| `floor` | 40 | 1.73 | 1.73 | 1.19 | 68 % / 67 % | 2.00 |
| `lam2` | 40 | 1.82 | 1.79 | 1.30 | 35 % / 37 % | 3.95 |
| `fadapt` | 40 | 2.12 | 2.16 | 1.34 | 30 % / 30 % | 2.02 |
| `floor2` | 40 | 3.00 | 2.97 | 1.50 | 5 % / 3 % | 1.93 |
| `lam0` | 20 | 4.00 | 4.00 | 1.58 | 0 % / 0 % | 0.00 |

The plan's prediction (+/- 0.15, right order) is **held** with margin: the largest gap
is 0.07 (`late`), the seven arms are in order, and the share of one-root runs is
predicted to within three points. Run by run, the correlation between the predicted expectation and
the observed value is 0.97 for `ctl`, 0.99 for `floor`, 0.95 for `fadapt`, 0.84 for `floor2`.
The number of lineages is therefore a function of the weights and of the number of resamplings,
not of the steering: it is the degeneracy of the paths, and the ESS of one step does not measure it
(`adapt` holds the ESS at 2 and loses 2.6 roots out of 4).

The fine point: with uniform weights the comb is the identity, the multinomial is not.
Four steps with flat weights under multinomial leave 1.58 roots out of 4 (`lam0`, multinomial
column): this is the neutral drift of population genetics, and the authors' code
undergoes it at every inert step their floor produces. On the weights of `ctl`, their
resampler would leave 1.03 roots against 1.07 with the comb: their repository collapses at least
as much, floor aside.

Pre-registered prediction for `thr05` (`docs/protocol_sd.md`, 19h55): 1.84 roots, 61 %
with one root, 1.12 resamplings, from the weights of `floor` and the rule ESS < k/2.
Far from the plan's 2.4-2.8: with the floor the first steps are inert, the accumulated weights
stay flat, and when the threshold finally fires, they are peaked and a single
pass of the comb takes almost everything.

## Night 2, the reference: R1 returned (22/09, 21h45), R0 running

**R1, `smc/fk.py` with the four choices of the released code** (floor at 0, statistical form,
multinomial at each scheduled step, guide decoded by the pipeline's VAE, indices
{20, 40, 60, 80, 99}), 100 prompts, seed 2024, paired:

| | prediction (19h35) | measurement | |
|---|---|---|---|
| `ir_max` R1 - `ctl` | -0.08 +/- 0.05 | **-0.067 +/- 0.055** (42/100 won) | held |
| `ir_max` R1 - `bon4` | | **-0.011 +/- 0.041** | |
| lineages | 1.0 to 1.3 | **1.19**, one root in 81 % | held |
| first step inert | ~90 % | 81 % | held, a little less |
| resamplings | 4 at each run | 4.00 | held |
| mean `ir` of the 4 | | 0.609 against 0.681 for `ctl` | |

Pre-registered rule: the gap to the paper is **bounded**, not closed (R1 - `bon4` = -0.01, threshold
+0.12). Reading: the four implementation choices of the released code, carried by `smc/`, return
exactly the level of best-of-4 on these prompts; `ctl` (the choices of this repository) keeps
+0.056 above. The floor and the multinomial draw with flat weights (the neutral drift of the
coalescence finding: 4 -> 3 -> 2 roots before any information, visible on F1) cost
the lead that the steering buys. Nothing here brings us closer to the published +0.161; everything moves away from it.

**R0, the authors' code, paper configuration**, 100 prompts, seed 42 (23h15):
mean `ir_max` **0.554**; against `bon4` **-0.216 +/- 0.061** (41/100 won), against `ctl`
-0.272 +/- 0.067, against `R1` -0.206 +/- 0.072, against a free draw `k1` +0.330. Mean
of the four finals 0.414; 3.81 distinct images out of 4 (the terminal step duplicates some in 8 %
of the runs); div_pix 0.104; 61 s per run. The prediction 0.77 [0.72, 0.82] is **missed from
below by 0.2**: their code, under the configuration of their table 1, returns here a little better than
best-of-2 and clearly less than best-of-4.

The diagnosis (`ref/diag_authors.py`, prompts 0 and 2, `out/diag_authors_*.json`) closes
reading (b): their `score_batched` returns the same values as the official ImageReward to the
third decimal, on the decoded x0 as on the finals. And it shows mechanism (a) as
written: at index 20 the four rewards are -2.2, floored to 0, flat weights, the multinomial
draw keeps [1, 0, 0, 3] (one root lost by pure chance); on prompt 2 the rewards
stay negative until index 60 and three flat draws in a row leave only two
roots before the first information; at index 80 the selection decides (ESS 1.05 on
prompt 0). Their FK is neutral drift followed by a late selection.

What remains to be settled, and is pre-registered (`protocol_sd.md`, 23h30): `R1` carries
the same choices and returns 0.756, that is +0.21 above `R0`. The only remaining difference
is the noise stream (seed 42 through the global RNG in their code, `seed_effective` by generator
here). `R0g` = their pipeline with our generator, 40 prompts, paired with `bon4`/`ctl`/`R1` by
x_T; running in `nuit2d.sh`. Prediction: if the two codes are equivalent, `R0g - R1`
within +/- 0.05; otherwise, their pipeline differs from `smc/` on something the table does not
list.

**The latents test (finding 11)**, two prompts (`out/latents_*.json`): x_T identical to the
bit between the pipeline and `initial_state`; per-slot correlation between the pipeline's latents
and those of our model: 1.0000 at indices 0, 1, 10, 50, and **0.984 to 0.9999 at index
99**. The prediction "chaotic divergence" is **missed**: the trajectories stay the
same to within 1 or 2 %. Yet `lam0` and `bon4` differ slot by slot (correlation 0.64
over 80 slots, 0.89 once sorted within the prompt), and on prompt 0 the largest
ImageReward gap (0.05 against 0.77) falls on the least correlated slot (0.984). Two readings,
pre-registered at 23h45: the `bon4` file of 20/09 does not replay today, or
ImageReward moves by 0.7 for an fp16 gap on the latents. `m_latents.py` now decodes and scores
both finals next to the recorded value; it runs again in `nuit2d.sh`.
In both cases, `lam0` is the free baseline paired by construction, and findings 3,
5 and 5 bis are recomputed against it.

*Replayed on 23/09 about 03h15 with ImageReward on the finals.* (a) **Held**: today's
pipeline returns the recorded rewards of `bon4` to the third decimal on both
prompts; the file of 20/09 replays. (b) **Missed in its strong form**: the finals
of our bare model differ from those of the pipeline by 0.17 to 0.34 (the slot at 0.984
correlation goes from 0.048 to -0.292), not by 0.7. And they also differ from the `lam0` records
of the probe (prompt 0: 1.11 / -0.29 / 0.69 / 0.87 today, 1.01 / 0.77 / 1.10 / 0.68
recorded): the `lam0` arm departs from the bare model somewhere, and the replay of `nuit2e.sh`
on this prompt says first whether the probe is deterministic. As long as this point is not cleared, neither
`bon4` nor `lam0` serves as a slot-by-slot pairing; findings 3, 5 and 5 bis stay
suspended.

*Replay of the six prompts of the grid, 23/09 about 04h00 (`--redo`).* The replayed `lam0` returns
**exactly** the finals of the bare model of `m_latents.py` (1.1131 / -0.2924 / 0.6885 / 0.871)
and the replayed `R1` returns exactly the `R1` of the night: the probe is deterministic, and the
`lam0` arm is indeed the bare model. But the replayed `ctl` **does not return** the `ctl` of 21/09
(`sd_ref_fields100.json`, which the probe of the morning of 22/09 reproduced to 0.0000), and the replayed `lam0`
does not return the `lam0` of the morning of 22/09. The diffusers pipeline, for its part, returns today
the `bon4` of 20/09 to the third decimal. So: deterministic within a session, not from one
session to another for the `smc.models.StableDiffusion` path (the fp16 kernels chosen at
run time are the suspect; nothing in `smc/` changed), and the rounding gap, amplified over
100 steps at eta = 1, changes the root kept (4 prompts out of 5) and `ir_max` by -0.4 to -1.6 on
three of the five replayed prompts (`007191-0058`: 0.21 tonight against 1.83 on 21/09): this is
not measurement noise, it is another draw. Consequence: a
**slot-by-slot** pairing between two files written at different times measures nothing
(finding 11, resolved by a third reading); the comparisons **by prompt** between FK arms
stay valid on average, the choice of root being close to chance anyway. To
keep findings 3, 5 and 5 bis, `ctl` and `lam0` are needed in the **same session**: 200 runs,
about 3 h 20 of T4, for the author to decide.

*Incident and correction, 23/09 about 04h20.* The `--redo` replay of the six prompts of the grid had
**replaced** in `probe.json` the `ctl`, `floor2` and `lam0` records of session A (21-22/09
morning) with those of session B (22-23/09 night), lower by 0.4 to 1.6 on three prompts: all
the "- ctl" columns of `r_solutions.py` had slipped by +0.08. The 18 records of session
B are renamed `ctl_b1`, `floor2_b1`, `lam0_b1` (they serve figures F1 and F2, with `R1`
which is from the same session); the six session A records of `floor2` and `lam0` are lost
(`floor2` goes to 34 prompts, `lam0` to 17); the six of `ctl` are **restored** from
`sd_ref_fields100.json` (identical to session A on `ir`, ESS, roots) without `logG` or
`r_phi` or `lineages_trace`, marked `partial_from`, and the scripts read the weights only on
the complete records. The six prompts are not arbitrary (chosen by rule as medians,
largest `ctl - bon4` gaps, best `floor2`): removing them would have biased all the
"- ctl" columns by about +0.05, hence the restoration. Backup of the contaminated file:
`out/probe_backup_2309_0630.json`. `floor2` now has only 34 prompts, and the six missing ones
are not arbitrary: recomputed on 34, `floor2 - ctl` goes from -0.083 to +0.009. **The
`floor2` figures at n = 40 of the partial verdict above, computed before the incident, remain
the ones to cite**; the current output of `r_solutions.py` for `floor2` is biased, and this is said here.
The times noted in this file and in `protocol_sd.md` are approximate; the order is
that of the `out/nuit2*.log` logs.

## The reference, continued (23/09, about 03h50): `R0g`, and their pipeline without FK

`R0g` = their code, paper configuration, **our generator**, 40 prompts: `ir_max`
**0.807**; against `R1` **-0.147 +/- 0.092**; against `bon4` **-0.039 +/- 0.073**; against
`ctl` -0.152 +/- 0.072; against `R0` (their seed path, same prompts) **+0.310 +/-
0.077**. The prediction "equivalent codes" (R0g - R1 within +/- 0.05) is **missed**; the one
on `bon4` is held. And the 0.31 between the two seed paths of the same code exceeds anything
the seeds do to best-of-4 (less than 0.02).

Their pipeline **without FK**, our generator, 5 prompts (`authors_free_g`): first read as
"does not return `bon4` slot by slot" (gaps up to 2.69). **My mistake**: these runs
were at seed 42000 + i (the driver's default), not 2024000 + i. Replayed with `--seed 2024`,
their pipeline without FK returns the four rewards of `bon4` **to the fourth decimal on the five
prompts** (20 slots at 0.0000): their base sampler is bit-compatible with the diffusers pipeline
it copies. Consequence: `R0g` (seed 42) pairs with `bon4`, `ctl`, `R1` only
by prompt; `R0g - authors_free_g` is paired by noise (same seed 42); an `R0g24`
(seed 2024, 40 prompts) is running for the comparison paired by x_T, prediction in
`protocol_sd.md`. The two free baselines (their seed, our generator, both at 42)
are identical to the bit with each other and have the distribution of `bon4`.

*`R0g24`, 23/09 about 06h30: their FK, paper configuration, seed 2024000 + i, 40 prompts,
paired by x_T and by DDIM noise with `bon4`, `ctl`, `R1`.* `ir_max` **0.720**; against `bon4`
**-0.126 +/- 0.070** (19/40 won); against `ctl` -0.238 +/- 0.103; against `R1` **-0.233 +/-
0.085**. Predictions (within +/- 0.06 of `bon4`, +/- 0.08 of `R1`): **missed**, both from
below; the duplicates at the terminal step (10 % of the runs): held. Per particle, their four finals
are worth 0.564 against 0.820 for `R1` on the same x_T: their loop steers less well than
`smc/` with the same four choices, on the same noise. The difference between the two codes is
therefore real and is not in the table of `reference_config.md`; the four choices tested one by
one in `smc/` cost only 0.05 to 0.06 each. To be looked for in their code, not here.

*Their free baseline, 40 prompts (23/09, about 05h20).* `authors_free` (their pipeline without FK,
their seed): `ir_max` **0.824** against 0.846 for `bon4` on the same prompts (-0.022 +/-
0.083), 0.295 per particle against 0.288: prediction **held**, their seed path is
sound and their free sampler has the distribution of ours. So, on an equal footing: **their FK, their seed,
returns 0.33 +/- 0.08 less than their own best-of-4** (`R0 - authors_free`), and their FK with
our generator 0.11 +/- 0.09 less than their corresponding free run (`R0g - authors_free_g`,
22 prompts, 40 running). The 0.31 between the two seed paths of the same FK remains without
explanation: the two free baselines coincide, only the FK loop differs between the two
paths. Reproducible (the smoke test and the night return the same figures), unexplained, to
be said as it is.

What the reference says at this stage, in one sentence: the released code, under the configuration of
table 1, returns on this hardware **less than best-of-4** (by 0.11 to 0.33 depending on the seed
path), never +0.16 above;
`smc/` with the same choices returns best-of-4 (R1); `smc/` with its own choices returns +0.056
(`ctl`). The gap to the paper is **bounded**, not closed, and it is not in the implementation:
neither of the two implementations produces it.

## Night 2, continued: bisection, `thr05`, `rise` (23/09, about 02h40)

Paired with `ctl`, 40 prompts except `rise` (20). Predictions from `protocol_sd.md` (22/09 19h35 and
19h55) alongside.

| arm | `ir_max` - `ctl` | prediction | lineages | prediction | resamp. | root(s) = `ctl` |
|---|---|---|---|---|---|---|
| `stat0` | -0.050 +/- 0.096 | -0.09 +/- 0.05 | 1.75 | ~1.7 | 1.88 | 25 % |
| `multi` | -0.061 +/- 0.081 | -0.02 +/- 0.04 | 1.00 | 1.0-1.1 | 4.00 | 38 % |
| `vae` | +0.009 +/- 0.083 | under 0.03 | 1.05 | | 3.73 | **30 %** (predicted >= 70 %) |
| `idx` | -0.051 +/- 0.085 | under 0.03 | 1.00 | | 3.73 | 22 % |
| `thr05` | -0.058 +/- 0.100 | -0.10 +/- 0.06 | **2.00** | **1.84** (model); 2.4-2.8 (plan) | **0.97** | 22 % |

- **`thr05`: the prediction of the coalescence model is held** (2.00 roots against 1.84,
  62 % with one root against 61 %, 0.97 resampling against 1.12), and that of the plan
  (2.4-2.8) is missed. Resampling less does not keep the lineages when the accumulated weights
  are peaked at the moment the threshold fires. Mean `ir` 0.643, between `floor`
  (0.671) and `floor2` (0.490), as predicted. The terminal ESS is under 1.5 in only 8 % of the runs
  (predicted more than half): the postponed selection does not collapse at the last
  step, it has already happened at the step where the threshold fired.
- **Bisection**: none of the four choices alone is worth more than 0.06, and their sum (-0.15)
  exceeds `R1` (-0.067): they do not add up. The guide's VAE is neutral on the
  reward (+0.009) but **changes the root kept in 70 % of the prompts**: the selection of the
  first informative step hangs on reward gaps of the order of a decoder's rounding.
  This is the most direct version of finding 3.
- The coalescence model, read on the column of the resampler actually used, holds on the
  fifteen arms: `R1` 1.25 predicted against 1.19 observed (multinomial), `multi` 1.02 against 1.00,
  `stat0` 1.84 against 1.75, `vae` and `idx` to within 0.05.

## Verdict (23/09, about 06h, everything returned)

**The starting question: do the corrections resolve the collapse?** Layer by layer.

- *The mechanism.* A single correction passes the criterion "less than 25 % of runs with one root":
  `floor2` (floor + lambda 2), 5 % at n = 40, 3.0 roots out of 4, pixel diversity 0.29
  against 0.09. It does so by changing the target (lambda 2), and it returns exactly the
  diversity that this target carries (weighted root ESS 2.7 for a ceiling of 2.7). Everything
  that keeps lambda = 10 stays between 62 and 100 % of runs with one root: `floor` 68 %,
  `fadapt` 30 % but 1.7 of weighted ESS for a ceiling of 1.5 (it returns roots that its
  target crushes), `thr05` 62 %, `late` 86 % at n = 300, `adapt` 60 %.
- *Why.* The number of final roots is predicted from the recorded weights and the
  resampler, without knowing anything about the steering, to within 0.05 on fifteen arms (`n_coalescence.py`):
  it is the degeneracy of the paths, and the ESS of one step does not measure it. With flat weights the
  comb is the identity and the multinomial is not (1.58 roots out of 4 after four flat
  steps): the released code loses roots before any information.
- *The price.* Every correction that keeps lineages at lambda = 10 returns `ir_max` at the level of
  best-of-4 (`floor` -0.09, `fadapt` -0.11, `adapt` -0.10, `stat0` -0.05, `multi` -0.06 against
  `ctl`; bootstrap CI at n = 40 covering zero one by one; mean of the seven corrections
  -0.069 [-0.176, +0.041] at n = 40, with `floor2` at 34 prompts biased upward, and -0.106
  [-0.193, -0.019] over the first five before the incident), and the mean `ir` of the four follows what
  is bought (-0.15 to -0.53). `late` alone does not pay (+0.04 +/- 0.03 at n = 300) and
  repairs nothing.

**The reference: the gap to the paper is bounded, not closed.** The paper's configuration is
the one of this repository. The released code under this configuration, over four runs (220 run-prompts),
returns on average **-0.11 +/- 0.04** against best-of-4, and its four means at 40 prompts range
from **-0.35 to +0.10** depending on the random stream, at equal x_T: its result depends on the draw by
more than the effect the paper reports. `smc/` with their four implementation choices returns
best-of-4 (`R1`, -0.01); with its own, +0.056 and +0.030 in two sessions. The +0.161 of
table 1 is reached by one run in four of their code (+0.10 +/- 0.06), and by no
mean. The diagnosis shows the mechanism of their
loop: floor, flat weights, drift, late selection; and on the same x_T their loop
returned 0.56 per particle against 0.82 for `smc/` with the same choices in one run, +0.10 over
best-of-4 in another. *(Reread on 23/09 in the afternoon: "steers less well" withdrawn; it is the
spread between runs of their filter, measured and not explained, see section C.)*

**What is held, missed, open** (predictions of `protocol_sd.md`):

| prediction | outcome |
|---|---|
| `floor2`, `lam2` at 40: lineages, one root, price | held |
| `late`: 1.4-1.8 lineages | missed (1.0-1.14); zero price: held |
| coalescence at +/- 0.15, right order | held (+/- 0.07) |
| `thr05`: 1.84 roots, 61 %, 1.1 resamp. (model) | held (2.00, 62 %, 0.97); plan (2.4-2.8) missed |
| latents: chaotic divergence | missed (corr > 0.98); x_T to the bit: held |
| `bon4` of 20/09 replayable | held (0.000) |
| `R0` = 0.77 [0.72, 0.82] | missed from below (0.554) |
| `R1` - `ctl` = -0.08 +/- 0.05 | held (-0.067) |
| `R0` within 0.03 of `R1` | missed: the four runs of their code spread from -0.35 to +0.10 against `bon4` where `R1` is at -0.01; the question has no single answer but a spread |
| `R0g24` within +/- 0.06 of `bon4` | missed (-0.13 +/- 0.07); terminal duplicates 5-15 %: held (10 %) |
| bisection: each choice under 0.06 | held; `vae` changes the root kept in 70 % of the prompts (predicted < 30 %): missed |
| `rise`: lineages 2.0-2.3, not above `ctl` | held (1.9-2.0; +0.01) |
| their free baseline = ours in distribution | held (0.824 / 0.846, 0.295 / 0.288) |

Open: (1) the spread of the authors' FK between random streams (four means at 40
prompts from -0.35 to +0.10, per-prompt standard deviation 0.25 between runs) where `ctl` moves by 0.03
between sessions: the multinomial with flat weights and the terminal duplication are its suspects,
not measured; (2) the cross-session non-reproducibility of the `smc.models` path (diffusers
pipeline replayable, our model not), cause not identified; (3) findings 3, 5 and 5 bis
suspended as long as `ctl` and `lam0` have not been replayed in one same session; (4) `floor2`
and `lam0` cut by 6 prompts by the replay incident.

## What this says about the "solutions"

`floor2` resolves the collapse in the strict sense (5 % of
runs with one root, diversity at the ceiling of its target) by changing the target; at lambda = 10,
nothing resolves it because the target accepts only 1.2 to 1.5 particles out of 4 (finding 15),
and the arms that keep roots at this lambda return particles that the target crushes
(`floor`, `fadapt`: weighted ESS above the ceiling). The rest (R0, R1, bisection,
`thr05`, `rise`): `docs/protocol_sd.md`, then here.
