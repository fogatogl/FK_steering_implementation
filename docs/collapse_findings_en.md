# Why the particle cloud collapses: final state of the collapse lab

At k = 4 and lambda = 10, no correction keeps four lineages. The target itself carries 1.2 to 1.5 of the 4 (finding 15), and the number of final roots is predicted from the weights and the resampler alone, to 0.05 across fifteen arms, without knowing anything about the steering. The only correction that brings the share of one-root runs under 25 % changes the target (`floor2`, lambda 2: 6 %, 2.9 roots) and returns exactly the diversity that target carries. Every correction that keeps lineages at lambda = 10 returns `ir_max` at the level of best-of-4: FK's gain over best-of-4 is the price of the concentration, and making it reversible costs that gain. On the reference side, the paper's configuration is the one of this repo, and the +0.161 of table 1 is produced neither by this repo (+0.056, +0.030 in two sessions) nor on average by the released code (-0.11 +/- 0.04 over four runs, 220 run-prompts). The released code's four 40-prompt means range from -0.35 to +0.10 with the random stream at equal x_T: one run in four reaches the paper's figure, and the spread of their filter exceeds the effect it reports.

## The corrections, paired to the reference

Session A, seed 2024, every arm paired to `ctl` by prompt.

| arm | n | one root | roots | weighted ESS / ceiling | div_pix | `ir_max` - `ctl` | mean `ir` - `ctl` | `ir_max` - `bon4` |
|---|---|---|---|---|---|---|---|---|
| `ctl` (reference) | 40 | 95 % | 1.06 | 1.01 / 1.51 | 0.092 | | | +0.112 +/- 0.091 |
| `late` (without t = 80) | 20; 300 | 100 %; 86 % | 1.00; 1.14 | 1.00 / 1.43 | 0.082 | +0.048 +/- 0.104; +0.039 +/- 0.033 | +0.081 | +0.187 |
| `adapt` (bisected lambda) | 40 | 60 % | 1.40 | 1.10 / 1.51 | 0.153 | -0.099 +/- 0.044 | -0.171 | +0.013 |
| `floor` (floor at 0) | 40 | 68 % | 1.73 | 1.64 / 1.51 | 0.150 | -0.091 +/- 0.055 | -0.153 | +0.021 |
| `lam2` | 40 | 35 % | 1.82 | 1.62 / 2.72 | 0.205 | -0.052 +/- 0.103 | -0.138 | +0.060 |
| `fadapt` (floor + bisected) | 40 | 30 % | 2.12 | 1.72 / 1.51 | 0.200 | -0.106 +/- 0.058 | -0.214 | +0.006 |
| `floor2` (floor + lambda 2) | 40* | 5 % | 3.00 | 2.73 / 2.72 | 0.286 | -0.083 +/- 0.093 | -0.334 | +0.029 |
| `thr05` (floor, ESS < k/2) | 40 | 62 % | 2.00 | 1.66 / 1.51 | 0.172 | -0.058 +/- 0.100 | -0.181 | +0.054 |
| `rise` (floor, bisected, lam_max 100) | 20 | 35 % | 1.95 | 1.63 / 1.43 | 0.170 | +0.007 +/- 0.109 | -0.135 | +0.146 |
| `lam0` (free) | 17 | 0 % | 4.00 | 4.00 / 4.00 | 0.338 | -0.070 +/- 0.082 | -0.557 | +0.034 |

\* `floor2`: figures at n = 40, read on 22/09 at 22h, before an incident lost six records; at n = 34 the current output of `r_solutions.py` is biased (+0.009 on `ir_max`). The mean of the seven corrections against `ctl` on `ir_max` is -0.069 [-0.176, +0.041]; on the first five, before the incident, -0.106 [-0.193, -0.019]. The price in mean `ir` follows the roots kept, from -0.14 to -0.56.

`thr05` (resample less often) returns 2.0 roots, not 2.4 to 2.8; `late` repairs nothing; `rise` does not beat `ctl`. The "threshold under the target" lead is importance sampling, settled without GPU. The ranking of the corrections holds: `floor2` > `fadapt` > `thr05` ~ `floor` ~ `lam2` > `adapt`.

## The released code's four implementation choices

Each choice of the released code ported alone into `smc/`, 40 prompts, paired to `ctl`.

| arm | choice | roots | `ir_max` - `ctl` | root kept = `ctl` |
|---|---|---|---|---|
| `stat0` | floor at 0 + statistic form | 1.75 | -0.050 +/- 0.096 | 25 % |
| `multi` | multinomial at every scheduled step | 1.00 | -0.061 +/- 0.081 | 38 % |
| `vae` | guide decoded by the pipeline's VAE | 1.05 | +0.009 +/- 0.083 | 30 % |
| `idx` | indices {20, 40, 60, 80, 99} | 1.00 | -0.051 +/- 0.085 | 22 % |
| `R1` | the four together (100 prompts) | 1.19 | -0.067 +/- 0.055 | |

No single choice is worth more than 0.06; the four together are worth `ctl`'s lead over best-of-4 (`R1 - bon4` = -0.011 +/- 0.041). The guide's VAE, neutral on the reward, changes the root kept in 70 % of the prompts: the selection hangs on the rounding of a decoder.

## The reference: the released code under the paper's configuration

Table 1 configuration, SD v1.5.

| run | code | seed | n | `ir_max` | against `bon4` | pairing |
|---|---|---|---|---|---|---|
| paper, table 1 | | | | 0.898 | +0.161 | |
| `ctl` | `smc/` | 2024 | 100 | 0.826 | +0.056 +/- 0.052 | x_T |
| `R1` | `smc/` + their 4 choices | 2024 | 100 | 0.756 | -0.011 +/- 0.041 | x_T |
| `R0g24` | their code | 2024, generator | 40 | 0.720 | -0.126 +/- 0.070 | x_T and DDIM noise |
| `R0g` | their code | 42, generator | 40 | 0.807 | -0.039 +/- 0.073 (prompt); -0.018 +/- 0.074 against their free sampler | noise (seed 42) |
| `R0` | their code | 42, global RNG | 100 | 0.554 | -0.216 +/- 0.061 (prompt); -0.328 +/- 0.083 against their free sampler | prompt |
| `R0` seed 2024 | their code | 2024, global RNG (= this repo's x_T) | 40 | 0.949 | +0.103 +/- 0.056 | x_T |
| their pipeline without FK | their code | 2024, generator | 5 | | 0.0000 gap on 20 slots | bit |
| their FK's four runs, grouped | their code | | 220 | 0.702 | -0.110 +/- 0.036 | |

On the same 40 prompts, the four runs of their FK return, against `bon4`: -0.35, -0.04, -0.13, +0.10 (+/- 0.06 to 0.09 each). Two of them share x_T and DDIM noise and differ only by the stream of the multinomial draw (-0.13 and +0.10); the standard deviation of `ir_max` between the four runs of one prompt has a median of 0.25. This repo, on the same 40 prompts and two sessions: +0.11 and +0.08. The paper's +0.161 is inside the range of what their code returns from one stream to the next, and 0.25 above its mean.

Their base sampler is bit-compatible with the diffusers pipeline, their ImageReward scorer matches the official one to the third decimal, and their FK loop does what it writes (floor, flat weights, multinomial drift, late selection). The spread between its runs is not reduced by the reading of the code (23/09: no reseeding, no bias between seed paths). Under the pre-registered rule the gap to the paper is bounded, not closed.

## The mechanism in three lines

**Weights.** At each scheduled step the weight is `exp(lambda r_phi)` up to a shared factor (findings 4, 8); at lambda = 10 the range of the first step is 9 nats on noise.

**Paths.** The number of final roots is a function of the recorded weights and of the resampler (`n_coalescence.py`, fifteen arms to 0.05, per-run correlation 0.87 to 0.99); the ESS of one step does not measure it (`adapt`: ESS 2.0, 1.4 roots). With flat weights the comb is the identity, the multinomial is not (1.58 roots out of 4 after four flat steps).

**Target.** At lambda = 10, four free draws reweighted by `exp(10 ir)` have an ESS of 1.2 to 1.5 (finding 15): keeping more roots means returning particles the target crushes (`floor`, `fadapt`, `thr05`, `rise`: weighted ESS above the ceiling).

## Session C: the valid pairing

`ctl`, `lam0` and `floor2` at 100 prompts in a single process (23/09), `out/probe_C.json`, readout `t_sessionC.py`. The predictions come from `docs/protocol_sd.md` ("Pre-registration of session C").

| measure | value | prediction | outcome |
|---|---|---|---|
| `lam0_C` against `bon4` (20/09), per slot | correlation 1.00, `ir_max` +0.009 +/- 0.006 | correlation under 0.7 | missed, in the right direction: the free path replays |
| `ctl_C` against `ctl` of 21/09, per slot | correlation 0.70, same root 30 %, `ir_max` -0.026 +/- 0.058 | mean within 0.05, decorrelated slots | held |
| `floor2 - ctl`, `ir_max`, x_T paired | -0.012 [-0.082, +0.060] | [-0.14, -0.02] | missed from above: the price is smaller than predicted |
| `floor2 - ctl`, mean `ir` of the 4 | -0.250 +/- 0.045 | | |
| `floor2` roots; one root | 3.03; 4 % | 2.8 to 3.1; 5 to 10 % | held |
| `ctl` roots; one root | 1.07; 93 % | 1.0 to 1.1 | held |
| `ctl - bon4` in this session | +0.030 +/- 0.037 | | (+0.056 in the session of 21/09) |
| finding 3, Kendall tau of r_phi(t = 80) against the free `ir` of the same root | +0.137 +/- 0.050; top-1 35 % | under 0.15 | held, narrowly |
| finding 5 bis, A - B | +0.313 +/- 0.042 | [0.25, 0.50] | held |

The decomposition uses the only valid per-slot pairing (`lam0` is the free continuation of each root, same process). A, the best free root: 0.779. M, the mean root: 0.233. B, the root `ctl` keeps, read free: 0.466. C, what `ctl` draws from it: 0.799. B - M = +0.233 +/- 0.047: the root kept beats a random draw by a third of the way to the best (mean rank 2.04 out of 4). A - B = +0.313: the collapse still costs 0.31 of root. C - B = +0.333: the steering returns a little more than that. C - A = +0.021: the balance over best-of-4 is what the steering adds minus what the early choice loses, and it is small. Finding 3 now reads: the first step carries a little information (tau 0.14, top-1 35 % against 25 %), not none. Finding 5 bis holds, with A - B at 0.31 against 0.41 in the cross-session reading of 22/09.

Paired by x_T, `floor2` costs -0.01 on `ir_max` instead of the -0.08 to -0.10 seen when paired by prompt: the price of the lineages is of the order of `ctl`'s lead over best-of-4 (0.03 to 0.06). The price on mean `ir` stays (-0.25).

## Findings 16 to 19

**16. The reference.** The paper's configuration (max, [0, 20, 40, 60, 80], lambda 10, k 4, DDIM eta 1, 100 steps, CFG 7.5) is the one of this repo; the released script's defaults (`diff`, 5-30-5) are another configuration, listed further down in the paper's appendix. Under the paper's configuration the released code returns, on the 100 benchmark prompts and SD v1.5, 0.554 (its own seed) and 0.720 (this repo's x_T, 40 prompts), against 0.770 for best-of-4 and 0.826 for this repo; `smc/` with its four implementation choices returns 0.756. The +0.161 appears in no reading; the gap is bounded, and it is not in the implementation. Sources: `docs/reference_config.md`, `results/sd_authors_R0.json`, `collapse_lab/ref/`.

**17. Coalescence reads on the weights.** Replaying the comb (or the multinomial) on the weights recorded at each step, integrating over `u`, predicts the number of final roots of fifteen arms to 0.05 and the share of one-root runs to three points (`n_coalescence.py`). The steering does not enter the prediction. `adapt` holds the ESS at 2 and loses 2.6 roots: the ESS is the degeneracy of the weights, not of the paths. With flat weights the authors' multinomial loses roots by pure drawing (4 to 1.58 in four steps).

**18. Resampling less often does not keep the lineages.** `thr05` (floor, ESS < k/2): 0.97 resampling per run, 2.0 roots, 62 % with one root, predicted at 1.84 / 61 % / 1.12 by finding 17 before the measurement. When the threshold fires, the accumulated weights are peaked and a single pass takes almost everything. `ir_max` -0.06 +/- 0.10 against `ctl`.

**19. Replayability.** The diffusers pipeline today returns the rewards of `bon4` (20/09) to the third decimal. The `smc.models.StableDiffusion` path, same weights and same code, returned the same figures on 21/09 and on the morning of 22/09, and other roots and other `ir_max` on the evening of 22/09 (up to 1.6 apart on one prompt): deterministic within a session, not between sessions. Session C separates two cases. The free path (`lam0`) matched `bon4`'s scores in session C (per-slot correlation 1.00) and did not in session B; at lambda = 0 no reward is read, so neither the model cache (`HF_HOME` differed between the two sessions, same revisions) nor the scorer can explain it: same code, same weights, same seed, another process, cause not identified. The resampling path (`ctl`) does not replay between sessions (per-slot correlation 0.70, same root kept in 30 % of the prompts); the candidate is the guide's reward stack (VAE ft-mse, ImageReward, BERT tokenizer) loaded from another cache path, where 1e-3 on one reward moves one tooth of the comb; not tested in isolation. Per-slot pairing between files from different days measures nothing.

The chronology, the predictions written before each run and the tables at full length are in `collapse_lab/ASSESSMENT.md` and `collapse_lab/FINDINGS.md` (in French) and in `docs/protocol_sd.md`.
