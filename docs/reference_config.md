# The reference configuration: what the paper states, what the released code does

Written 22/09 before the reference night, from the paper source
(arXiv:2501.06848v5, https://arxiv.org/abs/2501.06848v5) and the released repository
(`zacharyhorvitz/Fk-Diffusion-Steering`, commit 9413005 of 2025-06-25, read in
`/home/onyxia/work/fkd_ref/`, not copied into this repository). Each cell names its source; the line numbers of the released code
were rechecked at 9413005 on 25/09.
"This repo" is `smc/fk.py` as run by `scripts/run_sd_baseline.py --samplers fk4` and by
`collapse_lab/probe.py --arms ctl`.

## The number to reproduce

| | k = 1 | BoN(k = 4) | FK(lambda = 10, k = 4) | FK - BoN |
|---|---|---|---|---|
| paper, SD v1.5, IR of the best particle (`experiments_new.tex:82-85`) | 0.187 | 0.737 | 0.898 | **+0.161** |
| this repo, 100 prompts x 3 seeds (`results/sd_baseline.json`) | 0.237 | 0.758 | 0.820 | +0.062 +/- 0.024 |
| this repo, per seed 2024 / 2025 / 2026 | | | | +0.030 +/- 0.037, +0.089 +/- 0.034, +0.068 +/- 0.050 |
| this repo, current code, seed 2024 (`results/sd_ref_fields100.json`) | | 0.770 | 0.826 | +0.056 +/- 0.052 |

The paper reports "the performance of the highest reward particle" (`experiments_new.tex:45`),
which is `ir_max`. Its k = 1 and BoN rows sit 0.05 and 0.02 under this repo's, its FK row
0.08 above. The seed spread here is as large as the mean gain: one seed cannot close 0.10.

## Configuration, column by column

| item | paper (text) | released code (defaults, or the flags `launch.sh` uses) | this repo |
|---|---|---|---|
| model | SD v1.4, v1.5, v2.1, SDXL (`experiments_new.tex:76-97`) | `runwayml/stable-diffusion-v1-5`, fp16, no `variant` (`launch_eval_runs.py:86, 303`) | `stable-diffusion-v1-5/stable-diffusion-v1-5`, fp16 variant |
| sampler | DDIM, eta = 1, T = 100, CFG 7.5 (`experiments_new.tex:34`) | `DDIMScheduler.from_config`, `--eta 1.0`, `--num_inference_steps 100`, pipeline default `guidance_scale=7.5` (`launch_eval_runs.py:100, 254-256`; `fkd_pipeline_sd.py:225`) | same |
| reward | ImageReward on the denoised state `x0_hat` (`experiments_new.tex:32`) | `step_dict["pred_original_sample"]`, decoded by **the pipeline's own VAE**, then ImageReward-v1.0 `score_batched` (`fkd_pipeline_sd.py:539-545, 484, 607-647`; `rewards.py:do_image_reward`) | `predict_x0` (Tweedie, same formula), decoded by **`stabilityai/sd-vae-ft-mse`** (decision 6, `docs/protocol_sd.md`) |
| lambda, k | 10, 4 (`experiments_new.tex:35`) | `--lmbda 10.0`, `--num_particles` forced by `--model_idx % 4` (`launch_eval_runs.py:284-294`) | 10, 4 |
| schedule | `[0, 20, 40, 60, 80]`, "t = 0 is the terminal step" (`experiments_new.tex:35`) | loop indices `arange(t_start, t_end + 1, freq)` plus `time_steps - 1`; **defaults 5-30-5** = {5, 10, ..., 30, 99} (`fkd_class.py:98-101`; `launch_eval_runs.py:273-275`); `launch.sh` uses 20-80-20 = {20, 40, 60, 80, 99} | indices {19, 39, 59, 79, 99} (`t` -> `99 - t`), i.e. the paper's schedule one index earlier than `launch.sh` |
| potential | `G_t = exp(lambda max_{s >= t} r_phi(x_s))` (`experiments_new.tex:35`) | **default `diff`**; `launch.sh` uses `max` (`launch_eval_runs.py:276`) | `max` |
| max potential, weight | the statistic itself | `rs = max(r_t, population_rs)`, `w = exp(lambda * rs)`, `population_rs` initialised at **`reward_min_value = 0.0`** and set to the resampled `rs`, the floored running maximum (`fkd_class.py:77-79, 111-113, 148, 168`); before commit `699c929` (15/06/2025, "address max potential bug") it carried the raw current reward | increment form `exp(lambda (M_t - M_{t-1}))` with `M` initialised at -inf; `potential_form="statistic"` exists as a flag; no floor |
| terminal step | `G_0` closes the product (`method.tex`) | `w = exp(lambda rs) / product_of_potentials` with `rs` the floored running maximum at 9413005 (`r_0` itself before `699c929`), then **adaptive resampling at the last step if ESS < k/2** (`fkd_class.py:125-131, 136-163`): the four returned images can be duplicates | `logG_last = lambda r_0 - acc` under the statistic form; `resample_last=False`: the returned images are never resampled |
| resampler | multinomial at every step (Algorithm 1, `method.tex:75`) | `torch.multinomial(w, k, replacement=True)`, at **every** scheduled step when `adaptive_resampling` is off, uniform weights included (`fkd_class.py:144, 165`) | systematic comb, only if ESS < k (strict), so never at exactly uniform weights (`smc/resampling.py`, `smc/weights.py`) |
| adaptive resampling | appendix C.3 sketches it | off by default; on: resample only if ESS < k/2 (`fkd_class.py:136-146`) | `--fk-threshold`, default 1.0 |
| prompts | ImageReward benchmark prompts (`experiments_new.tex:104`) | `prompt_files/benchmark_ir.json`, 100 prompts | `data/imagereward-benchmark-prompts.json`: **byte-identical ids, order and text** |
| seeds | not stated | `torch.manual_seed(seed)` once per pass, seeds 42, 43, 44, no `generator` (`launch_eval_runs.py:50-52, 335`) | `seed_effective = seed * 1000 + i` per prompt, generator passed |
| what is scored | "the highest reward particle" (`experiments_new.tex:45`); the appendix's base rows at k = 4 carry Table 1's best-of-4 HPS as their maximum over the four (0.256, 0.263, 0.296 for v1.4, v2.1, SDXL; `appendix_experiments.tex:71,77,82`) | `do_eval` on the four final images: the maximum and the mean over the four of each metric, HPS included, independently of ImageReward, averaged over prompts (`fks_utils.py:40-103`; `launch_eval_runs.py:176-215`) | `ir_max`, `ir` per slot, HPS at `ir_max` |
| time | 8.1 s for FK k = 4 on SD v1.5 (`experiments_new.tex:134`) | not measured on this service | 62.5 s per run on the faster card (model not recorded; about 88 s on the A2) |

## What this settles before any run

- **The paper's configuration is this repository's configuration**: max potential,
  schedule `[0, 20, 40, 60, 80]`, lambda 10, k 4, DDIM eta 1, T 100, CFG 7.5, ImageReward
  on the Tweedie estimate. The released script's defaults (`diff`, 5-30-5) are a different
  configuration, and the paper's own appendix scores it lower: for SD v1.4, IR max 0.783 with
  5-30-5 against 0.927 with 20-80-20 (`appendix_experiments.tex:72-73`). The gap is not a
  matter of which schedule or potential the text names.
- **Four implementation choices differ**, none named in the text: the floor at 0 on the max
  statistic (which makes the first step inert in 90 % of runs, `collapse_lab/FINDINGS.md`
  constat 6), the multinomial resampler applied at every scheduled step, the adaptive
  resampling of the terminal population, and the VAE that decodes the guide's estimate.
- The appendix diversity tables say "Here we use the difference potential"
  (`appendix_experiments.tex:21`) yet their 20-80-20, lambda 10 row for SD v1.4 (0.927) is
  Table 1's max-potential number. Reported as is.
- **Issue #14 of the released repository**
  (https://github.com/zacharyhorvitz/Fk-Diffusion-Steering/issues/14, opened 18/12/2025, read
  through the GitHub API on 25/09 and 26/09/2026): SD v1.5, FK k = 4, `max`, lambda 10, schedule
  `[0, 20, 40, 60, 80]` with `resample_t_start` 0, seeds 42 to 44 and the GenEval prompt file
  (`geneval_metadata.jsonl`, not `benchmark_ir.json`) give a max reward of 0.61 to 0.64 against the
  paper's 0.898. The one reply from the authors' side (a collaborator of the repository,
  22/12/2025): "can you change the resample t start to 20. we start at 20." `launch.sh` and the
  runs of the post's section 7 start at 20.

## What the reference night measures

`R0`: the released code, paper configuration (`max`, 20-80-20), SD v1.5 fp16, 100 prompts,
seed 42, one pass. `R1`: `smc/fk.py` with the four implementation choices above, by wrappers
(`collapse_lab/probe.py --arms R1`). Then each choice alone against `ctl`: `stat0`, `multi`,
`vae`, `idx`. Predictions and decision rule: `docs/protocol_sd.md`, "Pre-registration of the
reference arms (22/09)".

**Outcome (23/09).** `R1 - bon4` = -0.011 +/- 0.041 on 100 prompts: bounded, by the rule.
The released code in four runs (its seeding at 42 on 100 prompts, our generator at 42 and at
2024 on 40, its seeding at 2024 on 40): 0.554, 0.807, 0.720, 0.949; against `bon4`, pooled
over 220 run-prompts, -0.110 +/- 0.036; on the common 40 prompts, -0.35, -0.04, -0.13, +0.10.
Its free sampler is bit-identical to this repository's best-of-4 given the same generator.
Full account: `collapse_lab/ASSESSMENT.md`, sections C and H; `docs/results.md` block 18.
