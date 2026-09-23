# Chronology: how the project got from one state to the next

The thread that joins the runs of `docs/results.md` and the entries of
`docs/decisions.md`. Each stage says what was put in place, why and what problem
it solved, what went wrong, and which question came out of it and pushed the
project to the next stage. Dates are 2026. Sources: the git log, the two
planning documents of 16/09 (`.claude/CONTEXTE_projet.md`,
`.claude/plan_code_fk_pgdlm.md`), `LEARNING.md`, and the run files.

## The frame, before any SMC code (August to 16/09)

**Put in place.** A DDPM with an 8.4 M-parameter U-Net trained from scratch on
CIFAR-10 (200 epochs, about 4 h 15 on the T4), a fine-tuning on one class with
FID 79.2 to 50.7 and a measured floor of 21.7, a masked diffusion language model
(MDLM) installed and patched for the T4 that generates grammatical English, a
checkpointed pipeline on S3 with idempotent resume. In parallel the theory of
Sequential Monte Carlo: Doucet and Johansen, Chopin and Papaspiliopoulos
chapters 5, 8 to 11.

**Why.** The goal fixed on 28/08 is a faithful reproduction from the equations
of two SMC papers for diffusion models, FK Steering (Singhal et al., ICML 2025)
then PG-DLM (Dang et al., 2025), on one code base, delivered as an arXiv report
and a public repository. The search for an original contribution was dropped
that day: without a senior supervisor it was judged too ambitious, and the
reproduction is what all the internship applications rest on. The text modality
(MDLM) was chosen as the one shared by both papers and the one whose model fits
the T4.

**Problems met.** No commit between 01/09 and 16/09: fifteen days of reading
alone, and a reading budget that no longer fit the five weeks left before the
20/10 delivery.

**What it raised.** On 16/09 four decisions that shape everything after: no more
hour of theory without a code target in front of it, reading becomes
just-in-time; delivery in two stages, v1 FK Steering on 20/10 and v2 PG-DLM
around 04/12; one visible commit per week; and the code contract, the SMC core
is written by the author and the assistant stays on plumbing, pointers and
review. The image block (SD v1.5 with ImageReward) was at that point placed
behind v2, December at best.

## Module 0: the four SMC bricks (16/09)

**Put in place.** `smc/weights.py` (normalised log-weights and ESS),
`smc/resampling.py` (multinomial and systematic resamplers that return ancestor
indices, not particles), tests against the `particles` library as oracle. The
first design entries: `right=False` in `searchsorted` so a point landing on an
upper edge stays in its slot; tolerances 0.02 for the multinomial and 2/k for
the systematic, since the comb bounds each frequency error by 1/k.

**Why.** Everything in log-space because exp(lambda r) overflows in fp32 as
soon as lambda r exceeds 88. Indices rather than particles because the
conditional SMC kernel of PG-DLM will need to pin one ancestor, and that
separation makes it almost free later.

**What it raised.** Nothing blocking: the module closed in a day and the
question moved to the model interface.

## Module 1 on CIFAR: the interface, the baseline, the toy reward (16/09)

**Put in place.** `DiffusionModel` with `initial_state`, `step`, `predict_x0`,
and `CifarDDPM` around the trained U-Net, with a non-regression test against the
notebook's sampler. `best_of_n` counting UNet calls. The red reward, deliberately
stupid and monotone. Figures 0 and 1 (runs 1 and 2 of `docs/results.md`).

**Why.** The interface is the bet of the whole project: if it holds, porting to
MDLM and later to Stable Diffusion changes no line of the FK loop. The reward is
stupid on purpose: if FK does not push a monotone reward up it is a bug, not a
subtlety. The compute unit, UNet calls, is fixed here because the paper's claim
is "better than best-of-N at equal compute" and that is the first thing a
reader checks.

**What it raised.** Two of the four steps of Algorithm 1 exist; the potentials
are next.

## `fk_steer` and the three potentials (17/09)

**Put in place.** `fk_steer` with `difference`, `max`, `sum` in log-space, the
adaptive rule ESS < k/2, ESS and resampling count in the diagnostics, the lambda
sweep script and figure 2 (run 3). The test of the product constraint: on
synthetic reward sequences, the sum of log G_t equals lambda r(x0) for the three
potentials. The decision "one target for the three potentials": the corrective
term at the terminal step is kept for all three, otherwise the final marginal
would be biased by the blurry Tweedie estimates at large t.

**Why.** The paper's specification is the product constraint; the three
potentials differ in when the guidance acts, not in what it targets. The test
is two hours of work and the proof, in an interview, that one understands why
the potentials are interchangeable.

**Problems met.** The EMA bug: `load_model` read the raw weights while the
notebook's FID used the EMA ones, so the sweep and figures 0 to 2 ran on a
different model than the FID anchors (max parameter gap 0.018). Found by
rereading the notebook, first line of `LEARNING.md`. Everything after switches
to EMA; the two raw-weight files stay as they are.

**What it raised.** The red reward is hackable. `sum` pushes it past its bound,
the images leave [-1, 1], at lambda 8 the samples are flat red squares, and a
calibrated ResNet-18 still gives them a mean p(cat) of 0.43. A reward that can
be gamed and a metric that follows it: the project needs a reward that means
something and a judge that did not guide.

## A classifier reward, and a guide that is not the judge (17/09)

**Put in place.** `r(x) = log p_A(cat | clamp(x))` with A a small VGG (89.4 %),
a second classifier B (ResNet-18, 93.2 %, temperature fitted on the test set)
that only judges, the sweep of run 5, figure 3, the class-3 FID pipeline (run 4:
base 80.2, fine-tuned 51.4, reproducing the notebook). A DDIM sampler with eta
and a timestep sub-sequence, since FID at T = 1000 costs 16 s per sample.

**Why.** Log-probability rather than a normalised score because with
`difference` the product telescopes to p(x) p_A(cat | x)^lambda, so lambda = 1
is the Bayes posterior under A, and the fine-tuned DDPM is a direct comparison
point (what retraining buys against what steering buys). Two architectures and
not two seeds for A and B, since twins share their blind spots. The clamp lives
in the reward because the classifier must never see values outside its training
support.

**Problems met.** p_B does not judge out-of-distribution images: on the red
squares it returns a number like on anything else. FID is the complement, and
neither suffices alone. A noise-conditioned classifier was considered and
deferred: the clean classifier on the Tweedie estimate is what the paper does.

**What it raised.** The ESS column of run 5, 16 to 1.05 across lambda, is the
first sighting of the collapse. At 32 px the damage lambda does to an image is
invisible, so the question became: what does it look like on a face.

## CelebA-HQ 256 from the Hub: the collapse becomes visible (18/09)

**Put in place.** `smc/pretrained.py`, a diffusers UNet
(`google/ddpm-ema-celebahq-256`, same linear beta schedule and T = 1000 as
CIFAR) behind the `CifarDDPM` interface, fp16, DDIM 50 steps eta 1. An
Eyeglasses classifier pair trained at 64 px with class weights (positives are
4.9 %). The two sweeps of run 6, the grids and figure 3b, the two-reference FID
of run 7 (7 h 35 min of GPU). The HF cache moved to the persistent volume and S3
because `~/.cache` died with the service. README along the CLAIRE template,
Onyxia setup document.

**Why.** Zero lines of `smc/` changed for a 256 px model from the Hub: the
interface held its first test. Two FID references because "closer to the target"
and "what the guidance costs" are different questions.

**Problems met.** On the 256 px model lambda must stay small: at lambda 1, 7 of
16 faces wear glasses for B on clean faces; at lambda 2, sixteen copies of one
pink blurred face, A says glasses, B says none, `ess_min` 1 from lambda 2 on. The
FID lots include the sixteen particles of each run, duplicates included, because
one particle per run would have been 2048 runs, two and a half days: the numbers
read as pessimistic bounds.

**What it raised.** The failure has a mechanism now: every particle descends
from one ancestor chosen on Tweedie estimates at large t where the guide is
fooled. And a second question: FK beats best-of-N at k = 16, but does the edge
survive when k grows.

## What k buys at a matched budget (19/09)

**Put in place.** The sweep script takes k as well as lambda, the ESS trace per
step lands in the JSON and the ancestor index in the `.pt`. Three files, run 8,
figure 5.

**Why.** The paper's claim is at equal compute; a curve in k with best-of-k
beside it is what makes the claim checkable.

**What it raised.** FK is far ahead at k = 4 and best-of-k has caught up by
k = 16: the steering pays where the budget is small, with enough independent
samples taking the max is enough. At k = 2 no resampling can ever fire under
ESS < k/2, so those points are pure importance weighting. This is the shape the
SD block would find again, and it is why the SD variant screen included k = 8.

## The turn to Stable Diffusion (18/09 to 19/09)

**Put in place.** The six SD decisions (reward evaluated at the five scheduled
steps only, which is the paper's interval resampling; fixed schedule for the
table and adaptive as a variant; schedule converted once at entry; the MAX
potential with the corrective term; budget matched on UNet rows counted by a
forward hook; `sd-vae-ft-mse` as the guide's decoder, flagged as a deviation).
A separate venv because ImageReward needs transformers 4.x and diffusers stays
at 0.31. The SD wrapper with the VAE inside the reward object, not the model, so
that `fk.py` does not know what domain `predict_x0` returns. A stopwatch (13.7 to
14.4 s per image at 100 steps, 2.16 GiB peak) and a verification that eta = 1
actually injects noise, since otherwise resampled clones could never diverge.
The judge pilot of run 9. The protocol written before the run
(`docs/protocol_sd.md`).

**Why the block moved forward.** The 16/09 plan had the image block behind the
PG-DLM delivery, December at best, and the text modality (MDLM) as the one
shared by both papers. It was pulled forward on 18/09 for two reasons, recorded
here on 21/09. First, the author knows the theory of image diffusion far better
than that of masked diffusion language models and of PG-DLM, and judged one
block developed in depth worth more than two blocks each left thin: the SD row
of the paper's table 1 is a real reproduction target, with two baselines that
can be checked against published numbers, where the text block would have been a
port plus a reimplementation of a paper without public code. Second, a venue
appeared: the Hi! PARIS Student Conference on AI, open to IP Paris students,
whose submission is a Markdown blog post (2,500 to 3,500 words, 6,000 max) backed
by a documented, licensed, public GitHub repository, in the model of the ICLR
Blogposts track that the plan already had in view for January. Its deadline,
7 October 2026, sits two weeks before the 20/10 arXiv date and asks for exactly
what the image block produces: a reproduction with a clean repository and a
result worth presenting. The two-paper comparison stays the long-term goal;
PG-DLM moves behind this deadline.

**Problems met.** The terminal-step defect: `fk_steer` identified the last step
by `t == 0`, which holds for the CIFAR DDIM sub-sequence by construction but not
for SD v1.5, whose diffusers timesteps end on 1 (`steps_offset = 1`). Wired as
is, the corrective branch would never have fired on SD, the target would
silently have become exp(lambda max_s r(x_s-hat)), and nothing would have
raised. Fixed by walking `model.timesteps` by position: the terminal step is the
last one, whatever its value, and the schedule must contain it or `fk_steer`
refuses to run. An earlier reading of the paper had claimed it defined no
corrective term; that reading was wrong and is corrected in `decisions.md`.
Also: `ImageReward.load` ignores `HF_HOME` and drops 1.7 GB on the ephemeral
overlay; the 40-prompt subset first planned was discarded when the stopwatch
showed the full 100 fit in a night, because the prompt index feeds the seed and
switching files renumbers every x_T.

**What it raised.** Run 9 said before anything ran that ImageReward and HPS
agree on the trend and not the ranking (Pearson 0.58), and that at lambda = 10 a
1.7-point gap is a weight ratio of e^17: the collapse was already in the scales.
C3 closed on 19/09 with the wrapper reproducing the pipeline bit for bit at
lambda = 0. The baselines were launched for the night.

## The baselines land, the FK row does not run (20/09 morning)

**Put in place.** Run 10: `k1` and `bon4` on the 100 prompts, three seeds,
within 0.3 to 0.6 standard errors of the paper on ImageReward.

**Problems met.** No `fk4` row: the SD venv had no editable install of the
project, and `fk4` is the only sampler that imports `smc`. The two baselines
never import it and ran the whole night without noticing. Second line of
`LEARNING.md`. Fixed, and the FK row ran 07:18 to 12:36 (run 11).

**What it raised.** FK beats best-of-4 at equal budget by +0.062 ImageReward,
2.4 standard errors on 100 paired prompts, 69 prompts won, where the paper has
+0.161. Both baselines are on the paper, so the missing 60 % is in the FK row
alone. Three candidates were named in the order to test them: the running max
over a five-point grid, the decoder, the fixed schedule.

## Reading the FK row: the particles collapse before the image exists (20/09, midday)

This is the reading written at the time, kept as the hinge of the block.

**The implementation is right.** The product constraint holds on the three
potentials (`tests/test_fk.py`), the SD wrapper reproduces the diffusers
pipeline bit for bit at lambda = 0, and the UNet budget of FK and best-of-4 is
the same measured number, 800 rows, on all 600 runs. The two baselines land on
the paper's table. Whatever is missing is not a bug in the weights, the
resampler or the potential.

**FK beats best-of-N at equal budget, and by less than the paper.** On SD +0.062
where the paper has +0.161. On the CIFAR k sweep the same shape: far ahead at
k = 4, caught up by k = 16.

**The gain does not reach the judge.** HPS moves by +0.0015, 1.4 standard
errors, 50 prompts each way; the paper's own table has FK slightly below
best-of-4 on HPS. An interviewer will ask whether +0.06 of ImageReward is worth
anything; the honest answer is that the metric that was not optimised did not
move.

**Where the missing 60 % most likely went.** At the first scheduled step, t = 80,
twenty of the hundred denoising steps have run and the Tweedie estimate is a
blur; the median ESS there is 1.18 out of 4. With threshold 1.0 the resample
fires immediately, so from step 20 on three of the four particles are copies of
one ancestor chosen on noise. At the terminal step the median ESS is 2.78: the
weights are well spread once the image is sharp, when they no longer have much
to select from. The per-prompt gain has a heavy left tail (minimum -1.30): on
one prompt best-of-4 draws a +0.92 particle and FK, on the same four x_T,
collapses to ESS 1.002 at the first step and ends below `k1`, on all three
seeds. The particle that would have won was killed at step 20. The decoder is
not the story (same particle first in 225 of 300 runs), and final particles are
not clones (no run ends with two bit-identical scores; the 9 of 300 counted here
on 21/09 agree only to three decimals): the damage is done by the selection,
not by a failure to diverge. Lambda = 10 on a reward whose useful
range spans two units means a 0.2 gap in early ImageReward is a weight ratio of
e^2, fine at the terminal step and destructive at step 20.

**What this ruled out.** The adaptive rule alone would not fix it, since 1.18 is
already below 2 and it fires at the same step. Replacing the decoder would move
a handful of rankings inside the noise. Neither is worth a night on its own.

**What it left open, in the order to test.** Whether the first scheduled step is
the problem (drop it), whether the weight sharpness is (lambda 2 to 20), whether
the five-point running max is, and whether the paper's number needs k > 4. All
one CLI flag away, costed in the protocol: the screen of the afternoon.

## The seven-variant screen: the schedule is the lever (20/09, 13:21 to 16:57)

**Put in place.** A threshold flag, k = 8 samplers, one JSON per variant so the
resume key does not swallow them, paired reading against the records that share
each variant's x_T (`compare_sd_variants.py`). Run 12.

**What it raised.** Dropping the t = 80 step (`S60`) doubles the gain over
best-of-4 on the 20 prompts; the three other lambdas all lose to 10; the ESS at
the first evaluation is 1.0 to 1.2 whatever step it falls on, so the collapse
follows the first evaluation and not the clock. `A05` helps a little, against
the protocol's prediction, by letting the weights carry across a step. `K8`
keeps the edge at twice the budget. Three times over: the first evaluation at
full lambda on a blurred image is what costs. A lambda that grows with the
denoising is the synthesis of `S60` (remove it), `A05` (resample less there) and
`L2` (soften it), and the FK formalism admits any sequence G_t whose product
along the lineage reaches lambda_T r(x0). That is a change to the potential,
hence to `smc/fk.py`, and it got its own protocol section before any code.

## A time-dependent lambda, and where its deficit is paid (20/09, afternoon and evening)

**Put in place.** `lam_schedule` in `fk_steer`, a lambda per scheduled step
built in the script (linear 2 to 10, quadratic 0.4 to 10, lambda_T = 10 so the
target is unchanged). Then, on review, the realisation that two FK models share
those schedules and are not the same experiment: the terminal placement pays
the whole deficit at the last step, whose weight nothing reads, so the images
are exactly those of a weaker steering; the tempering placement carries the
tempered previous max in the gate, G_t = pi_t / pi_{t-1} with
pi_t proportional to p(x) exp(lambda_t m_t), and pays the deficit at every
scheduled step where resampling can still act. Both behind `lam_placement`,
recorded in every output. Property tests before launch, 24 green: lineage sum
equal to lambda_T r(x0) under both placements for the three potentials,
constant schedule under tempering identical to the constant path, lambda = 0
neutral. The tempering launcher runs the tests first and refuses to start if
they are red.

**Problems met.** The change doubled `smc/fk.py` and the review produced the
code rules kept in `docs/protocol_sd.md`: one formula, one body (terminal is
tempering with lambda_prev = lambda plus one override); nothing that cannot run;
a comment states the non-obvious once; a property test is never deleted when a
mode is added; a branch no test reaches is tested or removed; a refactor is
gated by the same green and a byte-identical `logG` on the path the running
screen imports.

The smoke run of `T1` on prompt 0 showed the ramp cannot protect diversity on
its own: ESS 2.07 at t = 80, and still one surviving x_T among the four finals,
because at threshold 1.0 the cloud is resampled at every scheduled step whatever
the ESS and four systematic draws on k = 4 end on one root. Hence the cross with
the 0.5 threshold. And `ir_max` cannot see the collapse at all: four copies of
one image score like four images. Two fields were added, `n_lineages` from a
backward walk over the ancestors and `div_pix` as a pixel proxy, and the
reference was regenerated with them (run 14): `fk4` ends on one lineage in 20 of
20 prompts.

**What it raised.** Eight ramps queued on the T4 from 20:47 to 00:15 (runs 13
and 15), and two predictions written down before they ran: the ESS at t = 80
would be about 2.9 for the linear ramp and close to 4 for the quadratic one; if
the terminal placement beats `fk4` then tempering is worth its run, and if it
loses, the ramp is not the lever and tempering loses too.

## What the screen says, and the fork it leaves (21/09)

**Result.** The ESS prediction held exactly (2.88 and 3.92). On ImageReward
nothing beats `fk4`: all eight ramps sit inside one standard error of zero, and
`S60` at +0.094 stays the only candidate for the table. On the collapse the
improvement is real and measured: `T2tA05` doubles the lineages and closes 38 %
of the diversity gap to best-of-4 at 5 standard errors, and the tempering
placement wins on diversity in all four pairs. The second prediction was half
wrong: tempering rescued `T2` on ImageReward (-0.100 to +0.030) but not
`T2A05`, so the catch-up term and the skipped resamplings overlap rather than
add. Figure 6 shows it on a win, a median and a loss.

**The question this leaves, and it is the author's to answer before the next
GPU night.** The claim to defend is either `ir_max`, where `S60` is ahead and no
ramp is, or the collapse, where `T2tA05` is. They do not point at the same
variant and the write-up cannot have both. Whichever goes to 100 prompts x
3 seeds (5.3 h) enters the table as a stated deviation from the paper's
constant lambda. If diversity becomes the claim, `div_pix` needs a perceptual
metric behind it and neither LPIPS nor CLIP is in the venv.

## The collapse lab: cause first, then the corrections (21/09 to 22/09)

**Put in place.** `collapse_lab/`, a folder of analysis scripts that read the
recorded runs and call `smc.fk` read-only, with its own `probe.py` that runs the
paper's setting with the ancestor matrix kept and a set of named arms (the
released code's floor at 0, lambda 2, both, lambda bisected to ESS = k/2, a
schedule without t = 80, a threshold at k/2). The reference row was regenerated
with the collapse fields (`sd_ref_fields100.json`, run 14): 96 runs of 100 end on
one root. Criteria for "solved", predictions and outcomes went into
`collapse_lab/ASSESSMENT.md` before each run; the findings, numbered, into
`collapse_lab/FINDINGS.md`.

**Why.** The screen of 21/09 had said the schedule is the lever, and left the
question of which claim the post defends. Neither could be written without the
cause: whether the collapse is the weights (an ESS question) or the paths (a
genealogy question), and whether any correction keeps lineages without giving
the reward back.

**What came out.** Two degeneracies that the ESS does not separate (finding 15):
reweighting best-of-4's four free draws by exp(10 ir) already gives an ESS of
1.23 out of 4, so at lambda = 10 the target itself carries about one particle;
and the number of final roots is a function of the recorded weights and the
resampler alone, predicted within 0.07 on fifteen arms without knowing anything
about the steering (finding 17, `n_coalescence.py`). One arm passes the
criterion fixed before the runs, the floor with lambda = 2 (5 % single-root, 3.0
roots of 4), and it does so by changing the target. Resampling less does not
keep lineages (finding 18), and no arm moves `ir_max` by more than the noise
while the mean of the four falls by 0.15 to 0.56 with the roots kept.

**Problems met.** A rerun with `--redo` overwrote six records of three arms with
values from another session; `floor2` and `lam0` lost those six prompts, `ctl`
was restored from the reference file. Behind it, the fact the incident exposed:
the `smc.models` path with resampling returns identical numbers within one
session and different roots and rewards across sessions, same code, same
weights, same seed (finding 19). Slot pairing across files from different
sessions was invalid, and with it findings 3, 5 and 5 bis as first read.

## The reference night and session C (22/09 evening to 23/09 midday)

**Put in place.** The authors' released code cloned outside the repository and
driven from `collapse_lab/ref/run_authors.py` under the paper's configuration,
with a shim for a module it imports and never uses; the four implementation
choices it makes and the paper does not state (`docs/reference_config.md`)
wrapped one by one into this repository's filter (`stat0`, `multi`, `vae`,
`idx`) and together (`R1`). Then session C: `ctl`, `lam0` and `floor2` at 100
prompts in a single process (`probe_C.json`), so that slot pairing is valid
inside the file; and a fourth run of the released code at seed 2024 under its own
seeding path.

**Why.** The +0.062 against the paper's +0.161 needed a reference that was not
this repository's code, and the findings that rested on slot pairing needed a
pairing that holds.

**What came out.** The paper's configuration is this repository's. The released
code without its filter returns best-of-4's rewards to the fourth decimal; with
its filter its four runs on the same 40 prompts land at -0.35, -0.04, -0.13 and
+0.10 against best-of-4, pooled -0.110 +/- 0.036 over 220 run-prompts. Two of
those runs share x_T and the DDIM noise and differ only by the stream of the
multinomial draw. A reading of their code on 23/09 found no re-seeding and no
bias between the two seeding paths: without a generator the multinomial advances
the global stream and changes the later DDIM noise, with one it does not, and
the two paths are two random runs of the same filter. The spread between them is
measured, not explained. The gap is bounded, not closed, and it is not in the
implementation (finding 16). Session C: the free path (`lam0`) returns best-of-4
slot by slot at correlation 1.00, so the non-reproducibility sits in the
resampling path only; findings 3 and 5 bis hold in weaker form (tau +0.14,
A - B +0.31); `floor2` paired by x_T costs -0.01 on `ir_max` and -0.25 on the
mean of the four (finding 20).

**Problems met.** The pre-registered prediction for `floor2`'s price missed on
the shallow side and the one for the fourth run of the released code missed on
the high side, both recorded in `docs/protocol_sd.md`. The post's "flat price of
-0.10 on `ir_max`" from 22/09 does not survive session C and was rewritten:
nothing beats best-of-4 by more than the noise on the best image, while the
lineages and the mean move by a lot.

## Where this sits against the plan (21/09)

The 16/09 plan had S6 (16/09 to 04/10) for modules 0 and 1 on CIFAR, S7 (05 to
11/10) for the MDLM port, S8 for the reproduced text figure and the v1 report,
20/10 for arXiv v1 and the public repository. As of 21/09 modules 0 and 1 are
done with two extra models (CelebA-HQ 256 and SD v1.5) that the plan did not
have before December, and the MDLM port has not started. The interface bet has
been tested twice and held both times.

The plan is now re-anchored on the Hi! PARIS Student Conference on AI:

| date | what |
|---|---|
| 7 Oct 2026 | submission deadline: the blog post and the public repository |
| 12 to 23 Oct | reviewing period; every submitting author reviews two other submissions on OpenReview |
| 20 Oct | arXiv v1, the date of the 16/09 plan, now after the submission |
| 26 to 30 Oct | additional reviews if needed |
| 5 Nov | decisions announced (taken 2 to 4 Nov); accepted work is then sorted into posters and orals |
| final version, conference day | not yet announced |

The review is double blind on four public criteria: quality, clarity,
correctness, and whether the project's context and approach are made explicit.
A "context of creation" field is required, saying where and how the project was
carried out and with whom; for this repository that field states the code
contract of 16/09 (the SMC core written by the author, the assistant on
plumbing, pointers and review), which is what `docs/decisions.md` and
`LEARNING.md` already make checkable. The official submission is the rendered
web page, not a PDF; a submission without a public, documented, licensed
repository is ineligible, which this one already satisfies.

Sixteen days from today to the submission. What has to exist by then: the
claim named (`ir_max` with `S60`, or the collapse with `T2tA05`), its 100-prompt
x 3-seed run done and in the table as a stated deviation, the figures rebuilt
from the JSON, and the blog post written from `docs/results.md`,
`docs/decisions.md` and this file. v1 for FK on images, MDLM and PG-DLM after
the conference decision.

## Re-anchored after the collapse lab (23/09)

The fork of 21/09 is closed: the claim is the collapse, with its cause and its
price, and `ir_max` is reported as indistinguishable from best-of-4 across
every configuration tried. The reference is the released code, read as a mean
under best-of-4 with a run-to-run spread wider than the paper's effect.

Fourteen days to the submission. What is left is writing, not measuring:
`docs/paper.md` at pass 9 with sections 2 to 5 to tighten, F0 to extend with the
reference rows, `collapse_lab/README.md` to complete, a reproduction on a clean
machine, the anonymisation. The three-process determinism test ran on 23/09
afternoon (`collapse_lab/nuit4.sh`): every process since the 22/09 evening
agrees to the fourth decimal, the 21/09 machine does not, and the cause is the
machine's execution path, not the repository. One GPU item stays open and is
cheap, if wanted: one more paired run of the released code to put a standard
error on its spread. **Data freeze on 25/09 evening**: whatever is not measured by then
becomes a sentence in the limits section. `smc/` stays frozen until the
submission; the floor as a `reward_floor` parameter comes after it.
