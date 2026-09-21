# Plan of the blog post (21/09)

Target: Hi! PARIS Student Conference on AI, submission 7 October 2026, a
rendered Markdown page of 2 500 to 3 500 words plus the public repository.
Double blind, four public criteria: quality, clarity, correctness, and whether
the context and approach of the project are made explicit. The post itself is
`docs/paper.md`; this file is what it has to contain and what has to exist
before it can be written.

## The thesis

FK Steering claims that a particle filter along the denoising trajectory beats
best-of-N at equal budget. Reproduced from the equations on SD v1.5 with
ImageReward, the two baselines land on the paper's table and the FK row does
not: +0.062 over best-of-4 on 100 paired prompts where the paper has +0.161.
The post is about the missing 60 %. It is not a bug in the weights, the
resampler or the potential, all three of which are checked; it is a premature
collapse of the particle cloud. At the first scheduled step, twenty of the
hundred denoising steps in, the Tweedie estimate is a blur, the median ESS is
1.18 out of 4, and from there the four particles are one particle. Every
measurement in the post is a measurement of that mechanism, and the two changes
that act on it are the two results.

**What the mechanism claim may not say.** Within the paper's setting, the ESS at
the first scheduled step does not order the runs by how much they gain over
best-of-4: Spearman +0.018 on 300 runs, and the three ESS terciles give median
gains of +0.060, +0.042 and +0.083. The collapse is what a change of schedule
acts on, measured on the average across variants; it is not a per-run predictor
of failure. Figure 7 shows the flat cloud and the post says what it rules out.

## The two results, in order

**First, and it is the claim the table carries.** Dropping the first scheduled
step (`S60`, schedule `[0, 20, 40, 60]`) doubles the gain over best-of-4 on the
screen. What it buys is not fewer collapses but a collapse on a sharper
$\hat x_0$, so the surviving ancestor is a better one. A confirmatory run at
100 prompts x 3 seeds decides whether it holds, and it enters the table as one
stated deviation from the paper's schedule.

**Second, and it is a different claim.** `ir_max` cannot tell four copies of one
image from four images. Measured with `n_lineages` and `div_pix`, an FK run at
the paper's setting ends on one lineage in 20 of 20 prompts. A quadratic
$\lambda_t$ ramp under the tempering placement, crossed with the 0.5 threshold
(`T2tA05`), doubles the lineages and closes 38 % of the diversity gap to
best-of-4 at 5 standard errors, at no change in ImageReward. The post says
plainly that this is a claim about the cloud and not about the table's number.

## What is not claimed

Adaptive resampling by ESS threshold is textbook (Chopin and Papaspiliopoulos,
ch. 10) and sketched by the authors in their appendix C.3. The tempering
placement is the textbook Feynman-Kac sequence $G_t = \pi_t / \pi_{t-1}$
(ch. 17). Neither is presented as a contribution. They are implementations,
measured, of things the literature already has. The contribution of the post is
the reproduction and the mechanism behind its shortfall.

## Structure, with a word budget

Total about 3 300 words.

| # | section | words | rests on |
|---|---|---|---|
| 1 | What FK Steering claims, and the one figure that shows it | 350 | fig. 4, paper table 1 |
| 2 | The reproduction: the table, and the paired differences behind it | 450 | run 11, `make_table_sd.py` |
| 3 | Where the missing 60 % goes: ESS, lineages, the wrong root, and what the ESS does not predict | 550 | run 11 diagnostics, fig. 6, fig. 7 |
| 4 | Ablation of the schedule: `[0]`, `[0, 80]`, the paper's, `S60` | 450 | run 12, night 1 |
| 5 | $\lambda_t$: nothing on the reward, tempering on the diversity | 450 | runs 13 to 15, night 2 |
| 6 | The judge: HPS, and the same echo at three scales | 300 | run 9, runs 5 to 7 in annex |
| 7 | What was verified, and how | 400 | `tests/`, protocol, budget counters |
| 8 | Limits, and the context of creation | 350 | `decisions.md`, `LEARNING.md` |

Section 3 is the post. Sections 1 and 2 exist so that a reader who has not read
FK Steering can read section 3.

## Reviewer attacks, and the answer to each

Written now, so the post answers them in its own text rather than in a rebuttal.

**The winner's curse.** ImageReward both guides and scores, so FK is selected on
the metric it is judged by and best-of-4 is too. The answer is that both rows
are selected the same way on the same metric, which is the paper's own protocol,
and that the second metric, HPS, is reported and does not move (+0.0015,
1.4 standard errors). The post states that the +0.062 is a gain on the guiding
reward and nothing more.

**`D10` was never run.** The densest schedule is the one variant of the screen's
list that was costed and never launched, so the claim that the five-point
running max is not the suspect rests on `S60` and `S40` alone. Night 1 runs it.
If it is skipped, the post says the suspect list is not closed.

**The reward is the metric.** Section 6 answers with the three scales: the red
reward gamed past its own bound, the classifier reward where an independent
judge B confirms the gain, and ImageReward where HPS does not move. The same
shape three times is the argument; one scale would not be.

**`div_pix` is a pixel proxy.** It separates four copies from four images and
nothing more. Neither LPIPS nor CLIP was in the venv when the runs were made.
`div_clip`, a cosine distance between CLIP image embeddings, is added before
night 1 so the diversity claim rests on a perceptual metric as well. If it is
not in time, the diversity result is reported on `div_pix` alone and the post
says so.

**Three seeds on CIFAR.** The CIFAR numbers are three seeds and carry standard
deviations over seeds, not over prompts. They are in the annex and support the
mechanism, not the headline.

**FK is 11 % slower than best-of-4 at "matched budget".** Measured: 62.5 s
against 56.1 s per run, a ratio of 1.113, on a budget matched at 800 UNet rows
for both. The extra is the reward evaluations, five decodes plus five
ImageReward passes per particle. The honest statement of the comparison is
best-of-N at equal wall clock, which is best-of-4.45, and the post gives the
number rather than hiding behind the row count.

**The screen chose the variant and the confirmatory run tests it.** Eight ramps
and seven variants were read on the same 20 prompts before `S60` was picked.
That is a selection, and the confirmatory run at 100 prompts x 3 seeds on
prompts the screen did not see is what makes it a test. The post says which
numbers are screening and which are confirmatory, and the pre-registration in
`docs/protocol_sd.md` is dated before the run.

## What has to be added, and by whom

| what | cost | who | file | status |
|---|---|---|---|---|
| `root_slots`, the surviving root of each final particle | one line, no GPU | author | `scripts/run_sd_baseline.py` | to do before night 1 |
| `div_clip`, cosine distance between CLIP embeddings of the k finals | a function, one model load | author | `scripts/run_sd_baseline.py` | to do before night 1 |
| night 1: `S80`, `D10`, `S60` confirmatory, `fk4` reference with the new fields | 7.7 h GPU | author launches | `scripts/run_sd_night1.sh` | written |
| night 2: `T2tA05` at 100 x 3, `bon4` reference with `div_clip` | 6.9 h GPU | author launches | `scripts/run_sd_night2.sh` | written |
| ESS against gain, bootstrap, wrong-root rate, figure 7 | no GPU | assistant | `scripts/analyze_sd_collapse.py` | written |
| pre-registration of night 1 and night 2 | none | assistant | `docs/protocol_sd.md` | written |
| adaptive $\lambda_t$ by ESS target | a night, optional | author | `smc/fk.py` | pointer below |
| the post itself | the writing days | assistant, under the loop | `docs/paper.md` | not started |

## Sixteen days

| day | GPU at night | written during the day |
|---|---|---|
| 21/09 | night 1, 7.7 h | this plan, the launchers, the analysis script, the pre-registration |
| 22/09 | night 2, 6.9 h | night 1 read into `results.md`, fig. 7, sections 1 and 2 |
| 23/09 | free | night 2 read in, sections 3 and 4 |
| 24 to 27/09 | a third night, held for whatever night 1 raises | sections 5 to 8, first complete draft |
| 28/09 to 01/10 | free | loop passes on the draft, figures rebuilt from the JSON |
| 02 to 05/10 | free | README, reproduction lines, repository hygiene, loop passes |
| 06/10 | free | last pass, the double-blind check |
| 07/10 | submission |

**Order of cuts if it slips.** The third night goes first, then the adaptive
$\lambda_t$, then night 2. Night 1 is not cut: without it there is no
confirmatory run and the post has a screen and no test. The two record fields
are not cut either, since the images are not kept and a field added later means
re-running the night.

## Pointers, for what is the author's to write

**`root_slots`.** In `sample_fk`, the backward walk at lines 74 to 77 already
leaves `slots` as the vector of surviving roots; only `unique().numel()` is kept
today. The vector itself is what tells whether FK's surviving root is the one
best-of-4 would have chosen, since the two share slot $i$ and therefore the same
$x_T$.

**`div_clip`.** Next to `diversite`, same signature, `None` under two images.
Two ways in, and the choice is the author's: `clip.load("ViT-B/32")` with
`download_root` on the persistent cache, which is a fresh 350 MB download and an
independent encoder; or the ViT-H-14 that `hpsv2` already leaves in
`img_score.model_dict` after the first `hpsv2.score`, which costs nothing and
ties the diversity metric to the judge. The trade-off is independence against
cost and VRAM. Validate with `--limit 1` on `S80`, about a minute.

**Adaptive $\lambda_t$ by ESS target, optional and after the two nights.** At
each scheduled step, bisect on $\lambda_t$ so that the ESS after reweighting
equals $k/2$, with $\lambda_T = 10$ imposed so the target is unchanged. This is
adaptive tempering: Chopin and Papaspiliopoulos ch. 17, and Jasra, Stephens,
Doucet, Tsagaris 2011 for the ESS-targeting criterion. Where: `smc/fk.py`, at
the point where `lam_schedule[i]` is read, which means the schedule becomes a
callable or a mode rather than a list. The existing property test
`test_telescoping_time_dependent_lambda` has to stay green, and it is the test
that says whether the bisection broke the product constraint.

## Constraints on this plan

**The core is the author's.** `smc/weights.py`, `smc/resampling.py`,
`smc/fk.py`, the potentials, `smc/rewards*.py`, `smc/models.py`, and the tests
that check a mathematical property of those objects are written by the author,
as the code contract of 16/09 says. `sample_fk` and the two record fields are
the author's for the same reason: they are the experiment, not its plumbing.
The post's "context of creation" field states this contract, and
`docs/decisions.md` and `LEARNING.md` are what make it checkable by a reviewer.

**No generated-text texture, in the prose or in the code.** The rules are in
`.claude/skills/paper-loop/SKILL.md`, calibrated on what is already written
here: zero em dashes and zero `However` / `Moreover` / `Furthermore` in 70 kB of
`docs/`, bold as a paragraph label and not as emphasis, a sentence that could be
false or it does not get written. For the code, the rules of `CLAUDE.md` and the
ones the tempering review produced in `docs/protocol_sd.md`.

**The loop reviews itself.** The post is written and reviewed in passes against
the venue's four criteria and an integrity audit, logged in `docs/paper_log.md`,
without the author in the loop for each pass. What the loop may not do is
decide: a question that needs the author goes in the log and the pass moves on.

## Open questions, for the author

1. **Double blind against a public repository.** The call requires a public,
   documented, licensed repository, and the review is double blind. A link to
   this repository names its author. What the call says about this has to be
   read before the post is written, because the answer changes how sections 7
   and 8 are written. This is the one item that can make an otherwise finished
   submission ineligible.
2. **Which claim leads.** The plan above puts `ir_max` with `S60` in the table
   and the collapse with `T2tA05` as the second result. Night 1 and night 2 can
   overturn that order.
