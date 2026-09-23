# Log of the blog-post loop

One block per pass, most recent on top. The loop, its scope and its review grid
are `.claude/skills/paper-loop/SKILL.md`; what the post has to contain is
`docs/paper_plan.md`.

## Pass 9 (23/09, the fourth run of the released code read in: section 2's table and paragraph)

**Reviewed.** Block 18's reference table after the last arm, and section 2.

**Found.** Correctness: the sentence "neither of them returns the paper's number on this
material" was falsified by the fourth run of the released code (0.949, +0.10 +/- 0.06
over best-of-4 on the same x_T, within a standard error of +0.161). Quality: the four
runs of the same code on the same 40 prompts span -0.35 to +0.10, two of them on
identical x_T and noise; that spread is the finding, and the post now says it instead
of a single number. The pooled -0.110 +/- 0.036 over 220 run-prompts replaces the
per-run rows in the table.

**Corrected.** Section 2's table and the paragraph under it. The "bounded" reading
stands with its reason changed: the mean is under best-of-4 and the run-to-run spread
is wider than the paper's effect.

**Open, for the author.** The variance of the released filter (median 0.25 of `ir_max`
between its runs of one prompt, against 0.03 for `ctl` between sessions) has two
suspects, the multinomial draw at flat weights and the terminal duplication, neither
measured on its own at this variance; it is a night of GPU if wanted. Section 5's
`floor2` price by x_T (pass 8, open b) still waits on the author's choice.

## Pass 8 (23/09, session C read in: section 3's last paragraph and section 7's reproducibility paragraph)

**Reviewed.** Block 18's session C paragraph (new), then the two paragraphs of the
post that rest on it.

**Found.** The previous draft said the wrong-root question was not measurable;
session C measures it on a same-process pairing, and the free path turns out to
reproduce across sessions (correlation 1.00 with the 20/09 file), which narrows the
non-reproducibility to the resampling path. Quality: constats 3 and 5 bis return in
weaker, measured form (tau +0.14, A - B +0.31). Correctness: the earlier
"trajectories diverge enough that slot i is not the continuation of slot i" was true
of the file pairs compared then and false of a same-process pairing; replaced.

**Corrected.** Section 3's last paragraph rewritten on the decomposition (0.78 /
0.23 / 0.47 / 0.80); section 7's paragraph names the free path as reproducible and
the guide's fp16 reward as the suspect.

**Open.** (a) The authors' code at seed 2024 under its own seeding is running; its
number goes to `reference_config.md`'s table and to block 18, not to the post
unless it changes the reading. (b) `floor2`'s price paired by x_T is 0.01 on
`ir_max` where the prompt-paired readout said 0.08: section 5 quotes the pooled
-0.069 over seven corrections and should quote the x_T-paired `floor2` figure next
to it in the next pass; not done here because block 18 carries both and the
sentence needs the author's choice of which to lead with.

## Pass 7 (23/09, review of sections 1, 6, 7, 8; A5 added to F5)

**Reviewed.** The four sections pass 6 had not reread, against blocks 5, 6, 9, 18.

**Found.** Correctness: "fourteen SD arms" in section 8 against "fifteen" in
section 5 (fifteen is right). Clarity: section 7 stated the origin of the
lambda = 0 residue as a fact; it is the one measured input that differs (the text
embedding, 1.6e-2 in fp16), stated as such now. Nothing else in the four sections.

**Corrected.** The two sentences. `scripts/fig_three_scales.py` gains the A5
panel (ESS / k of four free draws reweighted by exp(lambda ir) against lambda,
with fk4's first-step ESS at lambda = 10), so F5 carries the target's ceiling
next to the two collapse curves.

**Open.** One pass with two findings, none of them a number: the loop stops here
until session C lands, which will touch section 3's last paragraph and possibly
restore constats 3 and 5 bis.

## Pass 6 (23/09, review of pass 5 with the grid, four corrections)

**Reviewed.** Sections 2 to 5 reread against blocks 11, 12, 15, 17, 18 of
`results.md`, figures checked by `ls`.

**Found.** Correctness: "60 runs of 100 on a single one" for an arm measured on 40
prompts (it is 60 % of its runs); "the prediction held on both" arms, where only
`thr05` had a pre-registered coalescence prediction (`R1`'s multinomial-column
match was read after the fact); "paired on 40 prompts" for arms that ran on 20
(the ramps, `lam2`, `rise`); two FK numbers in section 2 (0.820 from the 300-run
row of 20/09, 0.826 from the row regenerated on 21/09) with no clause saying
which code each comes from. Clarity: figure 7 (ESS against gain) is no longer
referenced in the text after the trim; it stays in the annex list.

**Corrected.** The four sentences above; the reference row of the table names its
provenance.

**Open.** No question to the author from this pass. Next pass after session C
lands: section 3's last paragraph and the two suspended constats.

## Pass 5 (23/09, sections 2 to 8 rewritten on the final state; block 18 of `results.md` added)

**Reviewed.** The whole post against the grid, after writing: the reference
paragraph and table of section 2 (block 18), the two-degeneracies and target
paragraphs of section 3 (blocks 14, 18), section 4 filled from blocks 12, 16, 17,
section 5 rewritten as "repairing the collapse, and what it costs" from blocks 13,
15, 18, the three-scales line of section 6 (block 18), the reproducibility
paragraphs of section 7, the limits of section 8. Quality 4: the reference run
and the coalescence model are the two strongest pieces; the reference is 40
prompts on the x_T-paired row. Clarity 3: section 3 now carries three ideas
(weights, paths, target) and a reader needs the two figures to follow. Correctness
4: every number was re-read from block 18 or an earlier block; the "bit for bit at
lambda = 0" claim of the previous draft was false and is replaced in two places by
the measured 0.984. Context 5: the contract paragraph now names the lab.

**Found.** (1) Word count 4 030 after writing, against 3 500: trimmed the
worst-prompt anecdote, the ESS-does-not-predict aside, the two CIFAR paragraphs
and section 7's bug paragraph. (2) Three numbers quoted in the post were not in
`results.md` (the 0.34 slot difference, the 1.2 decoder shift, the 0.31 between
seedings): added to block 18. (3) The previous draft's thesis sentence in
`paper_plan.md` ("the two changes that act on it are the two results") no longer
matches the post: the two results are now the bounded reference and the
mechanism. `paper_plan.md` not edited in this pass; question to the author below.

**Corrected.** The two "bit for bit" claims; the 20-of-20 lineage count replaced
by 96 of 100; the schedule section's promise of "the claim the table carries"
(`S60` over `fk4` is not settled) rewritten as measured.

**Open, for the author.** (a) `paper_plan.md`'s thesis and word budget are the
21/09 ones; the post now follows the 22/09 plan (reference first, collapse
second, repairs third). Rewrite the plan or accept the drift. (b) Section 3's
figure of the ancestry is one prompt (`005848-0000`); the six are in
`collapse_lab/out/`. (c) Session C (ctl, lam0, floor2 at 100 in one process) is
running; if it lands as predicted, constats 3 and 5 bis return to section 3 in
their reformulated form. (d) Whether the image grid at 5.3 MB stays in the repo.

## Pass 4 (22-23/09, `docs/reference_config.md` created, `docs/protocol_sd.md` extended; `paper.md` untouched)

**Reviewed.** Not a writing pass on the post: the reference night and the collapse
night of the 22/09 plan, run back to back. What entered `docs/`: `reference_config.md`
(paper text against released code against this repo, cell by cell with sources) and
the pre-registration blocks at the end of `protocol_sd.md` (latents test, `R0`, `R1`,
bisection, `B1` rule, `thr05` with the coalescence model's prediction, `rise`, then
`R0g`, `authors_free_g`, `authors_free`), each written before its run. Findings and
confrontations: `collapse_lab/ASSESSMENT.md`.

**Found.** (1) The paper's stated SD configuration is this repo's; the released
script's defaults are another configuration the paper's own appendix scores lower.
(2) The released code under the paper's configuration returns 0.554 on the 100
prompts with its own seeding and 0.807 with our generator on 40, never the 0.898 of
Table 1; `smc/` with its four implementation choices returns best-of-4 (`R1`); the
gap is bounded, not closed, and sits in neither implementation. (3) The latents test
holds $x_T$ at the bit and the trajectories at correlation 0.98 or more, so constat 11
has a third reading: the `smc` path reproduces within a session and not across
sessions, with identical code and weights; slot pairing across files from different
days measures nothing. (4) The coalescence model predicts the lineages of fifteen arms
from the recorded weights alone, and its pre-registered `thr05` prediction (1.84 roots)
held against the plan's guess (2.4 to 2.8).

**Corrected.** Nothing in the post. In the reference: the generator runs of the
released code (`R0g`, `authors_free_g`) were seeded `42000 + i` by the driver's default,
and a five-prompt "their pipeline does not pair with ours" verdict was drawn from that
mismatch; rerun at seed 2024 their pipeline without FK returns `bon4` to the fourth
decimal, the passages in `protocol_sd.md` and `ASSESSMENT.md` say so, and `R0g24` (their
FK at seed 2024, 40 prompts) runs for the $x_T$-paired number. In the lab: the `--redo` of the six grid prompts
had overwritten six session-A records of `ctl`, `floor2`, `lam0` with session-B values;
`ctl` restored from `sd_ref_fields100.json` (without weight fields), the six of `floor2`
and `lam0` lost, scripts read weights only where present.

**Open, for the author.** (a) Whether to rerun `ctl` and `lam0` in one session (200
runs, about 3 h 20) to recompute constats 3, 5, 5 bis on a valid slot pairing, or to
drop them. (b) Section 2 of the post now has a reference measurement to write from:
"bounded", with the released code at or under best-of-4 here; the wording of the
two seeding paths (0.554 against 0.807) waits for `authors_free` and `authors_free_g`
at 40 prompts. (c) The `_b1` arms and the image grid come from session B and pair
only with `R1`; F1 and F2 say so in their captions. (d) `smc/` untouched; the floor
as a `reward_floor` parameter stays post-submission.

## Pass 3 (21/09, section 6 re-derived from the JSON and the saved tensors)

**Reviewed.** Every number of section 6, recomputed from `results/*.json` and
from `samples/*.pt` with the judge weights, rather than quoted from
`docs/results.md`. Judge runs on CPU, since the T4 is on night 1.

**Found.** Four errors, two of them mine and two inherited.

1. **Mine.** The post read "the minimum ESS falls from 16 to 1" inside the
   $\lambda = 1$ clause. Measured, the minimum ESS is 2.1 at $\lambda = 1$ and
   1.05 at $\lambda = 4$. `docs/results.md` had it right as a column.
2. **Mine.** "Past $\lambda = 1$ the guide is being fooled and $B$ stops
   following" is contradicted by the data. On CIFAR the judge's cat count keeps
   rising: 11 of 48 free, 15 at $\lambda = 1$, 32 at $\lambda = 2$, 37 at
   $\lambda = 4$ (argmax; 9, 14, 32, 34 at $p > 0.5$). What it costs is
   measurable in the same tensors: the mean pairwise pixel distance between the
   sixteen finals falls from 0.339 to 0.231 to 0.183. The cats are bought by
   spending the cloud, which is the collapse of section 3 at a second scale and
   a better paragraph than the one it replaces.
3. **Inherited.** "10 of 16 cats at $\lambda = 1$ against 4 of 16" is seed 2024
   alone, and it is the favourable seed: the three seeds read 10, 0 and 5.
   `docs/results.md` says "of seed 2024" and is honest; `README.md` places the
   same count inside a "three seeds" sentence and was not. Both now carry the
   48-particle counts.
4. **Inherited.** The red reward's "mean $p(\text{cat})$ of 0.43" is the
   `difference` seed-2024 cell, not `sum`, which reads 0.66, 0.37, 0.60. Over
   three seeds either potential gives 0.54, against 0.27 for the free model. The
   bound the score cannot exceed on pixels inside $[-1, 1]$ is 10.19, computed
   from its definition; `sum` reaches 10.90 at $\lambda = 1$, and at
   $\lambda = 8$ the pixels span $[-1.26, 1.30]$, which is the claim "the images
   leave $[-1, 1]$" measured rather than asserted.

**Also checked, outside section 6.** The CelebA glasses count in `README.md`
carried the same single-seed framing as (3). Recounted: 2 of 48 free, 21 at
$\lambda = 1$ (7, 0 and 14 per seed), and **0 of 48 at $\lambda = 2$**, where
`results.md` only described one grid of sixteen copies of one face. The collapse
is total there and is now a number. Note that CIFAR and CelebA part company past
$\lambda = 1$: the cat count rises while the glasses count goes to zero.

**Fixed.** Section 6 of `docs/paper.md` rewritten on the measured numbers.
`README.md`: the red reward sentence, the CIFAR judge sentence, the CelebA
sentence. `docs/results.md`: run 3's misattributed 0.43, run 5 and run 6 given
their three-seed recounts, each marked as a 21/09 recount rather than silently
changed. `docs/chronology.md` and `docs/results.md` run 11: the nine runs with
"identical" scores, which hold only to three decimals.

**Open.** The FID numbers of run 7 and the CIFAR fine-tuning FID are still
quoted and not re-derived; they need the Inception statistics and a GPU, so they
wait for the T4. The post cites none of them today.

## Pass 2 (21/09, first draft of sections 1, 2, 3, 6, 7, 8)

**Reviewed.** `docs/paper.md` as written this pass, against the five gates.
2 233 words with sections 4 and 5 left as marked holes, so about 3 150 once the
two nights are in. Written anonymous: no name, no institution, no repository URL
in the text.

**Found.** Three numbers inherited from the docs, and one of them was wrong.

1. The standard errors of the three-row table (0.082, 0.069, 0.069) are the
   per-prompt convention, seeds averaged first, and match. Over the 300 runs the
   same quantities read 0.056, 0.043 and 0.044, so a reviewer recomputing them
   the obvious way gets a different number. The post now states the convention
   and gives both.
2. The heavy left tail's -1.30 is a per-prompt mean over three seeds, not a run.
   The worst individual run is -2.60. The post said "minimum", which reads as a
   run. Corrected, and the -2.60 is given.
3. **Wrong.** `docs/chronology.md` says nine of 300 FK runs end with two
   identical ImageReward scores, and the post repeated it. At full precision the
   count is zero; nine is what appears after rounding to three decimals. The
   claim it supports gets stronger, not weaker: particles sharing a lineage
   still diverge numerically, so the damage really is selection and not a
   failure to diverge. The post now says both counts. `docs/chronology.md` still
   carries the loose version and is the author's file to amend.

**Checked and clean.** The citation: title, the seven authors in order, and ICML
2025 confirmed against PMLR v267 (`singhal25b`) and the ICML 2025 poster listing,
not only against the arXiv abstract. The anti-slop grep over `docs/paper.md`
returns nothing. Every file, field and function named in the post exists.

**Fixed.** The three corrections above, in `docs/paper.md`.

**Open.**

1. The figure is referenced as `../figures/fig7_ess_vs_gain.png`, which resolves
   from `docs/` and will not resolve in whatever the venue renders. To settle
   when the submission format is known, with open question 1.
2. The CIFAR-10 and CelebA-HQ numbers of section 6 are quoted from
   `docs/results.md` and were not re-derived from the JSON this pass. They
   should be before the post is final.
3. Sections 4 and 5 wait on the two nights. Night 1 started 21/09 at 07:40.

## Pass 1 (21/09, review of the two record fields, and the two open questions answered)

**Reviewed.** `scripts/run_sd_baseline.py`, the `root_slots` and `div_clip`
fields, against the launchers that are about to read them. Plus the answers
brought back on open questions 1 and 2.

**Found.**

1. `root_slots` is right and is the vector the wrong-root rate needs.
2. `div_clip` returns `None` on every record. It looks up `model_dict` under the
   key `"HPS-v2.1"`; `hpsv2.img_score.initialize_model` stores the model under
   `"model"` and its transform under `"preprocess_val"`. The lookup misses, the
   function returns `None` by its own guard, and no error is raised. Night 1
   would write 440 records with `div_clip: null`.
3. Three separate paths in `div_clip` return `None` and nothing tells them
   apart: fewer than two images, the missing key, and any exception. The
   `--limit 1` check catches a total failure because a four-particle run must
   not produce `null`, but an intermittent failure during the night would be
   silent and unattributable.
4. The embedding, once the key is fixed, is the ViT-H after `hpsv2.score` has
   loaded the HPS v2.1 state dict into it, so `div_clip` would be measured in the
   judge's representation and not in vanilla CLIP. That is a statement the post
   has to make where it reports the diversity result, and it is the cost of the
   cheap option that was chosen over a separate `ViT-B/32`.

**Open question 1 is answered in mechanism and not in fact.** An anonymised
mirror (anonymous.4open.science) is the standard answer and strips commit
metadata and account names. It does not touch file contents, and six tracked
files name the author:

| file | what |
|---|---|
| `LICENSE` | `Copyright (c) 2026 Giulio Fogato`, and the call requires a licensed repository |
| `README.md` | the byline, line 3 |
| `docs/ONYXIA_setup.md` | GitHub owner, MinIO bucket, clone URL |
| `notebooks/demo_DDPM.ipynb` | W&B run URLs under `wandb.ai/fogatogl-ensae-fr`, in saved cell outputs |
| `scripts/onyxia_bootstrap.sh` | GitHub owner, bucket, a default git name and e-mail |
| `scripts/sync_s3.sh` | the bucket |

`.claude/` is gitignored, so the personal context file is not exposed. The
`LICENSE` line is the one that cannot be scrubbed without weakening the
licensing the call requires, so it is a question for the organisers rather than
a thing to fix alone.

**Open question 2 keeps its answer and loses its justification.** `S60` leads
and the collapse is the second result, which is what the plan already had. The
reasoning brought back with it carries a number that exists in no file,
"+0.062 to +0.134 via `S60`": the screen measured `S60` - `bon4` at
+0.187 +/- 0.049 and `S60` - `fk4` at +0.094 +/- 0.080 on 20 prompts, and `S60`
at 100 prompts x 3 seeds is exactly what night 1 is for. Two other things in it
do not go in the post: the collapse is at the first scheduled step, t = 80 in
the paper's reverse-time convention, which is twenty of the hundred denoising
steps and not eighty; and it is this reproduction that falls short of the
paper's table, not the paper that failed, which is a much stronger claim than
anything measured here supports.

**Open.** Items 1 and 4 of pass 0 move here in sharper form: the organisers'
answer on the licensed repository under double blind, and the `div_clip` key,
which blocks night 1. Item 3 of pass 0 (French axis titles on figure 4) is
unchanged.

## Pass 0 (21/09, the plan and the plumbing under it)

**Reviewed.** Nothing yet in `docs/paper.md`, which does not exist. This pass
built what the post will be written from: the plan, the two launchers, the
analysis script, the pre-registration.

**Found.** One finding, and it is a correctness finding against the plan as it
was written. The plan's section 3 was going to argue that the premature
collapse is where the missing 60 % went, using the ESS as the evidence.
Measured run by run on the 300 existing runs, the ESS at the first scheduled
step does not order the runs by their paired gain: Spearman +0.018, and the
three ESS terciles give median gains of +0.060, +0.042 and +0.083, which is not
an order. The sentence the diagnostics of run 11 invite, that a low ESS predicts
a bad run, is not supported by them. The mechanism claim survives, because it is
a claim about what changing the schedule does to the average and that is what
runs 12 and the confirmatory run measure, but it has to be written narrowly.
Recorded in `docs/protocol_sd.md` under the pre-registration and in the plan.

**Fixed.** `figures/fig7_ess_vs_gain.png` is the flat cloud, titled for what it
shows rather than for what section 3 wanted. The plan gained a paragraph saying
what the mechanism claim may not say.

**Open.**

1. Double blind against a public repository. The call requires a public,
   documented, licensed repository and reviews double blind; a link to this one
   names its author. Author to read the call. This is the item that can make a
   finished submission ineligible, so it is first.
2. `root_slots` and `div_clip` in `scripts/run_sd_baseline.py`, author's, before
   night 1, since the images are not kept and a field added later means
   re-running the night.
3. `figures/fig4_sd_ir_hps.png` carries French axis titles ("la reward qui
   guide", "le juge, que rien n'optimise") and the post is in English. Figure 5
   is already in English. To settle before the figures are rebuilt, and it is a
   one-line change in `scripts/plot_fig4_sd.py`.
4. The slot correspondence between `bon4` and `fk4`, on which the wrong-root
   rate rests, has never been checked directly. To check with `--limit 1` once
   `root_slots` exists.
