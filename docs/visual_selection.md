# Which prompts the post shows as images, and why

Written 23/09/2026, before session C. Nothing below looks at a reward. The appendix of
the post (A.3) points here.

## Why the choice is made after session C, not now

The images the post shows will come from session C (`ctl`, `lam0` and `floor2` on the
100 prompts, one process). The `smc.models` path gives the same numbers within a session
and different roots across sessions (finding 19), so a prompt chosen on the 21/09 `ctl`
file could be a gain there and a loss in the image on screen. The eligibility filter is
therefore fixed now, from the prompt text, and the numbers that rank the eligible
prompts are read from session C alone.

Session C saves, for every prompt and every arm, the four final images (512 px PNG) and
the four decoded Tweedie estimates at each of the five scheduled steps (128 px WebP).
Decoding is already done for the guide; saving costs disk, not GPU time. The selection
then needs no rerun.

In session C the free baselines are the `lam0` arm: slot 0 is the single free sample,
the maximum of the four slots is best-of-4, and both share their four `x_T` with `ctl`.
Before the first resampling (t = 80) the four `ctl` particles and the four `lam0`
particles are the same trajectories, which is what makes W2 valid.

## Eligibility (`data/visual_pool.json`)

A prompt is eligible if it names something a reader can check by eye (a subject, an
object, a colour, a material, a count, a spatial relation); names no person, artist,
brand or franchise; is not a human portrait; and is not abstract or style-only. Each
eligible prompt carries a category (animal, vehicle, food, object, scene, figure) and a
one-line note of what to check in the image. 46 prompts are eligible, 54 are excluded
with their reason code (one of them, `005895-0048`, classified on 23/09 after the first pass
missed it: three named artists, rule E2). The script refuses to run if any benchmark prompt is left
unclassified.

## The rule

With `free = lam0.ir[0]`, `bo4 = max(lam0.ir)`, `fk = ctl.ir_max`, seed 2024:

| role | how many | rule |
|---|---|---|
| gains | 8 | `fk > bo4`, ranked by `fk − free`, at most 2 per category |
| median | 1 | `fk − bo4` closest to the median over the eligible pool |
| loss | 1 | among `fk < bo4`, closest to the median of that subset (a typical loss) |
| worst loss | reported | the most negative `fk − bo4`, in the appendix table |

Ties go to the lower prompt id. Where each is used:

| use | prompts |
|---|---|
| F1, main text | the three best gains from three different categories |
| F1, appendix | all ten, with the median and the loss labelled as such |
| F5, main text | among the ten, `ctl` ends on one root and `floor2` on three or more; largest `fk − free` |
| W1 (genealogy explorer) | the ten |
| W2 (pick the winner) | 10 prompts drawn from the whole eligible pool with `random.Random(2026)` |

W2 does not use the ten. Its question is whether the ranking at t = 80 predicts the
final ranking; prompts selected for a large FK gain are prompts where the kept root
turned out well, which would bias the answer toward yes.

`funny peanut butter` (`010856-0009`) is eligible. If the rule does not select it, it
appears in the appendix under the label "chosen by hand, the example that started the
project", and nowhere in the main text.

## Amendment, 23/09/2026 18h49 UTC (written before any session D record was read)

What happened after the text above. Session C ran from about 07h30 to 12h30 and saved no
image: `probe.py` had no thumbnail code and the run did not pass `--save-images`. The rule
was run on session C's numbers at 17h34, and its output is kept as
`data/visual_selection_pre_amendment.json`. Its ten prompts have no image on disk, so the
figures cannot show them from session C. The pod restarted at 18h02 on an NVIDIA A2; the
GPU of the earlier sessions is not recorded.

What changes.

- The source. The rule reads session D (`collapse_lab/out/session_D/probe_D.json`,
  `collapse_lab/nuitD.sh`): `ctl`, `lam0` and `floor2` at the 100 prompts in one process,
  launched at 18h47, which saves the four finals and the guide's decoded Tweedie estimates
  at the five scheduled steps for every prompt and arm, and writes `session_id` and the
  device in every record. The numbers that rank a prompt and the images that show it then
  come from the same records. Session C stays the source of the post's statistics.
- Completeness. A prompt enters the ranking only if the three arms exist, carry every
  field the script requires (`REQUIRED`), share one `session_id`, and have their finals and
  thumbnails on disk. An incomplete prompt is replaced by the rule, never by hand: it
  leaves the ranking and the next one takes its place. The script lists each one in
  `excluded_incomplete`; the replacements are copied below with the date.
- W2. It takes the first ten complete prompts of a fixed permutation of the eligible pool
  (`random.Random(2026)` shuffling the sorted ids), instead of a sample of the complete
  pool, so one missing prompt shifts the list by one and changes nothing else.

The rest of the rule is unchanged. The prediction on how much the selection moves from
session C to session D is in `docs/protocol_sd.md` ("Pre-registration of session D").

Replacements: none. Run on 24/09 03h05 UTC on session D (one process, `session_id`
`2026-09-23T18:47:29+00:00/...`): 46 complete eligible prompts of 46, no prompt excluded. Five of
the eight gains differ from the pre-amendment selection read on session C, as the protocol
predicted for a change of machine. `funny peanut butter` is not selected and appears in the
appendix under its hand-picked label.

## Amendment, 25/09/2026 (F5 only, written after the images were seen)

On the first rule's F5 prompt (`010525-0074`, the retriever) the free sampler's row and
floor + lambda = 2's are nearly the same four images: the weights stay nearly flat at every
step (ESS 3.8 to 4.0 of 4), so the variant barely moves its four images, and the figure shows the collapse of FK but
not what keeping the lineages does. F5 is therefore chosen by a second rule, stated here after
the images were seen and applied to the records alone: among the 46 complete eligible prompts
where `ctl` ends on one root and `floor2` keeps four, the one where `floor2`'s mean ImageReward
rises most above `lam0`'s. It returns `004971-0071` ("a sports car, motion blur"), at +0.43.
The first rule's choice stays in `data/visual_selection.json` as `F5_first_rule`. F1, W1 and W2
are unchanged. The post says, in A.3, that F5 was chosen after the data.

## Amendment, 26/09/2026 (F1 only, chosen by the author after the images were seen)

The author looked at the 46 complete eligible prompts drawn as F1 rows (free sample, best-of-4, FK,
from the same four noises) and asked for the third row to show `007187-0044` ("an underwater
rollercoaster, cinematic, dramatic, -") in place of `007171-0104` (the scampi). Both are among the
eight gains. F1 stays the best gain of each of three categories; what changes is the categories,
now figure, animal and scene (`F1_CATS` in `scripts/select_visual_prompts.py`) where the rule took
the first three in the order of the gains (figure, animal, food). The rule's choice stays in
`data/visual_selection.json` as `F1_first_rule`. F5, W1 and W2 are unchanged. The post says, in F1's
caption and in A.3, that one category was chosen by eye.

## Amendment, 26/09/2026 (F5 and F4, chosen by the author after the images were seen)

F5 and F4 now show the free sampler against FK only: on both earlier picks floor + lambda = 2 stayed
close to the free sampler, and the author wants the two figures to show the collapse; floor + lambda =
2 stays in the text and in F7. The author looked at the 43 complete eligible prompts where `ctl` ends
on one root, drawn as two-row grids (`lam0`, `ctl`), and chose `007171-0040` ("A foggy forest with
cherry blossom leaves on the ground, liminal, quiet") for images that follow the prompt (`F5_CHOICE`
in `scripts/select_visual_prompts.py`). On it `ctl` keeps the best of the four roots from the first
step on, and its best image reads 0.09 under best-of-4 (fk - free +0.23). The second rule's choice
stays as `F5_second_rule` (`004971-0071`), the first rule's as `F5_first_rule`. F1, W1 and W2 are
unchanged. The post says, in F5's caption and in A.3, that the prompt was chosen by eye.

## Running it

    python scripts/select_visual_prompts.py collapse_lab/out/session_D/probe_D.json

writes `data/visual_selection.json` (ids, roles, the numbers behind each choice, the
incomplete prompts and what each misses) and prints one line per selected prompt.
`--no-files` skips the image check (for a dry run on numbers alone). The figure scripts
(`fig_hero_grid.py`, `q_image_grid.py`) and the demo read their prompt list from that
file and from nowhere else.
