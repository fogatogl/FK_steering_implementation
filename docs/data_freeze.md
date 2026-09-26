# Data freeze

Written 2026-09-24T03:14:48+00:00 by `scripts/data_freeze.py`; `python3 scripts/data_freeze.py --check` lists what changed since.
After the freeze no arm is added; reruns of incomplete prompts only, as dated amendments.

| file | sha256 | bytes | runs |
|---|---|---|---|
| `results/bestofn.json` | `5d09a536605de676fd91731201f3e784eed0b97c0f91853be10ff9d32902f0bb` | 2940 | 15 |
| `results/c2_ir_hps.json` | `7e128b52b1d2b63786189af97a08b205b43dca3135ecc5affa01d7709cfeda7e` | 7963 |  |
| `results/fid.json` | `4f6a6d7199d71930f9e75ce85658321f0a39528ea7f46c433651acb2c50e8e02` | 1931 |  |
| `results/free_samples_seed12345.json` | `33f2b8ee756bc151eaa11da044207fd7fca5ba6462069765fda33e793858f561` | 6581 |  |
| `results/paper_table1_sd15.json` | `993cfe07d88aaf23a4356c27637760f76021b42e8c6dc9e9e77d5796c51082dd` | 418 |  |
| `results/sd_authors_R0.json` | `171b1c6361ba62da1c85acffa6be9d002db1efcee59da3fa58ca9350b4014c7c` | 207102 | 305 |
| `results/sd_baseline.json` | `64344498a43be148a2b0cdf517266b33146c115cfbc2d8f07b27bd8164e8d902` | 872189 | 900 |
| `results/sd_baseline_div20.json` | `5e3938a3219bca1cc6f560cf40aa29cad2309a9d57366fec88098dce99f9faae` | 50462 | 40 |
| `results/sd_ref_fields100.json` | `c654dc8828fb9a28c86800df389eaa474a06bbf4c7d796e7c1b99e19cf21dcdf` | 169305 | 100 |
| `results/sd_s60_full.json` | `f61012d533b8fd1ab89db0a8f4d81526b3322a710d34de11e55ee23655140e10` | 491647 | 300 |
| `results/smoke_hub.json` | `bf61befc1b88026926d2617a5216d050b40f7ed517a2962b49813b1ed50ffb7a` | 1296 | 2 |
| `results/sweep_k_classifier.json` | `2aa1c5309923d8f99aa928b944620e0127166b30eb2690ff22f021f1dc91f0da` | 600932 | 48 |
| `results/sweep_k_hub_classifier.json` | `d3645d5ecfdf8028e395972a48b3e560245ca693c5e5799208a1ceecefb3425f` | 19120 | 18 |
| `results/sweep_k_max_classifier.json` | `47a1ca9b990769883f54a5356e33ffeed9cd063ba57d10f59df72680f3a446ad` | 454567 | 36 |
| `results/sweep_lambda.json` | `b9a9c91474a85945fa83150bb10b5a6d5789f7ec2f31801ae8f6e88c906ce85b` | 38760 | 111 |
| `results/sweep_lambda_classifier.json` | `3ad301614dbf947f47405363e8af9f3599e67c8e2c0514ed70864a119239f14d` | 7819 | 18 |
| `results/sweep_lambda_hub_classifier.json` | `3bf2e2730c9827b5a99ec6e2490d835f00722ff42c8740657c7c7ce74e376a6f` | 8417 | 18 |
| `results/sweep_lambda_hub_red.json` | `8fff6e33464fd74a65bfbd3e70821568f0448933b19cf3d590b9974e580c5809` | 9480 | 21 |
| `results/sd_variants/A05.json` | `216ff8479f1ba9b5bfa735a546c3bf803cdeea6273f282d20910dfd3806ca944` | 27984 | 20 |
| `results/sd_variants/D10.json` | `8690473d2d067f7ca2f274ca650b65693a55dc2ea86b0a2ef39818d02943cf95` | 39668 | 20 |
| `results/sd_variants/K8.json` | `c0b4b6350ef6daf60fded94e4d85272bd5b0199e37ae281d0019aa3dd80f8e69` | 55259 | 40 |
| `results/sd_variants/L2.json` | `71c05beaa05f650c97f83731cb82f6132d78aa535889aee5b71c3cb001c01e49` | 27947 | 20 |
| `results/sd_variants/L20.json` | `57087d7fd56da24ad1d74380d3059086256a88491b37921711d4cd5bce314d15` | 27885 | 20 |
| `results/sd_variants/L5.json` | `33886a60616bec7a2bf29c897bb8ecc4b41904c81da68b2de2bd43044c0243e5` | 27886 | 20 |
| `results/sd_variants/S40.json` | `8f8f5617499e5ac3b130a699f6586cf407de8cb01ac9cb6fbffabadc278a0745` | 26332 | 20 |
| `results/sd_variants/S60.json` | `93cb28d697e4db8e547171d85708320b554a43d704d36a3cfa7a5f545b0bb678` | 27086 | 20 |
| `results/sd_variants/S80.json` | `a06ce5f254fc1f7bf2b9248cc58d49f0af0eed3a2000372f1d6f5cbf1fbc61d6` | 31183 | 20 |
| `results/sd_variants/T1.json` | `3c169c07b4eba975a3ddd7136efd271a5cbebf1312bf564fb0e0c637378cabdd` | 31559 | 20 |
| `results/sd_variants/T1A05.json` | `bcc5b679b03d1b3066bc01ebd86f443a800dedbc556ad4ba57c57dc6b92d5154` | 32296 | 20 |
| `results/sd_variants/T1t.json` | `9e3424639c2ecadc68c854d9ceefc20e49cc5c18f668f5d732fbb5aecf1ecc07` | 32291 | 20 |
| `results/sd_variants/T1tA05.json` | `de1475cf82ceeffc3d77e360787f169202004fe3edb00e5c42a1addbc290d75a` | 32332 | 20 |
| `results/sd_variants/T2.json` | `c077bb80e7aabebffe35b0933b40b10b44b6ced58b23aa9c5231b0df04b2a252` | 32205 | 20 |
| `results/sd_variants/T2A05.json` | `417ed749f8faab0e4b8d0037e7dc8bdb3ec70af32797a7bd3cc3ce939f4d726d` | 32287 | 20 |
| `results/sd_variants/T2t.json` | `b99e44687d97b9b2268fd538863aa6fa3a9bec82838f5c3ddda84a01854706c5` | 32217 | 20 |
| `results/sd_variants/T2tA05.json` | `32f3ac9b16af653a523aab50f7dc792a54aa65de6016ce650d1c3c4b2958e254` | 32298 | 20 |
| `results/sd_variants/fk4_diff.json` | `60423db8ac286a831cf31cff98abf214606e01c18b6dad1d62a5a43a07518d49` | 45234 | 20 |
| `results/sd_variants/fk4_stat.json` | `4b17f30116ac3eeb402b8d402b80cf42a0ba0085dd3eb7a1eee6f191699aee0a` | 45095 | 20 |
| `collapse_lab/out/det_a.json` | `619fa01489ab0f11483e3c4d31821b8e90d9e642dc5513cf882ed7b12d482a63` | 5243 | 2 |
| `collapse_lab/out/det_b.json` | `e3838732daa276df29f85d739831bc73bc2b51a01bbc4a7013c358cb743b1dab` | 5243 | 2 |
| `collapse_lab/out/det_c.json` | `7fe1b67ac55006331246d19f1d761634c933cc1d024e862e300a070934976227` | 5243 | 2 |
| `collapse_lab/out/det_old.json` | `352a50197dc4fb73eee1ff5aa2a4ffdfe7fd29ab5c4d7b42a657e509432e41f0` | 4217 | 2 |
| `collapse_lab/out/diag_authors_0.json` | `4bd0fcb1b90d8eeb5d2065f0923c65eebdc46c793fc9ce0862db321ca2bdff77` | 2718 |  |
| `collapse_lab/out/diag_authors_0_s2024.json` | `ef3d804d1fc2e373d4b2c85624a94cc55101a147fe4560dac3cc4d101f187f45` | 2726 |  |
| `collapse_lab/out/diag_authors_2.json` | `5802d1fc559ed9d536d0380eb2743533ca5a2aef0de68b4e23b00395487d78d0` | 2763 |  |
| `collapse_lab/out/latents_0.json` | `c8c3ed11789d384d1e5b60e88b52cdb98f715310771c49f0247299d28fe99a30` | 1625 |  |
| `collapse_lab/out/latents_1.json` | `e19e94bc6d8017621c7c9a6d56b8d975062ffc39455ed086dcc49b688aa68bbe` | 1636 |  |
| `collapse_lab/out/probe.json` | `6c5c88b1a69bf7bdfb341217b039e5552d47b2f8453a84f08e46c03390c98442` | 1438138 | 609 |
| `collapse_lab/out/probe_C.json` | `ad75a88136d5d22e245e33708d824fea14f06a2ba83245da843c2757f855ee8c` | 773986 | 300 |
| `collapse_lab/out/session_D/hps_finals.json` | `d3d73025b95be207c69cb4fd89d97708901950de2820c68c7e44561ddddbc594` | 21221 |  |
| `collapse_lab/out/session_D/probe_D.json` | `d8116222c22b62477d7013aa483344e7580a22a42c293f593d58f81562224395` | 895589 | 300 |
| `data/imagereward-benchmark-prompts.json` | `34682e824ebb3ad8e2531b3cde1b362dc72e1e29946ff6fab77bb49e9edb4b21` | 14558 |  |
| `data/prompts_subset_40.json` | `698eb9ad38f5da06dfa6867e6c01456b1a645f95996f5d4ce6885fec983fd7b8` | 6075 |  |
| `data/visual_pool.json` | `6aaebfa03b037c3ca674da0f17d5752255e13126e675a9346dbdb19ccea948d2` | 7212 |  |
| `data/visual_selection.json` | `a57b04645262baf63f0b91027eaacd4157759962f0b58dd7d2703ffc751436ae` | 3362 |  |
| `data/visual_selection_pre_amendment.json` | `73e32ba4d2b9cb6da2ff483b53d1e64d8a70daa586c09866158b6a952df056f7` | 3191 |  |

## Amendment, 25/09/2026, after the runs of sessions E and F

Three records written after the freeze, each pre-registered in `docs/protocol_sd.md`. Session E
completes the released code's seed-2024 runs from the first 40 prompts to the 100 (a rerun of
incomplete prompts, which the freeze allows), and checks the machine on two prompts. Session F adds
one arm, the released code at its commit before the fix of the MAX potential, which the freeze does not
allow: it was added because the seventh cold review found the fix, and Table 1 predates it. One frozen
file changed: `data/visual_selection.json`, rewritten by the F5 amendment of 25/09
(`docs/visual_selection.md`: F5 moves to `004971-0071`, the first rule's pick kept as `F5_first_rule`);
its new sha256 is below. The other 54 are unchanged.

| file | sha256 | bytes | runs |
|---|---|---|---|
| `results/sd_authors_R0_100.json` | `21e2ba9e4468563e51de3227996fb2902ffb524182237ab287a28bca0ef33c89` | 136040 | 200 |
| `results/sd_authors_prefix.json` | `62abd4ac736cde8db67e3807ee445bec96edd4c9008764faecad1b912072a67b` | 69533 | 100 |
| `collapse_lab/out/session_E/t4_check.json` | `69d833a6d4440cdf9d8af4d6315e05eae2836f9e307d42e45c05137442c930b8` | 11933 | 4 |
| `data/visual_selection.json` | `0511d6e2bcc30173b07bee2900a7060cf8cda57bce2e18d5bbfa6903cf9ef1e8` | 3949 |  |

## Amendment, 26/09/2026, F1's third row chosen by the author

`data/visual_selection.json` rewritten by `scripts/select_visual_prompts.py` after the F1 amendment of
26/09 (`docs/visual_selection.md`): F1's third prompt moves from `007171-0104` to `007187-0044`, chosen
by eye among the 46 candidates, and the rule's pick is kept as `F1_first_rule`. No run record changed;
the other 57 frozen files are unchanged.

| file | sha256 | bytes | runs |
|---|---|---|---|
| `data/visual_selection.json` | `2d0e9b7dedc4a4b8395fba1a56d9b23c06fa86f5694a9b020aa084c06648806a` | 4031 |  |

## Amendment, 26/09/2026, F5 and F4 chosen by the author

`data/visual_selection.json` rewritten by `scripts/select_visual_prompts.py` after the F5 amendment of
26/09 (`docs/visual_selection.md`): F5, which F4 follows, moves from `004971-0071` to `007171-0040`,
chosen by eye among the 43 eligible prompts where FK ends on one root; the earlier picks are kept as
`F5_second_rule` and `F5_first_rule`. No run record changed; the other 57 frozen files are unchanged.

| file | sha256 | bytes | runs |
|---|---|---|---|
| `data/visual_selection.json` | `382e61fc63b38d9a5a6d0be5d783fd3dd52e10e30975238713ee4161288f63ba` | 4303 |  |
