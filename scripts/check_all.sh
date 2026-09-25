#!/bin/bash
# The five checks of the post in one pass (plan, annex F). Blocking: prose, numbers, format;
# figures and language list what they find on the files the post uses and cites.
# Usage: bash scripts/check_all.sh   (after scripts/post_numbers.py when the data changed)
cd "$(dirname "$0")/.."
P=docs/paper.md
V=/home/onyxia/work/.venvs/ddpm/bin/python
fail=0
run() { local name=$1; shift; local out; out=$("$@" 2>&1); local rc=$?
        printf '%-9s %s  %s\n' "$name" "$([ $rc = 0 ] && echo ok || echo FAIL)" "$(echo "$out" | tail -1)"
        [ $rc = 0 ] || echo "$out" | grep -E "FAIL|MISSING|BLOCKING|LEXICON|EM-DASH|Title Case" | head -8 | sed 's/^/          /'
        return $rc; }

figs=$(grep -oE '\]\(\.\./figures/[^)]+\.png\)' $P | sed -E 's/^\]\(\.\.\///; s/\)$//' | sort -u)
fig_scripts="scripts/fig_hero_grid.py scripts/fig_algorithm.py scripts/plot_fig4_sd.py collapse_lab/o_ancestry_fig.py
             collapse_lab/q_image_grid.py collapse_lab/s_coalescence_fig.py collapse_lab/p_two_rewards.py
             scripts/fig_three_scales.py scripts/figstyle.py"
# FINDINGS.md and ASSESSMENT.md are the lab's working notes, in French by design; A.5 quotes them in English
cited=$(grep -oE '`[A-Za-z0-9_./-]+\.(py|sh|md|json)`' $P | tr -d '`' | sort -u | grep -vE 'FINDINGS.md|ASSESSMENT.md' \
        | while read f; do [ -f "$f" ] && echo "$f"; done)

run prose    python3 scripts/check_prose.py $P || fail=1
run numbers  python3 scripts/check_numbers.py $P || fail=1
run build    $V scripts/build_post.py || fail=1
run format   python3 scripts/check_format.py || fail=1
run figures  python3 scripts/check_figures.py $(for f in $figs; do [ -f "$f" ] && echo "$f"; done) $fig_scripts
run language python3 scripts/check_language.py README.md $cited $fig_scripts
echo "blocking checks: $([ $fail = 0 ] && echo 'all green' || echo 'at least one FAILED')"
exit $fail
