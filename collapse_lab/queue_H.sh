#!/bin/bash
# Session H, what remains after H1, as a queue that can be relaunched at any time: after an Onyxia
# restart, /home/onyxia/work/ops/resume_H.sh. It waits while nuitH.sh or anything else holds the GPU,
# then skips what is on disk.
# H2: the six passes of nuitH.sh. A seed-once pass does not resume: a partial one is set aside in
#     $OUT/h2_interrupted_*.json and rerun from its first prompt, which reproduces it (amendment of 17h50).
# H1-bis: ctl lam0 free200 d200 on prompts 41-100; probe.py resumes by (prompt, arm). Skipped if
#     $OUT/NO_H1BIS exists.
# Stops without retrying if the GPU or a venv's torch build is not the session's.
#   setsid nohup bash collapse_lab/queue_H.sh >> collapse_lab/out/session_H/queue_H.log 2>&1 < /dev/null &
#   DRY=1 RES=<copy of the results> ASIDE=<dir>: prints the runs, touches only RES and ASIDE
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=/home/onyxia/work/hf_cache
OUT=collapse_lab/out/session_H
V=/home/onyxia/work/.venvs/sd/bin/python
VP=/home/onyxia/work/.venvs/fkd_pinned/bin/python
PREFIX=/home/onyxia/work/fkd_ref/prefix_6726324
DRY=${DRY:-0}
RES=${RES:-results/sd_authors_once.json}
ASIDE=${ASIDE:-$OUT}
FLAG=/home/onyxia/work/ddpm/queue_H.active
LOCK=/home/onyxia/work/ddpm/queue_H.lock
[ "$DRY" = 1 ] && LOCK=$LOCK.dry

exec 9>"$LOCK"
flock -n 9 || { echo "queue_H already running"; exit 0; }
echo "=== queue_H started $(date -u +%F_%T) UTC, DRY=$DRY"
[ "$DRY" = 1 ] || touch $FLAG

run() { if [ "$DRY" = 1 ]; then echo "DRY: $*"; else "$@"; fi; }
stop() { echo "STOP: $*"; [ "$DRY" = 1 ] || rm -f $FLAG; exit 1; }
backup() { [ "$DRY" = 1 ] && return; bash /home/onyxia/work/ops/s3_sync.sh repo >/dev/null 2>&1 \
    && echo "backup $(date -u +%T)" || echo "backup FAILED $(date -u +%T), the volume keeps the files"; }
busy() { pgrep -f '^bash collapse_lab/nuitH\.sh|python collapse_lab/(probe|ref/run_authors)\.py' >/dev/null \
    || nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; }
group() {  # group <python> <torch build> : the GPU and the build the session ran on
    local got
    got=$($1 -c "import torch; print(torch.cuda.get_device_name(0), torch.__version__)" 2>&1 | tail -1)
    [ "$got" = "NVIDIA A2 $2" ] || stop "this pod gives '$got', the session ran on 'NVIDIA A2 $2'"
}
count() { $V -c "import json, pathlib; p = pathlib.Path('$RES')
print(sum(r['sampler'] == '$1' and r['seed'] == $2 for r in json.loads(p.read_text())['runs']) if p.exists() else 0)"; }

if [ "$DRY" = 1 ]; then busy && echo "DRY: would wait, the GPU is busy"
else while busy; do sleep 120; done; fi
echo "GPU free $(date -u +%T)"

group $VP 2.4.0+cu124
for S in 42 43 44; do
    for P in paper free; do
        SAMPLER=authors_${P}_once_prefix
        n=$(count $SAMPLER $S)
        [ "$n" = 100 ] && { echo "H2 $SAMPLER seed $S: done"; continue; }
        if [ "$n" != 0 ]; then
            $V - "$RES" "$ASIDE" $SAMPLER $S <<'EOF' || stop "could not set aside the partial pass"
import json, sys, time
from pathlib import Path
res, aside, sampler, seed = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
runs = json.loads(res.read_text())["runs"]
part = [r for r in runs if r["sampler"] == sampler and r["seed"] == seed]
dest = aside / f"h2_interrupted_{sampler}_{seed}_{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}.json"
dest.write_text(json.dumps({"runs": part}, indent=1))
tmp = res.with_suffix(".json.tmp")
tmp.write_text(json.dumps({"runs": [r for r in runs if not (r["sampler"] == sampler and r["seed"] == seed)]}, indent=1))
tmp.replace(res)
print(f"H2 {sampler} seed {seed}: {len(part)} records of a partial pass moved to {dest}")
EOF
        fi
        [ $P = paper ] && FLAGS="--config paper" || FLAGS="--no-smc"
        FKD_ROOT=$PREFIX run $VP collapse_lab/ref/run_authors.py $FLAGS --seed-once --seed $S --limit 100 \
            --tag _prefix --out $RES
        [ "$DRY" = 1 ] && continue
        n=$(count $SAMPLER $S)
        [ "$n" = 100 ] || stop "H2 $SAMPLER seed $S: $n of 100 after the pass"
        backup
    done
done
echo "SESSION H2 DONE"

[ -e $OUT/NO_H1BIS ] && { echo "H1-bis skipped ($OUT/NO_H1BIS)"; [ "$DRY" = 1 ] || rm -f $FLAG; exit 0; }
group $V 2.14.0+cu132
$V -c "import json; json.load(open('$OUT/images/index.json'))" || stop "$OUT/images/index.json does not parse"
IDS=$($V -c "import json; d = json.load(open('data/imagereward-benchmark-prompts.json'))
d = d if isinstance(d, list) else next(v for v in d.values() if isinstance(v, list))
print(*[str(e.get('id', i)) for i, e in enumerate(d)][40:100])")
run $V collapse_lab/probe.py --arms ctl lam0 free200 d200 --limit 100 --prompt-ids $IDS --save-images \
    --out $OUT/probe_H1bis.json
if [ "$DRY" = 0 ]; then
    n=$($V -c "import json; print(len(json.load(open('$OUT/probe_H1bis.json'))['runs']))")
    [ "$n" = 240 ] || stop "H1-bis: $n of 240 runs"
    backup
    rm -f $FLAG
fi
echo "SESSION H1-BIS DONE"
