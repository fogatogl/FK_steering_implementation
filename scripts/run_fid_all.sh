#!/bin/bash
# The FID stages that do not depend on the classifier reward, sequentially.
# Rerunnable as is after a cut: each stage resumes where it stopped.
#
#   nohup scripts/run_fid_all.sh &      then      tail -f /home/onyxia/work/ddpm/fid/run_fid_all.log
set -euo pipefail
cd "$(dirname "$0")/.."
FT=/home/onyxia/work/ddpm/weights/ft_class3/ddpm_last.pt
LOG=/home/onyxia/work/ddpm/fid/run_fid_all.log
mkdir -p "$(dirname "$LOG")"
{
  echo "=== $(date) start"
  python scripts/run_fid.py refs
  python scripts/run_fid.py gen --tag base
  python scripts/run_fid.py gen --tag ft_class3 --weights "$FT"
  python scripts/run_fid.py fid --gen ref_class3_sub2048 --ref ref_class3
  python scripts/run_fid.py fid --gen ref_class3_test    --ref ref_class3
  python scripts/run_fid.py fid --gen gen_base           --ref ref_class3
  python scripts/run_fid.py fid --gen gen_ft_class3      --ref ref_class3
  python scripts/run_fid.py fid --gen gen_base           --ref ref_test
  echo "=== $(date) end, files in /home/onyxia/work/ddpm/fid"
} 2>&1 | tee -a "$LOG"
