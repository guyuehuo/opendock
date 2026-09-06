#!/usr/bin/env bash
# Parallel driver for the PDBbind CASF-2016 docking-power benchmark.
#
# Usage:
#   ./run_all.sh [opendock|idock|rmsd|all]     (default: all)
#
# Env knobs:
#   JOBS        number of parallel workers          (default: nproc)
#   STEPSCALE   OpenDock steps multiplier            (default: 1)
#   IDOCK_BIN   path to the idock binary             (default: $IDOCK_BIN)
#
# Preparation (01) and per-tool runs (02_*) can be called directly; this
# script only parallelises the docking runs and then chains 03/04.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="${PY:-python}"
JOBS="${JOBS:-$(nproc)}"
STEPSCALE="${STEPSCALE:-1}"

TOOL="${1:-all}"

# build the optional --steps-scale text injected into the runner invocation
STEPSCALE_TEXT=""
if [ "$STEPSCALE" != "1" ]; then
  STEPSCALE_TEXT=" --steps-scale $STEPSCALE"
fi

run_opendock() {
  echo "[run_all] OpenDock jobs (JOBS=$JOBS) ..."
  "$PY" 02_run_opendock.py --list-jobs |
    xargs -n 4 -P "$JOBS" bash -c \
      'exec "$0" 02_run_opendock.py --code "$1" --source "$2" --mode "$3" --cfg "$4"'"$STEPSCALE_TEXT" "$PY"
}

run_idock() {
  if [ -z "${IDOCK_BIN:-}" ]; then
    echo "[run_all] IDOCK_BIN not set; skipping idock jobs"
    return 0
  fi
  echo "[run_all] idock jobs (JOBS=$JOBS) ..."
  "$PY" 02_run_idock.py --list-jobs |
    xargs -n 4 -P "$JOBS" bash -c \
      'exec "$0" 02_run_idock.py --code "$1" --source "$2" --mode "$3" --cfg "$4"' "$PY"
}

run_rmsd() {
  echo "[run_all] computing RMSDs ..."
  "$PY" 03_compute_rmsd.py
  echo "[run_all] aggregating success rates ..."
  "$PY" 04_aggregate.py
}

case "$TOOL" in
  opendock) run_opendock ;;
  idock)    run_idock ;;
  rmsd)     run_rmsd ;;
  all)
    run_opendock
    run_idock
    run_rmsd
    ;;
  *)
    echo "unknown tool: $TOOL (opendock|idock|rmsd|all)" >&2
    exit 1
    ;;
esac
