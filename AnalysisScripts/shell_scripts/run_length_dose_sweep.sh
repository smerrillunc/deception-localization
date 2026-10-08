#!/usr/bin/env bash
set -euo pipefail

# Sweep steering length against dose for one circuit across the environments.
# Cells are claimed from a shared manifest, so running this on several GPUs (or
# several machines sharing the output directory) divides the work without any
# coordination beyond the lock directory.

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYSIS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$ANALYSIS_DIR/.." && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/Results}"

OUT_DIR="${OUT_DIR:-$RESULTS_ROOT/Steering/runs}"
LOG_DIR="${LOG_DIR:-$RESULTS_ROOT/Steering/logs}"
LOCK_DIR="${LOCK_DIR:-$RESULTS_ROOT/Steering/locks}"
VECTOR_PATH="${VECTOR_PATH:-$ANALYSIS_DIR/steering_support/bs_circuit.pt}"

ENVS="${ENVS:-bs gridworld interview car_sales advisor_audit}"
ALPHAS="${ALPHAS:-0.5 1.0 2.0}"
LENGTHS="${LENGTHS:-50 100 250 500 1000}"
GPUS="${GPUS:-0}"

DELTA_MODE="${DELTA_MODE:-relative}"
MAX_NEW="${MAX_NEW:-3072}"
TARGET_PREFIXES="${TARGET_PREFIXES:-8}"
SAMPLES="${SAMPLES:-50}"
SCREEN_SAMPLES="${SCREEN_SAMPLES:-16}"
MIN_SCREEN="${MIN_SCREEN:-0.80}"
DUMP_GENERATIONS="${DUMP_GENERATIONS:-1}"

usage() {
  cat <<'EOF'
Usage:
  run_length_dose_sweep.sh [--gpus "0 1 2 3"] [--envs "bs car_sales"] [-- extra args]

Every setting is also an environment variable, so a full override looks like:

  GPUS="0 1" ALPHAS="0.5 1.0" LENGTHS="100 500" ./run_length_dose_sweep.sh

Options:
  --gpus "LIST"        GPU ids to run on, one worker each.   Default: 0
  --envs "LIST"        Environments to sweep.                Default: all five
  --alphas "LIST"      Steering doses.                       Default: 0.5 1.0 2.0
  --lengths "LIST"     Steered decode steps before release.  Default: 50 100 250 500 1000
  --vector_path PATH   Circuit to steer with.                Default: steering_support/bs_circuit.pt
  --out_dir DIR        Where run JSONL is written.           Default: Results/Steering/runs
  --no_generations     Skip the raw-completion dumps (much smaller output).

Anything after `--` is passed through to prefix_steering.py.
EOF
}

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) GPUS="$2"; shift 2 ;;
    --envs) ENVS="$2"; shift 2 ;;
    --alphas) ALPHAS="$2"; shift 2 ;;
    --lengths) LENGTHS="$2"; shift 2 ;;
    --vector_path) VECTOR_PATH="$2"; shift 2 ;;
    --out_dir) OUT_DIR="$2"; shift 2 ;;
    --no_generations) DUMP_GENERATIONS=0; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    *) echo "unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

mkdir -p "$OUT_DIR" "$LOG_DIR" "$LOCK_DIR"

cells() {
  for env in $ENVS; do for a in $ALPHAS; do for len in $LENGTHS; do
    echo "$env $a $len"
  done; done; done
}

claim() {
  # one directory create per cell: the first worker to make it owns the cell
  while read -r env a len; do
    local tag="sw_${env}_a${a//./}_L${len}"
    [[ -s "$OUT_DIR/$tag.jsonl" ]] && continue
    mkdir "$LOCK_DIR/$tag" 2>/dev/null || continue
    echo "$env $a $len $tag"
    return 0
  done < <(cells)
  return 1
}

worker() {
  local gpu="$1"
  while true; do
    local got; got="$(claim)" || { echo "[gpu $gpu] nothing left"; return 0; }
    read -r env a len tag <<<"$got"
    echo "[gpu $gpu] start $tag"
    local dump=()
    [[ "$DUMP_GENERATIONS" == "1" ]] && dump=(--dump-generations "$OUT_DIR/$tag.gen.jsonl")
    CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$PYTHON_BIN" "$ANALYSIS_DIR/prefix_steering.py" \
        --env "$env" --vector-path "$VECTOR_PATH" \
        --alpha "$a" --window "fixed:${len}" --delta-mode "$DELTA_MODE" \
        --max-new "$MAX_NEW" --target-prefixes "$TARGET_PREFIXES" \
        --samples "$SAMPLES" --screen-samples "$SCREEN_SAMPLES" \
        --min-screen-deception "$MIN_SCREEN" \
        --out "$OUT_DIR/$tag.jsonl" "${dump[@]}" \
        "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
        >"$LOG_DIR/$tag.log" 2>&1 \
      || { echo "[gpu $gpu] FAILED $tag (see $LOG_DIR/$tag.log)"; rm -rf "$LOCK_DIR/$tag"; continue; }
    echo "[gpu $gpu] done  $tag ($(wc -l <"$OUT_DIR/$tag.jsonl") prefixes)"
  done
}

for gpu in $GPUS; do worker "$gpu" & sleep 2; done
wait
echo "sweep complete; aggregate with:"
echo "  $PYTHON_BIN $ANALYSIS_DIR/collect_steering_results.py --runs $OUT_DIR"
