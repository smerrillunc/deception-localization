#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/Results}"

ENVIRONMENT=""
MODEL_NAME=""
GPU_ID="${CUDA_VISIBLE_DEVICES:-}"
OUTPUT_DIR=""
RUN_TAG="${RUN_TAG:-$(date +%Y-%m-%d_%H-%M-%S)}"

SAMPLES_PER_STATE="${SAMPLES_PER_STATE:-1}"
TARGET_DECEPTIVE="${TARGET_DECEPTIVE:-3000}"
TARGET_TRUTHFUL="${TARGET_TRUTHFUL:-3000}"
MAX_GAMES="${MAX_GAMES:-1000}"
MAX_EPISODES="${MAX_EPISODES:-1000}"
MAX_TURNS="${MAX_TURNS:-1000}"
REASONING_MODE="${REASONING_MODE:-auto}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"

EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  run_deception_miner_single_gpu.sh --env ENV --model_name MODEL [options] [-- extra args]

Required:
  --env ENV                     One of: bs, gridworld, advisor_audit, interview, car_sales
  --model_name MODEL            Hugging Face / vLLM model name

Optional:
  --gpu GPU                     Optional single GPU id; otherwise uses existing CUDA_VISIBLE_DEVICES
  --results_root DIR            Default: <repo>/Results
  --output_dir DIR              Optional explicit miner output dir
  --run_tag TAG                 Default: timestamp
  --samples_per_state N         Default: 1
  --target_deceptive N          Default: 3000
  --target_truthful N           Default: 3000
  --max_games N                 Default: 1000 for non-advisor environments
  --max_episodes N              Default: 1000 for advisor_audit
  --max_turns N                 Default: 1000
  --reasoning MODE              One of: auto, on, off. Default: auto
  --help                        Show this message

Examples:
  bash run_deception_miner_single_gpu.sh --env bs --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --gpu 0
  bash run_deception_miner_single_gpu.sh --env interview --model_name MODEL --gpu 1 -- --interview_conversations_path /path/to/seeds.jsonl

Anything after `--` is forwarded directly to LocalizationScripts/deception_miner.py.
EOF
}

guess_reasoning_flag() {
  local mode="$1"
  local model_name="$2"
  local lower
  lower="$(printf '%s' "$model_name" | tr '[:upper:]' '[:lower:]')"

  if [[ "$mode" == "on" ]]; then
    printf '%s' "--is_reasoning_model"
    return
  fi

  if [[ "$mode" == "off" ]]; then
    return
  fi

  if [[ "$lower" == *"r1"* || "$lower" == *"qwq"* || "$lower" == *"gpt-oss"* || "$lower" == *"reason"* || "$lower" == *"thinking"* ]]; then
    printf '%s' "--is_reasoning_model"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      ENVIRONMENT="$2"
      shift 2
      ;;
    --model_name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --gpu)
      GPU_ID="$2"
      shift 2
      ;;
    --results_root)
      RESULTS_ROOT="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --run_tag)
      RUN_TAG="$2"
      shift 2
      ;;
    --samples_per_state)
      SAMPLES_PER_STATE="$2"
      shift 2
      ;;
    --target_deceptive)
      TARGET_DECEPTIVE="$2"
      shift 2
      ;;
    --target_truthful)
      TARGET_TRUTHFUL="$2"
      shift 2
      ;;
    --max_games)
      MAX_GAMES="$2"
      shift 2
      ;;
    --max_episodes)
      MAX_EPISODES="$2"
      shift 2
      ;;
    --max_turns)
      MAX_TURNS="$2"
      shift 2
      ;;
    --reasoning)
      REASONING_MODE="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$ENVIRONMENT" || -z "$MODEL_NAME" ]]; then
  usage >&2
  exit 1
fi

case "$ENVIRONMENT" in
  bs|gridworld|advisor_audit|interview|car_sales)
    ;;
  *)
    echo "Unsupported --env: $ENVIRONMENT" >&2
    exit 1
    ;;
esac

if [[ -n "$GPU_ID" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

MODEL_TAG="${MODEL_NAME##*/}"
if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$RESULTS_ROOT/DeceptionMining/$ENVIRONMENT/$MODEL_TAG/$RUN_TAG"
fi
mkdir -p "$OUTPUT_DIR"

REASONING_FLAG="$(guess_reasoning_flag "$REASONING_MODE" "$MODEL_NAME")"

CMD=(
  "$PYTHON_BIN" "$REPO_ROOT/LocalizationScripts/deception_miner.py"
  --game "$ENVIRONMENT"
  --model_name "$MODEL_NAME"
  --output_dir "$OUTPUT_DIR"
  --samples_per_state "$SAMPLES_PER_STATE"
  --target_deceptive "$TARGET_DECEPTIVE"
  --target_truthful "$TARGET_TRUTHFUL"
  --max_turns "$MAX_TURNS"
  --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION"
)

if [[ "$ENVIRONMENT" == "advisor_audit" ]]; then
  CMD+=(--max_episodes "$MAX_EPISODES")
else
  CMD+=(--max_games "$MAX_GAMES")
fi

if [[ -n "$REASONING_FLAG" ]]; then
  CMD+=("$REASONING_FLAG")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "Environment: $ENVIRONMENT"
echo "Model: $MODEL_NAME"
echo "Output dir: $OUTPUT_DIR"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
fi
echo "Running:"
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
