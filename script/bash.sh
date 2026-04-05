#!/bin/bash

#SBATCH --job-name=OODToolkit
#SBATCH --array=0-0
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  sbatch script/bash.sh [options]

Options:
  --modules LIST                       Comma-separated module names from src/splitters and src/models
  --splitters LIST                     Comma-separated splitter class names
  --models LIST                        Comma-separated model class names
  --require-eval BOOL                  true|false
  --splitwise-baseline-only BOOL       true|false
  --splitwise-include-variants BOOL    true|false
  --dataset-names LIST                 Comma-separated dataset names
  --python PATH                        Python executable to use. Default: python
  --help                               Show this message

Examples:
  sbatch script/bash.sh \
    --modules geometric_split,random_split,tree_models,statistical_models \
    --splitters RandomSplit \
    --models RFRegressor,LightGBMRegressor \
    --require-eval true \
    --splitwise-baseline-only false \
    --splitwise-include-variants true \
    --dataset-names bike

  sbatch --export=ALL,PYTHON_BIN=python3.11,CONDA_ENV_NAME=jupyter_env script/bash.sh \
    --modules geometric_split,marginal_distribution_shift,random_split,statistical_models,tree_models,resnet \
    --models HuberLinearRegressor,RFRegressor \
    --require-eval false \
    --dataset-names bike
EOF
}

to_lower() {
    printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

normalize_bool() {
    local value
    value="$(to_lower "${1:-}")"
    case "$value" in
        true|1|yes|y) printf 'True' ;;
        false|0|no|n) printf 'False' ;;
        "")
            printf '%s' "$2"
            ;;
        *)
            printf 'Invalid boolean value: %s\n' "$1" >&2
            exit 1
            ;;
    esac
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-jupyter_env}"

MODULES=""
SPLITTERS=""
MODELS=""
REQUIRE_EVAL=""
SPLITWISE_BASELINE_ONLY=""
SPLITWISE_INCLUDE_VARIANTS=""
DATASET_NAMES=""
PYTHON_BIN="${PYTHON_BIN:-python}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --modules)
            MODULES="${2:-}"
            shift 2
            ;;
        --splitters)
            SPLITTERS="${2:-}"
            shift 2
            ;;
        --models)
            MODELS="${2:-}"
            shift 2
            ;;
        --require-eval)
            REQUIRE_EVAL="${2:-}"
            shift 2
            ;;
        --splitwise-baseline-only)
            SPLITWISE_BASELINE_ONLY="${2:-}"
            shift 2
            ;;
        --splitwise-include-variants)
            SPLITWISE_INCLUDE_VARIANTS="${2:-}"
            shift 2
            ;;
        --dataset-names)
            DATASET_NAMES="${2:-}"
            shift 2
            ;;
        --python)
            PYTHON_BIN="${2:-}"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            printf 'Unknown argument: %s\n\n' "$1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

REQUIRE_EVAL="$(normalize_bool "$REQUIRE_EVAL" "False")"
SPLITWISE_BASELINE_ONLY="$(normalize_bool "$SPLITWISE_BASELINE_ONLY" "True")"
SPLITWISE_INCLUDE_VARIANTS="$(normalize_bool "$SPLITWISE_INCLUDE_VARIANTS" "False")"

T1=$(date +%s)

cd "${REPO_ROOT}"
mkdir -p logs results

if type module >/dev/null 2>&1; then
    module purge
fi

if [[ -f "${HOME}/.bashrc" ]]; then
    # Load shell init so `conda activate` works in batch jobs.
    source "${HOME}/.bashrc"
fi

if command -v conda >/dev/null 2>&1; then
    conda activate "${CONDA_ENV_NAME}"
fi

export MODULES
export SPLITTERS
export MODELS
export REQUIRE_EVAL
export SPLITWISE_BASELINE_ONLY
export SPLITWISE_INCLUDE_VARIANTS
export DATASET_NAMES
export REPO_ROOT

JOB_TMP_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
JOB_CACHE_DIR="${JOB_TMP_ROOT}/oodtoolkit-${SLURM_JOB_ID:-$$}"
mkdir -p "${JOB_CACHE_DIR}/matplotlib" "${JOB_CACHE_DIR}/cache"

export MPLCONFIGDIR="${JOB_CACHE_DIR}/matplotlib"
export XDG_CACHE_HOME="${JOB_CACHE_DIR}/cache"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

ENV_INFO="logs/env_info_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-nojob}}_${SLURM_ARRAY_TASK_ID:-0}.txt"

{
    echo "Task run at $(date)"
    echo "Host: $(hostname -s)"
    echo "PWD: $(pwd)"
    echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
    echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-}"
    echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
    echo "Python executable:"
    command -v "${PYTHON_BIN}" || true
    echo "Conda environment: ${CONDA_ENV_NAME}"
    if type module >/dev/null 2>&1; then
        module list
    fi
} > "${ENV_INFO}" 2>&1

cd "${REPO_ROOT}/src"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

printf 'Repository root: %s\n' "${REPO_ROOT}"
printf 'Python executable: %s\n' "${PYTHON_BIN}"
printf 'Job cache dir: %s\n' "${JOB_CACHE_DIR}"
printf 'Modules: %s\n' "${MODULES:-<default>}"
printf 'Splitters: %s\n' "${SPLITTERS:-<none>}"
printf 'Models: %s\n' "${MODELS:-<none>}"
printf 'Require eval: %s\n' "${REQUIRE_EVAL}"
printf 'Splitwise baseline only: %s\n' "${SPLITWISE_BASELINE_ONLY}"
printf 'Splitwise include variants: %s\n' "${SPLITWISE_INCLUDE_VARIANTS}"
printf 'Dataset names: %s\n' "${DATASET_NAMES:-<all>}"

"${PYTHON_BIN}" - <<'PY'
import os
from main import main


def parse_csv(value: str):
    if not value:
        return None
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    return parsed or None


main(
    modules=parse_csv(os.environ.get("MODULES", "")),
    splitters=parse_csv(os.environ.get("SPLITTERS", "")),
    models=parse_csv(os.environ.get("MODELS", "")),
    require_eval=os.environ["REQUIRE_EVAL"] == "True",
    splitwise_baseline_only=os.environ["SPLITWISE_BASELINE_ONLY"] == "True",
    splitwise_include_variants=os.environ["SPLITWISE_INCLUDE_VARIANTS"] == "True",
    dataset_names=parse_csv(os.environ.get("DATASET_NAMES", "")),
)
PY

T2=$(date +%s)
echo "Elapsed: $((T2 - T1)) seconds" >> "${REPO_ROOT}/${ENV_INFO}"
