#!/bin/bash

#SBATCH --job-name=SLipInt-0
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=Virtual
#SBATCH --time=48:00:00
#SBATCH --mem=10G
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

set -eo pipefail

usage() {
    cat <<'EOF'
Usage:
  sbatch script/bash.sh [--config FILE]

Options:
  --config FILE                        Path to a shell config file. Default: script/job.conf
  --help                               Show this message

Config file variables:
  MODULES="geometric_split,random_split,tree_models,slip_interpolant"
  # Available models:
  # HuberLinearRegressor, HuberPolynomialRegressor, SVMRegressor, KNNRegressor,
  # DTRegressor, RFRegressor, GBRegressor, ABRegressor, XGBRegressor, LightGBMRegressor,
  # ResnetRegressor, SLipInterpolant
  SPLITTERS="RandomSplit"
  MODELS="RFRegressor,LightGBMRegressor,SLipInterpolant"
  CPU_MODELS="RFRegressor,LightGBMRegressor"
  GPU_MODELS="ResnetRegressor"
  ARRAY_MODELS="RFRegressor,GBRegressor,ResnetRegressor"
  ARRAY_DATASETS="bike,concrete,energy"
  REQUIRE_EVAL="true"
  SPLITWISE_BASELINE_ONLY="false"
  SPLITWISE_INCLUDE_VARIANTS="true"
  MODELWISE_EVAL="false"
  PER_DATASET_TABLE_EVAL="false"
  DATASET_NAMES="bike"
  PYTHON_BIN="python"
  CONDA_ENV_NAME="jupyter_env"

Examples:
  sbatch script/bash.sh
  sbatch script/bash.sh --config script/job.conf
EOF
}

to_lower() {
    printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

normalize_bool() {
    # Normalize various true/false spellings to Python-friendly True/False strings.
    local value
    # Hold the lowercased version of the input flag value.
    value="$(to_lower "${1:-}")"
    # Map accepted boolean spellings to canonical values or fail on invalid input.
    case "$value" in
        true|1|yes|y) printf 'True' ;;
        false|0|no|n) printf 'False' ;;
        "")
            # Fall back to the provided default when the flag was omitted.
            printf '%s' "$2"
            ;;
        *)
            # Stop immediately if the user passed an unsupported boolean value.
            printf 'Invalid boolean value: %s\n' "$1" >&2
            exit 1
            ;;
    esac
}

SCRIPT_DIR="script/"
# Resolve the directory that contains this script.
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# Resolve the repository root as the parent directory of the script directory.
CONDA_ENV_NAME="${CONDA_ENV_NAME:-jupyter_env}"
# Use the requested Conda environment name, defaulting to jupyter_env.
CONFIG_FILE="${SCRIPT_DIR}/job.conf"
# Use script/job.conf by default for pipeline settings.

MODULES=""
# Store the comma-separated module list loaded from the config file.
SPLITTERS=""
# Store the comma-separated splitter class list loaded from the config file.
MODELS=""
# Store the comma-separated model class list loaded from the config file.
CPU_MODELS=""
# Store the comma-separated CPU-only model class list loaded from the config file.
GPU_MODELS=""
# Store the comma-separated GPU-only model class list loaded from the config file.
ARRAY_MODELS=""
# Store a comma-separated model sweep list where each Slurm array task runs one model.
ARRAY_DATASETS=""
# Store a comma-separated dataset sweep list where each Slurm array task runs one dataset.
REQUIRE_EVAL=""
# Store the requested evaluation flag before normalization.
SPLITWISE_BASELINE_ONLY=""
# Store the split-wise baseline-only flag before normalization.
SPLITWISE_INCLUDE_VARIANTS=""
# Store the split-wise include-variants flag before normalization.
MODELWISE_EVAL=""
# Store the model-wise evaluation flag before normalization.
PER_DATASET_TABLE_EVAL=""
# Store the per-dataset table evaluation flag before normalization.
DATASET_NAMES=""
# Store the comma-separated dataset list loaded from the config file.
PYTHON_BIN="${PYTHON_BIN:-python}"
# Use the requested Python executable, defaulting to python.

while [[ $# -gt 0 ]]; do
    # Parse the small set of wrapper arguments until none remain.
    case "$1" in
        --config)
            CONFIG_FILE="${2:-}"
            # Override the default config file path.
            shift 2
            # Consume the flag and its value.
            ;;
        --help|-h)
            usage
            # Print help text and stop without running the pipeline.
            exit 0
            ;;
        *)
            # Reject unknown flags so the job does not run with unintended settings.
            printf 'Unknown argument: %s\n\n' "$1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ ! -f "${CONFIG_FILE}" ]]; then
    # Stop early if the requested config file does not exist.
    printf 'Config file not found: %s\n' "${CONFIG_FILE}" >&2
    exit 1
fi

source "${CONFIG_FILE}"
# Load pipeline settings from the external shell config file.

MODULES="${MODULES:-}"
# Read the configured module list, or leave empty if omitted.
SPLITTERS="${SPLITTERS:-}"
# Read the configured splitter list, or leave empty if omitted.
MODELS="${MODELS:-}"
# Read the configured model list, or leave empty if omitted.
CPU_MODELS="${CPU_MODELS:-}"
# Read the configured CPU model list, or leave empty if omitted.
GPU_MODELS="${GPU_MODELS:-}"
# Read the configured GPU model list, or leave empty if omitted.
ARRAY_MODELS="${ARRAY_MODELS:-}"
# Read the configured array-model sweep list, or leave empty if omitted.
ARRAY_DATASETS="${ARRAY_DATASETS:-}"
# Read the configured array-dataset sweep list, or leave empty if omitted.
REQUIRE_EVAL="${REQUIRE_EVAL:-}"
# Read the configured evaluation flag before normalization.
SPLITWISE_BASELINE_ONLY="${SPLITWISE_BASELINE_ONLY:-}"
# Read the configured baseline-only flag before normalization.
SPLITWISE_INCLUDE_VARIANTS="${SPLITWISE_INCLUDE_VARIANTS:-}"
# Read the configured include-variants flag before normalization.
MODELWISE_EVAL="${MODELWISE_EVAL:-}"
# Read the configured model-wise evaluation flag before normalization.
PER_DATASET_TABLE_EVAL="${PER_DATASET_TABLE_EVAL:-}"
# Read the configured per-dataset table evaluation flag before normalization.
DATASET_NAMES="${DATASET_NAMES:-}"
# Read the configured dataset list, or leave empty if omitted.
PYTHON_BIN="${PYTHON_BIN:-python}"
# Allow the config file to override the Python executable.
CONDA_ENV_NAME="${CONDA_ENV_NAME:-jupyter_env}"
# Allow the config file to override the Conda environment name.

REQUIRE_EVAL="$(normalize_bool "$REQUIRE_EVAL" "False")"
# Default to skipping evaluation unless the user explicitly enables it.
SPLITWISE_BASELINE_ONLY="$(normalize_bool "$SPLITWISE_BASELINE_ONLY" "True")"
# Default to baseline-only split-wise comparison.
SPLITWISE_INCLUDE_VARIANTS="$(normalize_bool "$SPLITWISE_INCLUDE_VARIANTS" "False")"
# Default to excluding variants from split-wise comparison.
MODELWISE_EVAL="$(normalize_bool "$MODELWISE_EVAL" "False")"
# Default to skipping model-wise comparison unless explicitly enabled.
PER_DATASET_TABLE_EVAL="$(normalize_bool "$PER_DATASET_TABLE_EVAL" "False")"
# Default to skipping per-dataset table export unless explicitly enabled.

echo "REPO_ROOT=${REPO_ROOT}"
# Print the resolved repository root to the Slurm output stream for debugging.

T1=$(date +%s)
# Record the start time in Unix seconds for elapsed-time reporting.

cd "${REPO_ROOT}"
# Move to the repository root so relative paths resolve consistently.
mkdir -p logs
# Ensure the logs directory exist under OODToolkit before Slurm writes outputs.

module purge
# Clear inherited HPC environment modules before activating the Conda environment.
source ~/.bashrc
# Load shell startup so `conda activate` is available in the batch shell.
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    echo "conda command not found in PATH" >&2
    exit 1
fi
# Initialize Conda for this non-interactive Bash job shell.
conda activate "${CONDA_ENV_NAME}"
# Activate the Conda environment used for this job.
if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV_NAME}" ]]; then
    echo "Failed to activate Conda environment: ${CONDA_ENV_NAME}" >&2
    echo "Active env: ${CONDA_DEFAULT_ENV:-<none>}" >&2
    echo "CONDA_PREFIX=${CONDA_PREFIX:-<unset>}" >&2
    exit 1
fi
# Stop immediately if the requested environment did not become active.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
# Prefer the Conda environment's C++ runtime so compiled wheels like matplotlib use a compatible libstdc++.
python -m pip install -r "${REPO_ROOT}/requirements.txt"
# Install the Python dependencies listed in requirements.txt into the active environment.

export MODULES
# Make the module list available to the embedded Python process.
export SPLITTERS
# Make the splitter list available to the embedded Python process.
export MODELS
# Make the model list available to the embedded Python process.
export CPU_MODELS
# Make the CPU model list available to the phase runner.
export GPU_MODELS
# Make the GPU model list available to the phase runner.
export ARRAY_MODELS
# Make the model sweep list available to array-task selection logic.
export ARRAY_DATASETS
# Make the dataset sweep list available to array-task selection logic.
export REQUIRE_EVAL
# Make the normalized evaluation flag available to Python.
export SPLITWISE_BASELINE_ONLY
# Make the normalized baseline-only flag available to Python.
export SPLITWISE_INCLUDE_VARIANTS
# Make the normalized include-variants flag available to Python.
export MODELWISE_EVAL
# Make the normalized model-wise evaluation flag available to Python.
export PER_DATASET_TABLE_EVAL
# Make the normalized per-dataset table evaluation flag available to Python.
export DATASET_NAMES
# Make the dataset list available to Python.
export REPO_ROOT
# Expose the repository root in case downstream code needs it.

JOB_TMP_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
# Prefer Slurm's per-job temporary directory, otherwise fall back to TMPDIR or /tmp.
JOB_CACHE_DIR="${JOB_TMP_ROOT}/oodtoolkit-${SLURM_JOB_ID:-$$}"
# Create a job-specific cache directory keyed by Slurm job id or shell pid.
mkdir -p "${JOB_CACHE_DIR}/matplotlib" "${JOB_CACHE_DIR}/cache"
# Create writable cache locations for Matplotlib and other libraries.

export MPLCONFIGDIR="${JOB_CACHE_DIR}/matplotlib"
# Force Matplotlib to write config/cache files to a writable per-job directory.
export XDG_CACHE_HOME="${JOB_CACHE_DIR}/cache"
# Force general XDG cache usage into the job-specific writable cache directory.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
# Limit OpenMP thread count to the CPUs allocated by Slurm.
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
# Limit MKL thread count to the CPUs allocated by Slurm.
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
# Limit OpenBLAS thread count to the CPUs allocated by Slurm.

ENV_INFO="logs/env_info_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-nojob}}_${SLURM_ARRAY_TASK_ID:-0}.txt"
# Define a log file that captures runtime environment details for this array task.

{
    # Write diagnostic information that helps debug cluster environment issues.
    echo "Task run at $(date)"
    # Record the human-readable start time.
    echo "Host: $(hostname -s)"
    # Record the host machine name.
    echo "PWD: $(pwd)"
    # Record the current working directory.
    echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
    # Record the Slurm job id when present.
    echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-}"
    # Record the Slurm array master job id when present.
    echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
    # Record the current array index when present.
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
    # Record which GPU devices Slurm exposed to the job.
    echo "Python executable:"
    # Label the next line for readability.
    command -v "${PYTHON_BIN}" || true
    # Show the resolved Python executable path without failing the job if it is missing.
    echo "Conda environment: ${CONDA_ENV_NAME}"
    # Record the requested Conda environment name.
    if type module >/dev/null 2>&1; then
        # If the module system exists, capture the loaded module stack.
        module list
    fi
} > "${ENV_INFO}" 2>&1
# Save all environment diagnostics to the env-info log file.

cd "${REPO_ROOT}/src"
# Move into src so the toolkit's relative default paths behave as intended.
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
# Add src to PYTHONPATH so imports like `from main import main` work.

printf 'Repository root: %s\n' "${REPO_ROOT}"
# Echo the resolved repository root into the Slurm output log.
printf 'Config file: %s\n' "${CONFIG_FILE}"
# Echo the config file path used for this run.
printf 'Python executable: %s\n' "${PYTHON_BIN}"
# Echo the Python command being used.
printf 'Job cache dir: %s\n' "${JOB_CACHE_DIR}"
# Echo the per-job cache directory.
printf 'Modules: %s\n' "${MODULES:-<default>}"
# Echo the user-selected modules or show a placeholder if none were passed.
printf 'Splitters: %s\n' "${SPLITTERS:-<none>}"
# Echo the selected splitters or show a placeholder if none were passed.
printf 'Models: %s\n' "${MODELS:-<none>}"
# Echo the selected models or show a placeholder if none were passed.
printf 'CPU models: %s\n' "${CPU_MODELS:-<none>}"
# Echo the selected CPU models or show a placeholder if none were passed.
printf 'GPU models: %s\n' "${GPU_MODELS:-<none>}"
# Echo the selected GPU models or show a placeholder if none were passed.
printf 'Array models: %s\n' "${ARRAY_MODELS:-<none>}"
# Echo the model sweep list used for Slurm array sharding.
printf 'Array datasets: %s\n' "${ARRAY_DATASETS:-<none>}"
# Echo the dataset sweep list used for Slurm array sharding.
printf 'Require eval: %s\n' "${REQUIRE_EVAL}"
# Echo whether evaluation is enabled.
printf 'Splitwise baseline only: %s\n' "${SPLITWISE_BASELINE_ONLY}"
# Echo the split-wise baseline-only flag.
printf 'Splitwise include variants: %s\n' "${SPLITWISE_INCLUDE_VARIANTS}"
# Echo the split-wise include-variants flag.
printf 'Modelwise eval: %s\n' "${MODELWISE_EVAL}"
# Echo the model-wise evaluation flag.
printf 'Per-dataset table eval: %s\n' "${PER_DATASET_TABLE_EVAL}"
# Echo the per-dataset table evaluation flag.
printf 'Dataset names: %s\n' "${DATASET_NAMES:-<all>}"
# Echo the selected dataset names or show a placeholder if all datasets are used.

csv_pick_by_task() {
    # Pick one comma-separated value based on the current SLURM_ARRAY_TASK_ID.
    local csv_values="$1"
    local task_id="${SLURM_ARRAY_TASK_ID:-0}"
    local -a values
    local IFS=','
    read -r -a values <<< "${csv_values}"

    if (( ${#values[@]} == 0 )); then
        printf ''
        return
    fi

    if (( task_id < 0 || task_id >= ${#values[@]} )); then
        printf 'SLURM_ARRAY_TASK_ID=%s is out of range for list of size %s\n' "${task_id}" "${#values[@]}" >&2
        exit 1
    fi

    printf '%s' "$(printf '%s' "${values[${task_id}]}" | xargs)"
}

csv_contains() {
    # Check whether a CSV list contains an exact token.
    local csv_values="$1"
    local needle="$2"
    local IFS=','
    local token
    read -r -a _tokens <<< "${csv_values}"
    for token in "${_tokens[@]}"; do
        token="$(printf '%s' "${token}" | xargs)"
        if [[ "${token}" == "${needle}" ]]; then
            return 0
        fi
    done
    return 1
}

if [[ -n "${ARRAY_MODELS}" ]]; then
    # Shard by model: each array task runs exactly one model.
    SELECTED_MODEL="$(csv_pick_by_task "${ARRAY_MODELS}")"
    printf 'Array-selected model: %s\n' "${SELECTED_MODEL}"

    if [[ -n "${CPU_MODELS}" || -n "${GPU_MODELS}" ]]; then
        if csv_contains "${GPU_MODELS}" "${SELECTED_MODEL}"; then
            GPU_MODELS="${SELECTED_MODEL}"
            CPU_MODELS=""
        elif csv_contains "${CPU_MODELS}" "${SELECTED_MODEL}"; then
            CPU_MODELS="${SELECTED_MODEL}"
            GPU_MODELS=""
        else
            # Fall back to single-pass mode if the selected model is outside CPU/GPU buckets.
            MODELS="${SELECTED_MODEL}"
            CPU_MODELS=""
            GPU_MODELS=""
        fi
    else
        MODELS="${SELECTED_MODEL}"
    fi
fi

if [[ -n "${ARRAY_DATASETS}" ]]; then
    # Shard by dataset: each array task runs exactly one dataset.
    SELECTED_DATASET="$(csv_pick_by_task "${ARRAY_DATASETS}")"
    DATASET_NAMES="${SELECTED_DATASET}"
    printf 'Array-selected dataset: %s\n' "${SELECTED_DATASET}"
fi

run_main_phase() {
    # Run one pipeline phase with optional CPU-only device visibility.
    local phase_name="$1"
    local phase_splitters="$2"
    local phase_models="$3"
    local phase_require_eval="$4"
    local phase_device="$5"

    printf 'Running phase: %s\n' "${phase_name}"

    if [[ "${phase_device}" == "cpu" ]]; then
        # Hide GPUs from this subprocess so CPU-bound models never claim CUDA.
        CUDA_VISIBLE_DEVICES="" \
        PHASE_SPLITTERS="${phase_splitters}" \
        PHASE_MODELS="${phase_models}" \
        PHASE_REQUIRE_EVAL="${phase_require_eval}" \
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
    splitters=parse_csv(os.environ.get("PHASE_SPLITTERS", "")),
    models=parse_csv(os.environ.get("PHASE_MODELS", "")),
    require_eval=os.environ["PHASE_REQUIRE_EVAL"] == "True",
    splitwise_baseline_only=os.environ["SPLITWISE_BASELINE_ONLY"] == "True",
    splitwise_include_variants=os.environ["SPLITWISE_INCLUDE_VARIANTS"] == "True",
    modelwise_eval=os.environ["MODELWISE_EVAL"] == "True",
    per_dataset_table_eval=os.environ["PER_DATASET_TABLE_EVAL"] == "True",
    dataset_names=parse_csv(os.environ.get("DATASET_NAMES", "")),
)
PY
    else
        # Inherit Slurm GPU visibility for GPU-capable phases.
        PHASE_SPLITTERS="${phase_splitters}" \
        PHASE_MODELS="${phase_models}" \
        PHASE_REQUIRE_EVAL="${phase_require_eval}" \
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
    splitters=parse_csv(os.environ.get("PHASE_SPLITTERS", "")),
    models=parse_csv(os.environ.get("PHASE_MODELS", "")),
    require_eval=os.environ["PHASE_REQUIRE_EVAL"] == "True",
    splitwise_baseline_only=os.environ["SPLITWISE_BASELINE_ONLY"] == "True",
    splitwise_include_variants=os.environ["SPLITWISE_INCLUDE_VARIANTS"] == "True",
    modelwise_eval=os.environ["MODELWISE_EVAL"] == "True",
    per_dataset_table_eval=os.environ["PER_DATASET_TABLE_EVAL"] == "True",
    dataset_names=parse_csv(os.environ.get("DATASET_NAMES", "")),
)
PY
    fi
}

if [[ -n "${CPU_MODELS}" || -n "${GPU_MODELS}" ]]; then
    # Mixed scheduling mode: split once, train CPU/GPU model groups separately, then evaluate once.
    if [[ -n "${SPLITTERS}" ]]; then
        run_main_phase "split" "${SPLITTERS}" "" "False" "gpu"
    fi

    if [[ -n "${CPU_MODELS}" ]]; then
        run_main_phase "train-cpu-models" "" "${CPU_MODELS}" "False" "cpu"
    fi

    if [[ -n "${GPU_MODELS}" ]]; then
        run_main_phase "train-gpu-models" "" "${GPU_MODELS}" "False" "gpu"
    fi

    if [[ "${REQUIRE_EVAL}" == "True" ]]; then
        run_main_phase "evaluate" "" "" "True" "gpu"
    fi
else
    # Backward-compatible single-pass mode.
    run_main_phase "single-pass" "${SPLITTERS}" "${MODELS}" "${REQUIRE_EVAL}" "gpu"
fi

T2=$(date +%s)
# Record the finish time in Unix seconds.
echo "Elapsed: $((T2 - T1)) seconds" >> "${REPO_ROOT}/${ENV_INFO}"
# Append the total elapsed runtime to the environment-info log file.
