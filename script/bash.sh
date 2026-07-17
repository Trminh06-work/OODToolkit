#!/bin/bash
#SBATCH --job-name=Runtime_3d
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=Virtual
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
  RUN_MODE="pipeline"                  # pipeline or visualize
  MODULES="geometric_split,random_split,tree_models,slip_interpolant"
  # Available models:
  # HuberLinearRegressor, HuberPolynomialRegressor, SVMRegressor, KNNRegressor,
  # DTRegressor, RFRegressor, GBRegressor, ABRegressor, XGBRegressor, LightGBMRegressor,
  # ResnetRegressor, SLipInterpolant
  SPLITTERS="RandomSplit"
  MODELS="RFRegressor,LightGBMRegressor,SLipInterpolant"
  ARRAY_MODELS="RFRegressor,GBRegressor,ResnetRegressor"
  ARRAY_DATASETS="bike,concrete,energy"
  # When both ARRAY_MODELS and ARRAY_DATASETS are set, the Slurm array runs their
  # cartesian product: task t picks ARRAY_MODELS[t / n_datasets] and ARRAY_DATASETS[t % n_datasets].
  REQUIRE_EVAL="true"
  SPLITWISE_BASELINE_ONLY="false"
  SPLITWISE_INCLUDE_VARIANTS="true"
  MODELWISE_EVAL="false"
  PER_DATASET_TABLE_EVAL="false"
  RUNTIME_EVAL="false"
  DATASET_NAMES="bike"
  VISUALIZE_GRID_SIZE="100"
  VISUALIZE_PLOT_KINDS="test_only,model_only,absolute_error,test_and_model,train_and_model,train_test_and_model"
  VISUALIZE_SPLIT_NAMES="Random_Split"
  VISUALIZE_RUN_IDS="0"
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
RUN_MODE="pipeline"
# Select the wrapper action: pipeline uses run_main_phase, visualize calls main_visualize only.
SPLITTERS=""
# Store the comma-separated splitter class list loaded from the config file.
MODELS=""
# Store the comma-separated model class list loaded from the config file.
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
RUNTIME_EVAL=""
# Store the runtime evaluation flag before normalization.
DATASET_NAMES=""
# Store the comma-separated dataset list loaded from the config file.
VISUALIZE_GRID_SIZE="100"
# Store the Visualizer prediction grid resolution.
VISUALIZE_PLOT_KINDS="test_only,model_only,absolute_error,test_and_model,train_and_model,train_test_and_model"
# Store comma-separated visualization plot kinds.
VISUALIZE_SPLIT_NAMES="Random_Split"
# Store comma-separated split names to visualize; leave empty in config to include all split folders.
VISUALIZE_RUN_IDS="0"
# Store comma-separated run ids to visualize; leave empty in config to include all runs.
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

RUN_MODE="${RUN_MODE:-pipeline}"
# Read the selected wrapper action, defaulting to the normal pipeline.
MODULES="${MODULES:-}"
# Read the configured module list, or leave empty if omitted.
SPLITTERS="${SPLITTERS:-}"
# Read the configured splitter list, or leave empty if omitted.
MODELS="${MODELS:-}"
# Read the configured model list, or leave empty if omitted.
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
RUNTIME_EVAL="${RUNTIME_EVAL:-}"
# Read the configured runtime evaluation flag before normalization.
DATASET_NAMES="${DATASET_NAMES:-}"
# Read the configured dataset list, or leave empty if omitted.
VISUALIZE_GRID_SIZE="${VISUALIZE_GRID_SIZE:-100}"
# Read the visualization grid size.
VISUALIZE_PLOT_KINDS="${VISUALIZE_PLOT_KINDS:-}"
# Read the selected visualization plot kinds.
VISUALIZE_SPLIT_NAMES="${VISUALIZE_SPLIT_NAMES:-}"
# Read the selected visualization split names.
VISUALIZE_RUN_IDS="${VISUALIZE_RUN_IDS:-}"
# Read the selected visualization run ids.
PYTHON_BIN="${PYTHON_BIN:-python}"
# Allow the config file to override the Python executable.
CONDA_ENV_NAME="${CONDA_ENV_NAME:-jupyter_env}"
# Allow the config file to override the Conda environment name.

case "$(to_lower "${RUN_MODE}")" in
    pipeline|main)
        RUN_MODE="pipeline"
        ;;
    visualize|visualise|visualization|visualisation)
        RUN_MODE="visualize"
        ;;
    *)
        printf 'Invalid RUN_MODE: %s (expected pipeline or visualize)\n' "${RUN_MODE}" >&2
        exit 1
        ;;
esac
# Normalize accepted spelling variants to the internal mode names.

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
RUNTIME_EVAL="$(normalize_bool "$RUNTIME_EVAL" "False")"
# Default to skipping runtime table export unless explicitly enabled.

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
export RUN_MODE
# Make the selected wrapper action available to diagnostics and Python.
export SPLITTERS
# Make the splitter list available to the embedded Python process.
export MODELS
# Make the model list available to the embedded Python process.
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
export RUNTIME_EVAL
# Make the normalized runtime evaluation flag available to Python.
export DATASET_NAMES
# Make the dataset list available to Python.
export VISUALIZE_GRID_SIZE
# Make the visualization grid size available to Python.
export VISUALIZE_PLOT_KINDS
# Make the visualization plot kind filter available to Python.
export VISUALIZE_SPLIT_NAMES
# Make the visualization split filter available to Python.
export VISUALIZE_RUN_IDS
# Make the visualization run-id filter available to Python.
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
printf 'Run mode: %s\n' "${RUN_MODE}"
# Echo whether this job runs the pipeline or visualization only.
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
printf 'Runtime eval: %s\n' "${RUNTIME_EVAL}"
# Echo the runtime evaluation flag.
printf 'Dataset names: %s\n' "${DATASET_NAMES:-<all>}"
# Echo the selected dataset names or show a placeholder if all datasets are used.
printf 'Visualize grid size: %s\n' "${VISUALIZE_GRID_SIZE}"
# Echo the visualization grid resolution.
printf 'Visualize plot kinds: %s\n' "${VISUALIZE_PLOT_KINDS:-<default>}"
# Echo the selected visualization plot kinds or show default handling.
printf 'Visualize split names: %s\n' "${VISUALIZE_SPLIT_NAMES:-<all>}"
# Echo the selected visualization split names or show all.
printf 'Visualize run ids: %s\n' "${VISUALIZE_RUN_IDS:-<all>}"
# Echo the selected visualization run ids or show all.

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

csv_count() {
    # Count the comma-separated tokens in the given CSV string.
    local csv_values="$1"
    local IFS=','
    local -a tokens
    read -r -a tokens <<< "${csv_values}"
    printf '%s' "${#tokens[@]}"
}

csv_pick_at() {
    # Pick the value at the given index from a CSV list.
    local csv_values="$1"
    local idx="$2"
    local -a values
    local IFS=','
    read -r -a values <<< "${csv_values}"

    if (( idx < 0 || idx >= ${#values[@]} )); then
        printf 'Index %s is out of range for list of size %s\n' "${idx}" "${#values[@]}" >&2
        exit 1
    fi

    printf '%s' "$(printf '%s' "${values[${idx}]}" | xargs)"
}

if [[ -n "${ARRAY_DATASETS}" && -n "${ARRAY_MODELS}" ]]; then
    # Shard by (dataset, model) pair: each array task runs one cell of the cartesian product.
    # Iteration order: model is the outer loop, dataset is the inner loop.
    TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
    N_DATASETS="$(csv_count "${ARRAY_DATASETS}")"
    if (( N_DATASETS == 0 )); then
        echo "ARRAY_DATASETS is empty after parsing" >&2
        exit 1
    fi
    N_MODELS="$(csv_count "${ARRAY_MODELS}")"
    N_TOTAL=$(( N_DATASETS * N_MODELS ))
    if (( TASK_ID >= N_TOTAL )); then
        printf 'SLURM_ARRAY_TASK_ID=%s exceeds %s models * %s datasets = %s pairs\n' "${TASK_ID}" "${N_MODELS}" "${N_DATASETS}" "${N_TOTAL}" >&2
        exit 1
    fi
    MODEL_IDX=$(( TASK_ID / N_DATASETS ))
    DATASET_IDX=$(( TASK_ID % N_DATASETS ))
    SELECTED_DATASET="$(csv_pick_at "${ARRAY_DATASETS}" "${DATASET_IDX}")"
    SELECTED_MODEL="$(csv_pick_at "${ARRAY_MODELS}" "${MODEL_IDX}")"
    DATASET_NAMES="${SELECTED_DATASET}"
    MODELS="${SELECTED_MODEL}"
    printf 'Array-selected pair (task=%s): dataset=%s, model=%s\n' "${TASK_ID}" "${SELECTED_DATASET}" "${SELECTED_MODEL}"
elif [[ -n "${ARRAY_MODELS}" ]]; then
    # Shard by model: each array task runs exactly one model.
    SELECTED_MODEL="$(csv_pick_by_task "${ARRAY_MODELS}")"
    MODELS="${SELECTED_MODEL}"
    printf 'Array-selected model: %s\n' "${SELECTED_MODEL}"
elif [[ -n "${ARRAY_DATASETS}" ]]; then
    # Shard by dataset: each array task runs exactly one dataset.
    SELECTED_DATASET="$(csv_pick_by_task "${ARRAY_DATASETS}")"
    DATASET_NAMES="${SELECTED_DATASET}"
    printf 'Array-selected dataset: %s\n' "${SELECTED_DATASET}"
fi

run_main_phase() {
    # Run one pipeline phase with Slurm-provided GPU visibility.
    local phase_name="$1"
    local phase_splitters="$2"
    local phase_models="$3"
    local phase_require_eval="$4"

    printf 'Running phase: %s\n' "${phase_name}"

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
    runtime_eval=os.environ["RUNTIME_EVAL"] == "True",
    dataset_names=parse_csv(os.environ.get("DATASET_NAMES", "")),
)
PY
}

run_visualize_phase() {
    # Run only the visualization procedure from src/main.py.
    printf 'Running phase: visualize-only\n'

    "${PYTHON_BIN}" - <<'PY'
import os
from main import _string2class, main_visualize

def parse_csv(value: str):
    if not value:
        return None
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    return parsed or None

model_names = parse_csv(os.environ.get("MODELS", ""))
if model_names is None:
    raise ValueError(
        "Visualization mode requires MODELS or ARRAY_MODELS to select at least one model."
    )

model_classes = _string2class(
    modules=parse_csv(os.environ.get("MODULES", "")),
    inputs=model_names,
    conversion_engine="models",
)

main_visualize(
    model_classes,
    dataset_names=parse_csv(os.environ.get("DATASET_NAMES", "")),
    grid_size=int(os.environ.get("VISUALIZE_GRID_SIZE", "100")),
    plot_kinds=parse_csv(os.environ.get("VISUALIZE_PLOT_KINDS", "")),
    split_names=parse_csv(os.environ.get("VISUALIZE_SPLIT_NAMES", "")),
    run_ids=parse_csv(os.environ.get("VISUALIZE_RUN_IDS", "")),
)
PY
}

if [[ "${RUN_MODE}" == "visualize" ]]; then
    run_visualize_phase
else
    run_main_phase "single-pass" "${SPLITTERS}" "${MODELS}" "${REQUIRE_EVAL}"
fi

T2=$(date +%s)
# Record the finish time in Unix seconds.
echo "Elapsed: $((T2 - T1)) seconds" >> "${REPO_ROOT}/${ENV_INFO}"
# Append the total elapsed runtime to the environment-info log file.
