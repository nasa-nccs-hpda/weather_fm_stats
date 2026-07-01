#!/bin/bash

set -e  # Exit on error

module load python/GEOSpyD

# Create unique timestamp for job files
timestamp=$(date +%Y%m%d_%H%M%S)

YAML_FILENAME="stats_AIFS_ERA5_MAY_2024.yaml"

# Handle optional generate-only mode and YAML filename argument
generate_only=0
if [[ $# -ge 1 ]]; then
    if [[ "$1" == "--generate-only" ]]; then
        generate_only=1
        if [[ $# -ge 2 ]]; then
            filename="$2"
        else
            filename=$YAML_FILENAME
        fi
    else
        filename="$1"
    fi
else
    filename=$YAML_FILENAME
fi

# Check if YAML file exists before proceeding
if [[ ! -f "$filename" ]]; then
    echo "Error: YAML file '$filename' not found"
    exit 1
fi

echo "Using config file: $filename"

# Read and validate model names
for model_name in fcst_model ana_model clim_model; do
    model_value=$(grep -v '^[[:space:]]*#' "${filename}" | grep "${model_name}:" | sed "s/.*${model_name}:[[:space:]]*//" | sed 's/[[:space:]]*#.*//')
    
    # Check for non-alphanumeric characters
    if [[ ! "$model_value" =~ ^[[:alnum:]]+$ ]]; then
        echo "Error: $model_name contains non-alphanumeric characters or is empty"
        exit 1
    fi
    
    # Use dynamic variable assignment
    declare "$model_name=$model_value"
done

echo "Models: fcst=$fcst_model, ana=$ana_model, clim=$clim_model"

# Read parameters from config file
start_date=$(grep -v '^[[:space:]]*#' "${filename}" | grep "start_date:" | awk '{print $2}' | sed 's/[#].*//')
end_date=$(grep -v '^[[:space:]]*#' "${filename}" | grep "end_date:" | awk '{print $2}' | sed 's/[#].*//')

if [[ -z "$start_date" || -z "$end_date" ]]; then
    echo "Error: start_date and end_date must be specified in config file"
    exit 1
fi

echo "Date range: $start_date to $end_date"

# Make directories
work_dir=$(pwd)
out_dir="${work_dir}/outputs"
sts_dir="stats_${fcst_model}_${ana_model}_${clim_model}_${start_date}-${end_date}_${timestamp}"
log_dir="${work_dir}/outputs/${sts_dir}/logs"
job_dir="${work_dir}/outputs/${sts_dir}/jobs"
tmp_dir="${work_dir}/outputs/${sts_dir}/tmp"

mkdir -p "${out_dir}"
mkdir -p "${log_dir}"
mkdir -p "${job_dir}"
mkdir -p "${tmp_dir}"

echo "Created directory structure:"
echo "  Output dir: ${out_dir}/${sts_dir}"
echo "  Log dir: ${log_dir}"
echo "  Job dir: ${job_dir}"
echo "  Tmp dir: ${tmp_dir}"

# Create a versioned copy of the YAML file
versioned_yaml="${job_dir}/stats_${fcst_model}_${ana_model}_${clim_model}.yaml"
cp "$filename" "$versioned_yaml"
echo "Created versioned config: $versioned_yaml"

# Read SBATCH parameters from config file (we'll ignore most of these for direct execution)
account=$(grep -v '^[[:space:]]*#' "${versioned_yaml}" | grep "account:" | awk '{print $2}' | sed 's/[#].*//')
partition=$(grep -v '^[[:space:]]*#' "${versioned_yaml}" | grep "partition:" | awk '{print $2}' | sed 's/[#].*//')
qos=$(grep -v '^[[:space:]]*#' "${versioned_yaml}" | grep "qos:" | awk '{print $2}' | sed 's/[#].*//')
constraint=$(grep -v '^[[:space:]]*#' "${versioned_yaml}" | grep "constraint:" | awk '{print $2}' | sed 's/[#].*//')

# Resource parameters (informational only for direct execution)
pipeline_time=$(grep -v '^[[:space:]]*#' "${versioned_yaml}" | grep "pipeline_time:" | awk '{print $2}' | sed 's/[#].*//')
pipeline_mem=$(grep -v '^[[:space:]]*#' "${versioned_yaml}" | grep "pipeline_mem:" | awk '{print $2}' | sed 's/[#].*//')
pipeline_cpus=$(grep -v '^[[:space:]]*#' "${versioned_yaml}" | grep "pipeline_cpus:" | awk '{print $2}' | sed 's/[#].*//')
pipeline_gres=$(grep -v '^[[:space:]]*#' "${versioned_yaml}" | grep "pipeline_gres:" | awk '{print $2}' | sed 's/[#].*//')

# Set defaults if not specified
pipeline_time=${pipeline_time:-"08:00:00"}
pipeline_mem=${pipeline_mem:-"450G"}
pipeline_cpus=${pipeline_cpus:-"48"}

echo "Resource requirements (for reference):"
echo "  Time: $pipeline_time"
echo "  Memory: $pipeline_mem"
echo "  CPUs: $pipeline_cpus"
if [[ -n "$pipeline_gres" ]]; then
    echo "  GRES: $pipeline_gres"
fi

# Determine stats modes to run
stats_types=$(grep -v '^[[:space:]]*#' "${versioned_yaml}" | grep "stats_types:" | sed 's/.*stats_types:[[:space:]]*//' | sed 's/[[:space:]]*#.*//')
stats_types=${stats_types:-"both"}

run_regional=0
run_global=0
if [[ "$stats_types" == "regional" || "$stats_types" == "both" ]]; then
    run_regional=1
fi
if [[ "$stats_types" == "global" || "$stats_types" == "both" ]]; then
    run_global=1
fi

if [[ $run_regional -eq 0 && $run_global -eq 0 ]]; then
    echo "Error: stats_types must be one of regional, global, or both"
    exit 1
fi

echo "Stats types to run: $stats_types (regional=$run_regional, global=$run_global)"

# Build the execution script
pipeline_script="${job_dir}/stats_pipeline_direct.sh"
cat > "$pipeline_script" << EOF
#!/bin/bash
#
# Direct execution script for stats v4 pipeline
# Generated: $(date)
# Config: $versioned_yaml
#

set -e  # Exit on error
set -u  # Exit on undefined variable

echo "============================================"
echo "Stats v4 Pipeline - Direct Execution"
echo "Started: \$(date)"
echo "Host: \$(hostname)"
echo "User: \$(whoami)"
echo "Working Directory: \$(pwd)"
echo "============================================"
echo ""

# Load required modules
source /usr/share/lmod/lmod/init/bash
module load python/GEOSpyD

# Change to working directory
cd "${work_dir}"

# Log file
log_file="${log_dir}/stats_v4_pipeline_\$(date +%Y%m%d_%H%M%S).log"

echo "Log file: \$log_file"
echo ""

# Run the Python orchestrator with output redirection
echo "Running Python orchestrator..."
python -u "${work_dir}/stats_trimmed_v4.py" \\
    --config "${versioned_yaml}" \\
    --pipeline \\
    --stats_types "${stats_types}" \\
    --info_dir "${sts_dir}" \\
    2>&1 | tee "\$log_file"

exit_code=\${PIPESTATUS[0]}

echo ""
echo "============================================"
echo "Pipeline completed with exit code: \$exit_code"
echo "Finished: \$(date)"
echo "============================================"

exit \$exit_code
EOF

chmod +x "$pipeline_script"

if [[ $generate_only -eq 1 ]]; then
    echo ""
    echo "============================================"
    echo "Generate-only mode enabled."
    echo "Created pipeline execution script: $pipeline_script"
    echo "Created versioned config: $versioned_yaml"
    echo ""
    echo "To run the pipeline, execute:"
    echo "  bash $pipeline_script"
    echo ""
    echo "Or run directly with:"
    echo "  python -u ${work_dir}/stats_trimmed_v4.py \\"
    echo "    --config ${versioned_yaml} \\"
    echo "    --pipeline \\"
    echo "    --stats_types ${stats_types} \\"
    echo "    --info_dir ${sts_dir}"
    echo "============================================"
    exit 0
fi

# Execute the pipeline directly
echo ""
echo "============================================"
echo "Starting pipeline execution..."
echo "Execution script: $pipeline_script"
echo "============================================"
echo ""

# Run the pipeline script
bash "$pipeline_script"

exit_code=$?

echo ""
echo "============================================"
if [[ $exit_code -eq 0 ]]; then
    echo "✓ Pipeline completed successfully!"
else
    echo "✗ Pipeline failed with exit code: $exit_code"
fi
echo "============================================"
echo ""
echo "Output directory: ${out_dir}/${sts_dir}"
echo "Logs directory: ${log_dir}"
echo ""

exit $exit_code