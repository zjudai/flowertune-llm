#!/bin/zsh
# Wrapper script for running experiments

# Check if at least one task is provided, otherwise use general-nlp as default
if [ $# -eq 0 ]; then
    TASKS=("general-nlp")
else
    TASKS=("$@")
fi

# Create timestamp for log file
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="multi_task_evaluation_run_${timestamp}.log"

echo "Starting multi-task evaluation run at $(date)" | tee "$log_file"
echo "Tasks to run: ${TASKS[*]}" | tee -a "$log_file"
echo "Log file: $log_file" | tee -a "$log_file"

# Activate conda environment and set proxies
echo "Setting up environment..." | tee -a "$log_file"
# source ~/miniconda3/etc/profile.d/conda.sh
conda activate flwr
export http_proxy=http://10.72.74.124:7890
export https_proxy=http://10.72.74.124:7890

# Print environment information for debugging
echo "Python version:" | tee -a "$log_file"
python --version 2>&1 | tee -a "$log_file"
echo "Conda environment:" | tee -a "$log_file"
conda info 2>&1 | tee -a "$log_file"
echo "Proxy settings:" | tee -a "$log_file"
echo "HTTP_PROXY: $http_proxy" | tee -a "$log_file"
echo "HTTPS_PROXY: $https_proxy" | tee -a "$log_file"

# Run each task sequentially
for task in "${TASKS[@]}"; do
    echo "Starting task: $task" | tee -a "$log_file"
    python run_experiments.py --task "$task" --run-id "$timestamp" 2>&1 | tee -a "$log_file"
    echo "Completed task: $task" | tee -a "$log_file"
    echo "-----------------------------------------" | tee -a "$log_file"
done

echo "All task evaluations completed at $(date)" | tee -a "$log_file" 