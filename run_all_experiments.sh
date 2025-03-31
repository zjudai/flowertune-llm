#!/bin/zsh
# Wrapper script for running experiments

# Create timestamp for log file
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="evaluation_run_${timestamp}.log"

echo "Starting evaluation run at $(date)" | tee "$log_file"
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

# Pass all arguments to the Python script
echo "Running: python run_experiments.py $@" | tee -a "$log_file"
python run_experiments.py "$@" 2>&1 | tee -a "$log_file"

echo "Evaluation completed at $(date)" | tee -a "$log_file" 