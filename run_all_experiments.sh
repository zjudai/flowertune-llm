#!/bin/zsh
# Wrapper script for running experiments

# Default values
MODEL=""
TASK="general-nlp"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --task)
      TASK="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Validate required parameters
if [ -z "$MODEL" ]; then
  echo "Error: --model parameter is required"
  echo "Usage: $0 --model MODEL_NAME --task TASK_NAME"
  exit 1
fi

# Create timestamp for log file
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="multi_task_evaluation_run_${timestamp}.log"

echo "Starting evaluation run at $(date)" | tee "$log_file"
echo "Model: $MODEL" | tee -a "$log_file"
echo "Task: $TASK" | tee -a "$log_file"
echo "Log file: $log_file" | tee -a "$log_file"

# Activate conda environment
echo "Setting up environment..." | tee -a "$log_file"
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate flwr-tune

# Print environment information for debugging
echo "Python version:" | tee -a "$log_file"
python --version 2>&1 | tee -a "$log_file"
echo "Conda environment:" | tee -a "$log_file"
conda info 2>&1 | tee -a "$log_file"

# Run the experiment
echo "Starting experiment with model: $MODEL and task: $TASK" | tee -a "$log_file"
python run_experiments.py --models "$MODEL" --task "$TASK" --run-id "$timestamp" 2>&1 | tee -a "$log_file"
echo "Experiment completed" | tee -a "$log_file"

echo "Evaluation completed at $(date)" | tee -a "$log_file" 