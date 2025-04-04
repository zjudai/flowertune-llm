#!/usr/bin/env python3
"""
Federated Learning Experiment Runner

This script runs training and evaluation for various small language models across different domains:
- General NLP: evaluates on STEM, social sciences, and humanities
- Finance: evaluates on FPB, FIQA, and TFNS datasets
- Medical: evaluates on PubMedQA, MedMCQA, MedQA, and CareQA
- Code: evaluates on HumanEval, MBPP, multiple-js, and multiple-cpp

Prerequisites:
  - Conda environment 'flwr' must be active
  - Proxy settings must be configured:
    export http_proxy=http://10.72.74.124:7890 https_proxy=http://10.72.74.124:7890

Usage:
  # Recommended: Use the wrapper script that sets up environment automatically
  ./run_all_experiments.sh [task] [options]
  
  # Or manually run with proper environment setup:
  conda activate flwr && export http_proxy=http://10.72.74.124:7890 https_proxy=http://10.72.74.124:7890
  python run_experiments.py --task [task] [options]
  
  # Command options:
  python run_experiments.py --task general-nlp                      # Run training+evaluation for general-nlp task
  python run_experiments.py --task finance --eval-only              # Run only evaluation for finance task
  python run_experiments.py --task medical --models model1 model2   # Run medical task on specific models
  python run_experiments.py --task code --eval-only --models model1  # Evaluate only one model for code task

Available tasks:
  - general-nlp (default)
  - finance
  - medical
  - code

Available models:
  - meta-llama/Llama-3.2-1B-Instruct
  - Qwen/Qwen2.5-1.5B-Instruct
  - TinyLlama/TinyLlama-1.1B-Chat-v1.0
  - google/gemma-3-1b-it
  - facebook/MobileLLM-1B
  - mlx-community/Llama-3.2-1B-Instruct-4bit
  - deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
  - mistralai/Mistral-7B-v0.3
  - Qwen/Qwen2.5-7B-Instruct
  - mistralai/Mistral-7B-Instruct-v0.3
"""
import subprocess
import os
import csv
import re
import toml
import glob
import json
from datetime import datetime
import argparse
from pathlib import Path

# List of models to test
models = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Ministral-8B-Instruct-2410" 
    
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "google/gemma-3-1b-it",
    # "facebook/MobileLLM-1B",
    # "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
]

# Task-specific configurations
TASK_CONFIGS = {
    "general-nlp": {
        "dataset_name": "vicgalle/alpaca-gpt4",
        "llm_task": "generalnlp",
        "fraction_fit": 0.1,
        "num_supernodes": 20
    },
    "finance": {
        "dataset_name": "FinGPT/fingpt-sentiment-train",
        "llm_task": "finance",
        "fraction_fit": 0.1,
        "num_supernodes": 50
    },
    "medical": {
        "dataset_name": "medalpaca/medical_meadow_medical_flashcards",
        "llm_task": "medical",
        "fraction_fit": 0.1,
        "num_supernodes": 20
    },
    "code": {
        "dataset_name": "flwrlabs/code-alpaca-20k",
        "llm_task": "code",
        "fraction_fit": 0.2,
        "num_supernodes": 10
    }
}

# Evaluation types
EVAL_TYPES = ["general_nlp", "finance", "medical", "code"]

# Create results directory if it doesn't exist
results_dir = "experiment_results"
os.makedirs(results_dir, exist_ok=True)

# To be set in main() after parsing args
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
current_task = "general-nlp"  # Default task
results_file = None

def update_pyproject_toml(model_name, task_config):
    """Update the pyproject.toml file with the current model name and task-specific configurations."""
    pyproject_path = "pyproject.toml"
    config = toml.load(pyproject_path)
    
    # Update model name
    config["tool"]["flwr"]["app"]["config"]["model"]["name"] = model_name
    
    # Update task-specific configurations
    config["tool"]["flwr"]["app"]["config"]["static"]["dataset"]["name"] = task_config["dataset_name"]
    config["tool"]["flwr"]["app"]["config"]["strategy"]["fraction-fit"] = task_config["fraction_fit"]
    config["tool"]["flwr"]["federations"]["local-simulation"]["options"]["num-supernodes"] = task_config["num_supernodes"]
    
    with open(pyproject_path, 'w') as f:
        toml.dump(config, f)
    print(f"Updated pyproject.toml with model: {model_name} for task: {current_task}")
    print(f"  - Dataset: {task_config['dataset_name']}")
    print(f"  - LLM Task: {task_config['llm_task']}")
    print(f"  - Fraction Fit: {task_config['fraction_fit']}")
    print(f"  - Num Supernodes: {task_config['num_supernodes']}")

def update_eval_config_toml(model_name, peft_path, task=None, config_path="eval_config.toml"):
    """Update the eval_config.toml file with the latest PEFT path for the model."""
    if not os.path.exists(config_path):
        print(f"Warning: {config_path} not found, cannot update PEFT path")
        return False
        
    try:
        config = toml.load(config_path)
        
        # Check if the models section exists, if not create it
        if "models" not in config:
            config["models"] = {}
        
        # If the task parameter is provided, store paths in a task-specific structure
        if task:
            # Create task-specific section if it doesn't exist
            if "tasks" not in config:
                config["tasks"] = {}
                
            if task not in config["tasks"]:
                config["tasks"][task] = {}
                
            if "models" not in config["tasks"][task]:
                config["tasks"][task]["models"] = {}
                
            # Update task-specific model path
            if model_name not in config["tasks"][task]["models"]:
                config["tasks"][task]["models"][model_name] = {}
                
            config["tasks"][task]["models"][model_name]["peft_path"] = peft_path
            
            print(f"Updated {config_path} with task-specific PEFT path for {task}/{model_name}: {peft_path}")
        
        # Also update in the general models section
        if model_name not in config["models"]:
            config["models"][model_name] = {}
        
        # Update the PEFT path for this model
        config["models"][model_name]["peft_path"] = peft_path
        
        # Write the updated config back to the file
        with open(config_path, 'w') as f:
            toml.dump(config, f)
            
        print(f"Updated {config_path} with PEFT path for {model_name}: {peft_path}")
        return True
    except Exception as e:
        print(f"Error updating {config_path}: {e}")
        return False

def get_latest_peft_path(task_name=None, model_name=None, model_timestamp=None):
    """Find the most recent peft directory in the results folder based on config."""
    base_results_dir = "/home/st/flwr-nlp/results"
    
    # Read the number of rounds from pyproject.toml
    pyproject_path = "pyproject.toml"
    config = toml.load(pyproject_path)
    
    # Get the total rounds from config
    num_server_rounds = config.get("tool", {}).get("flwr", {}).get("app", {}).get("config", {}).get("num-server-rounds", 100)
    
    # Get the save frequency
    save_every_round = config.get("tool", {}).get("flwr", {}).get("app", {}).get("config", {}).get("train", {}).get("save-every-round", 5)
    
    # Calculate the highest round number that should be saved based on the save frequency
    highest_saved_round = (num_server_rounds // save_every_round) * save_every_round
    peft_dir_name = f"peft_{highest_saved_round}"
    
    print(f"Looking for PEFT directory: {peft_dir_name} (based on {num_server_rounds} rounds, saving every {save_every_round} rounds)")
    
    # If we have all the specific information, we can locate the exact directory
    if task_name and model_name and model_timestamp:
        model_id = model_name.replace("/", "_")
        specific_dir = f"{task_name}_{model_id}_{model_timestamp}"
        result_dir = os.path.join(base_results_dir, specific_dir)
        peft_path = os.path.join(result_dir, peft_dir_name)
        
        print(f"Looking for specific PEFT path: {peft_path}")
        
        if os.path.exists(peft_path):
            print(f"Found specific PEFT path: {peft_path}")
            return peft_path
        else:
            # Try to find any peft directory in this specific folder
            peft_dirs = sorted(glob.glob(os.path.join(result_dir, "peft_*")))
            if peft_dirs:
                peft_path = peft_dirs[-1]  # Get the highest available peft directory
                print(f"Found alternative PEFT path: {peft_path}")
                return peft_path
            else:
                print(f"No PEFT directories found in {result_dir}")
    
    # If we don't have specific information or couldn't find the specific directory,
    # try a more general search pattern
    result_dirs_pattern = os.path.join(base_results_dir, "????-??-??_??-??-??")
    
    # Refine search pattern based on available information
    if task_name:
        if model_name:
            model_id = model_name.replace("/", "_")
            # Look for task and model specific directories
            result_dirs_pattern = os.path.join(base_results_dir, f"{task_name}_{model_id}_*")
        else:
            # Look for task-specific directories
            result_dirs_pattern = os.path.join(base_results_dir, f"{task_name}_*")
    
    print(f"Searching for PEFT directories with pattern: {result_dirs_pattern}")
    result_dirs = sorted(glob.glob(result_dirs_pattern))
    
    if not result_dirs:
        print(f"No result directories found with pattern: {result_dirs_pattern}")
        # Try without task prefix if no results found
        if task_name:
            print("Trying to find results with only timestamp pattern...")
            result_dirs = sorted(glob.glob(os.path.join(base_results_dir, "????-??-??_??-??-??")))
    
    if not result_dirs:
        print("No result directories found!")
        return None
    
    # Get the most recent directory
    latest_dir = result_dirs[-1]
    peft_path = os.path.join(latest_dir, peft_dir_name)
    
    if not os.path.exists(peft_path):
        print(f"PEFT directory {peft_dir_name} not found in the latest results folder: {latest_dir}")
        # Fall back to listing available peft directories
        peft_dirs = sorted(glob.glob(os.path.join(latest_dir, "peft_*")))
        if peft_dirs:
            peft_path = peft_dirs[-1]  # Get the highest available peft directory
            print(f"Using fallback PEFT directory: {peft_path}")
        else:
            return None
        
    print(f"Found PEFT path: {peft_path}")
    return peft_path

def run_training(model_name, task_results_dir):
    """Run the federated learning training for the current model."""
    print(f"Starting training for model: {model_name} with task: {current_task}")
    
    # Get task-specific configuration
    task_config = TASK_CONFIGS[current_task]
    
    # Update configuration file
    update_pyproject_toml(model_name, task_config)
    
    # Create a model-specific timestamp for unique folder naming
    model_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create log file path for training
    train_log_path = os.path.join(task_results_dir, f"{model_name.replace('/', '_')}_train.log")
    
    # Run federated learning
    try:
        # Write training header to log file
        with open(train_log_path, 'w') as log_file:
            log_file.write(f"=== TRAINING LOG: {model_name} for task {current_task} ===\n")
            log_file.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"Task: {current_task}\n")
            log_file.write(f"LLM Task: {task_config['llm_task']}\n")
            log_file.write(f"Dataset: {task_config['dataset_name']}\n")
            log_file.write(f"Model timestamp: {model_timestamp}\n")
            log_file.write(f"Command: flwr run\n")
            log_file.write(f"{'='*60}\n\n")
        
        # Set environment variable for the llm_task to be used in the client code
        os.environ["FLWR_LLM_TASK"] = task_config["llm_task"]
        # Set environment variable for task name to be included in saved model directories
        os.environ["FLWR_TASK_NAME"] = current_task
        # Set environment variable for model-specific timestamp
        os.environ["FLWR_RUN_TIMESTAMP"] = model_timestamp
        # Set environment variable for model name
        os.environ["FLWR_MODEL_NAME"] = model_name.replace("/", "_")
        
        print(f"Set environment variables:")
        print(f"  FLWR_LLM_TASK={task_config['llm_task']}")
        print(f"  FLWR_TASK_NAME={current_task}")
        print(f"  FLWR_RUN_TIMESTAMP={model_timestamp}")
        print(f"  FLWR_MODEL_NAME={model_name.replace('/', '_')}")
        
        # Start the process with real-time output
        process = subprocess.Popen(
            "flwr run",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=os.environ.copy() # Use the current environment with the added vars
        )
        
        # Read and print output in real-time, also save to log file
        with open(train_log_path, 'a') as log_file:
            for line in iter(process.stdout.readline, ''):
                print(line, end='')  # Print to console in real-time
                log_file.write(line)  # Save to log file
            
            # Add training completion marker
            log_file.write(f"\n{'='*60}\n")
            log_file.write(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"=== END OF TRAINING LOG ===\n")
        
        # Wait for process to complete and get return code
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, "flwr run")
            
        print(f"Training completed for {model_name} on task {current_task}. Log saved to {os.path.abspath(train_log_path)}")
        
        # After successful training, get the latest PEFT path for this model
        model_results_dir = f"{current_task}_{model_name.replace('/', '_')}_{model_timestamp}"
        peft_path = get_latest_peft_path(task_name=current_task, model_name=model_name, model_timestamp=model_timestamp)
        if peft_path:
            # Update the eval_config.toml with the new PEFT path
            update_eval_config_toml(model_name, peft_path, task=current_task)
        else:
            print(f"Warning: Could not find PEFT path after training for {model_name}")
            
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during training for {model_name} on task {current_task}: {e}")
        
        # Log the error
        try:
            with open(train_log_path, 'a') as log_file:
                log_file.write(f"\nERROR: {str(e)}\n")
                log_file.write(f"Finished with error at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.write(f"=== END OF TRAINING LOG WITH ERROR ===\n")
        except Exception as log_err:
            print(f"Failed to write error to log file: {log_err}")
            
        return False
    except Exception as e:
        print(f"Unexpected error during training for {model_name} on task {current_task}: {str(e)}")
        
        # Log the error
        try:
            with open(train_log_path, 'a') as log_file:
                log_file.write(f"\nUNEXPECTED ERROR: {str(e)}\n")
                log_file.write(f"Finished with error at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.write(f"=== END OF TRAINING LOG WITH ERROR ===\n")
        except Exception as log_err:
            print(f"Failed to write error to log file: {log_err}")
            
        return False

def run_evaluation(model_name, eval_type="general_nlp", eval_config=None):
    """Run evaluation for the model and extract results for a specific evaluation type."""
    print(f"Starting evaluation for {model_name} - Type: {eval_type}")
    
    # Create log file path
    eval_log_path = os.path.join(results_dir, f"{model_name.replace('/', '_')}_{eval_type}_eval.log")
    
    try:
        # Get default model config
        default_config = eval_config.get("default", {})
        model_config = eval_config.get("models", {}).get(model_name, {})
        eval_type_config = eval_config.get(eval_type, {})
        
        # Check for task-specific model PEFT path
        task_specific_peft_path = None
        if "tasks" in eval_config and current_task in eval_config["tasks"]:
            if "models" in eval_config["tasks"][current_task] and model_name in eval_config["tasks"][current_task]["models"]:
                task_specific_peft_path = eval_config["tasks"][current_task]["models"][model_name].get("peft_path")
                if task_specific_peft_path:
                    print(f"Found task-specific PEFT path for {current_task}/{model_name}: {task_specific_peft_path}")
        
        # Get model-specific parameters, prioritizing task-specific paths
        peft_path = task_specific_peft_path or model_config.get("peft_path", default_config.get("default_peft_path", None))
        trust_remote_code = model_config.get("trust_remote_code", False)
        quantization = model_config.get("quantization", 4)
        
        # Get evaluation type parameters
        working_dir = eval_type_config.get("working_dir", default_config.get("working_dir", f"flowertune-eval-{eval_type}"))
        batch_size = eval_type_config.get("batch_size", default_config.get("batch_size", 16))
        script = eval_type_config.get("script", "eval.py")
        run_name = default_config.get("run_name", "fl")
        
        # Check for PEFT path
        if not peft_path:
            print(f"No PEFT path specified for {model_name}. Trying to find latest...")
            peft_path = get_latest_peft_path(task_name=current_task, model_name=model_name)
            if not peft_path:
                print(f"Could not find a valid PEFT path for evaluation. Skipping evaluation for {model_name}")
                return {
                    "model": model_name,
                    "eval_type": eval_type,
                    "scores": {}
                }
        else:
            print(f"Using PEFT path from config: {peft_path}")
        
        # Write evaluation header to log file
        with open(eval_log_path, 'w') as log_file:
            log_file.write(f"=== EVALUATION LOG: {model_name} - {eval_type} ===\n")
            log_file.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"Task: {current_task}\n")
            log_file.write(f"PEFT path: {peft_path}\n")
            log_file.write(f"Working directory: {working_dir}\n")
            log_file.write(f"Batch size: {batch_size}\n")
            log_file.write(f"{'='*60}\n\n")
        
        # Build command based on evaluation type
        cmd = ""
        
        if eval_type == "code":
            # Build code evaluation command
            tasks = eval_type_config.get("tasks", ["humaneval"])
            max_length = eval_type_config.get("max_length_generation", 1024)
            allow_exec = "--allow_code_execution" if eval_type_config.get("allow_code_execution", True) else ""
            save_gen = "--save_generations" if eval_type_config.get("save_generations", True) else ""
            save_ref = "--save_references" if eval_type_config.get("save_references", True) else ""
            use_auth = "--use_auth_token" if eval_type_config.get("use_auth_token", True) else ""
            
            # Run each task separately
            results = {}
            for task in tasks:
                task_cmd = (
                    f"cd {working_dir} && python {script} "
                    f"--model={model_name} "
                    f"--peft_model={peft_path} "
                    f"--max_length_generation={max_length} "
                    f"--batch_size={batch_size} "
                    f"{allow_exec} {save_gen} {save_ref} {use_auth} "
                    f"--tasks={task} "
                    f"--metric_output_path=./evaluation_results_{task}.json"
                )
                
                print(f"Running code evaluation command for task {task}: {task_cmd}")
                
                # Run the process
                task_log_path = os.path.join(results_dir, f"{model_name.replace('/', '_')}_{eval_type}_{task}_eval.log")
                process = subprocess.Popen(
                    task_cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                task_log_content = ""
                with open(task_log_path, 'w') as log_file:
                    log_file.write(f"=== CODE EVALUATION LOG: {model_name} - {task} ===\n")
                    log_file.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    log_file.write(f"Command: {task_cmd}\n")
                    log_file.write(f"{'='*60}\n\n")
                    
                    for line in iter(process.stdout.readline, ''):
                        print(line, end='')
                        log_file.write(line)
                        task_log_content += line
                    
                    log_file.write(f"\n{'='*60}\n")
                    log_file.write(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    log_file.write(f"=== END OF LOG ===\n")
                
                # Parse results from the output JSON file
                results[task] = extract_code_results(working_dir, task)
            
            return {
                "model": model_name,
                "eval_type": eval_type,
                "scores": results
            }
        else:
            # Build NLP evaluation command (general, finance, medical)
            extra_args = ""
            if trust_remote_code:
                extra_args += " --trust-remote-code"
                
            if eval_type == "general_nlp":
                category = eval_type_config.get("category", "stem,social_sciences,humanities")
                cmd = (
                    f"cd {working_dir} && python {script} "
                    f"--base-model-name-path {model_name} "
                    f"--peft-path {peft_path} "
                    f"--run-name {run_name} "
                    f"--batch-size {batch_size} "
                    f"--quantization {quantization} "
                    f"--category {category}{extra_args}"
                )
            else:  # finance or medical
                datasets = eval_type_config.get("datasets", "")
                cmd = (
                    f"cd {working_dir} && python {script} "
                    f"--base-model-name-path {model_name} "
                    f"--peft-path {peft_path} "
                    f"--run-name {run_name} "
                    f"--batch-size {batch_size} "
                    f"--quantization {quantization} "
                    f"--datasets {datasets}{extra_args}"
                )
            
            print(f"Running evaluation command: {cmd}")
            
            # Run the process
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            log_content = ""
            with open(eval_log_path, 'a') as log_file:
                log_file.write(f"Command: {cmd}\n\n")
                
                for line in iter(process.stdout.readline, ''):
                    print(line, end='')
                    log_file.write(line)
                    log_content += line
                
                log_file.write(f"\n{'='*60}\n")
                log_file.write(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.write(f"=== END OF LOG ===\n")
            
            # Wait for process to complete
            return_code = process.wait()
            if return_code != 0:
                print(f"Warning: Command returned non-zero exit code {return_code}, but continuing to parse output")
            
            print(f"Evaluation command completed. Log saved to {eval_log_path}")
            
            # Parse results based on evaluation type
            if eval_type == "general_nlp":
                categories = ["stem", "social_sciences", "humanities"]
                scores = {}
                
                for cat in categories:
                    score = extract_score(log_content, cat)
                    if score is not None:
                        print(f"Found {cat} score: {score:.4f}")
                        scores[cat] = score
                    else:
                        print(f"Could not find score for {cat}")
                        scores[cat] = None
                
                # Calculate average
                valid_scores = [s for s in scores.values() if s is not None]
                avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else None
                
                if avg_score is not None:
                    print(f"Average score: {avg_score:.4f}")
                    scores["average"] = avg_score
                else:
                    print("Could not calculate average score - no valid scores found")
                    scores["average"] = None
                    
                return {
                    "model": model_name,
                    "eval_type": eval_type,
                    "scores": scores
                }
            else:  # finance or medical
                datasets = eval_type_config.get("datasets", "").split(",")
                scores = {}
                
                for dataset in datasets:
                    score = extract_score(log_content, dataset)
                    if score is not None:
                        print(f"Found {dataset} score: {score:.4f}")
                        scores[dataset] = score
                    else:
                        print(f"Could not find score for {dataset}")
                        scores[dataset] = None
                
                # Calculate average
                valid_scores = [s for s in scores.values() if s is not None]
                avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else None
                
                if avg_score is not None:
                    print(f"Average score: {avg_score:.4f}")
                    scores["average"] = avg_score
                else:
                    print("Could not calculate average score - no valid scores found")
                    scores["average"] = None
                
                return {
                    "model": model_name,
                    "eval_type": eval_type,
                    "scores": scores
                }
    
    except Exception as e:
        print(f"Unexpected error during evaluation for {model_name} - {eval_type}: {str(e)}")
        
        # Log the error
        try:
            with open(eval_log_path, 'a') as log_file:
                log_file.write(f"\nERROR: {str(e)}\n")
                log_file.write(f"Finished with error at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.write(f"=== END OF LOG WITH ERROR ===\n")
        except Exception as log_err:
            print(f"Failed to write error to log file: {log_err}")
            
        return {
            "model": model_name,
            "eval_type": eval_type,
            "scores": {}
        }

def extract_score(log_content, category):
    """Extract score for a specific category from the log content."""
    # Convert category to lowercase for consistent matching
    category = category.lower()
    log_content = log_content.lower()
    
    # Different extraction patterns based on task type
    if category in ["fpb", "fiqa", "tfns"]:  # Finance-specific datasets
        # Try to match the finance dataset specific patterns
        finance_patterns = [
            rf"{category}.*?accuracy:?\s+(\d+\.\d+)",
            rf"dataset:?\s*{category}.*?accuracy:?\s+(\d+\.\d+)",
            rf"evaluating.*?{category}.*?accuracy:?\s+(\d+\.\d+)",
            rf"score.*?{category}[^0-9]*(\d+\.\d+)"
        ]
        
        for pattern in finance_patterns:
            matches = re.findall(pattern, log_content, re.IGNORECASE | re.DOTALL)
            if matches:
                return float(matches[-1])  # Return the last match (likely the final result)
                
    elif category in ["pubmedqa", "medmcqa", "medqa", "careqa"]:  # Medical-specific datasets
        # Try to match the medical dataset specific patterns
        medical_patterns = [
            rf"{category}.*?accuracy:?\s+(\d+\.\d+)",
            rf"dataset:?\s*{category}.*?accuracy:?\s+(\d+\.\d+)",
            rf"evaluating.*?{category}.*?accuracy:?\s+(\d+\.\d+)",
            rf"score.*?{category}[^0-9]*(\d+\.\d+)"
        ]
        
        for pattern in medical_patterns:
            matches = re.findall(pattern, log_content, re.IGNORECASE | re.DOTALL)
            if matches:
                return float(matches[-1])  # Return the last match (likely the final result)
                
    elif category in ["humaneval", "mbpp", "multiple-js", "multiple-cpp"]:  # Code-specific datasets
        # For code datasets, check specific patterns for pass@1 or other metrics
        code_patterns = [
            rf"{category}.*?pass@1:?\s+(\d+\.\d+)",
            rf"{category}.*?pass@1 \(%\):?\s+(\d+\.\d+)",
            rf"{category}.*?pass_at_1:?\s+(\d+\.\d+)",
            rf"{category}.*?score:?\s+(\d+\.\d+)"
        ]
        
        for pattern in code_patterns:
            matches = re.findall(pattern, log_content, re.IGNORECASE | re.DOTALL)
            if matches:
                return float(matches[-1])  # Return the last match
    else:
        # For general NLP and other categories
        patterns = [
            # 模式1: 类别后紧跟准确度
            rf"(?:category|dataset|running).*?{category}.*?accuracy:?\s+(\d+\.\d+)",
            # 模式2: 类别和准确度在同一段但不一定紧挨着
            rf"{category}(?:.*?\n)*?.*?accuracy:?\s+(\d+\.\d+)",
            # 模式3: 类别开始的段落到下一个类别之间包含准确度
            rf"{category}.*?(?=(?:{category}|stem|social_sciences|humanities|other|$)).*?accuracy:?\s+(\d+\.\d+)"
        ]
        
        # 尝试所有模式
        for pattern in patterns:
            matches = re.findall(pattern, log_content, re.IGNORECASE | re.DOTALL)
            if matches:
                # 找到匹配项，返回最后一个（通常是最终结果）
                return float(matches[-1])
    
    # 如果没有找到特定类别的匹配，尝试通用的准确度提取
    general_pattern = r"accuracy:?\s+(\d+\.\d+)"
    matches = re.findall(general_pattern, log_content, re.IGNORECASE)
    
    if matches:
        print(f"Warning: Could not find specific accuracy for '{category}', using general accuracy")
        # 如果有多个准确度值，则根据类别选择不同的值
        categories = ["stem", "social_sciences", "humanities", "other"]
        if category in categories:
            idx = categories.index(category)
            if idx < len(matches):
                return float(matches[idx])
        # 如果没有足够的值或类别不在列表中，返回最后一个
        return float(matches[-1])
    
    # 如果没有找到任何准确度，返回None
    print(f"Warning: Could not find any accuracy value for '{category}'")
    return None

def extract_code_results(working_dir, task):
    """Extract code evaluation results from the JSON output file."""
    json_path = os.path.join(working_dir, f"evaluation_results_{task}.json")
    
    if not os.path.exists(json_path):
        print(f"Warning: Results file {json_path} not found")
        return None
    
    try:
        with open(json_path, 'r') as f:
            results = json.load(f)
        
        # Extract the primary metric (pass@1 for HumanEval, etc.)
        if "pass@1" in results:
            return results["pass@1"]
        elif "pass@1 (%)" in results:  # Sometimes expressed as percentage
            return results["pass@1 (%)"] / 100.0
        elif "pass_at_1" in results:  # Alternative format
            return results["pass_at_1"]
        elif "score" in results:  # Generic score
            return results["score"]
        else:
            # If no specific score format is found, return the entire results
            return results
    except Exception as e:
        print(f"Error parsing code results from {json_path}: {e}")
        return None

def save_results(results, results_file=None):
    """Save results to separate CSV files for each evaluation type."""
    model_name = results["model"]
    eval_type = results["eval_type"]
    scores = results["scores"]
    
    # Use evaluation-specific results file if not provided
    if results_file is None:
        results_file = os.path.join(results_dir, f"{eval_type}_results_{timestamp}.csv")
    
    # Read existing data to avoid duplicates
    existing_models = []
    if os.path.exists(results_file) and os.path.getsize(results_file) > 0:
        try:
            with open(results_file, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                header = next(reader, None)  # Skip header
                for row in reader:
                    if row and len(row) > 0:
                        existing_models.append(row[0])
        except Exception as e:
            print(f"Warning: Error reading CSV file: {e}")
    
    # Check if model results already exist
    if model_name in existing_models:
        print(f"Results for model {model_name} already exist in {results_file}. Skipping write.")
        return
    
    try:
        # Create file with header if it doesn't exist
        if not os.path.exists(results_file) or os.path.getsize(results_file) == 0:
            with open(results_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Create header based on evaluation type
                if eval_type == "general_nlp":
                    writer.writerow(["Model", "STEM", "Social Sciences", "Humanities", "Average"])
                elif eval_type == "finance":
                    writer.writerow(["Model", "FPB", "FIQA", "TFNS", "Average"])
                elif eval_type == "medical":
                    writer.writerow(["Model", "PubMedQA", "MedMCQA", "MedQA", "CareQA", "Average"])
                elif eval_type == "code":
                    writer.writerow(["Model", "HumanEval", "MBPP", "Multiple-JS", "Multiple-CPP", "Average"])
        
        # Append results
        with open(results_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            if eval_type == "general_nlp":
                writer.writerow([
                    model_name,
                    format_score(scores.get("stem")),
                    format_score(scores.get("social_sciences")),
                    format_score(scores.get("humanities")),
                    format_score(scores.get("average"))
                ])
            elif eval_type == "finance":
                writer.writerow([
                    model_name,
                    format_score(scores.get("fpb")),
                    format_score(scores.get("fiqa")),
                    format_score(scores.get("tfns")),
                    format_score(scores.get("average"))
                ])
            elif eval_type == "medical":
                writer.writerow([
                    model_name,
                    format_score(scores.get("pubmedqa")),
                    format_score(scores.get("medmcqa")),
                    format_score(scores.get("medqa")),
                    format_score(scores.get("careqa")),
                    format_score(scores.get("average"))
                ])
            elif eval_type == "code":
                writer.writerow([
                    model_name,
                    format_score(scores.get("humaneval")),
                    format_score(scores.get("mbpp")),
                    format_score(scores.get("multiple-js")),
                    format_score(scores.get("multiple-cpp")),
                    format_score(scores.get("average"))
                ])
            
            print(f"Results for {model_name} ({eval_type}) saved to {results_file}")
    except Exception as e:
        print(f"Error writing results to CSV: {e}")

def format_score(score):
    """Format score for CSV output."""
    if score is None:
        return "N/A"
    try:
        return f"{float(score):.4f}"
    except (ValueError, TypeError):
        return str(score)

def print_summary(all_results):
    """Print a summary of all model results by evaluation type."""
    print("\n\n" + "="*100)
    print(f"EXPERIMENT RESULTS SUMMARY".center(100))
    print("="*100)
    
    # Group results by evaluation type
    results_by_type = {}
    for result in all_results:
        eval_type = result["eval_type"]
        if eval_type not in results_by_type:
            results_by_type[eval_type] = []
        results_by_type[eval_type].append(result)
    
    # Print summary for each evaluation type
    for eval_type, results in results_by_type.items():
        print(f"\n{eval_type.upper()} EVALUATION RESULTS:".center(100))
        print("-"*100)
        
        if eval_type == "general_nlp":
            # Print header
            print(f"{'Model':<50} | {'STEM':<10} | {'Social Sciences':<10} | {'Humanities':<10} | {'Average':<10}")
            print("-"*100)
            
            for result in results:
                model_short = result["model"].split("/")[-1]
                scores = result["scores"]
                print(f"{model_short:<50} | {format_score(scores.get('stem')):<10} | {format_score(scores.get('social_sciences')):<10} | {format_score(scores.get('humanities')):<10} | {format_score(scores.get('average')):<10}")
        
        elif eval_type == "finance":
            # Print header
            print(f"{'Model':<50} | {'FPB':<10} | {'FIQA':<10} | {'TFNS':<10} | {'Average':<10}")
            print("-"*100)
            
            for result in results:
                model_short = result["model"].split("/")[-1]
                scores = result["scores"]
                print(f"{model_short:<50} | {format_score(scores.get('fpb')):<10} | {format_score(scores.get('fiqa')):<10} | {format_score(scores.get('tfns')):<10} | {format_score(scores.get('average')):<10}")
        
        elif eval_type == "medical":
            # Print header
            print(f"{'Model':<50} | {'PubMedQA':<10} | {'MedMCQA':<10} | {'MedQA':<10} | {'CareQA':<10} | {'Average':<10}")
            print("-"*100)
            
            for result in results:
                model_short = result["model"].split("/")[-1]
                scores = result["scores"]
                print(f"{model_short:<50} | {format_score(scores.get('pubmedqa')):<10} | {format_score(scores.get('medmcqa')):<10} | {format_score(scores.get('medqa')):<10} | {format_score(scores.get('careqa')):<10} | {format_score(scores.get('average')):<10}")
        
        elif eval_type == "code":
            # Print header
            print(f"{'Model':<50} | {'HumanEval':<10} | {'MBPP':<10} | {'Multiple-JS':<10} | {'Multiple-CPP':<10} | {'Average':<10}")
            print("-"*100)
            
            for result in results:
                model_short = result["model"].split("/")[-1]
                scores = result["scores"]
                print(f"{model_short:<50} | {format_score(scores.get('humaneval')):<10} | {format_score(scores.get('mbpp')):<10} | {format_score(scores.get('multiple-js')):<10} | {format_score(scores.get('multiple-cpp')):<10} | {format_score(scores.get('average')):<10}")
    
    print("\n" + "="*100)
    print(f"Results saved to: {results_dir}")
    print("="*100)

def main():
    """Main function to run the experiment pipeline."""
    global current_task, timestamp, results_file
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run training and/or evaluation for language models')
    parser.add_argument('--eval-only', action='store_true', help='Run only evaluation without training')
    parser.add_argument('--models', type=str, nargs='+', help='Specific models to run (default: all models)')
    parser.add_argument('--config', type=str, default='eval_config.toml', help='Path to evaluation config file')
    parser.add_argument('--eval-types', type=str, default=','.join(EVAL_TYPES), 
                        help=f'Comma-separated list of evaluation types to run (default: all). Available: {",".join(EVAL_TYPES)}')
    parser.add_argument('--task', type=str, default='general-nlp', choices=TASK_CONFIGS.keys(), help='Task to run')
    parser.add_argument('--run-id', type=str, default=None, help='Run ID for grouping multiple tasks together')
    args = parser.parse_args()
    
    # Set global task
    current_task = args.task
    
    # Set timestamp - use provided run-id if available
    if args.run_id:
        timestamp = args.run_id
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Parse evaluation types
    eval_types_to_run = args.eval_types.split(',')
    invalid_types = [t for t in eval_types_to_run if t not in EVAL_TYPES]
    if invalid_types:
        print(f"Warning: Invalid evaluation types specified: {', '.join(invalid_types)}")
        print(f"Valid types are: {', '.join(EVAL_TYPES)}")
        eval_types_to_run = [t for t in eval_types_to_run if t in EVAL_TYPES]
        
    if not eval_types_to_run:
        print("No valid evaluation types specified. Exiting.")
        return
    
    # Select models to run
    models_to_run = args.models if args.models else models
    
    # Setup results directory and files with task name
    task_results_dir = os.path.join(results_dir, current_task, timestamp)
    os.makedirs(task_results_dir, exist_ok=True)
    
    # Set results file path for this task
    results_file = os.path.join(task_results_dir, f"{current_task}_results.csv")
    
    # Initialize results file with appropriate headers
    with open(results_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if current_task == 'general-nlp':
            writer.writerow(["Model", "STEM", "Social Sciences", "Humanities", "Average"])
        elif current_task == 'finance':
            writer.writerow(["Model", "FPB", "FIQA", "TFNS", "Average"])
        elif current_task == 'medical':
            writer.writerow(["Model", "PubMedQA", "MedMCQA", "MedQA", "CareQA", "Average"])
        elif current_task == 'code':
            writer.writerow(["Model", "HumanEval", "MBPP", "Multiple-JS", "Multiple-CPP", "Average"])
    
    # Load evaluation config
    eval_config = None
    if os.path.exists(args.config):
        try:
            eval_config = toml.load(args.config)
            print(f"Loaded evaluation config from {args.config}")
            
            # If models are specified in config and user didn't specify models, use config models
            if not args.models and "models" in eval_config and args.eval_only:
                config_models = list(eval_config.get("models", {}).keys())
                if config_models:
                    models_to_run = config_models
                    print(f"Using models from config file: {models_to_run}")
        except Exception as e:
            print(f"Error loading config file {args.config}: {e}")
            print("Continuing with default parameters...")
    else:
        print(f"Warning: Config file {args.config} not found. Using default parameters.")
    
    print(f"\n{'='*80}")
    print(f"Running {'evaluation only' if args.eval_only else 'training and evaluation'} for task: {current_task}")
    print(f"Models: {models_to_run}")
    print(f"Evaluation types: {eval_types_to_run}")
    print(f"Task config: {TASK_CONFIGS[current_task]}")
    print(f"All logs and results will be saved to: {os.path.abspath(task_results_dir)}")
    print(f"{'='*80}")
    
    # Dictionary to track training success for each model
    training_results = {}
    all_results = {}
    
    # PHASE 1: TRAINING ALL MODELS
    if not args.eval_only:
        print(f"\n\n{'#'*100}")
        print(f"TRAINING PHASE: Starting training for all models on task {current_task}")
        print(f"{'#'*100}")
        
        for model in models_to_run:
            print(f"\n{'-'*80}")
            print(f"TRAINING MODEL: {model} for task {current_task}")
            print(f"{'-'*80}")
            train_log_path = os.path.join(task_results_dir, f"{model.replace('/', '_')}_train.log")
            print(f"Training logs will be saved to: {os.path.abspath(train_log_path)}")
            training_success = run_training(model, task_results_dir)
            training_results[model] = training_success
            print(f"{'-'*80}")
            print(f"TRAINING COMPLETED FOR {model}: {'SUCCESS' if training_success else 'FAILED'}")
            print(f"{'-'*80}")
            
        print(f"\n{'#'*100}")
        print(f"TRAINING PHASE COMPLETED")
        print(f"Training summary:")
        for model, success in training_results.items():
            print(f"  - {model}: {'SUCCESS' if success else 'FAILED'}")
        print(f"{'#'*100}")
    
    # PHASE 2: EVALUATING ALL MODELS
    # Only run the evaluation type that matches the current task
    task_to_eval_type = {
        'general-nlp': 'general_nlp',
        'finance': 'finance',
        'medical': 'medical',
        'code': 'code'
    }
    
    # Get the evaluation type corresponding to the current task
    task_eval_type = task_to_eval_type.get(current_task)
    if task_eval_type not in eval_types_to_run:
        print(f"Warning: Evaluation type {task_eval_type} for task {current_task} not in specified eval types.")
        print(f"Adding {task_eval_type} to evaluation types.")
        eval_types_to_run = [task_eval_type]
    else:
        # Only evaluate on the current task's evaluation type
        eval_types_to_run = [task_eval_type]
    
    print(f"\n\n{'#'*100}")
    print(f"EVALUATION PHASE: Starting evaluation for {current_task} models on {task_eval_type}")
    print(f"{'#'*100}")
    
    # Create type-specific results file
    type_results_file = os.path.join(task_results_dir, f"{current_task}_results.csv")
    
    # Run evaluation for all models on this evaluation type
    for model in models_to_run:
        # Skip evaluation if training failed and we're not in eval-only mode
        if not args.eval_only and not training_results.get(model, True):
            print(f"\n{'-'*80}")
            print(f"SKIPPING EVALUATION FOR {model} ON {task_eval_type}: Training failed")
            print(f"{'-'*80}")
            continue
            
        print(f"\n{'-'*80}")
        print(f"EVALUATING MODEL: {model} ON {task_eval_type}")
        print(f"{'-'*80}")
        
        # Initialize model results if not exists
        if model not in all_results:
            all_results[model] = {}
            
        eval_log_path = os.path.join(task_results_dir, f"{model.replace('/', '_')}_{task_eval_type}_eval.log")
        print(f"Evaluation logs will be saved to: {os.path.abspath(eval_log_path)}")
        
        results = run_evaluation(model, task_eval_type, eval_config)
        
        # Store results
        all_results[model][task_eval_type] = results
        
        # Save results to type-specific file
        save_results(results, type_results_file)
        
        print(f"{'-'*80}")
        print(f"EVALUATION COMPLETED: {model} ON {task_eval_type}")
        print(f"{'-'*80}")
    
    print(f"\n{'#'*80}")
    print(f"ALL MODELS EVALUATED ON {task_eval_type}")
    print(f"{'#'*80}")
    
    print(f"\n{'#'*100}")
    print(f"EVALUATION PHASE COMPLETED FOR TASK {current_task}")
    print(f"Results saved to: {type_results_file}")
    print(f"{'#'*100}")
    
    # Convert all_results to format expected by print_summary
    summary_results = []
    for model, eval_types in all_results.items():
        for eval_type, results in eval_types.items():
            summary_results.append(results)
    
    # Print summary
    print_summary(summary_results)

if __name__ == "__main__":
    main() 