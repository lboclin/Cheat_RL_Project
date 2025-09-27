"""
This module provides a script to automate and manage multiple training runs.

It creates a timestamped parent directory for each batch of experiments and then
iterates a specified number of times, launching each training session in its own
isolated subdirectory to prevent interference between runs.
"""
import os
import subprocess
import argparse
from datetime import datetime

def run_training_sessions(num_runs: int):
    """
    Runs multiple training experiments in sequence.

    For each run, it creates a unique directory to store the results (logs and
    checkpoints) and invokes the main.py script, passing the directory path as
    a command-line argument.

    Args:
        num_runs (int): The number of independent training sessions to execute.
    """
    # --- 1. SETUP PARENT DIRECTORY FOR ALL EXPERIMENTS ---
    # Create a main folder for this batch of experiments, timestamped for uniqueness.
    # e.g., 'experiments_2025-09-21_21-30-00'
    base_dir_name = f"experiments_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(base_dir_name, exist_ok=True)
    print(f"Results parent directory created at: {base_dir_name}")

    for i in range(1, num_runs + 1):
        # --- 2. PREPARE DIRECTORY AND COMMAND FOR THIS RUN ---
        # Create a subdirectory for this specific run (e.g., 'run_01', 'run_02').
        run_name = f"run_{i:02d}"
        run_dir = os.path.join(base_dir_name, run_name)
        os.makedirs(run_dir, exist_ok=True)

        print("\n" + "="*50)
        print(f"--- STARTING EXPERIMENT {i}/{num_runs} ---")
        print(f"Results directory: {run_dir}")
        print("="*50)

        # Assemble the command to call main.py.
        # The output directory is passed as a command-line argument.
        command = [
            "python",
            "main.py",
            "--output_dir",
            run_dir
        ]

        # --- 3. EXECUTE THE TRAINING SCRIPT ---
        try:
            # Execute main.py and wait for it to complete.
            # 'check=True' will raise an exception if main.py returns an error code.
            subprocess.run(command, check=True)
            print(f"--- EXPERIMENT {i}/{num_runs} FINISHED SUCCESSFULLY ---")

        except subprocess.CalledProcessError as e:
            print(f"!!! ERROR: EXPERIMENT {i}/{num_runs} FAILED. Error code: {e.returncode} !!!")
        except KeyboardInterrupt:
            print("\n!!! Training interrupted by user. Halting automation. !!!")
            break
        except FileNotFoundError:
            print("!!! ERROR: 'main.py' not found. Ensure this script is in the same directory. !!!")
            break

    print("\n" + "="*50)
    print("ALL EXPERIMENTS CONCLUDED.")
    print("="*50)

if __name__ == "__main__":
    # --- 4. PARSE COMMAND-LINE ARGUMENTS ---
    # Set up the argument parser to accept the number of runs.
    parser = argparse.ArgumentParser(
        description="Script to automate the execution of multiple RL training runs."
    )
    parser.add_argument(
        "num_runs",
        type=int,
        help="The total number of training sessions to execute in series."
    )
    args = parser.parse_args()

    run_training_sessions(args.num_runs)