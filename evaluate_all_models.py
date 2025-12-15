"""
Mass Evaluation Script for PPO Models.

This script loads all model checkpoints saved in the 'ppo_models' directory,
runs a fixed number of games (e.g., 500) for each, and generates a statistical
Win Rate report comparing the Agent against different Bot strategies.
"""
import os
import torch
import numpy as np
from collections import defaultdict
import time

# Local imports from the project
from cheat_env.environment import CheatEnviroment
from agents_ppo.ppo_actor_critic import PPOActorCritic

def evaluate_directory(models_dir="ppo_models", num_games=500):
    # 1. Find all .pth model files
    if not os.path.exists(models_dir):
        print(f"Directory '{models_dir}' not found.")
        return

    model_files = [f for f in os.listdir(models_dir) if f.endswith(".pth")]
    
    # Sort by modification time (or name) for easier reading
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)))

    if not model_files:
        print("No .pth models found.")
        return

    print(f"Found {len(model_files)} models. Starting evaluation of {num_games} games per model...\n")

    # 2. Setup Environment and Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize environment WITHOUT visualization for speed
    env = CheatEnviroment(players_names=["RL_Agent", "Bot_1", "Bot_2"], visualize=False)
    
    # Configure Agent (Neural Network)
    temp_state = env.reset()
    state_size = len(temp_state)
    agent = PPOActorCritic(state_size).to(device)

    # 3. Loop through Models
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        print(f"--- Evaluating Model: {model_file} ---")
        
        try:
            # Load the model weights
            agent.load_state_dict(torch.load(model_path, map_location=device))
            agent.eval() # Set to evaluation mode
        except Exception as e:
            print(f"Error loading {model_file}: {e}")
            continue

        # Counters for this model
        # Using defaultdict to handle dynamic bot names
        win_counts = defaultdict(int)
        
        start_time = time.time()

        # 4. Games Loop (500 games)
        for i in range(num_games):
            state = env.reset()
            terminated = False
            truncated = False
            
            while not terminated and not truncated:
                # Prepare state for GPU
                state_tensor = torch.Tensor(state).to(device).unsqueeze(0)
                valid_actions = env.get_valid_actions()

                # Agent chooses action
                with torch.no_grad():
                    full_action, _, _, _, _ = agent.get_action_and_value(
                        state_tensor, 
                        valid_actions=valid_actions
                    )

                # Environment processes (Agent Turn + Bot Turns)
                state, _, terminated, truncated, info = env.step(full_action)

            # End of game: Register winner
            if env.winner:
                if env.winner.name == "RL_Agent":
                    win_counts["RL_Agent"] += 1
                else:
                    # Get STRATEGY name (e.g., "Bot 80/20") instead of "Bot_1"
                    strategy_name = env.bot_strategy_names[env.winner.name]
                    win_counts[strategy_name] += 1
            
            # (Optional) Simple progress bar
            if (i + 1) % 100 == 0:
                print(f"   Played {i + 1}/{num_games}...")

        elapsed_time = time.time() - start_time
        
        # 5. Calculate and Print Statistics
        print(f"\nResults for {model_file} ({elapsed_time:.2f}s):")
        print("-" * 40)
        print(f"{'Player / Strategy':<25} | {'Wins':<10} | {'Win Rate':<10}")
        print("-" * 40)
        
        # Sort results to show who won the most
        sorted_results = sorted(win_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Ensure RL_Agent appears even if it has 0 wins
        if "RL_Agent" not in win_counts:
            sorted_results.append(("RL_Agent", 0))

        total_wins_check = 0
        for name, count in sorted_results:
            percentage = (count / num_games) * 100
            print(f"{name:<25} | {count:<10} | {percentage:.2f}%")
            total_wins_check += count
            
        # Calculate truncated games (draws/time limit)
        truncated_games = num_games - total_wins_check
        if truncated_games > 0:
            print(f"{'Draw/Truncated':<25} | {truncated_games:<10} | {(truncated_games/num_games)*100:.2f}%")
            
        print("-" * 40 + "\n")

if __name__ == "__main__":
    # You can change num_games here if desired
    evaluate_directory(num_games=500)