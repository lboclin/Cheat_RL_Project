"""
This module contains the main script to run the training process for the RL agent.

It sets up the environment, initializes the agent and bot opponents, and manages
the main training loop, including logging, checkpointing, and episode management.

[UPDATED VERSION]
This script has been adapted to work with the Gym-style 'environment.py',
where the environment's 'step()' function handles all bot turns internally.
"""
import random
import torch
import os
import csv
import argparse
from cheat_env.environment import CheatEnviroment
from bots.strategies import bot_strategy_80_20, bot_strategy_one_third, bot_strategy_100_0, bot_strategy_60_40, bot_strategy_challenger
from agents_dqn.rl_agent import RLAgent 

def main():
    """
    Main function to execute the DQN agent training loop.
    """
    # --- 0. PARSE COMMAND-LINE ARGUMENTS ---
    parser = argparse.ArgumentParser(description="RL Agent (DQN) training script for Cheat.")
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save logs and checkpoints.')
    
    # 'visualize' is a boolean flag (True if present, False if absent).
    parser.add_argument('--visualize', action='store_true', help='Enable game logs in the console (default: False).')
    parser.add_argument('--num_episodes', type=int, default=1, help='Total number of episodes to train.')
    
    args = parser.parse_args()

    # --- 1. TRAINING AND LOGGING SETUP ---
    player_names = ["RL_Agent", "Bot_1", "Bot_2"]
    max_turns = 250 # The environment also has a 'max_episode_steps' check.
    BATCH_SIZE = 128
    LOG_INTERVAL = 500
    CHECKPOINT_INTERVAL = 500
    
    # Map bot strategy functions to readable names for logging.
    bot_pool_names = {
        bot_strategy_80_20: 'Bot 80/20',
        bot_strategy_one_third: 'Bot 1/3',
        bot_strategy_100_0: 'Bot Honest',
        bot_strategy_60_40: 'Bot 60/40',
        # bot_strategy_challenger: 'Bot Challenger'
    }
    
    # --- Data Logging Setup ---
    log_file_path = os.path.join(args.output_dir, 'win_rate_log.csv')
    CHECKPOINT_PATH = os.path.join(args.output_dir, "training_checkpoint.pth")

    log_header = ['Episode', 'RL_Agent'] + list(bot_pool_names.values()) + ['Epsilon']
    win_rate_log_history = []

    all_strategies = ['RL_Agent'] + list(bot_pool_names.values())
    interval_win_counts = {name: 0 for name in all_strategies}
    interval_games_played = {name: 0 for name in all_strategies}

    # --- 2. ENVIRONMENT AND AGENT INITIALIZATION ---
    # Initialize the Gym-style environment with visualization options.
    env = CheatEnviroment(
        players_names=player_names, 
        max_episode_steps=max_turns, 
        visualize=args.visualize
    )
    
    state = env.reset() # Returns the initial state vector (numpy array).
    state_size = len(state)
    agent = RLAgent(input_size=state_size)

    # --- 3. LOAD CHECKPOINT (If available) ---
    start_episode = 0
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH)
        agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        agent.target_net.load_state_dict(checkpoint['policy_net_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.epsilon = checkpoint['epsilon']
        start_episode = checkpoint['episode'] + 1
        print(f"Checkpoint loaded. Resuming from episode {start_episode} with epsilon={agent.epsilon:.4f}")
        
        # Load existing log data to maintain history consistency.
        if os.path.exists(log_file_path):
            with open(log_file_path, 'r') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header:
                    for row in reader:
                        parsed_row = [float(val) for val in row]
                        win_rate_log_history.append(parsed_row)
            print(f"Loaded {len(win_rate_log_history)} records from {log_file_path}")

    else:
        print("No checkpoint found. Starting training from scratch.")
        with open(log_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(log_header)

    # --- 4. MAIN TRAINING LOOP ---
    for episode in range(start_episode, args.num_episodes):
        state = env.reset() # Resets the environment for a new episode.
        terminated = False
        truncated = False

        # Note: Bot strategy assignment is now handled internally by env.reset().
        
        # Retrieve the names of the current bots from the environment for logging purposes.
        bot_1_name = env.bot_strategy_names.get("Bot_1", "Unknown Bot")
        bot_2_name = env.bot_strategy_names.get("Bot_2", "Unknown Bot")

        interval_games_played['RL_Agent'] += 1
        if bot_1_name in interval_games_played: interval_games_played[bot_1_name] += 1
        if bot_2_name in interval_games_played: interval_games_played[bot_2_name] += 1

        
        # --- Game Loop (Single Episode) ---
        while not terminated and not truncated:
            
            # The environment ensures it is always the agent's turn (index 0) here.
            
            # 1. Select an action using the agent's policy.
            valid_actions = env.get_valid_actions()
            action = agent.choose_action(state, valid_actions)

            # 2. Execute the action in the environment.
            # The 'step' method processes the agent's move AND all subsequent bot turns.
            next_state, reward, terminated, truncated, info = env.step(action)

            # 3. Store the transition in replay memory.
            done = terminated or truncated
            agent.memory.push(state, action, next_state, reward, done)
            
            # 4. Perform a learning step (backpropagation).
            agent.learn(BATCH_SIZE)
            
            # 5. Update the current state.
            state = next_state

        # --- End of Episode: Record Statistics ---
        if terminated:
            winner_name = env.winner.name
            if winner_name == "RL_Agent":
                interval_win_counts['RL_Agent'] += 1
            elif winner_name == "Bot_1":
                if bot_1_name in interval_win_counts: interval_win_counts[bot_1_name] += 1
            elif winner_name == "Bot_2":
                if bot_2_name in interval_win_counts: interval_win_counts[bot_2_name] += 1
        
        # --- 5. LOGGING AND CHECKPOINTING ---
        if (episode + 1) % LOG_INTERVAL == 0:
            print(f"\n--- End of Interval (Episode {episode + 1}) ---")
            
            win_rates = {}
            for name in all_strategies:
                wins = interval_win_counts[name]
                played = interval_games_played[name]
                win_rates[name] = (wins / played * 100) if played > 0 else 0.0
            
            # Prepare log row, ensuring alignment with the header.
            win_rate_values = [win_rates.get(name, 0.0) for name in log_header if name not in ['Episode', 'Epsilon']]
            log_row = [episode + 1] + win_rate_values + [agent.epsilon]
            
            # Handle potential column mismatches with older log files.
            if win_rate_log_history and len(win_rate_log_history[0]) < len(log_header):
                for old_row in win_rate_log_history:
                    if len(old_row) < len(log_header):
                        old_row.append(0.0) 
            
            win_rate_log_history.append(log_row)

            with open(log_file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(log_header)
                
                formatted_history = []
                for row_data in win_rate_log_history:
                    episode_num = int(row_data[0])
                    win_rates_part = [f"{val:.2f}" for val in row_data[1:-1]] 
                    epsilon_part = f"{row_data[-1]:.6f}"
                    formatted_history.append([episode_num] + win_rates_part + [epsilon_part])
                writer.writerows(formatted_history)
            
            print(f"Win rates saved to {log_file_path}. Epsilon: {agent.epsilon:.4f}")
            print({k: f"{v:.2f}%" for k, v in win_rates.items()})
            
            # Reset interval counters.
            interval_win_counts = {name: 0 for name in all_strategies}
            interval_games_played = {name: 0 for name in all_strategies}

        if (episode + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoint = {
                'episode': episode,
                'policy_net_state_dict': agent.policy_net.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'epsilon': agent.epsilon
            }
            torch.save(checkpoint, CHECKPOINT_PATH)
            print(f"Checkpoint saved at episode {episode + 1}\n")

    print("--- Training Finished ---")

if __name__ == "__main__":
    main()