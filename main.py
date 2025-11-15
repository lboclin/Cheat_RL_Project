"""
This module contains the main script to run the training process for the RL agent.

It sets up the environment, initializes the agent and bot opponents, and manages
the main training loop, including logging, checkpointing, and episode management.
"""
import random
import torch
import os
import csv
import argparse
from cheat_env.environment import CheatEnviroment
from agents.bots import bot_strategy_80_20, bot_strategy_one_third, bot_100_0, bot_strategy_60_40, bot_strategy_challenger
from agents.rl_agent import RLAgent

def main():
    """
    Main function to run the training loop for the RL agent against various bots.
    """
    # --- 0. PARSE COMMAND-LINE ARGUMENTS ---
    parser = argparse.ArgumentParser(description="RL Agent training script for the game of Cheat.")
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save logs and checkpoints.')
    args = parser.parse_args()

    # --- 1. TRAINING AND LOGGING SETUP ---
    player_names = ["RL_Agent", "Bot_1", "Bot_2"]
    max_turns = 250
    num_episodes = 1
    BATCH_SIZE = 128
    LOG_INTERVAL = 500
    CHECKPOINT_INTERVAL = 500
    VISUALIZE_GAMES = True

    # --- Bot Strategy Pool and Naming ---
    bot_pool = {
        bot_strategy_80_20: 'Bot 80/20',
        bot_strategy_one_third: 'Bot 1/3',
        bot_100_0: 'Bot Honest',
        bot_strategy_60_40: 'Bot 60/40',
        bot_strategy_challenger: 'Bot Challenger'
    }
    
    # --- Data Logging Setup ---
    log_file_path = os.path.join(args.output_dir, 'win_rate_log.csv')
    CHECKPOINT_PATH = os.path.join(args.output_dir, "training_checkpoint.pth")

    # MODIFICATION: Added 'Epsilon' to the log header.
    log_header = ['Episode', 'RL_Agent'] + list(bot_pool.values()) + ['Epsilon']
    win_rate_log_history = []

    all_strategies = ['RL_Agent'] + list(bot_pool.values())
    interval_win_counts = {name: 0 for name in all_strategies}
    interval_games_played = {name: 0 for name in all_strategies}

    card_values = ["Joker", "Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King"]
    value_to_index = {value: i for i, value in enumerate(card_values)}

    # --- 2. ENVIRONMENT AND AGENT INITIALIZATION ---
    env = CheatEnviroment(players_names=player_names, max_episode_steps=max_turns)
    state_size = len(env.reset())
    agent = RLAgent(input_size=state_size)

    # --- 3. LOAD CHECKPOINT IF IT EXISTS ---
    start_episode = 0
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH)
        agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        agent.target_net.load_state_dict(checkpoint['policy_net_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.epsilon = checkpoint['epsilon']
        start_episode = checkpoint['episode'] + 1
        print(f"Checkpoint loaded. Resuming from episode {start_episode} with epsilon={agent.epsilon:.4f}")
        
        if os.path.exists(log_file_path):
            with open(log_file_path, 'r') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header:
                    for row in reader:
                        # This logic handles both old and new log formats.
                        parsed_row = [float(val) for val in row]
                        win_rate_log_history.append(parsed_row)
            print(f"Loaded {len(win_rate_log_history)} records from {log_file_path}")

    else:
        print("No checkpoint found. Starting training from scratch.")
        with open(log_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(log_header)

    # --- 4. MAIN TRAINING LOOP ---
    for episode in range(start_episode, num_episodes):
        state = env.reset()
        terminated = False
        truncated = False

        opponent_1_strategy_func = random.choice(list(bot_pool.keys()))
        opponent_2_strategy_func = random.choice(list(bot_pool.keys()))

        interval_games_played['RL_Agent'] += 1
        interval_games_played[bot_pool[opponent_1_strategy_func]] += 1
        interval_games_played[bot_pool[opponent_2_strategy_func]] += 1

        if VISUALIZE_GAMES:
            print(f"bot 1: {bot_pool[opponent_1_strategy_func]}, bot 2: {bot_pool[opponent_2_strategy_func]}")
            print(f"--- Episode {episode+1} ---")
        
        # --- Game Loop (Single Episode) ---
        while not terminated and not truncated:
            current_player = env.players[env.current_player_index]
            state = env._get_state()
            action = None

            if VISUALIZE_GAMES:
                print("-" * 30)
                print(f"Turn: {env.turn_count} | Player's Turn: {current_player.name}")
                
                for p in env.players:
                    # Create a frequency vector for the player's hand
                    hand_vector = [0] * 14
                    for card in p.hand:
                        if card.value in value_to_index:
                            hand_vector[value_to_index[card.value]] += 1
                    
                    print(f"  - {p.name}: {len(p.hand)} cards -> Frequencies: {hand_vector}")
                


                print(f"Rank to Play: '{env.current_rank_to_play}' | Cards in Pile: {len(env.round_discard_pile)}")

            if current_player.name == "RL_Agent":
                action = agent.choose_action(env._get_state(), env.get_valid_actions())
            elif current_player.name == "Bot_1":
                action = opponent_1_strategy_func(current_player, env.current_rank_to_play, env.last_number_of_cards_played)
            elif current_player.name == "Bot_2":
                action = opponent_2_strategy_func(current_player, env.current_rank_to_play, env.last_number_of_cards_played)

            if VISUALIZE_GAMES:
                action_type, cards, rank = action
                print(f"Action chosen by {current_player.name}: Type={action_type}, Cards={cards}, Rank='{rank}'")

            next_state, reward, terminated, truncated, info = env.step(action)

            if current_player.name == "RL_Agent":
                done = terminated or truncated
                agent.memory.push(state, action, next_state, reward, done)
                agent.learn(BATCH_SIZE)

        # --- End of Episode: Record Winner ---
        if terminated:

            if VISUALIZE_GAMES:
                print(f"GAME OVER! The winner is: {env.winner.name}")
                print("=" * 30)
                print("\n")

            winner_name = env.winner.name
            if winner_name == "RL_Agent":
                interval_win_counts['RL_Agent'] += 1
            elif winner_name == "Bot_1":
                strategy_name = bot_pool[opponent_1_strategy_func]
                interval_win_counts[strategy_name] += 1
            elif winner_name == "Bot_2":
                strategy_name = bot_pool[opponent_2_strategy_func]
                interval_win_counts[strategy_name] += 1
        
        # --- 5. LOGGING AND CHECKPOINTING ---
        if (episode + 1) % LOG_INTERVAL == 0:
            print(f"\n--- End of Interval (Episode {episode + 1}) ---")
            
            win_rates = {}
            for name in all_strategies:
                wins = interval_win_counts[name]
                played = interval_games_played[name]
                win_rates[name] = (wins / played * 100) if played > 0 else 0.0
            
            win_rate_values = [win_rates.get(name, 0.0) for name in log_header if name not in ['Episode', 'Epsilon']]
            log_row = [episode + 1] + win_rate_values + [agent.epsilon]
            
            # Pad old history if resuming from a log without the Epsilon column.
            if win_rate_log_history and len(win_rate_log_history[0]) < len(log_header):
                for old_row in win_rate_log_history:
                    old_row.append(0.0) # Placeholder for missing epsilon
            
            win_rate_log_history.append(log_row)

            with open(log_file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(log_header)
                
                formatted_history = []
                for row_data in win_rate_log_history:
                    episode_num = int(row_data[0])
                    win_rates_part = [f"{val:.2f}" for val in row_data[1:-1]]
                    epsilon_part = f"{row_data[-1]:.6f}" # More precision for epsilon
                    formatted_history.append([episode_num] + win_rates_part + [epsilon_part])
                writer.writerows(formatted_history)
            
            print(f"Win rates saved to {log_file_path}. Epsilon: {agent.epsilon:.4f}")
            print({k: f"{v:.2f}%" for k, v in win_rates.items()})
            
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