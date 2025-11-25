"""
This script facilitates the evaluation of a trained PPO agent.

It loads a saved model checkpoint, initializes the environment with visualization
enabled, and runs a full game episode. This allows for qualitative analysis of the
agent's learned strategy and behavior against bot opponents.
"""
import torch
import argparse
import time
import numpy as np

from cheat_env.environment import CheatEnviroment
from agents_ppo.ppo_actor_critic import PPOActorCritic

def evaluate_agent(model_path: str):
    """
    Loads a trained PPO model and executes a single evaluation episode.

    Args:
        model_path: The file path to the saved PyTorch model checkpoint (.pth).
    """
    print(f"Loading model from: {model_path}")

    # --- 1. Environment Setup ---
    # Initialize the environment with 'visualize=True' to enable detailed console logs.
    env = CheatEnviroment(
        players_names=["RL_Agent", "Bot_1", "Bot_2"],
        visualize=True 
    )
    
    state = env.reset() 
    state_size = len(state)
    
    # Automatically detect available device (CUDA or CPU).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Model Initialization and Loading ---
    agent = PPOActorCritic(state_size).to(device)
    
    try:
        # Load the state dictionary into the model. 
        # map_location ensures compatibility if loading a GPU model on a CPU.
        agent.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    # Set the agent to evaluation mode (disables dropout/batch norm layers if present).
    agent.eval()

    print("\n--- STARTING EVALUATION EPISODE ---")

    # --- 3. Evaluation Loop ---
    terminated = False
    truncated = False

    while not terminated and not truncated:
        # Prepare state tensor.
        state_tensor = torch.Tensor(state).to(device).unsqueeze(0)
        
        # Retrieve valid actions for masking.
        valid_actions = env.get_valid_actions()
        
        # Select an action using the agent's policy (inference only, no gradient tracking).
        with torch.no_grad():
            full_action, _, _, _, _ = agent.get_action_and_value(
                state_tensor, 
                valid_actions=valid_actions
            )
        
        # Execute the action in the environment.
        # 'state' é atualizado aqui para o próximo loop
        state, reward, terminated, truncated, info = env.step(full_action)
        
        # Pause briefly to allow the user to read the console output.
        time.sleep(1.0) 

    print(f"--- EVALUATION FINISHED ---")
    if env.winner:
        print(f"Winner: {env.winner.name}")
    else:
        print("Game ended due to turn limit truncation.")

if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO agent.")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True, 
        help="Path to the trained model checkpoint (.pth file)."
    )
    args = parser.parse_args()
    
    evaluate_agent(args.model_path)