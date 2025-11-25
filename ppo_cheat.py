"""
This script executes the Proximal Policy Optimization (PPO) training loop for the 
Cheat RL agent.

It integrates the environment, the PPO Actor-Critic network, and the CleanRL-based
training logic. It handles hyperparameter configuration, rollout data collection,
policy updates via backpropagation, and logging of training metrics to TensorBoard.

Key Features:
- Hierarchical action handling for the complex Cheat game space.
- Automatic GPU detection and utilization.
- Modular reward shaping toggles.
- Model checkpointing for later evaluation.
"""
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

# Local imports from the project structure
from agents_ppo.ppo_actor_critic import PPOActorCritic
from cheat_env.environment import CheatEnviroment

@dataclass
class Args:
    """
    Configuration class for hyperparameters and experiment settings.
    
    Uses 'tyro' for command-line argument parsing.
    """
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """The name of this experiment for logging purposes."""
    seed: int = 1
    """Random seed for reproducibility."""
    torch_deterministic: bool = True
    """If True, sets torch.backends.cudnn.deterministic=False for consistent results."""
    cuda: bool = True
    """If True, enables CUDA (GPU) acceleration if available."""
    track: bool = False
    """If True, tracks the experiment with Weights and Biases (wandb)."""
    wandb_project_name: str = "cleanRL"
    """The wandb project name."""
    wandb_entity: str = None
    """The wandb entity (team/user)."""

    # --- Game-Specific Arguments ---
    env_name: str = "Cheat_v1"
    """Name of the environment."""
    visualize: bool = False
    """If True, enables console logging of game turns."""
    max_cards_played: int = 6
    """Maximum number of cards allowed in a single play action."""
    save_model: bool = False
    """If True, saves the trained model state dict (.pth) at the end."""
    reward_shaping: bool = False
    """If True, enables intermediate shaping rewards in the environment."""
    save_path: str = "ppo_models"
    """Directory to save trained models."""

    # --- PPO Algorithm Arguments ---
    total_timesteps: int = 256
    """Total number of timesteps (environment interactions) to train for."""
    learning_rate: float = 2.5e-4
    """The learning rate for the Adam optimizer."""
    num_envs: int = 1
    """Number of parallel game environments (Keep at 1 for this non-vectorized env)."""
    num_steps: int = 256
    """Number of steps to run in each environment per policy rollout (collection phase)."""
    anneal_lr: bool = True
    """If True, linearly anneals the learning rate from initial value to 0."""
    gamma: float = 0.99
    """Discount factor for future rewards."""
    gae_lambda: float = 0.95
    """Lambda parameter for General Advantage Estimation (GAE)."""
    num_minibatches: int = 4
    """Number of mini-batches to split the rollout data into for updating."""
    update_epochs: int = 4
    """Number of epochs to update the policy using the collected rollout data."""
    norm_adv: bool = True
    """If True, normalizes advantages (mean 0, std 1) during updates."""
    clip_coef: float = 0.2
    """Clipping coefficient for the PPO surrogate objective."""
    clip_vloss: bool = True
    """If True, uses clipped value function loss to improve stability."""
    ent_coef: float = 0.01
    """Coefficient for the entropy bonus (encourages exploration)."""
    vf_coef: float = 0.5
    """Coefficient for the value function loss in the total loss equation."""
    max_grad_norm: float = 0.5
    """Maximum norm for gradient clipping."""
    target_kl: float = None
    """Target KL divergence threshold for early stopping of updates."""

    # --- Runtime Computed Variables ---
    batch_size: int = 0
    """Total batch size (num_envs * num_steps). Computed at runtime."""
    minibatch_size: int = 0
    """Size of each mini-batch. Computed at runtime."""
    num_iterations: int = 0
    """Number of PPO iterations (total_timesteps // batch_size). Computed at runtime."""


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Initializes a neural network layer with orthogonal initialization.
    
    This is a standard practice for PPO to improve training stability.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


if __name__ == "__main__":
    # --- 1. Argument Parsing and Setup ---
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_name}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    # --- 2. Logging Setup ---
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # --- 3. Seeding for Reproducibility ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Automatically detect CUDA device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # --- 4. Environment Initialization ---
    # Initialize the custom Cheat environment with configured settings.
    envs = CheatEnviroment(
        players_names=["RL_Agent", "Bot_1", "Bot_2"], 
        visualize=args.visualize, 
        max_episode_steps=args.num_steps, 
        reward_shaping=args.reward_shaping
    )
    temp_state = envs.reset()
    state_size = len(temp_state)

    # --- 5. Agent (Actor-Critic) Initialization ---
    agent = PPOActorCritic(state_size).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # --- 6. Rollout Buffer Initialization ---
    # Pre-allocate tensors on the device to store data collected during rollouts.
    obs = torch.zeros((args.num_steps, args.num_envs, state_size)).to(device)
    
    # Separate buffers for each component of the hierarchical action space.
    action_types = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long).to(device)
    action_ranks_announced = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long).to(device)
    action_quantities = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long).to(device)
    
    # Buffer for the specific cards played. Uses -1 for padding.
    # Shape: [Steps, Envs, Max Cards]
    action_card_ranks_played = torch.full(
        (args.num_steps, args.num_envs, args.max_cards_played), -1, dtype=torch.long
    ).to(device)
    
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    
    # --- 7. Main Training Loop ---
    global_step = 0
    start_time = time.time()
    
    next_obs = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device).unsqueeze(0)
    next_done = torch.zeros(args.num_envs).to(device)
    current_episode_return = 0.0

    for iteration in range(1, args.num_iterations + 1):
        # Learning Rate Annealing
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # --- A. Rollout Phase (Data Collection) ---
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # Action Selection (No Gradient)
            with torch.no_grad():
                valid_actions = envs.get_valid_actions()
                
                # The agent returns the action tuple, total log probability, entropy,
                # value estimate, and the separated indices for buffer storage.
                full_action_tuple, total_logprob, entropy, value, action_indices = agent.get_action_and_value(
                    next_obs, 
                    valid_actions=valid_actions
                )
                
                # Store value and total log_prob
                values[step] = value.flatten()
                logprobs[step] = total_logprob 
                
                # Store action components in their respective buffers
                action_types[step] = action_indices["type"]
                action_ranks_announced[step] = action_indices["rank_announced"]
                action_quantities[step] = action_indices["quantity"]
                
                # Handle padding for variable-length card plays
                played_card_ranks_list = action_indices["card_ranks_played"] # List of tensors
                num_cards_played = len(played_card_ranks_list)
                if num_cards_played > 0:
                    # Stack tensors and fill the fixed-size buffer
                    padded_ranks = torch.full((args.max_cards_played,), -1, dtype=torch.long).to(device)
                    padded_ranks[:num_cards_played] = torch.stack(played_card_ranks_list).squeeze()
                    action_card_ranks_played[step] = padded_ranks
                # If 0 cards played (Pass/Doubt), buffer remains filled with -1

            # Execute Action in Environment
            next_obs_val, reward_val, terminated, truncated, info = envs.step(full_action_tuple)
            
            current_episode_return += reward_val
            next_done_val = terminated or truncated
            rewards[step] = torch.tensor([reward_val]).to(device)
            next_obs = torch.Tensor(next_obs_val).to(device).unsqueeze(0)
            next_done = torch.Tensor([next_done_val]).to(device)

            # Logging at Episode End
            if next_done_val:
                print(f"global_step={global_step}, episodic_return={current_episode_return}")
                writer.add_scalar("charts/episodic_return", current_episode_return, global_step)

                # Log the 'clean' win rate for the agent
                if "is_win" in info:
                    writer.add_scalar("charts/win_rate/RL_Agent", info.get("is_win", 0.0), global_step)
                
                if "winner_name" in info:
                    winner = info["winner_name"]
                    if winner != "RL_Agent":
                        writer.add_scalar(f"charts/win_rate_opponent/{winner}", 1, global_step)
                
                current_episode_return = 0.0
                next_obs = envs.reset()
                next_obs = torch.Tensor(next_obs).to(device).unsqueeze(0)

        # --- B. Advantage Estimation (GAE) ---
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # --- C. Flatten Batch for Training ---
        b_obs = obs.reshape((-1, state_size))
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        
        # Flatten action buffers
        b_action_types = action_types.reshape(-1)
        b_action_ranks_announced = action_ranks_announced.reshape(-1)
        b_action_quantities = action_quantities.reshape(-1)
        # Card ranks buffer becomes [batch_size, max_cards_played]
        b_action_card_ranks_played = action_card_ranks_played.reshape((-1, args.max_cards_played))

        # --- D. Optimization Phase (Policy Update) ---
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # Prepare action dictionary for re-evaluation
                mb_action_indices = {
                    "type": b_action_types[mb_inds],
                    "rank_announced": b_action_ranks_announced[mb_inds],
                    "quantity": b_action_quantities[mb_inds],
                    "card_ranks_played": b_action_card_ranks_played[mb_inds] 
                }

                # Re-evaluate actions to get new log_probs, entropy, and values
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], 
                    action=mb_action_indices
                )

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # Calculate KL divergence for monitoring
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # --- Calculate Losses ---
                
                # Policy Loss (Clipped Surrogate Objective)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss (MSE with Clipping)
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Entropy Loss (Bonus for exploration)
                entropy_loss = entropy.mean()
                
                # Total Loss
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                # --- Backpropagation and Optimization ---
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            # Early stopping based on KL divergence (optional)
            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # --- E. Metric Logging ---
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    # --- 8. Model Saving ---
    if args.save_model:
        os.makedirs(args.save_path, exist_ok=True)
        model_path = os.path.join(args.save_path, f"{run_name}.pth")
        torch.save(agent.state_dict(), model_path)
        print(f"\n--- Model saved to: {model_path} ---")

    # --- End of Training ---
    writer.close()
    print("--- PPO Training Finished ---")