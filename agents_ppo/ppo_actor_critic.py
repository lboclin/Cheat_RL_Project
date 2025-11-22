"""
This module defines the PPO Actor-Critic neural network for the Cheat RL agent.

The architecture consists of a shared backbone for feature extraction, followed
by separate heads for the critic (value function) and the actor (policy).
The actor is further divided into multiple heads to handle the hierarchical
nature of the game's action space (type, rank, quantity, and card selection).
"""
import torch
import torch.nn as nn
import numpy as np
import random
from torch.distributions.categorical import Categorical

from cheat_env.card import Card, Suit # For type hinting

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Initializes network layer weights using orthogonal initialization.
    
    This is a standard practice for PPO to maintain gradient stability and
    ensure consistent signal propagation through the network.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPOActorCritic(nn.Module):
    """
    The PPO Actor-Critic network architecture.
    
    This class encapsulates both the policy (Actor) and value function (Critic).
    It is designed to process the game state and output both a scalar value estimate
    and a set of probability distributions corresponding to the hierarchical decisions
    required to play a turn in "Cheat".
    """
    def __init__(self, state_size):
        """
        Initializes the network layers and decision heads.

        Args:
            state_size (int): The dimension of the input state vector.
        """
        super().__init__()

        # --- Card Mappings (Must match environment.py) ---
        self.card_values = ["Joker", "Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King"]
        # Map card values to indices (0-13) for card selection (includes Joker)
        self.rank_to_index_14 = {value: i for i, value in enumerate(self.card_values)}
        # Map card values to indices (0-12) for rank announcement (excludes Joker)
        self.rank_to_index_13 = {value: i for i, value in enumerate(self.card_values[1:])}
        
        # Maximum number of cards that can be played in a single turn
        self.max_cards_played = 6 


        # --- Critic Network (Value Function) ---
        # Estimates the value of state V(s) used for advantage calculation.
        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_size, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )

        # --- Actor Network (Policy Function) ---
        # Shared feature extractor for all actor heads
        self.actor_shared = nn.Sequential(
            layer_init(nn.Linear(state_size, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
        )

        # Decision Heads (Actors)
        self.action_type_head = layer_init(nn.Linear(128, 3), std=0.01)     # 0:Doubt, 1:Pass, 2:Play
        self.announced_rank_head = layer_init(nn.Linear(128, 13), std=0.01) # 0-12 (Ace-King)
        self.quantity_head = layer_init(nn.Linear(128, 6), std=0.01)      # 0-5 (1-6 cards)
        self.card_selection_head = layer_init(nn.Linear(128, 14), std=0.01) # 0-13 (Joker, Ace-King)

    def get_value(self, state):
        """
        Forward pass for the Critic network.

        Args:
            state: The input state tensor.

        Returns:
            The estimated value V(s) of the state.
        """
        return self.critic(state)

    def get_action_and_value(self, state, valid_actions=None, action=None):
        """
        Processes the state to generate action distributions and value estimates.

        This method operates in two modes:
        1. Sample Mode (action=None): Used during rollouts to sample new actions from the policy.
        2. Evaluate Mode (action!=None): Used during training to calculate log_probs and entropy 
           for actions that were previously taken.

        Args:
            state: The input state tensor.
            valid_actions: (Optional) Dictionary of valid actions for masking (Sample Mode).
            action: (Optional) Dictionary containing action components for evaluation (Training Mode).

        Returns:
            Tuple containing (full_action_tuple, total_log_prob, entropy, value, action_indices).
        """
        device = state.device
        batch_size = state.shape[0]

        # --- Forward Pass: Get Logits and Value ---
        actor_features = self.actor_shared(state)
        critic_value = self.critic(state) 

        type_logits = self.action_type_head(actor_features)
        rank_logits = self.announced_rank_head(actor_features)
        quantity_logits = self.quantity_head(actor_features)
        card_logits = self.card_selection_head(actor_features)

        # --- Initialize Accumulators ---
        total_log_probs = torch.zeros(batch_size, device=device)
        total_entropy = torch.zeros(batch_size, device=device)

        # --- 1. Decision: Action Type (Doubt, Pass, Play) ---
        if valid_actions is not None: # Sample Mode
            type_mask = torch.full_like(type_logits, -torch.inf, device=device)
            type_mask[0, valid_actions["types"]] = 0.0
            type_dist = Categorical(logits=type_logits + type_mask)
        else: # Evaluate Mode
            type_dist = Categorical(logits=type_logits)

        if action is None:
            action_type_idx = type_dist.sample()
        else:
            action_type_idx = action["type"]

        total_log_probs += type_dist.log_prob(action_type_idx)
        total_entropy += type_dist.entropy()

        # --- 2. Decision: Rank to Announce (Ace-King) ---
        if valid_actions is not None: # Sample Mode
            rank_mask = torch.full_like(rank_logits, -torch.inf, device=device)
            if valid_actions["is_starter"]:
                valid_rank_indices = [self.rank_to_index_13[r] for r in valid_actions["ranks"]]
                rank_mask[0, valid_rank_indices] = 0.0
            else:
                current_rank = valid_actions["current_rank"]
                if current_rank != "Open":
                    # Must follow the current rank
                    valid_rank_indices = [self.rank_to_index_13[current_rank]]
                    rank_mask[0, valid_rank_indices] = 0.0
                else:
                    # Open play but not starter (e.g., after a pass). Can play any rank.
                    rank_mask[0, :] = 0.0 
            rank_dist = Categorical(logits=rank_logits + rank_mask)
        else: # Evaluate Mode
            rank_dist = Categorical(logits=rank_logits)
        
        if action is None:
            action_rank_idx = rank_dist.sample() # 0-12
        else:
            action_rank_idx = action["rank_announced"]
        
        total_log_probs += rank_dist.log_prob(action_rank_idx)
        total_entropy += rank_dist.entropy()

        # --- 3. Decision: Quantity to Announce (1-6) ---
        if valid_actions is not None: # Sample Mode
            quantity_mask = torch.full_like(quantity_logits, -torch.inf, device=device)
            quantity_mask[0, valid_actions["quantities"]] = 0.0 # 0-5
            quantity_dist = Categorical(logits=quantity_logits + quantity_mask)
        else: # Evaluate Mode
            quantity_dist = Categorical(logits=quantity_logits)

        if action is None:
            action_quantity_idx = quantity_dist.sample() # 0-5
        else:
            action_quantity_idx = action["quantity"]

        total_log_probs += quantity_dist.log_prob(action_quantity_idx)
        total_entropy += quantity_dist.entropy()

        # --- 4. Decision: Card Selection (Joker, Ace-King) ---
        # This step requires sequential sampling or joint probability calculation.
        card_log_prob_sum = torch.zeros(batch_size, device=device)
        sampled_card_ranks = [] # List of rank tensors

        if action is not None:
            # --- Evaluate Mode ---
            play_mask = (action["type"] == 2) # Only calculate for 'Play' actions
            if play_mask.sum() > 0:
                # Recalculate log_probs using the stored hand state and played cards sequence
                card_log_prob_sum[play_mask] = self.recalculate_card_logprobs(
                    card_logits[play_mask],
                    state[play_mask], 
                    action["quantity"][play_mask],
                    action["card_ranks_played"][play_mask] 
                )
        
        elif action_type_idx.item() == 2: 
            # --- Sample Mode ---
            quantity_to_play = action_quantity_idx.item() + 1
            
            # Extract hand vector from state (batch_size=1 for sampling)
            hand_vector = state[0, :14].cpu().numpy() 
            
            sampled_card_ranks, card_log_prob_sum = self.sample_cards(
                card_logits, 
                quantity_to_play, 
                hand_vector
            )
            total_log_probs += card_log_prob_sum 

        # --- Build Returns ---
        if action is None:
            # --- Sample Mode Return ---
            full_action_tuple = self._build_full_action(
                action_type_idx.item(),
                action_rank_idx.item(),
                action_quantity_idx.item(),
                sampled_card_ranks, 
                valid_actions["player_hand"] 
            )
            
            action_indices = {
                "type": action_type_idx,
                "rank_announced": action_rank_idx,
                "quantity": action_quantity_idx,
                "card_ranks_played": sampled_card_ranks 
            }

            return full_action_tuple, total_log_probs, total_entropy, critic_value, action_indices
        
        else: 
            # --- Evaluate Mode Return ---
            total_log_probs += card_log_prob_sum
            # Note: Calculating entropy for sequential selection is computationally expensive
            # and is omitted here to prioritize training speed.
            
            return None, total_log_probs, total_entropy, critic_value
    
    def sample_cards(self, card_logits, quantity_to_play, hand_vector):
        """
        Samples 'N' card ranks from the hand sequentially, respecting available hand counts.

        Args:
            card_logits: The output logits from the card selection head.
            quantity_to_play: The number of cards to sample.
            hand_vector: The frequency vector of the player's hand.

        Returns:
            sampled_ranks_tensors: List of selected rank indices.
            log_prob_sum: The sum of log probabilities for the sampled sequence.
        """
        device = card_logits.device
        batch_size = card_logits.shape[0]
        
        # Create a mutable copy of the hand counts to track usage during sampling
        hand_rank_counts = hand_vector.copy()
            
        sampled_ranks_tensors = []
        
        # Initialize log_prob_sum with correct batch shape
        log_prob_sum = torch.zeros(batch_size, device=device) 
        
        current_card_logits = card_logits.clone()

        for _ in range(quantity_to_play):
            # Mask unavailable cards (count == 0)
            card_mask = torch.full_like(current_card_logits, -torch.inf, device=device)
            
            available_indices = np.where(hand_rank_counts > 0)[0]
            card_mask[0, available_indices] = 0.0
            
            # Sample a rank from the valid distribution
            card_dist = Categorical(logits=current_card_logits + card_mask)
            selected_rank_idx_tensor = card_dist.sample() 
            
            # Accumulate log_prob
            log_prob_sum += card_dist.log_prob(selected_rank_idx_tensor) 
            selected_rank_idx_int = selected_rank_idx_tensor.item()
            sampled_ranks_tensors.append(selected_rank_idx_tensor)

            # Decrement virtual hand count for the selected card
            hand_rank_counts[selected_rank_idx_int] -= 1
        
        return sampled_ranks_tensors, log_prob_sum

    def recalculate_card_logprobs(self, card_logits, state, quantities, card_ranks_played):
        """
        Recalculates the log probability of a stored sequence of card plays.

        This is critical for the PPO update step. It reconstructs the probability
        distribution at each step of the card selection process using the stored state,
        correctly handling padding and batch processing.

        Args:
            card_logits: Logits from the network.
            state: The state tensor containing the hand vector.
            quantities: Tensor of quantities played.
            card_ranks_played: Tensor of specific ranks played [Batch, Max_Cards].

        Returns:
            total_card_log_probs: Tensor of summed log probs for the batch.
        """
        device = card_logits.device
        batch_size = card_logits.shape[0]
        
        # Extract hand vector from state (first 14 elements)
        hand_rank_counts = state[:, :14].clone() 
        
        total_card_log_probs = torch.zeros(batch_size, device=device)
        
        # Iterate through each possible card slot in the sequence (up to max_cards_played)
        for k in range(self.max_cards_played):
            
            current_rank_idx = card_ranks_played[:, k] # [B], may contain -1 (padding)
            
            # Create mask for active samples (those that actually played a card in this slot)
            play_mask = (current_rank_idx != -1)
            
            # If no sample in the batch played a card in this slot, we can stop.
            if play_mask.sum() == 0:
                break 
            
            # Filter data to process only active samples
            valid_logits = card_logits[play_mask]         
            valid_hand_counts = hand_rank_counts[play_mask] 
            valid_rank_indices = current_rank_idx[play_mask] 
            
            # Mask unavailable cards based on current hand counts
            # log(0.0) = -inf, log(1.0) = 0.0
            card_mask = torch.log((valid_hand_counts > 0).float()) 
            
            # Get distribution for valid samples
            card_dist = Categorical(logits=valid_logits + card_mask) 
            
            # Calculate log_prob for the played ranks
            log_prob = card_dist.log_prob(valid_rank_indices)
            
            # Add log_prob to the total, respecting the mask
            total_card_log_probs[play_mask] += log_prob
            
            # Update hand counts for valid samples (decrement count of played card)
            valid_hand_counts.scatter_add_(
                1, 
                valid_rank_indices.unsqueeze(1), 
                torch.full_like(valid_rank_indices.unsqueeze(1), -1, dtype=torch.float) 
            )
            
            # Update the main hand counts tensor for the next iteration
            hand_rank_counts[play_mask] = valid_hand_counts
            hand_rank_counts.clamp_(min=0) 

        return total_card_log_probs

    def _build_full_action(self, type_idx, rank_idx, quantity_idx, card_rank_tensors, player_hand):
        """
        Converts raw network output indices into a structured, game-compatible action tuple.
        """
        
        action_type = type_idx
        announced_rank = self.card_values[rank_idx + 1] # +1 to skip 'Joker' in rank announcement

        cards_to_play = []
        if action_type == 2: # 'Play'
            quantity_to_play = quantity_idx + 1 # 1-6
            
            # Convert index tensors back to card value strings
            sampled_rank_strings = [self.card_values[tensor.item()] for tensor in card_rank_tensors]
            
            hand_copy = list(player_hand)
            
            # Find matching Card objects in the player's hand
            for rank_str in sampled_rank_strings:
                for card in hand_copy:
                    if card.value == rank_str:
                        cards_to_play.append(card)
                        hand_copy.remove(card)
                        break
            
            # Safety fallback (should not trigger if logic is correct)
            if len(cards_to_play) != quantity_to_play:
                if len(player_hand) >= quantity_to_play:
                     cards_to_play = random.sample(player_hand, k=quantity_to_play)
                else:
                     cards_to_play = list(player_hand) 
        
        return (action_type, cards_to_play, announced_rank)