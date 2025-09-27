"""
This module defines the RLAgent class, the core of the reinforcement learning player.

It encapsulates the agent's brain (a multi-head Q-Network), its memory of past
experiences (ReplayMemory), and the logic for making decisions and learning from them.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from .q_network import Q_Network
from .replay_memory import ReplayMemory, Transition

class RLAgent:
    """
    The main agent class that learns and plays the game of Cheat.

    This agent uses a Deep Q-Network (DQN) with a multi-head architecture to handle
    the game's complex action space. It learns through experience replay and an
    epsilon-greedy policy for exploration.
    """
    def __init__(self, input_size: int, epsilon: float = 1.0):
        """
        Initializes the agent and its components.

        Sets up the epsilon-greedy parameters, the neural networks (policy and target),
        the optimizer, and the replay memory.

        Args:
            input_size: The dimension of the state vector provided by the environment.
            epsilon: The initial value for epsilon in the epsilon-greedy strategy.
        """
        # --- Epsilon-Greedy Parameters ---
        self.epsilon = epsilon
        self.epsilon_decay = 0.999995
        self.epsilon_min = 0.01

        # Mappings to translate between card ranks and network output indices.
        self.card_values = ["Joker", "Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King"]
        self.rank_to_index = {value: i for i, value in enumerate(self.card_values)}

        # --- DQN Architecture ---
        # The policy_net is trained continuously, while the target_net is updated
        # periodically to provide stable TD targets.
        self.policy_net = Q_Network(input_size)
        self.target_net = Q_Network(input_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only for inference.

        # --- Learning Components ---
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001)
        self.memory = ReplayMemory(10000)

    def choose_action(self, state: np.ndarray, valid_actions: dict) -> tuple:
        """
        Selects an action based on the current state using an epsilon-greedy policy.

        With probability epsilon, it explores by choosing a random valid action.
        Otherwise, it exploits its current knowledge by using the policy network
        to make a hierarchical decision: first choosing an action type, then
        the details of that action (rank, quantity, cards).

        Args:
            state: The current state vector from the environment.
            valid_actions: A dictionary detailing all legal moves.

        Returns:
            A tuple representing the structured action to be taken.
        """
        player_hand = valid_actions["player_hand"]

        # --- 1. Exploration vs. Exploitation ---
        if random.random() < self.epsilon:
            return self._choose_random_valid_action(valid_actions, player_hand)

        # --- 2. Exploitation: Use the network to decide the best action ---
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_tensor)

            # --- 3. Hierarchical Decision Process ---

            # First, decide the action type (Doubt, Pass, Play) by masking illegal moves.
            action_type_q = q_values["action_type"][0]
            masked_action_q = torch.full_like(action_type_q, -torch.inf)
            masked_action_q[valid_actions["types"]] = action_type_q[valid_actions["types"]]
            action_type = torch.argmax(masked_action_q).item()

            cards_to_play = []
            announced_rank = None

            # If the network chose to 'Play', decide the specifics of the play.
            if action_type == 2:
                # Decide which rank to announce.
                if valid_actions["is_starter"]:
                    # If starting the round, the agent can choose any rank.
                    rank_q = q_values["rank_claim"][0]
                    announced_rank_idx = torch.argmax(rank_q).item()
                    announced_rank = self.card_values[announced_rank_idx + 1]  # +1 to skip Joker
                else:
                    # Otherwise, it must follow the current round's rank.
                    announced_rank = valid_actions["current_rank"]

                # Decide how many cards to play.
                quantity_q = q_values["quantity_claim"][0]
                masked_quantity_q = torch.full_like(quantity_q, -torch.inf)
                masked_quantity_q[valid_actions["quantities"]] = quantity_q[valid_actions["quantities"]]
                quantity_to_play = torch.argmax(masked_quantity_q).item() + 1

                # Decide which specific cards from the hand to play.
                cards_to_play = self._select_cards_with_rank_strategy(
                    q_values["rank_selection"][0],
                    player_hand,
                    quantity_to_play
                )

            return (action_type, cards_to_play, announced_rank)

    def _select_cards_with_rank_strategy(self, rank_q_values, player_hand, quantity):
        """
        Selects specific cards from the hand to play based on rank Q-values.

        The strategy is to sort the ranks present in the hand by their Q-values
        (from highest to lowest) and then greedily pick cards from these preferred
        ranks until the desired quantity is met. This allows the agent to discard
        cards it has learned are less valuable to hold.
        """
        hand_ranks = {card.value for card in player_hand}

        # Create a list of (Q-value, rank) tuples for sorting.
        rank_preferences = []
        for rank in hand_ranks:
            rank_index = self.rank_to_index[rank]
            rank_preferences.append((rank_q_values[rank_index].item(), rank))

        # Sort ranks by their Q-value in descending order (best ranks first).
        rank_preferences.sort(key=lambda x: x[0], reverse=True)

        # Build the hand to play by iterating through the sorted rank preferences.
        chosen_cards = []
        for _, rank in rank_preferences:
            cards_of_rank = [card for card in player_hand if card.value == rank]
            needed = quantity - len(chosen_cards)
            chosen_cards.extend(cards_of_rank[:needed])

            if len(chosen_cards) == quantity:
                break

        return chosen_cards

    def _choose_random_valid_action(self, valid_actions: dict, player_hand: list) -> tuple:
        """
        Selects a random but valid action, used for exploration.
        """
        action_type = random.choice(valid_actions["types"])
        cards_to_play = []
        announced_rank = None

        if action_type == 2:  # Play
            if valid_actions["is_starter"]:
                announced_rank = random.choice(valid_actions["ranks"])
            else:
                announced_rank = valid_actions["current_rank"]

            quantity = random.choice(valid_actions["quantities"]) + 1

            if len(player_hand) >= quantity:
                cards_to_play = random.sample(player_hand, k=quantity)

        return (action_type, cards_to_play, announced_rank)

    def learn(self, batch_size: int):
        """
        Samples a batch of experiences from memory and updates the network weights.

        This function implements the core DQN learning algorithm, including the
        calculation of the TD target and the hierarchical loss for the multi-head
        network.
        """
        if len(self.memory) < batch_size:
            return

        # 1. Sample a batch of transitions from the replay memory.
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        # 2. Convert batch data into PyTorch tensors.
        state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32)
        next_state_batch = torch.tensor(np.array(batch.next_state), dtype=torch.float32)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32)
        done_batch = torch.tensor(batch.done, dtype=torch.bool)

        # 3. Unpack the complex action tuples from the batch.
        action_types = torch.tensor([a[0] for a in batch.action], dtype=torch.int64).view(-1, 1)

        # For play-specific actions, use placeholders for non-'play' actions.
        announced_ranks_idx = []
        quantities_idx = []
        played_ranks_idx = []
        for action_tuple in batch.action:
            action_type, cards, rank = action_tuple
            if action_type == 2:  # Is a 'play' action
                # Convert rank name to a 0-12 index for the network.
                announced_ranks_idx.append(self.rank_to_index.get(rank, 1) - 1)
                # Convert quantity 1-6 to a 0-5 index for the network.
                quantities_idx.append(len(cards) - 1 if cards else 0)

                unique_ranks_idx = {self.rank_to_index[card.value] for card in cards}
                played_ranks_idx.append(list(unique_ranks_idx))

            else:
                announced_ranks_idx.append(-1)  # Placeholder
                quantities_idx.append(-1)     # Placeholder
                played_ranks_idx.append([])     # Placeholder

        announced_ranks_idx = torch.tensor(announced_ranks_idx, dtype=torch.int64).view(-1, 1)
        quantities_idx = torch.tensor(quantities_idx, dtype=torch.int64).view(-1, 1)

        # 4. Calculate the TD Target using the target network.
        with torch.no_grad():
            next_q_values_dict = self.target_net(next_state_batch)
            max_next_q_values = next_q_values_dict["action_type"].max(1)[0]
            # The TD target is the immediate reward + the discounted value of the best future action.
            target_q_values = reward_batch + (0.99 * max_next_q_values * ~done_batch)  # gamma=0.99

        # --- 5. Calculate the Hierarchical Loss ---
        q_values_dict = self.policy_net(state_batch)

        # The loss is calculated hierarchically. The action_type loss is always
        # computed. Losses for other heads are only computed for 'play' actions.
        predicted_q_for_action_type = q_values_dict["action_type"].gather(1, action_types)
        total_loss = F.smooth_l1_loss(predicted_q_for_action_type, target_q_values.unsqueeze(1))

        # Create a boolean mask to identify which transitions were 'play' actions.
        play_mask = (action_types == 2).squeeze()

        if play_mask.sum() > 0:
            # Loss for Rank Claim
            predicted_q_rank = q_values_dict["rank_claim"][play_mask].gather(1, announced_ranks_idx[play_mask])
            total_loss += F.smooth_l1_loss(predicted_q_rank, target_q_values[play_mask].unsqueeze(1))

            # Loss for Quantity Claim
            predicted_q_qty = q_values_dict["quantity_claim"][play_mask].gather(1, quantities_idx[play_mask])
            total_loss += F.smooth_l1_loss(predicted_q_qty, target_q_values[play_mask].unsqueeze(1))

            # Loss for Rank Selection
            # This loss is calculated only for the Q-values of the ranks that were actually played
            predicted_q_rank_selection = q_values_dict["rank_selection"][play_mask]
            target_q_values_for_plays = target_q_values[play_mask].unsqueeze(1)

            selection_loss = torch.tensor(0.0)
            num_valid_selections = 0

            # Filter the list of played ranks to correspond to the 'play' actions in the batch.
            played_ranks_for_this_batch = [ranks for i, ranks in enumerate(played_ranks_idx) if play_mask[i]]

            for i in range(predicted_q_rank_selection.size(0)):
                rank_indices_for_this_action = played_ranks_for_this_batch[i]

                if rank_indices_for_this_action:
                    predicted_q_for_played_ranks = predicted_q_rank_selection[i, rank_indices_for_this_action]
                    target = target_q_values_for_plays[i].expand_as(predicted_q_for_played_ranks)
                    selection_loss += F.smooth_l1_loss(predicted_q_for_played_ranks, target)
                    num_valid_selections += 1

            if num_valid_selections > 0:
                total_loss += (selection_loss / num_valid_selections)

        # --- 6. Perform backpropagation ---
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Decay epsilon to reduce exploration over time.
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay