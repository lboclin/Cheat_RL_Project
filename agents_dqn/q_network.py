import torch
import torch.nn as nn

class Q_Network(nn.Module):
    """
    Defines the multi-head Q-Network architecture for the Cheat RL agent.
    
    This network processes the game state through a shared body and then splits into
    four separate heads to estimate Q-values for different components of a complex action:
    1. Action Type: The primary strategic choice (e.g., Challenge, Pass, Play).
    2. Rank Claim: The card rank to announce if playing cards.
    3. Quantity Claim: The number of cards to announce.
    4. Rank Selection: The desirability of using each rank from the hand to form the play.
    """
    def __init__(self, input_size: int):
        """
        Initializes the network layers.

        Args:
            input_size (int): The length of the state vector from the environment.
        """
        super(Q_Network, self).__init__()
        
        # A shared feature extractor for all decision heads.
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # --- Decision Heads ---
        self.action_type_head = nn.Linear(128, 3)     # Outputs Q-values for {Challenge, Pass, Play}
        self.rank_claim_head = nn.Linear(128, 13)    # Outputs Q-values for ranks to announce {Ace..King}
        self.quantity_claim_head = nn.Linear(128, 6) # Outputs Q-values for quantities {1..6}
        self.rank_selection_head = nn.Linear(128, 14) # Outputs Q-values for ranks to use {Joker, Ace..King}

    def forward(self, state: torch.Tensor) -> dict:
        """
        Performs the forward pass through the network.

        Args:
            state (torch.Tensor): The input state tensor.
            
        Returns:
            dict: A dictionary of raw Q-value tensors (logits) from each head.
        """
        features = self.shared_layers(state)
        
        q_values = {
            "action_type": self.action_type_head(features),
            "rank_claim": self.rank_claim_head(features),
            "quantity_claim": self.quantity_claim_head(features),
            "rank_selection": self.rank_selection_head(features)
        }
        
        return q_values