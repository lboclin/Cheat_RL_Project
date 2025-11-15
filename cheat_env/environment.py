"""
This module defines the reinforcement learning environment for the card game "Cheat".

It manages the game state, rules, player actions, and rewards, following a
structure similar to standard RL environments like Gymnasium.
"""
from .card import Card, Suit
from .deck import Deck
from .player import Player
import numpy as np


class CheatEnviroment:
    """
    Manages the game logic, state transitions, and reward signals for the Cheat game.

    Attributes:
        players (list[Player]): A list of Player objects participating in the game.
        rl_agent (Player): A reference to the agent being trained (always players[0]).
        deck (Deck): The deck of cards used for the game.
        current_player_index (int): The index of the player whose turn it is.
        current_rank_to_play (str): The rank that must be played in the current round.
        ...and other state-tracking attributes.
    """

    def __init__(self, players_names: list, max_episode_steps=250):
        """
        Initializes the game environment.

        Args:
            players_names (list): A list of strings with the names of the players.
            max_episode_steps (int): The maximum number of turns before an episode
                                     is truncated.
        """
        if len(players_names) < 2:
            raise ValueError("The game need at least 2 players.")

        self.players = [Player(name) for name in players_names]
        self.rl_agent = self.players[0]
        self.deck = None
        self.current_player_index = 0
        self.current_rank_to_play = None
        self.last_player_who_played_index = None
        self.round_discard_pile = []
        self.pass_counter = 0
        self.winner = None
        self.starter_player_index = None
        self.max_episode_steps = max_episode_steps
        self.turn_count = 1

    def _get_state(self):
        """
        Constructs a numeric state vector representing the current game view
        for the active player.

        The state vector is composed of:
        1.  Player's hand (card counts per rank).
        2.  Number of cards in each opponent's hand.
        3.  The current rank to be played (one-hot encoded).
        4.  Size of the discard pile.
        5.  A flag indicating if a round has just started.

        Returns:
            A flat NumPy array representing the game state.
        """
        current_player = self.players[self.current_player_index]
        card_values = ["Joker", "Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King"]
        value_to_index = {value: i for i, value in enumerate(card_values)}

        # 1. Current player's hand vector.
        hand_vector = np.zeros(14)
        for card in current_player.hand:
            if card.value in value_to_index:
                hand_vector[value_to_index[card.value]] += 1
        
        # 2. Number of cards for each opponent.
        opponent_card_counts = [len(p.hand) for p in self.players if p is not current_player]

        # 3. Current rank to play (one-hot encoded).
        rank_vector = np.zeros(14)
        if self.current_rank_to_play in value_to_index:
            rank_vector[value_to_index[self.current_rank_to_play]] = 1

        # 4. Other game information.
        discard_pile_size = [len(self.round_discard_pile)]
        if self.starter_player_index == None :
            is_starting_play = [0.0]
        else:
            is_starting_play = [1.0]

        state_vector = np.concatenate([
            hand_vector,
            np.array(opponent_card_counts),
            rank_vector,
            np.array(discard_pile_size),
            np.array(is_starting_play)
        ]).astype(np.float32)
        
        return state_vector

    def get_valid_actions(self):
        """
        Determines the set of legal actions for the current player.

        This is used by the agent for action masking, ensuring it only chooses
        from possible moves.

        Returns:
            A dictionary containing lists of valid action types, ranks, and
            quantities, along with other contextual information for the agent.
        """
        valid_actions = {
            "types": [],
            "ranks": [],
            "quantities": [],
            "is_starter": False,
            "current_rank": self.current_rank_to_play,
            "player_hand": self.players[self.current_player_index].hand
        }
        
        current_player = self.players[self.current_player_index]
        is_starter = (self.starter_player_index == self.current_player_index)
        valid_actions["is_starter"] = is_starter

        if is_starter:
            valid_actions["types"].append(2) # Must play
        else:
            valid_actions["types"] = [0, 1, 2]

        if is_starter:
            valid_actions["ranks"] = ["Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King"]
        
        max_qty = min(len(current_player.hand), 6)
        if max_qty > 0:
            valid_actions["quantities"] = list(range(max_qty)) # 0-5, corresponds to 1-6 cards

        return valid_actions

    def _deal_cards(self):
        """Initializes, shuffles, and deals a new deck to all players."""
        self.deck = Deck()
        self.deck.shuffle()

        for player in self.players:
            player.hand = []

        card_to_deal = self.deck.get_card()
        player_index = 0
        while card_to_deal is not None:
            self.players[player_index % len(self.players)].receive_card(card_to_deal)
            card_to_deal = self.deck.get_card()
            player_index += 1

    def check_game_over(self):
        """
        Checks if any player has an empty hand and sets them as the winner.

        Returns:
            True if the game has a winner, False otherwise.
        """
        for player in self.players:
            if len(player.hand) == 0:
                self.winner = player 
                return True
        return False

    def reset(self):
        """
        Resets the environment to the beginning of a new episode.

        Deals new cards, resets all game state variables, and starts a new round.

        Returns:
            The initial state vector of the new game.
        """
        self._deal_cards()
        self.current_player_index = 0
        self.current_rank_to_play = None
        self.starter_player_index = None
        self.last_player_who_played_index = None
        self.last_number_of_cards_played = None
        self.round_discard_pile = []
        self.pass_counter = 0
        self.turn_count = 1

        self._start_new_round(self.current_player_index)

        return self._get_state()

    def step(self, action: tuple):
        """
        Executes one time step in the environment based on the agent's action.
        """
        action_type, cards_to_play, announced_rank = action
        acting_player_index = self.current_player_index
        
        round_ended_by_challenge = False
        
        # --- REWARD SHAPING ---
        reward = 0.0
        terminated = False

        if action_type == 0:
            reward = self._resolve_challenge(acting_player_index, self.last_player_who_played_index)
            round_ended_by_challenge = True
        elif action_type == 1:
            reward = self._handle_pass(acting_player_index)
        else: # action_type == 2
            reward = self._play_cards(acting_player_index, cards_to_play, announced_rank)
        
        self.turn_count += 1
        terminated = self.check_game_over()

        truncated = False
        if not terminated and self.turn_count >= self.max_episode_steps:
            truncated = True
            
        if terminated:
            if self.winner == self.rl_agent:
                reward = 1.0
            else:
                reward = -1.0
            
        if not terminated:
            if self.pass_counter >= (len(self.players)-1):
                self._start_new_round(self.last_player_who_played_index)
            elif round_ended_by_challenge:
                pass
            else:
                if self.starter_player_index == None :
                    self.current_player_index = (self.current_player_index+1) % len(self.players)

        state = self._get_state()
        info = {}
        return state, reward, terminated, truncated, info

    def _start_new_round (self, starting_player_index: int):
        """Resets the round state and sets the new starting player."""
        self.round_discard_pile = []
        self.last_number_of_cards_played = None
        self.last_player_who_played_index = None
        self.current_rank_to_play = "Open"

        self.starter_player_index = starting_player_index
        self.current_player_index = starting_player_index

    def _resolve_challenge (self, current_player_index, last_player_who_played_index):
        """
        Handles the logic when a player doubts the previous play and returns an
        intermediate reward based on the outcome for the RL agent.
        """
        current_player = self.players[current_player_index]
        last_player_who_played = self.players[last_player_who_played_index]
        self.pass_counter = 0
        
        # --- REWARD SHAPING ---
        reward = 0.0

        got_the_cheat = False
        for i in range(self.last_number_of_cards_played):
            current_card_to_analise = self.round_discard_pile[len(self.round_discard_pile)-i-1].value
            if current_card_to_analise != self.current_rank_to_play and current_card_to_analise != "Joker":
                got_the_cheat = True
                break

        if got_the_cheat:
            for card in self.round_discard_pile:
                last_player_who_played.receive_card(card)
            
            # --- REWARD SHAPING ---
            if current_player == self.rl_agent: 
                reward = 0.1
            elif last_player_who_played == self.rl_agent:
                reward = -0.1

            self._start_new_round (current_player_index)
        else:
            for card in self.round_discard_pile:
                current_player.receive_card(card)

            # --- REWARD SHAPING ---
            if current_player == self.rl_agent: 
                reward = -0.1
            elif last_player_who_played == self.rl_agent: 
                reward = 0.1

            self._start_new_round (last_player_who_played_index)
        
        return reward

    def _handle_pass(self, current_player_index):
        """Processes a 'pass' action for the current player."""
        current_player = self.players[current_player_index]
        self.pass_counter += 1

        if current_player == self.rl_agent:
            return -0.01
        return 0.0

    def _play_cards(self, current_player_index, cards_to_play, announced_rank):
        """
        Processes a 'play' action and returns a reward if the agent wins/loses on a lie.
        """
        current_player = self.players[current_player_index]
        self.current_rank_to_play = announced_rank

        for card in cards_to_play:
            current_player.hand.remove(card)
            self.round_discard_pile.append(card)

        self.last_player_who_played_index = current_player_index
        self.last_number_of_cards_played = len(cards_to_play)
        self.pass_counter = 0
        self.starter_player_index = None

        reward = 0.0


        if (len(current_player.hand) == 0) :
            reward = self._last_play_judge(current_player_index, cards_to_play)

        
        return reward

    def _last_play_judge(self, current_player_index, cards_to_play) :
        """
        Checks if a winning move was a lie and returns a penalty if so.
        """
        current_player = self.players[current_player_index]
        was_a_lie = False
        for card in cards_to_play :
            if card.value != self.current_rank_to_play and card.value != "Joker":
                was_a_lie = True
                break

        if was_a_lie :
            for card_from_pile in self.round_discard_pile:
                current_player.receive_card(card_from_pile)
            
            next_player = (self.current_player_index+1) % len(self.players)
            self._start_new_round(next_player)

            # --- REWARD SHAPING ---
            if current_player == self.rl_agent:
                return -0.1
        
        return 0.0