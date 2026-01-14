"""
This module defines the reinforcement learning environment for the card game "Cheat".

It manages the game state, rules, player actions, and rewards, following a
structure similar to standard RL environments like Gymnasium. The environment
supports customizable reward shaping and optional game logging for debugging
and analysis.
"""
from .card import Card, Suit
from .deck import Deck
from .player import Player
import numpy as np
import random 

from bots.strategies import (
    bot_strategy_80_20, 
    bot_strategy_one_third, 
    bot_strategy_100_0, 
    bot_strategy_60_40, 
    bot_strategy_challenger
)


class CheatEnviroment:
    """
    Manages the game logic, state transitions, and reward signals for the Cheat game.

    This environment encapsulates the entire game state, including players' hands,
    the discard pile, and the current game phase. It provides an interface for
    an RL agent to interact with the game and receive feedback.
    """

    def __init__(self, players_names: list, max_episode_steps=250, visualize=False, reward_shaping=False):
        """
        Initializes the Cheat environment.

        Args:
            players_names: A list of names for the players participating in the game.
            max_episode_steps: The maximum number of turns allowed per episode before truncation.
            visualize: If True, enables printing of detailed game logs to the console.
            reward_shaping: If True, enables intermediate rewards to guide the agent's learning.
        """
        if len(players_names) < 2:
            raise ValueError("The game needs at least 2 players.")
            
        self.VISUALIZE_GAMES = visualize
        self.reward_shaping = reward_shaping
        
        self.card_values = ["Joker", "Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King"]
        self.value_to_index = {value: i for i, value in enumerate(self.card_values)}

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
        
        # --- Reward Configuration ---
        
        # Final Rewards (Always active)
        self.REWARD_WIN = 1.0
        self.REWARD_LOSE = -1.0
        
        # Shaping Rewards (Initialized to 0.0)
        self.SHAPING_AGENT_CHALLENGE_SUCCESS = 0.0 
        self.SHAPING_AGENT_CHALLENGE_FAIL = 0.0    
        self.SHAPING_OPPONENT_CAUGHT_AGENT_LIE = 0.0 
        self.SHAPING_OPPONENT_FAILED_CHALLENGE = 0.0 
        self.SHAPING_AGENT_PASS = 0.0               
        self.SHAPING_AGENT_FINAL_LIE_PENALTY = 0.0  
        
        # Apply values if reward shaping is enabled
        if self.reward_shaping:
            self.SHAPING_AGENT_CHALLENGE_SUCCESS = 0.0
            self.SHAPING_AGENT_CHALLENGE_FAIL = -0.5
            self.SHAPING_OPPONENT_CAUGHT_AGENT_LIE = 0.0
            self.SHAPING_OPPONENT_FAILED_CHALLENGE = 0.0
            self.SHAPING_AGENT_PASS = -0.05
            self.SHAPING_AGENT_FINAL_LIE_PENALTY = -0.2
            

        # --- Bot Configuration ---
        self.bot_pool_dict = {
            bot_strategy_80_20: 'Bot 80/20',
            bot_strategy_one_third: 'Bot 1/3',
            bot_strategy_100_0: 'Bot Honest',
            bot_strategy_60_40: 'Bot 60/40'
            #bot_strategy_challenger: 'Bot Challenger'
        }
        self.bot_pool_funcs = list(self.bot_pool_dict.keys())
        
        self.bot_strategies = {} 
        self.bot_strategy_names = {}
        
        for player in self.players[1:]:
            self.bot_strategies[player.name] = random.choice(self.bot_pool_funcs)
            self.bot_strategy_names[player.name] = "None (reset)"



    def _get_state(self):
        """
        Constructs the state vector from the perspective of the RL agent (Player 0).

        Returns:
            A numpy array representing the current game state, including hand composition,
            opponent card counts, the current rank to play, pile size, and game phase info.
        """
        rl_player = self.rl_agent 
        
        # 1. RL Agent's Hand Vector (Frequency count of each rank)
        hand_vector = np.zeros(14)
        for card in rl_player.hand:
            if card.value in self.value_to_index:
                hand_vector[self.value_to_index[card.value]] += 1
        
        # 2. Opponents' Card Counts
        opponent_card_counts = [len(p.hand) for p in self.players if p is not rl_player]

        # 3. Current Rank Vector (One-hot encoding)
        rank_vector = np.zeros(14)
        if self.current_rank_to_play in self.value_to_index:
            rank_vector[self.value_to_index[self.current_rank_to_play]] = 1

        # 4. Discard Pile Size
        discard_pile_size = [len(self.round_discard_pile)]
        
        # 5. Starter Flag (1.0 if it is the agent's turn to start a new round)
        is_starting_play = [0.0]
        if self.starter_player_index == self.current_player_index and self.current_player_index == 0:
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
        Determines the set of legal actions available to the RL agent for the current turn.

        This function should only be called when it is the RL agent's turn.

        Returns:
            A dictionary containing lists of valid action types, ranks, quantities,
            and other context needed for decision masking.
        """
        if self.current_player_index != 0:
            raise Exception("get_valid_actions() called outside of RL Agent's turn.")

        valid_actions = {
            "types": [],
            "ranks": [],
            "quantities": [],
            "is_starter": False,
            "current_rank": self.current_rank_to_play,
            "player_hand": self.rl_agent.hand
        }
        
        is_starter = (self.starter_player_index == self.current_player_index)
        valid_actions["is_starter"] = is_starter

        if is_starter:
            valid_actions["types"].append(2) # Must Play
        else:
            valid_actions["types"] = [0, 1, 2] # Doubt, Pass, Play

        if is_starter:
            valid_actions["ranks"] = ["Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King"]
        
        max_qty = min(len(self.rl_agent.hand), 6)
        if max_qty > 0:
            valid_actions["quantities"] = list(range(max_qty)) 
        elif is_starter:
             valid_actions["quantities"] = []

        return valid_actions

    def _deal_cards(self):
        """
        Resets the deck, shuffles, and deals cards to all players.
        """
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
        Checks if any player has emptied their hand, signaling the end of the game.

        Returns:
            True if a winner is found, False otherwise.
        """
        for player in self.players:
            if len(player.hand) == 0:
                self.winner = player 
                return True
        return False

    def reset(self):
        """
        Resets the environment to start a new episode.

        Deals new cards, resets game state variables, and randomly assigns new
        strategies to the bot opponents.

        Returns:
            The initial state vector for the new episode.
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

        # Assign new random strategies to bots for each episode
        for player in self.players[1:]:
            strategy_func = random.choice(self.bot_pool_funcs)
            self.bot_strategies[player.name] = strategy_func
            self.bot_strategy_names[player.name] = self.bot_pool_dict[strategy_func]

        self._start_new_round(self.current_player_index)
        
        if self.VISUALIZE_GAMES:
            print(f"\n--- New Episode (Reward Shaping: {self.reward_shaping}) ---")
            for player in self.players[1:]:
                 print(f"{player.name}: {self.bot_strategy_names[player.name]}")

        return self._get_state()

    def _process_turn(self, action: tuple):
        """
        Processes a single turn for any player (Agent or Bot).

        Executes the chosen action, updates the game state, and calculates rewards
        and termination flags.

        Args:
            action: A tuple containing (action_type, cards_to_play, announced_rank).

        Returns:
            A tuple of (reward, terminated, truncated, info).
        """
        action_type, cards_to_play, announced_rank = action
        acting_player_index = self.current_player_index
        current_player = self.players[acting_player_index]
        
        info = {} 
        
        if self.VISUALIZE_GAMES:
            print("-" * 30)
            print(f"Turn: {self.turn_count} | Player's Turn: {current_player.name}")
            
            for p in self.players:
                hand_vector = [0] * 14
                for card in p.hand:
                    if card.value in self.value_to_index:
                        hand_vector[self.value_to_index[card.value]] += 1
                print(f"  - {p.name}: {len(p.hand)} cards -> Freq: {hand_vector}")
            
            print(f"Rank to Play: '{self.current_rank_to_play}' | Cards in Pile: {len(self.round_discard_pile)}")
            print(f"Action by {current_player.name}: Type={action_type}, Cards={cards_to_play}, Rank='{announced_rank}'")

        
        round_ended_by_challenge = False
        reward = 0.0
        terminated = False

        if action_type == 0: # Doubt
            reward = self._resolve_challenge(acting_player_index, self.last_player_who_played_index)
            round_ended_by_challenge = True
        elif action_type == 1: # Pass
            reward = self._handle_pass(acting_player_index)
        else: # Play
            reward = self._play_cards(acting_player_index, cards_to_play, announced_rank)
        
        self.turn_count += 1
        terminated = self.check_game_over()

        truncated = False
        if not terminated and self.turn_count >= self.max_episode_steps:
            truncated = True
            
        if terminated:
            if self.winner == self.rl_agent:
                reward = self.REWARD_WIN
                info["is_win"] = 1.0
            else:
                reward = self.REWARD_LOSE
                info["is_win"] = 0.0
            
            if self.winner == self.rl_agent:
                winner_name = "RL_Agent"
            else:
                winner_name = self.bot_strategy_names[self.winner.name]
            
            info["winner_name"] = winner_name
            
            if self.VISUALIZE_GAMES:
                print(f"GAME OVER! The winner is: {self.winner.name} ({winner_name})")
                print("=" * 30 + "\n")
        
        elif truncated:
            info["is_win"] = 0.0

        if not terminated:
            # Determine next player
            if self.pass_counter >= (len(self.players)-1):
                self._start_new_round(self.last_player_who_played_index, clear_pile=False)
            elif round_ended_by_challenge:
                pass
            else:
                if self.starter_player_index == None :
                    self.current_player_index = (self.current_player_index + 1) % len(self.players)

        return reward, terminated, truncated, info

    def step(self, agent_action: tuple):
        """
        Executes a complete game step, starting with the RL agent's action.

        This function processes the RL agent's move and then automatically
        executes the turns of all subsequent bot opponents until it is the
        RL agent's turn again or the game ends.

        Args:
            agent_action: The action tuple selected by the RL agent.

        Returns:
            A tuple of (next_state, reward, terminated, truncated, info).
        """
        
        if self.current_player_index != 0:
             raise Exception("Environment 'step' called outside of RL Agent's turn.")

        agent_reward, terminated, truncated, info = self._process_turn(agent_action)

        # Execute bot turns until it's the agent's turn again or game ends
        while self.current_player_index != 0 and not terminated and not truncated:
            
            current_bot_player = self.players[self.current_player_index]
            bot_strategy_func = self.bot_strategies[current_bot_player.name]
            
            bot_action = bot_strategy_func(
                current_bot_player, 
                self.current_rank_to_play, 
                self.last_number_of_cards_played
            )
            
            reward_bot, terminated, truncated, info = self._process_turn(bot_action)

            if terminated:
                # If a bot wins, the agent loses
                agent_reward = self.REWARD_LOSE
                break
        
        state = self._get_state()
        return state, agent_reward, terminated, truncated, info


    def _start_new_round (self, starting_player_index: int, clear_pile: bool = True):
        """
        Resets round-specific variables to start a fresh round of play.
        If clear_pile is False, the discard pile is kept (used when everyone passes).
        """
        if clear_pile:
            self.round_discard_pile = []
        self.last_number_of_cards_played = None
        self.last_player_who_played_index = None
        self.current_rank_to_play = "Open"
        self.starter_player_index = starting_player_index
        self.current_player_index = starting_player_index

    def _resolve_challenge (self, current_player_index, last_player_who_played_index):
        """
        Handles the logic for a 'Doubt' action (Challenge).

        Verifies if the last play was a bluff. Distributes cards and rewards accordingly.
        """
        current_player = self.players[current_player_index]
        last_player_who_played = self.players[last_player_who_played_index]
        self.pass_counter = 0
        reward = 0.0

        got_the_cheat = False
        # Safety check for invalid challenge
        if not self.last_number_of_cards_played:
             self._start_new_round(current_player_index)
             if current_player == self.rl_agent:
                 return self.SHAPING_AGENT_CHALLENGE_FAIL
             return 0.0
             
        # Check the last N cards played
        for i in range(self.last_number_of_cards_played):
            card_idx = len(self.round_discard_pile)-i-1
            if card_idx < 0: continue
            current_card_to_analise = self.round_discard_pile[card_idx].value
            if current_card_to_analise != self.current_rank_to_play and current_card_to_analise != "Joker":
                got_the_cheat = True
                break

        if got_the_cheat:
            # Challenger wins
            for card in self.round_discard_pile:
                last_player_who_played.receive_card(card)
            
            if current_player == self.rl_agent: 
                reward = self.SHAPING_AGENT_CHALLENGE_SUCCESS
            elif last_player_who_played == self.rl_agent: 
                reward = self.SHAPING_OPPONENT_CAUGHT_AGENT_LIE
            self._start_new_round (current_player_index)
        else:
            # Challenger loses
            for card in self.round_discard_pile:
                current_player.receive_card(card)

            if current_player == self.rl_agent: 
                reward = self.SHAPING_AGENT_CHALLENGE_FAIL
            elif last_player_who_played == self.rl_agent: 
                reward = self.SHAPING_OPPONENT_FAILED_CHALLENGE
            self._start_new_round (last_player_who_played_index)
        
        return reward

    def _handle_pass(self, current_player_index):
        """
        Handles the logic for a 'Pass' action.
        """
        current_player = self.players[current_player_index]
        self.pass_counter += 1
        
        if current_player == self.rl_agent:
            return self.SHAPING_AGENT_PASS
        return 0.0

    def _play_cards(self, current_player_index, cards_to_play, announced_rank):
        current_player = self.players[current_player_index]
        if not cards_to_play:
            return self._handle_pass(current_player_index)

        self.current_rank_to_play = announced_rank
        
        for card in list(cards_to_play): 
            if card in current_player.hand:
                current_player.hand.remove(card)
                self.round_discard_pile.append(card)
            elif self.VISUALIZE_GAMES:
                print(f"** BUG ALERT: {current_player.name} tried to play {card} but did not have it! **")

        reward = 0.0
        self.last_player_who_played_index = current_player_index
        self.last_number_of_cards_played = len(cards_to_play)
        self.pass_counter = 0
        self.starter_player_index = None

        if (len(current_player.hand) == 0):
            reward = self._last_play_judge(current_player_index, cards_to_play)
        
        return reward

    def _last_play_judge(self, current_player_index, cards_to_play) :
        """
        Verifies the final play of a player. If they lied to win, the play is invalid.
        """
        current_player = self.players[current_player_index]
        was_a_lie = False
        for card in cards_to_play :
            if card.value != self.current_rank_to_play and card.value != "Joker":
                was_a_lie = True
                break

        if was_a_lie :
            if self.VISUALIZE_GAMES:
                print(f"** ILLEGAL FINAL PLAY by {current_player.name}! Picking up pile. **")
                
            for card_from_pile in self.round_discard_pile:
                current_player.receive_card(card_from_pile)
            
            next_player = (self.current_player_index+1) % len(self.players)
            self._start_new_round(next_player)

            if current_player == self.rl_agent:
                return self.SHAPING_AGENT_FINAL_LIE_PENALTY
        
        return 0.0