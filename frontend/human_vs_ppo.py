import torch
import random
import time
import numpy as np
import os
from cheat_env.environment import CheatEnviroment
from agents_ppo.ppo_actor_critic import PPOActorCritic

def get_state_for_ai(env, ai_index):
    """Generates the relative state vector for the AI (Size 32)."""
    player = env.players[ai_index]
    hand_vector = np.zeros(14)
    for card in player.hand:
        if card.value in env.value_to_index:
            hand_vector[env.value_to_index[card.value]] += 1
            
    opp_1 = env.players[(ai_index + 1) % 3]
    opp_2 = env.players[(ai_index + 2) % 3]
    opponent_card_counts = [len(opp_1.hand), len(opp_2.hand)]

    rank_vector = np.zeros(14)
    if env.current_rank_to_play in env.value_to_index:
        rank_vector[env.value_to_index[env.current_rank_to_play]] = 1

    discard_pile_size = [len(env.round_discard_pile)]
    is_starting_play = [1.0] if env.starter_player_index == ai_index else [0.0]

    return np.concatenate([
        hand_vector, np.array(opponent_card_counts), rank_vector,
        np.array(discard_pile_size), np.array(is_starting_play)
    ]).astype(np.float32)

def get_valid_actions_for_any_player(env, player_idx):
    """Calculates valid actions for any player index without breaking the environment logic."""
    player = env.players[player_idx]
    is_starter = (env.starter_player_index == player_idx) or (env.current_rank_to_play is None)
    
    # Basic Masks: 0:Doubt, 1:Pass, 2:Play
    types = [2] if is_starter else [0, 1, 2]
    ranks = ["Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King"] if is_starter else []
    
    # Quantity Mask (1 to 4 or hand total if smaller)
    max_q = min(len(player.hand), 4)
    quantities = list(range(1, max_q + 1))
    
    # Selection Mask
    valid_cards_mask = np.ones(len(player.hand))

    return {
        "types": types,
        "ranks": ranks,
        "quantities": quantities,
        "is_starter": is_starter,
        "current_rank": env.current_rank_to_play,
        "player_hand": player.hand,
        "valid_cards_mask": valid_cards_mask
    }

def print_board(env, human_idx):
    """Displays the board state with sorted cards for clarity."""
    # Sort human hand by real value (Ace, 2, 3...)
    env.players[human_idx].hand.sort(key=lambda c: env.value_to_index[c.value])
    
    print("\n" + "="*70)
    print(f"TURN: {env.turn_count} | Current Rank: '{env.current_rank_to_play}' | Pile: {len(env.round_discard_pile)}")
    print("-" * 35)
    
    for i, p in enumerate(env.players):
        if i == human_idx:
            print(f"-> {p.name} (YOU): {len(p.hand)} cards")
            freq = {val: 0 for val in env.card_values}
            for card in p.hand: freq[card.value] += 1
            summary = " | ".join([f"{k}:{v}" for k, v in freq.items() if v > 0])
            print(f"   Hand: [ {summary} ]")
            print(f"   Indices: {[f'{idx}:{c.value}' for idx, c in enumerate(p.hand)]}")
        else:
            print(f"   {p.name} (AI): {len(p.hand)} cards")
    print("="*70)

def play_game(path1, path2):
    """Main terminal loop for playing against the AI."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    roles = [{"name": "Human", "path": None}, {"name": "AI_1", "path": path1}, {"name": "AI_2", "path": path2}]
    random.shuffle(roles)
    human_idx = next(i for i, r in enumerate(roles) if r["name"] == "Human")
    
    env = CheatEnviroment(players_names=[r["name"] for r in roles], visualize=False)
    state_size = 32
    
    for r in roles:
        if r["path"]:
            r["model"] = PPOActorCritic(state_size).to(device)
            r["model"].load_state_dict(torch.load(r["path"], map_location=device))
            r["model"].eval()

    env.reset()
    while True:
        curr_idx = env.current_player_index
        role = roles[curr_idx]
        print_board(env, human_idx)
        
        # Capture data before action for challenge feedback
        last_play_idx = env.last_player_who_played_index
        pile_before = len(env.round_discard_pile)
        hands_before = [len(p.hand) for p in env.players]

        if role["name"] == "Human":
            valid = get_valid_actions_for_any_player(env, curr_idx)
            print(f"\nActions: 0:Doubt, 1:Pass, 2:Play")
            a_type = int(input("Select action: "))
            
            if a_type == 2: # Play
                if valid['is_starter']:
                    print(f"Ranks: {valid['ranks']}")
                    rank = input("Announce rank: ").capitalize()
                else:
                    rank = env.current_rank_to_play
                indices = input("Card indices (e.g., 0,1): ")
                cards = [env.players[curr_idx].hand[int(i)] for i in indices.split(',')]
                action = (a_type, cards, rank)
            else:
                action = (a_type, [], env.current_rank_to_play)
        else:
            print(f"\n{role['name']} is thinking...")
            time.sleep(1.2)
            state = torch.Tensor(get_state_for_ai(env, curr_idx)).to(device).unsqueeze(0)
            valid = get_valid_actions_for_any_player(env, curr_idx)
            with torch.no_grad():
                action, _, _, _, _ = role["model"].get_action_and_value(state, valid_actions=valid)
        
        # Feedback on the current action
        act_names = ["DOUBTED", "PASSED", "PLAYED"]
        print(f"\n>>>> {role['name']} {act_names[action[0]]} {len(action[1]) if action[0]==2 else ''} {'card(s) as ' + action[2] if action[0]==2 else ''}")

        _, terminated, truncated, _ = env._process_turn(action)

        if action[0] == 0: # Challenge Result Logic
            time.sleep(1.0)
            victim = env.players[last_play_idx]
            if len(victim.hand) > hands_before[last_play_idx]:
                print(f"  [!] SUCCESS: {victim.name} LIED! They picked up {pile_before} cards.")
            else:
                print(f"  [X] FAILURE: {victim.name} told the TRUTH! {role['name']} picked up {pile_before} cards.")
            time.sleep(1.5)
        
        if terminated or truncated: break

    print(f"\nWINNER: {env.winner.name}")

if __name__ == "__main__":
    # Model paths based on project results
    MODEL_1 = r"results\ppo_agent\no_reward_shaping\hard_random_pool\ppo_models\Cheat_v1__ppo_cheat__1__1764353370.pth"
    MODEL_2 = r"results\ppo_agent\no_reward_shaping\hard_random_pool\ppo_models\Cheat_v1__ppo_cheat__2__1764376593.pth"
    play_game(MODEL_1, MODEL_2)