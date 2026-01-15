import flet as ft
import torch
import random
import time
import numpy as np
import os
import sys
import threading

# Add root directory to path to import cheat_env and agents_ppo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cheat_env.environment import CheatEnviroment
from agents_ppo.ppo_actor_critic import PPOActorCritic
from cheat_env.card import Suit

# --- Helper Logic (Mirrored from human_vs_ppo.py for independence) ---

def get_state_for_ai(env, ai_index):
    """Generates the relative state vector for the AI (Size 32)."""
    player = env.players[ai_index]
    hand_vector = np.zeros(14)
    for card in player.hand:
        if card.value in env.value_to_index:
            hand_vector[env.value_to_index[card.value]] += 1
            
    opp_idx1 = (ai_index + 1) % 3
    opp_idx2 = (ai_index + 2) % 3
    opponent_card_counts = [len(env.players[opp_idx1].hand), len(env.players[opp_idx2].hand)]

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
    """Calculates valid actions for any player index without breaking the environment."""
    player = env.players[player_idx]
    is_starter = (env.starter_player_index == player_idx) or (env.current_rank_to_play is None)
    
    types = [2] if is_starter else [0, 1, 2] # 0:Doubt, 1:Pass, 2:Play
    ranks = ["Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King"] if is_starter else []
    
    max_q = min(len(player.hand), 6) # Max 6 cards per play as requested
    quantities = list(range(max_q))
    
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

# --- UI Components ---

class PlayingCard(ft.GestureDetector):
    def __init__(self, card, on_click):
        super().__init__()
        self.card = card
        self.selected = False
        self.on_tap = self._handle_click
        self.click_callback = on_click
        self.mouse_cursor = ft.MouseCursor.CLICK
        
        suit_symbols = {
            Suit.SPADES: "♠",
            Suit.HEARTS: "♥",
            Suit.DIAMONDS: "♦",
            Suit.CLUBS: "♣",
            Suit.JOKER: "J"
        }
        suit_colors = {
            Suit.SPADES: ft.Colors.WHITE_70,
            Suit.HEARTS: ft.Colors.RED_400,
            Suit.DIAMONDS: ft.Colors.RED_400,
            Suit.CLUBS: ft.Colors.WHITE_70,
            Suit.JOKER: ft.Colors.PURPLE_400
        }
        
        symbol = suit_symbols.get(card.suit, "?")
        color = suit_colors.get(card.suit, ft.Colors.GREY_400)
        
        self.content = ft.Container(
            content=ft.Stack([
                # Mini rank top-left
                ft.Container(
                    content=ft.Text(card.value, size=12, weight=ft.FontWeight.BOLD, color=color),
                    left=5, top=5
                ),
                # Central suit symbol
                ft.Container(
                    content=ft.Text(symbol, size=34, color=color),
                    alignment=ft.Alignment.CENTER
                ),
                # Mini rank bottom-right
                ft.Container(
                    content=ft.Text(card.value, size=12, weight=ft.FontWeight.BOLD, color=color),
                    right=5, bottom=5
                ),
            ]),
            width=80,
            height=120,
            bgcolor=ft.Colors.GREY_900,
            border_radius=10,
            border=ft.Border.all(1, ft.Colors.WHITE_10),
            shadow=ft.BoxShadow(blur_radius=10, color=ft.Colors.BLACK_54),
            animate=ft.Animation(200, ft.AnimationCurve.EASE_OUT),
            on_hover=self._on_hover
        )

    def _handle_click(self, e):
        self.click_callback(self)

    def _on_hover(self, e):
        if not self.selected:
            self.content.scale = 1.1 if e.data == "true" else 1.0
            self.content.border = ft.Border.all(1, ft.Colors.WHITE_54) if e.data == "true" else ft.Border.all(1, ft.Colors.WHITE_10)
            self.content.update()

    def toggle_select(self):
        self.selected = not self.selected
        if self.selected:
            self.content.bgcolor = ft.Colors.BLUE_900
            self.content.border = ft.Border.all(2, ft.Colors.BLUE_400)
            self.content.scale = 1.05
            self.content.shadow = ft.BoxShadow(blur_radius=15, color=ft.Colors.BLUE_900)
        else:
            self.content.bgcolor = ft.Colors.GREY_900
            self.content.border = ft.Border.all(1, ft.Colors.WHITE_10)
            self.content.scale = 1.0
            self.content.shadow = ft.BoxShadow(blur_radius=10, color=ft.Colors.BLACK_54)
        self.update()

# --- Main Application ---

def main(page: ft.Page):
    page.title = "Cheat RL - Interface"
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 0
    page.bgcolor = "#0f111a"
    page.window_width = 1100
    page.window_height = 850

    # --- Game State ---
    game_state = {
        "env": None,
        "human_idx": 0,
        "roles": [],
        "selected_cards": [],
        "is_human_turn": False,
        "terminated": False,
        "started": False
    }

    # --- UI References ---
    narrative_text = ft.Text("Welcome!", size=20, color=ft.Colors.BLUE_200, italic=True)
    
    # Game Log (Chat style)
    game_log_column = ft.Column(scroll=ft.ScrollMode.ALWAYS, spacing=5)
    game_log_container = ft.Container(
        content=game_log_column,
        width=250,
        height=200,
        bgcolor="#1a1c29cc", # 80% opacity
        border_radius=10,
        padding=10,
        border=ft.Border.all(1, ft.Colors.WHITE_10)
    )

    pile_count = ft.Text("0", size=40, weight=ft.FontWeight.BOLD)
    current_rank_label = ft.Text("Open", size=24, color=ft.Colors.AMBER_400, weight=ft.FontWeight.BOLD)
    
    hand_row = ft.Row(wrap=True, spacing=10, alignment=ft.MainAxisAlignment.CENTER)
    
    ai1_cards = ft.Text("0", size=30)
    ai1_name = ft.Text("AI 1", size=18)
    ai2_cards = ft.Text("0", size=30)
    ai2_name = ft.Text("AI 2", size=18)
    
    # Control Buttons
    btn_doubt = ft.ElevatedButton("Doubt", icon=ft.Icons.QUESTION_MARK, disabled=True, on_click=lambda _: handle_action(0))
    btn_pass = ft.ElevatedButton("Pass", icon=ft.Icons.SKIP_NEXT, disabled=True, on_click=lambda _: handle_action(1))
    btn_play = ft.ElevatedButton("Play", icon=ft.Icons.PLAY_ARROW, disabled=True, bgcolor=ft.Colors.GREEN_700, color=ft.Colors.WHITE, on_click=lambda _: handle_action(2))
    
    rank_dropdown = ft.Dropdown(
        label="Announce Rank",
        width=150,
        options=[ft.DropdownOption(r) for r in ["Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King"]],
        visible=False
    )

    def restart_click(e):
        game_over_dialog.open = False
        page.update()
        init_game()

    def show_game_over(winner_name):
        game_over_dialog.content = ft.Text(f"The winner is: {winner_name}", size=20)
        game_over_dialog.open = True
        page.update()

    game_over_dialog = ft.AlertDialog(
        modal=True,
        title=ft.Text("Game Over!"),
        actions=[
            ft.TextButton("Play Again", on_click=restart_click)
        ],
        actions_alignment=ft.MainAxisAlignment.END,
    )
    page.overlay.append(game_over_dialog)

    def update_narrative(msg, color=ft.Colors.BLUE_200, add_to_log=True):
        narrative_text.value = msg
        narrative_text.color = color
        
        if add_to_log and game_state["started"]:
            # Add to game log
            game_log_column.controls.append(
                ft.Text(f"> {msg}", size=12, color=color)
            )
            # Maintain only last 15 messages for performance
            if len(game_log_column.controls) > 15:
                game_log_column.controls.pop(0)
            
        page.update()

    def update_ui():
        env = game_state["env"]
        human_idx = game_state["human_idx"]
        
        # Refresh human hand
        hand_row.controls.clear()
        # Sort hand
        env.players[human_idx].hand.sort(key=lambda c: env.value_to_index[c.value])
        for card in env.players[human_idx].hand:
            hand_row.controls.append(PlayingCard(card, on_card_click))
        
        # Update AI info
        ai_indices = [(human_idx + 1) % 3, (human_idx + 2) % 3]
        ai1_idx, ai2_idx = ai_indices
        ai1_name.value = env.players[ai1_idx].name
        ai1_cards.value = str(len(env.players[ai1_idx].hand))
        ai2_name.value = env.players[ai2_idx].name
        ai2_cards.value = str(len(env.players[ai2_idx].hand))
        
        # Update Pile and Rank
        pile_count.value = str(len(env.round_discard_pile))
        current_rank_label.value = f"RANK: {env.current_rank_to_play if env.current_rank_to_play != 'Open' else 'Open'}"
        current_rank_label.color = ft.Colors.AMBER_400 if env.current_rank_to_play != "Open" else ft.Colors.GREY_400
        
        # Update Buttons
        if game_state["is_human_turn"]:
            valid = get_valid_actions_for_any_player(env, human_idx)
            btn_doubt.disabled = 0 not in valid["types"]
            btn_pass.disabled = 1 not in valid["types"]
            refresh_play_button()
            
            rank_dropdown.visible = valid["is_starter"]
            if valid["is_starter"] and not rank_dropdown.value:
                rank_dropdown.value = "Ace"
        else:
            btn_doubt.disabled = True
            btn_pass.disabled = True
            btn_play.disabled = True
            rank_dropdown.visible = False
            
        page.update()

    def refresh_play_button():
        count = len(game_state["selected_cards"])
        btn_play.disabled = not (game_state["is_human_turn"] and 1 <= count <= 6)
        page.update()

    def on_card_click(card_control):
        if not game_state["is_human_turn"]: return
        
        if card_control.selected:
            game_state["selected_cards"].remove(card_control.card)
            card_control.toggle_select()
        else:
            if len(game_state["selected_cards"]) < 6:
                game_state["selected_cards"].append(card_control.card)
                card_control.toggle_select()
        
        refresh_play_button()

    def handle_action(a_type):
        if not game_state["is_human_turn"]: return
        
        env = game_state["env"]
        human_idx = game_state["human_idx"]
        
        rank = env.current_rank_to_play
        cards = []
        
        if a_type == 2: # Play
            cards = game_state["selected_cards"]
            if not cards: return
            if rank_dropdown.visible:
                rank = rank_dropdown.value
        
        action = (a_type, cards, rank)
        process_turn(action)

    def process_turn(action):
        game_state["is_human_turn"] = False
        game_state["selected_cards"] = []
        
        env = game_state["env"]
        curr_idx = env.current_player_index
        player_name = env.players[curr_idx].name
        
        # Action name for logs
        act_name = ["DOUBTED", "PASSED", "PLAYED"][action[0]]
        details = f"{len(action[1])} card(s) as {action[2]}" if action[0] == 2 else ""
        
        update_narrative(f"{player_name} {act_name} {details}")
        
        # Capture state before processing for challenge feedback
        last_play_idx = env.last_player_who_played_index
        pile_before = len(env.round_discard_pile)
        hands_before = [len(p.hand) for p in env.players]

        _, term, trunc, _ = env._process_turn(action)
        game_state["terminated"] = term or trunc
        
        update_ui()
        
        # Challenge Feedback Logic
        if action[0] == 0: # Doubt
            time.sleep(1.0)
            victim = env.players[last_play_idx]
            if len(victim.hand) > hands_before[last_play_idx]:
                update_narrative(f"SUCCESS! {victim.name} was lying and picked up {pile_before} cards!", ft.Colors.GREEN_400)
            else:
                update_narrative(f"FAILURE! {victim.name} was telling the truth. {player_name} picked up {pile_before} cards.", ft.Colors.RED_400)
            time.sleep(1.5)
            update_narrative("Next play...", add_to_log=False)

        if game_state["terminated"]:
            update_narrative(f"GAME OVER! Winner: {env.winner.name}", ft.Colors.YELLOW_400)
            show_game_over(env.winner.name)
            return

        # Trigger next turn
        run_game_loop()

    def run_game_loop():
        if game_state["terminated"]: return
        
        env = game_state["env"]
        curr_idx = env.current_player_index
        role = game_state["roles"][curr_idx]
        
        # Identify human if role has no model path
        if role["path"] is None:
            game_state["is_human_turn"] = True
            update_ui()
            update_narrative("Your turn! Select cards to play or doubt the last play.")
        else:
            # AI playing
            update_ui()
            update_narrative(f"{role['name']} is thinking...")
            
            # Run AI logic in separate thread to avoid freezing UI
            def ai_think():
                try:
                    time.sleep(1.5) # Narrative delay
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    state = torch.Tensor(get_state_for_ai(env, curr_idx)).to(device).unsqueeze(0)
                    valid = get_valid_actions_for_any_player(env, curr_idx)
                    
                    # Ensure safety limits for action indexing
                    valid["quantities"] = [q for q in valid["quantities"] if 0 <= q < 6]
                    
                    with torch.no_grad():
                        action, _, _, _, _ = role["model"].get_action_and_value(state, valid_actions=valid)
                    
                    # Sync back with main thread
                    page.run_thread(process_turn, action)
                except Exception as e:
                    import traceback
                    error_msg = f"AI ERROR ({role['name']}): {str(e)}"
                    print(error_msg)
                    traceback.print_exc()
                    page.run_thread(update_narrative, error_msg, ft.Colors.RED_700)
            
            threading.Thread(target=ai_think).start()

    # --- Game Initialization ---
    def init_game(e=None):
        start_overlay.visible = False
        game_state["terminated"] = False
        game_state["started"] = True
        game_state["selected_cards"] = []
        game_log_column.controls.clear()
        
        update_narrative("Shuffling positions and dealing cards...", add_to_log=False)
        
        # Hardcoded paths based on project structure
        MODEL_P1 = r"results\ppo_agent\no_reward_shaping\hard_random_pool\ppo_models\Cheat_v1__ppo_cheat__1__1764353370.pth"
        MODEL_P2 = r"results\ppo_agent\no_reward_shaping\hard_random_pool\ppo_models\Cheat_v1__ppo_cheat__2__1764376593.pth"
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        roles = [
            {"name": "Human Player", "path": None},
            {"name": "Alpha", "path": MODEL_P1},
            {"name": "Beta", "path": MODEL_P2}
        ]
        random.shuffle(roles)
        
        game_state["roles"] = roles
        game_state["human_idx"] = next(i for i, r in enumerate(roles) if r["path"] is None)
        
        env = CheatEnviroment(players_names=[r["name"] for r in roles], visualize=False)
        game_state["env"] = env
        
        state_size = 32 # Relative state size based on environment
        for r in roles:
            if r["path"]:
                r["model"] = PPOActorCritic(state_size).to(device)
                r["model"].load_state_dict(torch.load(r["path"], map_location=device))
                r["model"].eval()
        
        env.reset()
        run_game_loop()

    # --- Layout Definitions ---
    start_overlay = ft.Container(
        content=ft.Column([
            ft.Text("CHEAT RL", size=50, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_ACCENT),
            ft.Text("A bluffing game against Reinforcement Learning agents", size=18, italic=True, color=ft.Colors.WHITE_70),
            ft.Divider(height=40, color=ft.Colors.TRANSPARENT),
            ft.ElevatedButton(
                "Start Match",
                width=250,
                height=60,
                on_click=init_game,
                style=ft.ButtonStyle(
                    bgcolor=ft.Colors.BLUE_700,
                    color=ft.Colors.WHITE,
                    shape=ft.RoundedRectangleBorder(radius=10)
                )
            )
        ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
        expand=True,
        bgcolor="#0f111a",
        visible=True,
    )

    page.add(
        ft.Stack([
            ft.Container(
                content=ft.Column(
                    [
                        # Top Section: Narrative and Log
                        ft.Stack([
                            ft.Container(
                                content=narrative_text,
                                alignment=ft.Alignment.CENTER,
                                padding=10,
                                bgcolor="#1a1c29",
                                border_radius=10,
                                width=1000
                            ),
                            game_log_container,
                        ], width=1000, height=200),
                        ft.Divider(height=40, color=ft.Colors.TRANSPARENT),
                        
                        # Middle Area (Rivals and Pile)
                        ft.Stack([
                            ft.Row(
                                [
                                    # AI 1 Card View
                                    ft.Container(
                                        content=ft.Column([
                                            ft.Icon(ft.Icons.SMART_TOY, size=40, color=ft.Colors.BLUE_ACCENT),
                                            ai1_name,
                                            ft.Row([ft.Icon(ft.Icons.COPY_ALL, size=20), ai1_cards], alignment=ft.MainAxisAlignment.CENTER)
                                        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                                        padding=20, bgcolor="#1e202e", border_radius=15, width=180
                                    ),
                                    
                                    # Center Pile Area
                                    ft.Column([
                                        ft.Text("CURRENT RANK", size=12, color=ft.Colors.WHITE_54, weight=ft.FontWeight.W_300),
                                        current_rank_label,
                                        ft.Stack([
                                            ft.Container(
                                                content=ft.Column([
                                                    ft.Text("PILE", size=14, weight=ft.FontWeight.BOLD),
                                                    pile_count
                                                ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                                                width=120, height=160,
                                                bgcolor=ft.Colors.GREY_800,
                                                border=ft.Border.all(2, ft.Colors.WHITE_24),
                                                border_radius=10,
                                            ),
                                        ], alignment=ft.Alignment.CENTER),
                                    ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=10),
                                    
                                    # AI 2 Card View
                                    ft.Container(
                                        content=ft.Column([
                                            ft.Icon(ft.Icons.SMART_TOY, size=40, color=ft.Colors.ORANGE_ACCENT),
                                            ai2_name,
                                            ft.Row([ft.Icon(ft.Icons.COPY_ALL, size=20), ai2_cards], alignment=ft.MainAxisAlignment.CENTER)
                                        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                                        padding=20, bgcolor="#1e202e", border_radius=15, width=180
                                    ),
                                ],
                                alignment=ft.MainAxisAlignment.SPACE_EVENLY,
                                width=1000
                            ),
                        ]),
                        
                        ft.Divider(height=40, color=ft.Colors.WHITE_12),
                        
                        # Human Player Area
                        ft.Column(
                            [
                                # Player Hand
                                ft.Container(
                                    content=hand_row,
                                    height=150,
                                    padding=10
                                ),
                                
                                # Interaction Controls
                                ft.Row(
                                    [
                                        rank_dropdown,
                                        btn_doubt,
                                        btn_pass,
                                        btn_play,
                                    ],
                                    alignment=ft.MainAxisAlignment.CENTER,
                                    spacing=20
                                )
                            ],
                            horizontal_alignment=ft.CrossAxisAlignment.CENTER
                        )
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    
                ),
                padding=20,
                expand=True
            ),
            start_overlay
        ], expand=True)
    )

if __name__ == "__main__":
    ft.run(main)