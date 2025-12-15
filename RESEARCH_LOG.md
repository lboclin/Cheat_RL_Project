## Phase 4: PPO Implementation & Algorithmic Breakthrough

-   **Date:** December 15, 2025
-   **Objective:** Implement Proximal Policy Optimization (PPO) using the `cleanrl` framework, adapt the environment for a multi-head actor-critic architecture, and evaluate the agent against the full stochastic bot pool in both "Easy" (No Suspect Play) and "Hard" (Suspect Play Active) modes.

### 1. Implementation: PPO & Multi-Head Architecture
-   **Transition:** Shifted from DQN to PPO to handle the environment's high variance and stochasticity.
-   **Architecture:** Implemented a Multi-Head Actor-Critic network. The "Actor" makes four simultaneous decisions:
    1.  **Action Type:** Pass, Play, or Doubt.
    2.  **Rank Selection:** Which rank to announce.
    3.  **Quantity:** How many cards to play.
    4.  **Card Selection:** Which specific cards to remove from the hand (handled via a custom sequential probability logic).

### 2. Experiment A: "Easy Mode" Evaluation
-   **Methodology:**
    -   **Opponents:** Full mixed pool (`honest`, `80_20`, `one_third`, `60_40`).
    -   **Constraint:** `suspect_play = False` (Bots rely on random probability to doubt; they do not calculate hand/pile logic).
    -   **Training:** 3 distinct seeds, 3,000,000 steps each.
-   **Quantitative Results:**
    -   **Seed 1:** ~79.2% Win Rate.
    -   **Seed 2:** ~79.8% Win Rate.
    -   **Seed 3:** ~83.6% Win Rate.
-   **Qualitative Analysis (Emergent Behaviors):**
    -   **Strategic Dumping:** The agent learned that in "Easy Mode," the probability of being caught is low. It prioritized playing the maximum allowed cards (6) to empty its hand quickly.
    -   **Truth-Finisher:** The agent learned to "clean" its hand of bad cards by lying early, saving true cards for the final turn to guarantee a win.
    -   **The "Anchor" Strategy (Seeds 2 & 3):** The agent developed a fascination with specific ranks (Jack in Seed 2, 10 in Seed 3). It would hold these cards and the Joker until the very end, effectively "anchoring" its game plan around a guaranteed final play.
    -   **Joker Mastery:** The agent learned to use the Joker specifically as the final card to secure the win.
    -   **Logic Detection:** The agent learned to doubt when:
        1.  The pile became statistically too large.
        2.  It held all 4 cards of a specific rank, and an opponent claimed to play that rank (impossible scenario).

### 3. Experiment B: "Hard Mode" Evaluation
-   **Methodology:**
    -   **Opponents:** Full mixed pool.
    -   **Constraint:** `suspect_play = True` (Bots calculate logic; if the agent plays a card the bot holds, or if the pile is too big, the bot *will* doubt).
    -   **Training:** 3 distinct seeds, 3,000,000 steps each.
-   **Quantitative Results:**
    -   **Seed 1:** ~47.0% Win Rate.
    -   **Seed 2:** ~44.4% Win Rate.
    -   **Seed 3:** ~43.0% Win Rate.
-   **Qualitative Analysis (Emergent Behaviors):**
    -   **Baiting Strategy:** Unlike Easy Mode, the agent plays conservatively (1-2 cards) at the start. It learned to "bait" bots into using their true cards. Once the pile grows and bots lose track of the game state, the agent switches to aggression (lying 1-4 cards).
    -   **The "4-of-a-Kind" Bluff:** The agent discovered a loophole in the bots' logic. If the agent holds 4 Kings, it can repeatedly lie and announce "Kings." The bots, holding no Kings, cannot mathematically prove the agent is lying via `suspect_play`. The agent mimics human deception by bluffing with cards it actually possesses.
    -   **Counter-Suspect:** The agent developed its own version of `suspect_play`, learning to doubt aggressively when the pile size suggests a high probability of opponent dishonesty.
    -   **Exploiting Honesty:** The agent identified that `honest_bots` lose their ability to doubt effectively after the first few turns (as they play their cards and lose information). The agent exploits this window to lie repeatedly.
    -   **Bluff Pattern Recognition**: The agent identified a predictable flaw in the bots' bluffing logic. The bots are encoded to lie primarily using small quantities (1 or 2 cards) to appear "safe." The agent cracked this pattern, learning that a 1 or 2-card play often signals a bluff, and aggressively targeted these specific plays with doubts, effectively punishing the bots for their predictable behavior.

### 4. Comparison with DQN
-   **Performance:**
    -   **DQN:** Failed to achieve meaningful win rates (~0-12%) in stochastic environments.
    -   **PPO:** Achieved ~80% in Easy Mode and ~45% in Hard Mode.
-   **Capability:** The PPO agent defeated the "Hard Mode" bots, which strictly follow logical rules. The DQN agent previously failed to win a single match against these logic-driven opponents. This confirms that the PPO algorithm successfully bridged the gap identified in Phase 3.

### 5. Next Steps
-   [ ] **Deterministic Benchmarking:** Run PPO against fixed, deterministic bots to generate a direct 1:1 comparison with the Phase 3 DQN tests.
-   [ ] **Reward Shaping Tests:** Experiment with granular reward shaping to see if the learning speed or final win rate can be further optimized.