## Phase 5: Deterministic Benchmarking & Reward Shaping Analysis

-   **Date:** January 14, 2026
-   **Objective:** Conduct controlled experiments against fixed strategies (`Bot Honest` and `Bot Challenger`) to evaluate sample efficiency, final performance, and the impact of reward shaping. This phase aims to generate a direct comparison with previous DQN benchmarks.

### 1. Experiment A: Honest Bot (Easy & Hard)
-   **Methodology:**
    -   **Opponents:** Two fixed bots using `bot_strategy_100_0`.
    -   **Modes:** "Easy" (standard rules) and "Hard" (advanced logic).
-   **Quantitative Results:**
    -   **Easy Mode:** Achieved **100% win rate** in approximately **25,000 timesteps**.
    -   **Hard Mode:** Achieved **100% win rate** in approximately **35,000 timesteps** without requiring reward shaping.
-   **Comparison with DQN:**
    -   PPO demonstrated a massive gap in sample efficiency, training roughly **20x faster** than DQN.
    -   DQN failed to reach a 100% win rate in either mode, with its best performance topping out at 95% in Easy Mode.
    -   In Hard Mode, DQN was not able to win a single match, whereas all PPO models converged to 100% efficiency.

### 2. Experiment B: Challenger Bot (No Reward Shaping)
-   **Methodology:**
    -   **Opponents:** One `Bot Challenger` (always doubts) and one `Bot Honest`.
-   **Qualitative Analysis (Suboptimal Strategies):**
    -   **Game Control for Draws:** The agent learned to control the game state but prioritized avoiding defeat over securing a win. 
    -   **Card Accumulation:** To mitigate the complexity of aligning truth cards with announced ranks, the agent learned to accumulate as many cards as possible to increase the mathematical probability of a valid play.
    -   **Strategic Sabotage:** Some models purposely fed the Challenger bot lies to prevent it from winning, effectively trapping the game in a cycle of draws.
-   **Results:** Even after **5,000,000 timesteps**, no model achieved a victory, likely due to environment complexity and the difficulty of identifying a winning path under constant doubt.

### 3. Experiment C: Challenger Bot (With Reward Shaping)
-   **Methodology:**
    -   **Training:** 7 distinct training runs across 3 reward shaping variations.
    -   **Target:** Use intermediate rewards to guide the agent toward truthful play sequences.
-   **Quantitative Results:** **0% Win Rate** across all 7 runs.
-   **Failure Analysis (Reward Hacking):**
    -   The agent developed a "Reward Hacking" behavior. Because the environment offered **+0.5** for an opponent failing a challenge but only **+1.0** for a full game victory, the agent found it more optimal to prolong the game indefinitely. 
    -   Logs showed episodic returns reaching **~6.95**, indicating the agent was farming intermediate rewards through sporadic truthful plays rather than seeking the terminal win state.

### 4. Comparative Analysis: PPO vs. DQN (The Challenger Paradox)
A critical finding of this phase is that **DQN successfully defeated the Challenger bot (with reward shaping) while PPO failed completely.**

#### A. Card Selection Complexity
-   **DQN Structure:** The DQN implementation utilized a more atomic action structure, making "truthful play" a lower-dimensional optimization task.
-   **PPO Structure:** The PPO implementation uses a **Multi-Head architecture**. For every play, the agent must synchronize four separate decisions: Action Type, Rank, Quantity, and specific **Card Selection** (choosing 1-4 specific cards out of 54 options). 
-   **The "Needle in a Haystack":** Because the `Bot Challenger` always doubts, a single card selection error results in immediate, severe punishment. PPO's high-dimensional card selection head makes the probability of "stumbling" into a perfect truthful sequence statistically improbable during early exploration.

#### B. Algorithmic Sensitivity
-   **Policy vs. Value:** PPO (Policy Gradient) is highly sensitive to the initial "Wall of Punishment" presented by the Challenger. Once the agent identifies that playing cards leads to being caught, the policy for "Play" collapses to near-zero probability.
-   **Exploration:** DQN's $\epsilon$-greedy exploration forced it to keep trying actions until it identified the high value of truthful play, whereas PPO's entropy-based exploration was insufficient to overcome the immediate negative reinforcement of the Challenger bot.

### 5. Next Steps
- [ ] **Human-AI Interface Development:** Design and implement a graphical user interface (GUI) to facilitate direct interaction between human players and trained models, enabling empirical validation of agent behavior.
- [ ] **Self-Play Curriculum for Superhuman Mastery:** Leverage PPO’s demonstrated superiority in stochastic environments to transition into a Self-Play framework. By allowing the agent to compete against increasingly advanced versions of itself, the objective is to evolve complex bluffing dynamics and achieve superhuman-level strategic proficiency.