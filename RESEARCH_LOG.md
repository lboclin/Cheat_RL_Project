## Phase 3: Diagnosing Stochasticity & Proving Agent Capability

-   **Date:** November 14, 2025
-   **Objective:** Build upon the "vision-enabled" agent from Phase 2. The primary goals were to (1) implement reward shaping, (2) diagnose the agent's failure in the default stochastic environment, and (3) determine if the DQN agent is fundamentally capable of learning complex policies.

### 1. Experiment: Reward Shaping (Stochastic Environment)

-   **Objective:** Test the "Next Step" from Phase 2 by implementing a simple reward shaping mechanism to guide the agent.
-   **Methodology:**
    -   Reward: `+0.05` for successfully making an opponent draw cards.
    -   Penalty: `-0.05` for drawing cards from the pile.
    -   Environment: The default stochastic opponent pool (bots change each match, and the bots themselves are probabilistic).
-   **Results:**
    -   Performance was poor across 5 training runs.
    -   Maximum win rate achieved: **~12%**.
-   **Analysis:** The environment is excessively stochastic. The DQN algorithm (which is value-based) cannot learn a stable policy.
-   **Key Problem:** The agent faced contradictory rewards. For the exact same state (S) and action (A), it could receive `+0.05` in one game and `-0.05` in another due to the random nature of the bots. This makes it impossible for the Q-function to converge on whether action A is "good" or "bad."

### 2. Hypothesis & Isolation Tests

-   **Question:** Is the agent failing because (A) the environment is too complex/stochastic, or (B) the agent's code/architecture is fundamentally broken?

#### Test A: Isolate Bot Pool (e.g., vs. 2x 80/20 Bots)

-   **Objective:** Reduce environmental stochasticity by *removing* the changing bot pool.
-   **Methodology:** The agent was trained against a *static* pool of opponents (e.g., two `bot_strategy_80_20` or two `bot_strategy_60_40`).
-   **Result:** Failure. Performance remained poor, similar to the mixed-pool environment.
-   **Conclusion:** The **internal stochasticity** of the bots (their probabilistic decision-making) is, by itself, still too high for the DQN to handle.

#### Test B: Deterministic Environment (vs. 2x Honest Bots)

-   **Objective:** Prove that the agent's architecture *can* learn by placing it in a 100% deterministic environment.
-   **Methodology:** Agent plays against two `honest_bot` (who only play truth and never challenge).
-   **Expected Outcome:** If the agent is working, it should learn the simple, optimal policy: always play the maximum number of cards to win.
-   **Result:** **Total Success.** Achieved a **98% win rate** across two training runs. The ~2% loss is attributed to the remaining `epsilon` exploration.
-   **Conclusion:** **The agent's architecture and learning code are correct.** The DQN *works* perfectly in a deterministic environment.

### 3. Experiment: Complex Deterministic Policy (Forced Honesty)

-   **Hypothesis:** We've proven the agent fails in stochastic environments. Can it learn a *complex* but *deterministic* policy?
-   **Objective:** Force the agent to learn to "only tell the truth."
-   **Methodology:**
    -   **New Bot:** Created a `challenger_bot` (or `always_doubt_bot`) which always tells the truth and **always doubts the agent**.
    -   **New Reward Shaping:**
        -   Positive reward for telling the truth.
        -   High penalty for lying (which is always caught by the bot).
        -   Penalty for `PASS` (to prevent inaction).
    -   **Environment:** Agent vs. `challenger_bot`. The only path to victory is to tell the truth on every single turn.
-   **Results:**
    -   Training was extended to 150,000 episodes.
    -   The learning curve was incredibly slow, proving the policy's complexity.
    -   **First victory achieved at episode 130,000.**
    -   Reached a **20% win rate** by episode 150,000, showing a clear, though difficult, learning trend.
    -   *Note: This checkpoint was subsequently lost due to a bug, preventing further training.*
-   **Analysis:** This was a critical success. It proved two things:
    1.  The agent *can* learn highly complex policies.
    2.  Learning complex policies requires a *massive* number of episodes.

### 4. Phase 3 Conclusions & Next Steps

-   **Final Diagnosis:** The **DQN algorithm** is the bottleneck, not the agent's architecture. DQN is fundamentally ill-equipped to handle the high-stochasticity, high-variance environment of "Cheat" when played against probabilistic opponents. It excels *only* in deterministic (or near-deterministic) scenarios, as proven by Test B and Test C.
-   **Next Steps:**
    -   [x] Conclude experimentation with DQN.
    -   [ ] **Implement PPO (Proximal Policy Optimization).** PPO is a policy-gradient algorithm well-suited for stochastic environments and is the next logical step to solving the core problem identified in this phase. We are optimistic that PPO will yield better results in the original, stochastic environment.