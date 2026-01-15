# Research Log: Comparative Analysis of DQN vs. PPO in High-Stochasticity Environments

**Project:** Cheat_RL_Project  
**Focus:** Algorithmic Benchmarking for Superhuman Strategy Development  

---

### 1. Executive Summary
The objective of this research phase was to evaluate the performance, stability, and sample efficiency of value-based (DQN) versus policy-based (PPO) reinforcement learning algorithms. [cite_start]While DQN provided an initial baseline, PPO demonstrated a fundamental superiority in handling the stochastic nature of "Cheat," exhibiting a 20x improvement in sample efficiency and the ability to converge in environments where DQN failed catastrophically[cite: 164, 175].

### 2. Algorithmic Comparison Matrix

| Metric | DQN (Deep Q-Network) | PPO (Proximal Policy Optimization) |
| :--- | :--- | :--- |
| **Architecture** | [cite_start]Value-based (Q-Learning) [cite: 11] | [cite_start]Policy-based (Actor-Critic) [cite: 130] |
| **Action Space** | [cite_start]Atomic / Low-Dimensional [cite: 189] | [cite_start]Multi-Head High-Dimensional [cite: 130, 191] |
| **Stochastic Resilience** | [cite_start]Low (Policy collapse in random pools) [cite: 124] | [cite_start]High (Stable updates in varying environments) [cite: 129] |
| **Sample Efficiency** | [cite_start]Slow (~150k+ episodes for convergence) [cite: 117] | [cite_start]High (100% win rate in <35k steps) [cite: 175] |
| **Best Win Rate (Mixed)** | [cite_start]~12% [cite: 93] | [cite_start]~80% (Easy) / ~45% (Hard) [cite: 164] |
| **Exploration** | [cite_start]Epsilon-Greedy ($\epsilon$) [cite: 16, 196] | [cite_start]Entropy-based [cite: 196] |

---

### 3. Technical Deep-Dive: DQN Limitations
DQN served as the initial baseline but revealed critical flaws for complex card games:
* [cite_start]**The Stochastic Bottleneck:** DQN failed to learn stable policies when faced with probabilistic opponents (mixed bot pools), peaking at only ~12% win rate[cite: 93, 94]. [cite_start]The Q-function struggled to converge because identical state-action pairs yielded contradictory rewards due to opponent randomness[cite: 97, 98].
* [cite_start]**High Variance & Failure Rates:** In Phase 2, 7 out of 9 DQN training runs converged to a 0% win rate, indicating extreme sensitivity to initial weight initialization and replay buffer stochasticity[cite: 71, 74].
* [cite_start]**Deterministic Specialization:** DQN only achieved high performance (98%) in 100% deterministic environments (e.g., against Honest Bots), proving it lacks the robustness required for human-level play[cite: 107, 109].

### 4. Technical Deep-Dive: PPO Breakthroughs
PPO bypassed the limitations of DQN through several key mechanisms:
* [cite_start]**Resilience to POMDPs:** As a policy-gradient method, PPO is theoretically better suited for Partially Observable Markov Decision Processes (POMDPs) like "Cheat"[cite: 13, 129].
* **Superior Logic Discovery:** In "Hard Mode" (bots using logical suspect-play), the PPO agent discovered advanced human-like strategies, such as:
    * [cite_start]**The "4-of-a-Kind" Bluff:** Bluffing with ranks the agent held all four cards of to ensure the bots could not mathematically prove the lie[cite: 154, 157].
    * [cite_start]**Strategic Dumping & Anchoring:** Identifying specific cards (like Jails or Jokers) to hold as "anchors" for a guaranteed win[cite: 141, 142].
* [cite_start]**Sample Efficiency:** PPO reached a 100% win rate against fixed honest strategies 20x faster than DQN, demonstrating its ability to rapidly solve the credit assignment problem[cite: 175].

---

### 5. The "Challenger Paradox" and Self-Play Justification
[cite_start]During deterministic benchmarking, an anomaly was noted: PPO failed to defeat a "Constant-Doubt" bot (0% win rate) while DQN eventually succeeded (20% win rate)[cite: 182, 185, 189]. 

[cite_start]**Analysis:** PPO’s Multi-Head architecture makes "truthful play" a high-dimensional "needle in a haystack"[cite: 191, 193]. [cite_start]Under extreme punishment (constant doubting), PPO's policy for "Play" collapsed because it couldn't find a winning sequence fast enough during entropy-based exploration[cite: 195, 196].

**Conclusion for Self-Play:**
This paradox justifies the move to **Self-Play**. [cite_start]While static "Challenger" bots provide a wall of punishment that collapses a fixed policy, Self-Play allows for a **Co-evolutionary Curriculum**[cite: 198]. The agent will face versions of itself that are not 100% punitive but increasingly difficult, allowing the Multi-Head architecture to gradually align its Rank, Quantity, and Card Selection heads without premature policy collapse.

### 6. Final Recommendation for Superhuman Mastery
[cite_start]For the objective of achieving superhuman strategy, **PPO is the only viable candidate**[cite: 127, 164]. [cite_start]Its ability to exploit logical flaws in rule-based bots (e.g., detecting predictable bluff patterns in "Hard Mode") and its stability in stochastic environments make it the perfect foundation for the Self-Play phase[cite: 162, 165].

* **Final Decision:** Phase 6 will utilize PPO with a Multi-Head Actor-Critic architecture.
* [cite_start]**Adjustment:** Implement Victory-Centric Reward Scaling to eliminate "Reward Hacking" (prolonging games for intermediate points) observed in Phase 5[cite: 186, 187].