# Algorithm Design Details

## MCTS (Monte Carlo Tree Search)
PIMC-MCTS (Perfect Information Monte Carlo) with K determinizations per `act()` call.
- **Tree expansion**: `TrucoEnv.step_from_state(state, action)` — stateless Go transition (~43μs/call)
- **Simulation**: single `TrucoEnv.rollout_from_state(state, policy_id)` call — entire heuristic rollout in Go (~98μs/call vs ~1s for Python loop)
- **State storage**: each `MCTSNode.state` holds the Go GameState dict directly. Never replay the path from root — `deal_new_hand` calls `random.shuffle`, making replay non-deterministic.
- **policyID**: `0` = random, `1` = heuristic (default). Passed as `C.int` to `RolloutFromState`.
- Implemented in `agents/mcts_agent.py`. Constants: `ROLLOUT_POLICY_RANDOM = 0`, `ROLLOUT_POLICY_HEURISTIC = 1`.

## REINFORCE (Policy Gradient)
Full-game trajectory collection with a single policy update at game end.
- **Returns**: reward-to-go with γ=0.99, computed backward over the full step sequence (all steps, not just p0's turns)
- **Baseline**: cross-episode EMA (`α=0.05`) on G_0 (total game return). No within-episode std normalization — normalizing a single monotonically-discounted sequence distorts advantage signs.
- **Gradient clipping**: `clip_grad_norm_(parameters, max_norm=1.0)`
- **Opponent curriculum**: HeuristicAgent until 65% rolling win rate (500-episode window), then 50/50 mix of frozen snapshot + heuristic to prevent catastrophic forgetting.
- **Frozen snapshot**: updated every 500 episodes once self-play activates.
- **Periodic evaluation**: every 500 episodes — 100 games vs RandomAgent + 100 vs HeuristicAgent.
- Implemented in `agents/reinforce_agent.py` and `train.py`.

## CFR (Counterfactual Regret Minimization)
Not yet implemented. Planned: game-theoretic approach to approximate Nash Equilibrium by minimizing counterfactual regret.

## HMM (Hidden Markov Models)
Not yet implemented. Planned: temporal modeling to track opponent betting behaviors and bluffing profiles.
