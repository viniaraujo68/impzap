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
Not yet implemented. Next algorithm on the roadmap.

**Why CFR over more REINFORCE training**: REINFORCE plateaued at ~26% win rate vs MCTS after 50k episodes (no improvement over 10k). The ceiling is architectural — a static policy with no lookahead cannot match a search algorithm. CFR is game-theoretically sound for imperfect information games and converges to a Nash Equilibrium strategy rather than optimizing against a fixed opponent.

**Planned design**:
- Algorithm: Chance Sampling CFR (external sampling variant) — handles large state space without full tree traversal
- Information set key: `(player_hand, vira, table_cards, round_history, current_bet, score, round_wins)` — everything observable by the acting player
- State abstraction: card suit abstraction (suits are symmetric in Truco except for the manilha rule), bucket current_bet into ladder levels
- Storage: `regret_sum` and `strategy_sum` dicts keyed by `(info_set_key, action)`, persisted to disk as JSON or pickle
- Interface: `CFRAgent` in `agents/cfr_agent.py` with `train(num_iterations)` and `act(state, info)` methods
- Training: self-play only (CFR does not require an opponent — it converges by playing against itself)
- Use `TrucoEnv.step_from_state()` for tree traversal — stateless, no global state mutation

## HMM (Hidden Markov Models)
Not yet implemented. Planned: temporal modeling to track opponent betting behaviors and bluffing profiles.
