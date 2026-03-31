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
External Sampling CFR with Go-native traversal for high-performance training.

**Why CFR over more REINFORCE training**: REINFORCE plateaued at ~26% win rate vs MCTS after 50k episodes (no improvement over 10k). The ceiling is architectural — a static policy with no lookahead cannot match a search algorithm. CFR is game-theoretically sound for imperfect information games and converges to a Nash Equilibrium strategy rather than optimizing against a fixed opponent.

**Implementation**:
- **Go-native traversal** in `engine/cfr.go`: direct `GameState` struct manipulation with `deepCopyState()` for tree branching — no JSON marshaling overhead (~2.3ms/iteration vs ~seconds in Python)
- **CGO exports** in `engine/cfr_exports.go`: `CFRTrain`, `CFRSave`, `CFRLoad`
- **Python agent** in `agents/cfr_agent.py`: `act()` uses average strategy at play time, loads gzip JSON from Go training
- **Training script**: `train_cfr.py` — calls Go CFR via ctypes
- **Info set key**: `(hand_buckets, table_buckets, played_buckets, current_bet, pending_bet, current_round)` — no score (keeps space ~50k info sets)
- **5 strength buckets**: trash(0-3), low(4-6), mid(7-8), high(9), manilha(10+)
- **Action abstraction**: rank-ordered play actions (abstract 0=weakest, 2=strongest)
- **Regret pruning**: skip actions with cumulative regret below -300.0
- **Storage**: gzip-compressed JSON with string action keys (Python-compatible)
- **Training**: self-play only, ~39 min for 1M iterations

## HMM (Hidden Markov Models)
Not yet implemented. Planned: temporal modeling to track opponent betting behaviors and bluffing profiles.
