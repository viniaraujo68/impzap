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
- **8 strength buckets**: weak-trash(0-1), strong-trash(2-3), low(4-5), mid(6:K), mid-high(7:A), high(8:2), top(9:3), manilha(10+)
- **Action abstraction**: rank-ordered play actions (abstract 0=weakest, 2=strongest)
- **Regret pruning**: skip actions with cumulative regret below -300.0
- **Storage**: gzip-compressed JSON with string action keys (Python-compatible)
- **Training**: self-play only, ~4 min for 1M iterations (Go-native)

**Convergence results** (8 buckets):
- v4@1M (0.5 visits/set): 87.5% vs Random, 49.9% vs Heuristic, 48% vs MCTS, 69% vs REINFORCE
- v4@11M (5.4 visits/set): 87.4% vs Random, 48.5% vs Heuristic, 52% vs MCTS, 63.7% vs REINFORCE
- Strategy converges around 1M iterations; more iterations don't meaningfully improve results
- 8 buckets closed the heuristic gap (41% with 5 buckets -> ~50% with 8) but further granularity would explode the info set space (already 2M info sets)
- Remaining ceiling is the abstraction itself — within-bucket card distinction is lost

**Version history** (models are gitignored, stored locally in `models/`):
- `cfr_v3_5buck_1M.json.gz` — 5 buckets, 1M iters, 183k info sets
- `cfr_v3_5buck_6M.json.gz` — 5 buckets, 6M iters, 184k info sets (converged, no improvement over 1M)
- `cfr_v4_8buck_1M.json.gz` — 8 buckets, 1M iters, 1.97M info sets
- `cfr_v4_8buck_11M.json.gz` — 8 buckets, 11M iters, 2.05M info sets (converged)

## HMM (Hidden Markov Models)
Not yet implemented. Planned as an **opponent modeling layer** on top of existing agents (CFR/MCTS), not a standalone agent.
- Track opponent betting patterns over time to detect tendencies (bluffing frequency, aggression level)
- Hidden states represent unobservable opponent "modes" (aggressive, passive, tilted)
- Temporal transitions capture how opponent behavior shifts during a game
- Feed inferred opponent profile into agent decisions to shift away from Nash equilibrium and exploit detected weaknesses
