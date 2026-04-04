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
- **Info set key** (v8): `(hand_sorted, my_table_bucket, opp_table_bucket, round_history, current_bet, pending_bet)`
- `hand_sorted`: sorted 8-bucket strengths of remaining hand cards
- `my_table_bucket` / `opp_table_bucket`: ordered table (distinguishes who played which card this round; -1 if not played)
- `round_history`: tuple of `(outcome, opp_bucket)` per completed round. `outcome`: 0=I_won, 1=opp_won, 2=tie. `opp_bucket`: full 8-bucket value of opp's face-up card (-1 if facedown). This is the key structural improvement over v4: explicit round outcomes + per-round opp card history.
- **8 strength buckets**: weak-trash(0-1), strong-trash(2-3), low(4-5), mid(6:K), mid-high(7:A), high(8:2), top(9:3), manilha(10+)
- **Action abstraction**: rank-ordered play actions (abstract 0=weakest, 2=strongest)
- **Regret pruning**: skip actions with cumulative regret below -300.0
- **Storage**: gzip-compressed JSON with string action keys (Python-compatible)
- **Training**: self-play only, ~6 min for 2M iterations (Go-native)

**Why v4 key was limited**:
- `played_buckets` (flat list of all played cards) had no explicit round-winner encoding — CFR had to infer round context from card positions, which it couldn't reliably
- `table_buckets` was sorted, losing the distinction between "I'm winning this round" vs "I'm losing this round"
- Both players' cards were in the flat list, inflating info sets without proportional gain

**v8 convergence results** (84K info sets, 23 visits/set at 2M iters):
- 89.3% vs Random (+1.9 pp vs v4@11M)
- 46.7–48.5% vs Heuristic (on par with v4@11M, within noise)
- 55.0% vs MCTS (+3 pp vs v4@11M)
- Beats v4@11M in head-to-head: 87.5%
- 24x fewer info sets, 5.5x fewer training iterations than v4@11M

**Version history** (models are gitignored, stored locally in `models/`):
- `cfr_v3_5buck_1M.json.gz` — 5 buckets, 1M iters, 183k info sets
- `cfr_v3_5buck_6M.json.gz` — 5 buckets, 6M iters, 184k info sets (converged, no improvement over 1M)
- `cfr_v4_8buck_1M.json.gz` — 8 buckets, 1M iters, 1.97M info sets
- `cfr_v4_8buck_11M.json.gz` — 8 buckets, 11M iters, 2.05M info sets (converged)
- `cfr_v8_fullbucket_2M.json.gz` — v8 key (round history), 2M iters, 84K info sets (**current**)

## HMM (Hidden Markov Model)
Standalone opponent-modeling agent that infers behavioral state and exploits detected tendencies.

**Architecture**: 3-state HMM with per-hand observations and Bayesian belief updates.

**Hidden states (3)**:
- Aggressive (0): raises often with real cards, disciplined (R_LOSS ~1%)
- Passive (1): folds easily, rarely raises
- Bluffing (2): raises frequently with weak hands, elevated R_LOSS

**Observations (5 per hand)**: FOLD, PASSIVE_LOSS, PASSIVE_WIN, RAISE_WIN, RAISE_LOSS. Extracted from opponent actions within each hand (did they raise, fold, win/lose).

**Emission matrix**: Calibrated from 2000-game tournaments against HeuristicAgent (Aggressive archetype) and RandomAgent (Bluffing archetype). Passive row is theoretical (no natural agent to calibrate against).

**Belief update**: Forward algorithm after each hand. O(S^2) per update = negligible with S=3.

**Confidence threshold**: 0.45. Below this, falls back to neutral heuristic play.

**Exploitation policy**:
- vs Passive: bluff-raise at 20% even with weak hands (they fold easily), raise with 1+ strong at 60%
- vs Bluffing: widen calling range (call with max_str >= 6), keep re-raise strict (they never fold re-raises)
- vs Aggressive: no exploit (falls through to neutral) — too balanced to exploit at raise/fold level

**Online adaptation**: Implemented but disabled by default (adapt=False). EMA update of dominant state's emission row. Prone to row collapse when playing a single opponent type — all rows converge toward the observed distribution. Mitigated by dominant-only adaptation + emission floor (0.02), but still degrades over long sessions. Needs further work.

**Benchmark results** (5000 games, vs Heuristic baseline):

| Opponent | HMM | Heuristic | Diff |
|---|---|---|---|
| AlwaysFold | 89.7% | 77.5% | **+12.2%** |
| Heuristic | 51.6% | 49.5% | **+2.1%** |
| AlwaysRaise | 73.5% | 74.9% | -1.4% |
| Random | 86.1% | 88.4% | -2.2% |

**Key findings**:
- HMM excels against exploitable opponents (Passive: +12.2%)
- Roughly neutral against balanced opponents (Heuristic: +2.1%, within noise)
- Small leak against non-exploitable opponents where modeling adds no value (Random: -2.2%)
- Online adaptation hurts more than helps in current form — feedback loop causes emission row collapse
- HeuristicAgent observation profile: FOLD=21%, P_LOSS=27%, P_WIN=30%, R_WIN=20%, R_LOSS=1%
- RandomAgent observation profile: FOLD=13%, P_LOSS=29%, P_WIN=14%, R_WIN=34%, R_LOSS=10%

**Implementation**: `agents/hmm_agent.py` — `HMMModel` (belief updates, matrices) + `HMMAgent` (tracking, action selection)
