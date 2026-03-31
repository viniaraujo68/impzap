# Truco Paulista AI Project

## Project Overview
This project develops an autonomous AI agent capable of playing the Brazilian card game Truco Paulista (1v1). It tackles imperfect information, stochasticity, and bluffing mechanics.
The architecture uses a high-performance Go simulation engine coupled with a Python environment (Gymnasium wrapper) to train and evaluate AI agents.

The Go engine (`engine/`) exposes CGO exports to Python:
- `InitGame()` — creates a new game, returns a View JSON.
- `Step(actionID)` — advances the authoritative global game state.
- `StepFromState(stateJSON, actionID)` — stateless transition on a full GameState dict, used by MCTS tree expansion.
- `RolloutFromState(stateJSON, policyID)` — runs a full heuristic or random rollout to terminal inside Go, used by MCTS simulation. Returns `{"winner": int, "score": [int, int]}`.
- `CFRTrain(numIterations, resumePath)` — runs Go-native CFR training. Returns `{"iterations": int, "info_sets": int}`.
- `CFRSave(path)` / `CFRLoad(path)` — persist/load CFR tables as gzip JSON.

## Build and Execution Commands
* **Build Go Engine**: `make build`
* **Run REINFORCE Training**: `python train.py`
* **Run CFR Training**: `python train_cfr.py --iterations 1000000 --output models/cfr_NAME.json.gz`
* **Run Match/Tournament**: `python play.py`

## File Structure
- `engine/trucolib.go`: Core game engine — rules, state struct, CGO exports (`InitGame`, `Step`, `StepFromState`).
- `engine/rollout.go`: Rollout policies (`heuristicAction`, `randomAction`) and the `RolloutFromState` export.
- `engine/cfr.go`: Go-native CFR traversal — tables, tree search, save/load.
- `engine/cfr_exports.go`: CGO exports for CFR (`CFRTrain`, `CFRSave`, `CFRLoad`).
- `agents/`: All AI agent implementations.
  - `card_utils.py`: Single source of truth for card constants and Go card dict conversion (`card_to_go`, `go_to_card`).
  - `cfr_agent.py`: CFR agent — loads gzip JSON from Go training, uses average strategy at play time.
  - `mcts_agent.py`: PIMC-MCTS agent.
  - `heuristic_agent.py`, `random_agent.py`, `reinforce_agent.py`: Baseline agents.
- `truco_env/env.py`: `TrucoEnv` — thin ctypes wrapper around the Go shared library.
- `truco_env/wrappers.py`: `TrucoVectorObservation` — converts raw state dicts to fixed-size numpy vectors.
- `models/`: Saved model files — `.pth` (PyTorch/REINFORCE), `.json.gz` (gzip JSON/CFR). Gitignored due to size.

## Code Style
* **Language**: All code, comments, terminal prints, and documentation MUST be written in English.
* **Visuals**: Absolutely NO emojis are allowed in code comments, terminal output logs, or assistant responses.
* **Python**: PEP8, comprehensive type hinting, PyTorch `nn.Module` best practices. Use `numpy` and `matplotlib`.
* **Go**: Standard `gofmt`. CGO exports must be properly memory-managed. New CGO exports go in dedicated files, not `trucolib.go`.

## Development Guidelines
* Collect `reward_p0` at every step (including opponent turns) for correct credit assignment. Use `p0_step_indices` to align with `saved_log_probs`.
* Always base reward assignment on actual scoreboard delta, not on `current_player` turn.
* Ensure action masking is strictly applied to Neural Network outputs to prevent illegal actions.
* Access the underlying `TrucoEnv` via `env.raw_env` (typed property), not `env.unwrapped`.
* Use `TrucoEnv.step_from_state()` for MCTS tree expansion and `TrucoEnv.rollout_from_state()` for simulation. Never call `Step()` during search — it mutates global engine state.
* MCTS nodes must store the Go GameState dict directly. Do not replay the path from root.
* Card strings use the canonical format `"RANK_SUIT"` (e.g. `"3_CLUBS"`). Use `card_to_go()` / `go_to_card()` from `card_utils.py` for conversions.

## See Also
- `.claude/rules/algorithms.md` — MCTS and REINFORCE design details
- `.claude/rules/training.md` — training curriculum and hyperparameter decisions
- `.claude/rules/go-engine.md` — CGO interface and engine architecture notes
