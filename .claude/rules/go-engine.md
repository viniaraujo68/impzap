# Go Engine Architecture

## CGO Interface
The shared library (`truco_env/trucolib.so`) is built from all `.go` files in `engine/` via:
```
cd engine && go build -buildmode=c-shared -o ../truco_env/trucolib.so .
```
Always use `make build`. Editing any `engine/*.go` file requires a rebuild before Python can use the changes.

## Export Locations
- `trucolib.go`: core engine logic — game rules, state struct, `InitGame`, `Step`, `StepFromState`
- `rollout.go`: agent-facing logic — `heuristicAction`, `randomAction`, `RolloutFromState`
- New CGO exports belong in dedicated files, not `trucolib.go`

## Memory Management
- All CGO exports that return strings use `C.CString(...)` — Python must call `lib.free_string(ptr)` after reading
- The global `gameState` in `trucolib.go` is mutated by `Step()` and `InitGame()` only
- `StepFromState` and `RolloutFromState` are fully stateless (unmarshal → compute → marshal)

## State JSON Round-trip
- `StepFromState`: ~43μs/call
- `RolloutFromState`: ~98μs/call (entire heuristic rollout)
- All `Card` fields use json tags: `rank` (int), `suit` (int), `facedown` (bool)
- All `GameState` fields have json tags — see struct definition in `trucolib.go`

## Raise Ladder
Progression: 1 → 3 → 6 → 9 → 12
- `requestBet()` when `!WaitingForBet`: sets `PendingBet` to next value above `CurrentBet` (not hardcoded 3)
- `requestBet()` when `WaitingForBet` (re-raise): advances `PendingBet` only, never touches `CurrentBet`
- `acceptBet()`: the only function that updates `CurrentBet = PendingBet`
- `updateLegalActions`: raise allowed when `PendingBet < 12` (during negotiation) or `CurrentBet < 12` (outside negotiation)
