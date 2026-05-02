package main

import (
	"compress/gzip"
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"sort"
	"strings"
	"time"
)

// ---------------------------------------------------------------------------
// CFR tables — global singleton used by CGO exports
// ---------------------------------------------------------------------------

var cfrTables *CFRTables

// CFRTables holds cumulative regret and strategy sums keyed by info set.
type CFRTables struct {
	RegretSum   map[string]map[int]float64 `json:"regret_sum"`
	StrategySum map[string]map[int]float64 `json:"strategy_sum"`
	Iterations  int                        `json:"iterations"`
}

func newCFRTables() *CFRTables {
	return &CFRTables{
		RegretSum:   make(map[string]map[int]float64),
		StrategySum: make(map[string]map[int]float64),
	}
}

// ---------------------------------------------------------------------------
// Card strength helpers
// ---------------------------------------------------------------------------

// suitPower returns the manilha tie-breaking power for a suit.
// Diamonds=1, Spades=2, Hearts=3, Clubs=4.
func suitPower(s Suit) int {
	switch s {
	case Diamonds:
		return 1
	case Spades:
		return 2
	case Hearts:
		return 3
	case Clubs:
		return 4
	}
	return 0
}

// cfrCardStrength returns the numeric strength of a card given the vira.
// Regular cards: rank index 0-9. Manilhas: 10 + suitPower (11-14).
// Facedown cards return -1.
func cfrCardStrength(card Card, vira Card) int {
	if card.Facedown {
		return -1
	}
	if IsManilha(card, vira) {
		return 10 + suitPower(card.Suit)
	}
	return card.Rank.Index()
}

// cfrScoreDeltaBucket maps (my_score - opp_score) to 5 buckets.
// 0=far behind(delta<=-7), 1=behind(-6..-2), 2=even(-1..1), 3=ahead(2..6), 4=far ahead(>=7).
func cfrScoreDeltaBucket(delta int) int {
	if delta <= -7 {
		return 0
	}
	if delta <= -2 {
		return 1
	}
	if delta <= 1 {
		return 2
	}
	if delta <= 6 {
		return 3
	}
	return 4
}

// cfrStrengthBucket maps card strength to 8 buckets.
// -1=facedown, 0=weak trash(0-1: 4,5), 1=strong trash(2-3: 6,7),
// 2=low(4-5: Q,J), 3=mid(6: K), 4=mid-high(7: A),
// 5=high(8: 2), 6=top(9: 3), 7=manilha(10+).
func cfrStrengthBucket(strength int) int {
	if strength < 0 {
		return -1
	}
	if strength <= 1 {
		return 0
	}
	if strength <= 3 {
		return 1
	}
	if strength <= 5 {
		return 2
	}
	if strength == 6 {
		return 3
	}
	if strength == 7 {
		return 4
	}
	if strength == 8 {
		return 5
	}
	if strength == 9 {
		return 6
	}
	return 7
}

// ---------------------------------------------------------------------------
// Info set key (must match Python _info_set_key output exactly)
// ---------------------------------------------------------------------------

// tupleStr formats an int slice as a Python tuple string.
func tupleStr(vals []int) string {
	if len(vals) == 0 {
		return "()"
	}
	parts := make([]string, len(vals))
	for i, v := range vals {
		parts[i] = fmt.Sprintf("%d", v)
	}
	if len(parts) == 1 {
		return "(" + parts[0] + ",)"
	}
	return "(" + strings.Join(parts, ", ") + ")"
}

// cfrOppCardBucket returns the opponent's card bucket for the round history:
// -1 = facedown/unknown, 0-7 = card strength bucket (same as cfrStrengthBucket).
func cfrOppCardBucket(card Card, vira Card) int {
	if card.Facedown {
		return -1
	}
	return cfrStrengthBucket(cfrCardStrength(card, vira))
}

// cfrFormatRoundHistory formats a list of (outcome, opp_coarse) pairs as a
// Python-compatible tuple string.
// Example: [(0,1),(1,0)] -> "((0, 1), (1, 0))"
// Single element: [(0,1)] -> "((0, 1),)"
// Empty: [] -> "()"
func cfrFormatRoundHistory(parts []string) string {
	if len(parts) == 0 {
		return "()"
	}
	if len(parts) == 1 {
		return "(" + parts[0] + ",)"
	}
	return "(" + strings.Join(parts, ", ") + ")"
}

// cfrInfoSetKey builds the info set key from a full GameState for the given
// player. Uses 8-bucket card strength abstraction.
//
// Key format: (hand_sorted, my_table_bucket, opp_table_bucket, round_history, current_bet, pending_bet)
//   - hand_sorted: sorted bucketed strengths of remaining hand cards
//   - my_table_bucket: bucket of player's card on the table this round (-1 if not played)
//   - opp_table_bucket: bucket of opponent's card on the table (-1 if not played or facedown)
//   - round_history: tuple of (outcome, opp_coarse) per completed round, from player's POV.
//     outcome: 0=won, 1=lost, 2=tie.
//     opp_coarse: -1=facedown, 0=trash(bucket 0-3), 1=strong(bucket 4-7).
//   - current_bet, pending_bet: bet state
//
// Output format matches Python's str(info_tuple).
func cfrInfoSetKey(s *GameState, player int) string {
	vira := s.Vira

	// Player's hand as sorted bucketed strengths.
	handBuckets := make([]int, 0, len(s.Hands[player]))
	for _, card := range s.Hands[player] {
		handBuckets = append(handBuckets, cfrStrengthBucket(cfrCardStrength(card, vira)))
	}
	sort.Ints(handBuckets)

	// Table card: ordered (my card vs opponent's card).
	// At any decision point, table has 0 or 1 card.
	// The card belongs to the round starter of the current round.
	myTableBucket := -1
	oppTableBucket := -1
	if len(s.TableCards) == 1 {
		roundStarter := s.RoundStarter[s.CurrentRound]
		b := cfrStrengthBucket(cfrCardStrength(s.TableCards[0], vira))
		if roundStarter == player {
			myTableBucket = b
		} else {
			oppTableBucket = b
		}
	}

	// Round history: (outcome, opp_card_coarse) per completed round.
	// outcome: 0=I_won, 1=opp_won, 2=tie.
	// opp_card_coarse: coarsened view of what opponent played face-up (-1=facedown, 0=trash, 1=strong).
	roundHistoryParts := make([]string, 0, s.CurrentRound)
	for r := 0; r < s.CurrentRound; r++ {
		w := s.RoundWinners[r]
		var outcome int
		if w == -1 {
			outcome = 2
		} else if w == player {
			outcome = 0
		} else {
			outcome = 1
		}

		// Opponent's card in round r: starter plays first (index 0), other plays second (index 1).
		rnd := s.RoundHistory[r]
		starter := s.RoundStarter[r]
		var oppCard Card
		if starter == player {
			oppCard = rnd[1] // I started, opp played second.
		} else {
			oppCard = rnd[0] // Opp started, their card is first.
		}

		oppCoarse := cfrOppCardBucket(oppCard, vira)
		roundHistoryParts = append(roundHistoryParts, fmt.Sprintf("(%d, %d)", outcome, oppCoarse))
	}

	scoreDeltaBucket := cfrScoreDeltaBucket(s.Score[player] - s.Score[1-player])
	maoDeOnze := 0
	if s.WaitingForMaoDeOnze {
		maoDeOnze = 1
	}
	return fmt.Sprintf("(%s, %d, %d, %s, %d, %d, %d, %d)",
		tupleStr(handBuckets), myTableBucket, oppTableBucket,
		cfrFormatRoundHistory(roundHistoryParts), s.CurrentBet, s.PendingBet,
		scoreDeltaBucket, maoDeOnze)
}

// ---------------------------------------------------------------------------
// Action abstraction — rank-ordered play actions
// ---------------------------------------------------------------------------

// cfrBuildActionMaps remaps play actions by strength rank.
// Abstract action 0 = play weakest, 2 = play strongest.
// Non-play actions (3,4,5) keep their identity.
func cfrBuildActionMaps(legalActions []int, handStrengths []int) (
	r2a map[int]int, a2r map[int]int, abstractActions []int,
) {
	// Rank hand indices by strength: rank 0 = weakest.
	type indexedStrength struct {
		idx      int
		strength int
	}
	indexed := make([]indexedStrength, len(handStrengths))
	for i, s := range handStrengths {
		indexed[i] = indexedStrength{i, s}
	}
	sort.Slice(indexed, func(i, j int) bool {
		return indexed[i].strength < indexed[j].strength
	})
	indexToRank := make(map[int]int)
	for rank, is := range indexed {
		indexToRank[is.idx] = rank
	}

	r2a = make(map[int]int)
	a2r = make(map[int]int)

	for _, a := range legalActions {
		if a >= 0 && a <= 2 {
			absA := indexToRank[a]
			r2a[a] = absA
			a2r[absA] = a
		} else if a >= 6 && a <= 8 {
			handIdx := a - 6
			absA := 6 + indexToRank[handIdx]
			r2a[a] = absA
			a2r[absA] = a
		} else {
			r2a[a] = a
			a2r[a] = a
		}
	}

	abstractActions = make([]int, len(legalActions))
	for i, a := range legalActions {
		abstractActions[i] = r2a[a]
	}
	return
}

// cfrHandStrengths returns the strength of each card in a player's hand.
func cfrHandStrengths(s *GameState, player int) []int {
	strengths := make([]int, len(s.Hands[player]))
	for i, card := range s.Hands[player] {
		strengths[i] = cfrCardStrength(card, s.Vira)
	}
	return strengths
}

// ---------------------------------------------------------------------------
// Deep copy GameState (needed for tree branching)
// ---------------------------------------------------------------------------

func deepCopyState(s *GameState) *GameState {
	cp := *s
	cp.Hands[0] = append([]Card(nil), s.Hands[0]...)
	cp.Hands[1] = append([]Card(nil), s.Hands[1]...)
	cp.TableCards = append([]Card(nil), s.TableCards...)
	cp.LegalActions = append([]int(nil), s.LegalActions...)
	for i := 0; i < 3; i++ {
		cp.RoundHistory[i] = append([]Card(nil), s.RoundHistory[i]...)
	}
	return &cp
}

// ---------------------------------------------------------------------------
// Apply action directly on a GameState (no JSON round-trip)
// ---------------------------------------------------------------------------

// cfrApplyAction applies an action to the state in place.
// Assumes the action is legal. This is the core speedup over StepFromState:
// no JSON marshal/unmarshal overhead.
func cfrApplyAction(s *GameState, actionID int) {
	if s.ResetRewardFlag {
		s.Reward = [2]float64{0.0, 0.0}
		s.ResetRewardFlag = false
	}
	s.executeAction(actionID)
	s.updateLegalActions()
}

// ---------------------------------------------------------------------------
// Strategy computation
// ---------------------------------------------------------------------------

const cfrPruneThreshold = -300.0

// cfrGetStrategy returns the current strategy via regret matching.
func (t *CFRTables) cfrGetStrategy(infoKey string, actions []int) map[int]float64 {
	regrets := t.RegretSum[infoKey]
	positive := make(map[int]float64, len(actions))
	total := 0.0
	for _, a := range actions {
		v := 0.0
		if regrets != nil {
			if r, ok := regrets[a]; ok && r > 0 {
				v = r
			}
		}
		positive[a] = v
		total += v
	}
	strategy := make(map[int]float64, len(actions))
	if total > 0 {
		for _, a := range actions {
			strategy[a] = positive[a] / total
		}
	} else {
		uniform := 1.0 / float64(len(actions))
		for _, a := range actions {
			strategy[a] = uniform
		}
	}
	return strategy
}

// ---------------------------------------------------------------------------
// CFR traversal (External Sampling)
// ---------------------------------------------------------------------------

// cfrTraverse performs one CFR traversal for a single hand.
// Returns utility for the traversing player.
func (t *CFRTables) cfrTraverse(
	s *GameState, traversingPlayer int,
	reachP0, reachP1 float64,
) float64 {
	// Terminal check: non-zero reward or game over.
	reward := s.Reward
	hasReward := reward[0] != 0.0 || reward[1] != 0.0
	if s.IsTerminal || hasReward {
		return reward[traversingPlayer]
	}

	currentPlayer := s.CurrentPlayer
	legalActions := s.LegalActions
	if len(legalActions) == 0 {
		return 0.0
	}

	// Build abstract action mapping.
	handStrengths := cfrHandStrengths(s, currentPlayer)
	r2a, a2r, abstractActions := cfrBuildActionMaps(legalActions, handStrengths)
	_ = r2a

	infoKey := cfrInfoSetKey(s, currentPlayer)
	strategy := t.cfrGetStrategy(infoKey, abstractActions)

	if currentPlayer == traversingPlayer {
		// Traversing player: explore all actions (with regret pruning).
		regrets := t.RegretSum[infoKey]
		exploreActions := make([]int, 0, len(abstractActions))
		for _, a := range abstractActions {
			r := 0.0
			if regrets != nil {
				r = regrets[a]
			}
			if r > cfrPruneThreshold {
				exploreActions = append(exploreActions, a)
			}
		}
		if len(exploreActions) == 0 {
			exploreActions = abstractActions
		}

		actionValues := make(map[int]float64, len(exploreActions))
		for _, absA := range exploreActions {
			realA := a2r[absA]
			nextState := deepCopyState(s)
			cfrApplyAction(nextState, realA)
			if currentPlayer == 0 {
				actionValues[absA] = t.cfrTraverse(
					nextState, traversingPlayer,
					reachP0*strategy[absA], reachP1,
				)
			} else {
				actionValues[absA] = t.cfrTraverse(
					nextState, traversingPlayer,
					reachP0, reachP1*strategy[absA],
				)
			}
		}

		// Compute node value from explored actions.
		exploredValue := 0.0
		exploredWeight := 0.0
		for _, a := range exploreActions {
			exploredValue += strategy[a] * actionValues[a]
			exploredWeight += strategy[a]
		}
		nodeValue := exploredValue
		if exploredWeight > 0 && exploredWeight < 1.0 {
			nodeValue = exploredValue / exploredWeight
		}

		// Update regrets.
		opponentReach := reachP1
		if currentPlayer == 1 {
			opponentReach = reachP0
		}
		if t.RegretSum[infoKey] == nil {
			t.RegretSum[infoKey] = make(map[int]float64)
		}
		for _, absA := range exploreActions {
			regret := opponentReach * (actionValues[absA] - nodeValue)
			t.RegretSum[infoKey][absA] += regret
		}

		return nodeValue

	} else {
		// Opponent: sample one action, update strategy sum.
		myReach := reachP0
		if currentPlayer == 1 {
			myReach = reachP1
		}
		if t.StrategySum[infoKey] == nil {
			t.StrategySum[infoKey] = make(map[int]float64)
		}
		for _, a := range abstractActions {
			t.StrategySum[infoKey][a] += myReach * strategy[a]
		}

		// Sample action from strategy.
		r := rand.Float64()
		cumulative := 0.0
		chosenAbs := abstractActions[len(abstractActions)-1]
		for _, a := range abstractActions {
			cumulative += strategy[a]
			if r < cumulative {
				chosenAbs = a
				break
			}
		}

		realAction := a2r[chosenAbs]
		nextState := deepCopyState(s)
		cfrApplyAction(nextState, realAction)
		if currentPlayer == 0 {
			return t.cfrTraverse(
				nextState, traversingPlayer,
				reachP0*strategy[chosenAbs], reachP1,
			)
		}
		return t.cfrTraverse(
			nextState, traversingPlayer,
			reachP0, reachP1*strategy[chosenAbs],
		)
	}
}

// createGameWithRandomScore creates a fresh hand starting at a uniformly random
// score so that CFR sees all score_delta_bucket values during training.
// When exactly one player is at 11, WaitingForMaoDeOnze is set appropriately.
func createGameWithRandomScore() *GameState {
	gs := createNewGame()
	gs.Score[0] = rand.Intn(12) // 0-11
	gs.Score[1] = rand.Intn(12) // 0-11
	if gs.Score[0] == 11 && gs.Score[1] < 11 {
		gs.WaitingForMaoDeOnze = true
		gs.CurrentPlayer = 0
	} else if gs.Score[1] == 11 && gs.Score[0] < 11 {
		gs.WaitingForMaoDeOnze = true
		gs.CurrentPlayer = 1
	}
	gs.updateLegalActions()
	return gs
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

func (t *CFRTables) train(numIterations int) {
	logInterval := numIterations / 20
	if logInterval < 1 {
		logInterval = 1
	}
	startTime := time.Now()

	for i := 1; i <= numIterations; i++ {
		gs := createGameWithRandomScore()

		// Traverse as player 0, then as player 1.
		t.cfrTraverse(gs, 0, 1.0, 1.0)

		// Fresh game with same deal for player 1 traversal.
		// Reset rewards since traversal may have modified them.
		gs2 := createGameWithRandomScore()
		t.cfrTraverse(gs2, 1, 1.0, 1.0)

		t.Iterations++

		if i%logInterval == 0 {
			elapsed := time.Since(startTime).Seconds()
			infoSets := len(t.RegretSum)
			fmt.Printf("[CFR-Go] Iteration %d/%d (%.1fs) | Info sets: %d\n",
				i, numIterations, elapsed, infoSets)
		}
	}

	elapsed := time.Since(startTime).Seconds()
	fmt.Printf("[CFR-Go] Training complete. %d iterations in %.1fs. Info sets: %d\n",
		numIterations, elapsed, len(t.RegretSum))
}

// ---------------------------------------------------------------------------
// Persistence — gzip-compressed JSON, readable by Python
// ---------------------------------------------------------------------------

// cfrSaveJSON is the JSON-friendly format with string keys for actions.
type cfrSaveJSON struct {
	RegretSum   map[string]map[string]float64 `json:"regret_sum"`
	StrategySum map[string]map[string]float64 `json:"strategy_sum"`
	Iterations  int                           `json:"iterations"`
}

func (t *CFRTables) save(path string) error {
	// Convert int action keys to string for JSON compatibility.
	saveData := cfrSaveJSON{
		RegretSum:   make(map[string]map[string]float64, len(t.RegretSum)),
		StrategySum: make(map[string]map[string]float64, len(t.StrategySum)),
		Iterations:  t.Iterations,
	}
	for key, actions := range t.RegretSum {
		m := make(map[string]float64, len(actions))
		for a, v := range actions {
			m[fmt.Sprintf("%d", a)] = v
		}
		saveData.RegretSum[key] = m
	}
	for key, actions := range t.StrategySum {
		m := make(map[string]float64, len(actions))
		for a, v := range actions {
			m[fmt.Sprintf("%d", a)] = v
		}
		saveData.StrategySum[key] = m
	}

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	gz := gzip.NewWriter(f)
	defer gz.Close()

	enc := json.NewEncoder(gz)
	return enc.Encode(saveData)
}

func (t *CFRTables) load(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	gz, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	defer gz.Close()

	var saveData cfrSaveJSON
	if err := json.NewDecoder(gz).Decode(&saveData); err != nil {
		return err
	}

	t.Iterations = saveData.Iterations
	t.RegretSum = make(map[string]map[int]float64, len(saveData.RegretSum))
	t.StrategySum = make(map[string]map[int]float64, len(saveData.StrategySum))

	for key, actions := range saveData.RegretSum {
		m := make(map[int]float64, len(actions))
		for aStr, v := range actions {
			var a int
			fmt.Sscanf(aStr, "%d", &a)
			m[a] = v
		}
		t.RegretSum[key] = m
	}
	for key, actions := range saveData.StrategySum {
		m := make(map[int]float64, len(actions))
		for aStr, v := range actions {
			var a int
			fmt.Sscanf(aStr, "%d", &a)
			m[a] = v
		}
		t.StrategySum[key] = m
	}

	return nil
}
