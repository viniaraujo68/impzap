package main

// #include <stdlib.h>
import "C"

import (
	"encoding/json"
	"math/rand"
)

// Rollout policy identifiers passed as policyID to RolloutFromState.
const (
	rolloutPolicyRandom    = 0
	rolloutPolicyHeuristic = 1
)

// Rollout tuning constants.
const (
	rolloutStrongThreshold = 9   // minimum card strength to be considered "strong"
	rolloutTrucoProb       = 0.4 // probability of requesting truco when holding a strong card
	rolloutMaxSteps        = 300 // depth guard to prevent infinite rollouts
)

// cardStrength returns the numeric strength of a card given the vira.
// Regular cards return their rank index [0..9] (Four=0, Three=9).
// Manilhas return 10 + suit power [11..14] (Diamonds=11, Clubs=14).
// Face-down cards return -1.
func cardStrength(card Card, vira Card) int {
	if card.Facedown {
		return -1
	}
	if IsManilha(card, vira) {
		manilhaPower := map[Suit]int{Clubs: 4, Hearts: 3, Spades: 2, Diamonds: 1}
		return 10 + manilhaPower[card.Suit]
	}
	return card.Rank.Index()
}

// randomAction picks a uniformly random legal action.
func randomAction(s *GameState) int {
	return s.LegalActions[rand.Intn(len(s.LegalActions))]
}

// heuristicAction selects an action using the same policy as the Python
// heuristic agent: play the strongest face-up card, accept bets when
// holding a strong card, and request truco with probability
// rolloutTrucoProb when the strongest card meets rolloutStrongThreshold.
func heuristicAction(s *GameState) int {
	hand := s.Hands[s.CurrentPlayer]
	legal := s.LegalActions

	if s.WaitingForMaoDeOnze || s.WaitingForBet {
		hasStrong := false
		for _, c := range hand {
			if cardStrength(c, s.Vira) >= rolloutStrongThreshold {
				hasStrong = true
				break
			}
		}
		if hasStrong {
			for _, a := range legal {
				if a == 4 {
					return 4
				}
			}
		}
		for _, a := range legal {
			if a == 5 {
				return 5
			}
		}
		return legal[0]
	}

	bestPlay := -1
	bestStrength := -2
	for _, a := range legal {
		if a >= 0 && a <= 2 {
			if strength := cardStrength(hand[a], s.Vira); strength > bestStrength {
				bestStrength = strength
				bestPlay = a
			}
		}
	}

	if bestPlay == -1 {
		return randomAction(s)
	}

	for _, a := range legal {
		if a == 3 && bestStrength >= rolloutStrongThreshold {
			if rand.Float64() < rolloutTrucoProb {
				return 3
			}
			break
		}
	}

	return bestPlay
}

type rolloutResult struct {
	Winner int    `json:"winner"`
	Score  [2]int `json:"score"`
}

// RolloutFromState runs a complete rollout from the given GameState JSON
// to a terminal state (or rolloutMaxSteps depth limit) using the specified
// policy, and returns {"winner": int, "score": [int, int]}.
//
// policyID values:
//
//	0 (rolloutPolicyRandom)    — uniformly random legal action each step
//	1 (rolloutPolicyHeuristic) — heuristic policy matching HeuristicAgent
//
//export RolloutFromState
func RolloutFromState(stateJSON *C.char, policyID C.int) *C.char {
	var s GameState
	if err := json.Unmarshal([]byte(C.GoString(stateJSON)), &s); err != nil {
		data, _ := json.Marshal(rolloutResult{Winner: -1})
		return C.CString(string(data))
	}

	if s.ResetRewardFlag {
		s.Reward = [2]float64{0.0, 0.0}
		s.ResetRewardFlag = false
	}

	selectAction := randomAction
	if int(policyID) == rolloutPolicyHeuristic {
		selectAction = heuristicAction
	}

	for steps := 0; steps < rolloutMaxSteps && !s.IsTerminal; steps++ {
		if len(s.LegalActions) == 0 {
			break
		}
		action := selectAction(&s)
		if !s.isActionLegal(action) {
			action = s.LegalActions[0]
		}
		s.executeAction(action)
		s.updateLegalActions()
	}

	data, _ := json.Marshal(rolloutResult{Winner: s.Winner, Score: s.Score})
	return C.CString(string(data))
}
