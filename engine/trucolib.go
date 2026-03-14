package main

// #include <stdlib.h>
import "C"

import (
	"encoding/json"
	"math/rand"
	"unsafe"
)

type Suit int

const (
	Clubs Suit = iota
	Spades
	Hearts
	Diamonds
)

type Rank int

const (
	Four Rank = iota
	Five
	Six
	Seven
	Queen
	Jack
	King
	Ace
	Two
	Three
)

type Card struct {
	Rank     Rank
	Suit     Suit
	Facedown bool
}

type Deck []Card

func (r Rank) Index() int {
	regularRanks := []Rank{Four, Five, Six, Seven, Queen, Jack, King, Ace, Two, Three}
	for i, rank := range regularRanks {
		if rank == r {
			return i
		}
	}
	return -1
}

func IsManilha(card Card, vira Card) bool {
	viraIdx := vira.Rank.Index()
	nextIdx := (viraIdx + 1) % 10
	regularRanks := []Rank{Four, Five, Six, Seven, Queen, Jack, King, Ace, Two, Three}
	return card.Rank == regularRanks[nextIdx]
}

func Compare(c1, c2 Card, vira Card) int {
	if c1.Facedown && c2.Facedown {
		return 0
	}
	if c1.Facedown {
		return -1
	}
	if c2.Facedown {
		return 1
	}

	c1M, c2M := IsManilha(c1, vira), IsManilha(c2, vira)
	if c1M && !c2M {
		return 1
	}
	if !c1M && c2M {
		return -1
	}
	if c1M && c2M {
		manilhaOrder := []Suit{Clubs, Hearts, Spades, Diamonds}
		for _, suit := range manilhaOrder {
			if c1.Suit == suit {
				return 1
			}
			if c2.Suit == suit {
				return -1
			}
		}
		return 0
	}
	regularRanks := []Rank{Four, Five, Six, Seven, Queen, Jack, King, Ace, Two, Three}
	for _, rank := range regularRanks {
		if c1.Rank == rank {
			return -1
		}
		if c2.Rank == rank {
			return 1
		}
	}
	return 0
}

func (c Card) String() string {
	if c.Facedown {
		return "FACEDOWN"
	}
	rankNames := []string{"4", "5", "6", "7", "Q", "J", "K", "A", "2", "3"}
	suitNames := []string{"CLUBS", "SPADES", "HEARTS", "DIAMONDS"}
	return rankNames[c.Rank] + "_" + suitNames[c.Suit]
}

func NewDeck() Deck {
	var deck Deck
	ranks := []Rank{Four, Five, Six, Seven, Queen, Jack, King, Ace, Two, Three}
	suits := []Suit{Clubs, Spades, Hearts, Diamonds}
	for _, r := range ranks {
		for _, s := range suits {
			deck = append(deck, Card{Rank: r, Suit: s, Facedown: false})
		}
	}
	return deck
}

func (d Deck) Shuffle() Deck {
	rand.Shuffle(len(d), func(i, j int) { d[i], d[j] = d[j], d[i] })
	return d
}

type GameState struct {
	IsTerminal          bool   `json:"is_terminal"`
	CurrentPlayer       int    `json:"current_player"`
	Score               [2]int `json:"score"`
	Hands               [2][]Card
	Vira                Card   `json:"vira"`
	TableCards          []Card `json:"table_cards"`
	CurrentBet          int    `json:"current_bet_value"`
	PendingBet          int
	WaitingForBet       bool
	WaitingForMaoDeOnze bool `json:"waiting_for_mao_de_onze"`
	TrucoHolder         int
	OriginalTurn        int
	RoundWins           [2]int
	CurrentRound        int
	RoundHistory        [3][]Card
	RoundStarter        [3]int
	RoundWinners        [3]int
	Reward              float64 `json:"reward"`
	Winner              int     `json:"winner"`
	LegalActions        []int   `json:"legal_actions"`
	HandJustEnded       bool
	ResetRewardFlag     bool
	HandStarter         int
}

type View struct {
	IsTerminal          bool     `json:"is_terminal"`
	CurrentPlayer       int      `json:"current_player"`
	Score               [2]int   `json:"score"`
	Hand                []string `json:"hand"`
	Vira                string   `json:"vira"`
	TableCards          []string `json:"table_cards"`
	CurrentBet          int      `json:"current_bet_value"`
	WaitingForMaoDeOnze bool     `json:"waiting_for_mao_de_onze"`
	LegalActions        []int    `json:"legal_actions"`
	Reward              float64  `json:"reward"`
	Winner              int      `json:"winner"`
}

var gameState *GameState

//export InitGame
func InitGame() *C.char {
	gameState = createNewGame()
	return C.CString(gameState.Marshal())
}

func createNewGame() *GameState {
	deck := NewDeck().Shuffle()
	viraIdx := len(deck) - 1
	vira := deck[viraIdx]
	handCards := deck[:viraIdx]
	return &GameState{
		IsTerminal: false, CurrentPlayer: 0, Score: [2]int{0, 0},
		Hands: [2][]Card{handCards[:3], handCards[3:6]}, Vira: vira,
		TableCards: []Card{}, CurrentBet: 1, PendingBet: 0, WaitingForBet: false, WaitingForMaoDeOnze: false,
		TrucoHolder: -1, OriginalTurn: -1, RoundWins: [2]int{0, 0}, CurrentRound: 0,
		RoundHistory: [3][]Card{{}, {}, {}}, RoundStarter: [3]int{-1, -1, -1},
		RoundWinners: [3]int{-1, -1, -1}, Reward: 0.0, Winner: -1,
		LegalActions: []int{0, 1, 2, 3}, HandJustEnded: false, ResetRewardFlag: true,
		HandStarter: 0,
	}
}

//export Step
func Step(actionID C.int) *C.char {
	if gameState == nil {
		return C.CString(`{"error":"Game not initialized"}`)
	}
	if gameState.IsTerminal {
		return C.CString(`{"error":"Game already ended"}`)
	}

	if gameState.ResetRewardFlag {
		gameState.Reward = 0.0
		gameState.ResetRewardFlag = false
	}

	if !gameState.isActionLegal(int(actionID)) {
		winner := 1 - gameState.CurrentPlayer
		gameState.Score[winner] += gameState.CurrentBet
		gameState.IsTerminal = true
		gameState.Winner = winner
		gameState.Reward = float64(gameState.CurrentBet) * float64(winner*2-1)
		return C.CString(gameState.Marshal())
	}

	gameState.executeAction(int(actionID))
	gameState.updateLegalActions()
	return C.CString(gameState.Marshal())
}

//export FreeString
func FreeString(str *C.char) { C.free(unsafe.Pointer(str)) }

func (s *GameState) isActionLegal(action int) bool {
	for _, legal := range s.LegalActions {
		if legal == action {
			return true
		}
	}
	return false
}

func (s *GameState) executeAction(action int) {
	switch {
	case action >= 0 && action <= 2:
		s.playCard(action, false)
	case action >= 6 && action <= 8:
		s.playCard(action-6, true)
	case action == 3:
		s.requestBet()
	case action == 4:
		s.acceptBet()
	case action == 5:
		s.fold()
	}

	s.checkGameEnd()
}

func (s *GameState) playCard(index int, facedown bool) {
	hand := s.Hands[s.CurrentPlayer]
	card := hand[index]
	card.Facedown = facedown
	s.Hands[s.CurrentPlayer] = append(hand[:index], hand[index+1:]...)

	if len(s.TableCards) == 0 {
		s.RoundStarter[s.CurrentRound] = s.CurrentPlayer
	}

	s.TableCards = append(s.TableCards, card)
	s.RoundHistory[s.CurrentRound] = append(s.RoundHistory[s.CurrentRound], card)

	if len(s.TableCards) == 2 {
		s.resolveRound()
	} else {
		s.CurrentPlayer = 1 - s.CurrentPlayer
	}
}

func (s *GameState) resolveRound() {
	c1, c2 := s.TableCards[0], s.TableCards[1]
	result := Compare(c1, c2, s.Vira)
	var roundWinner int

	switch {
	case result > 0:
		roundWinner = s.RoundStarter[s.CurrentRound]
	case result < 0:
		roundWinner = 1 - s.RoundStarter[s.CurrentRound]
	default:
		roundWinner = -1
	}

	s.RoundWinners[s.CurrentRound] = roundWinner

	if roundWinner != -1 {
		s.RoundWins[roundWinner]++
	}

	if roundWinner != -1 {
		s.CurrentPlayer = roundWinner
	} else {
		s.CurrentPlayer = s.RoundStarter[s.CurrentRound]
	}

	s.TableCards = []Card{}
	s.CurrentRound++
}

func (s *GameState) fold() {
	if s.WaitingForMaoDeOnze {

		winner := 1 - s.CurrentPlayer
		s.Score[winner] += 1

		if winner == 0 {
			s.Reward = 1.0
		} else {
			s.Reward = -1.0
		}

		s.WaitingForMaoDeOnze = false

		if s.Score[winner] >= 12 {
			s.IsTerminal = true
			s.Winner = winner
		} else {
			s.HandJustEnded = true
			s.ResetRewardFlag = true
		}
		return
	}

	if !s.WaitingForBet {
		return
	}

	winner := 1 - s.CurrentPlayer
	s.Score[winner] += s.CurrentBet

	if winner == 0 {
		s.Reward = float64(s.CurrentBet)
	} else {
		s.Reward = -float64(s.CurrentBet)
	}

	s.WaitingForBet = false

	if s.Score[winner] >= 12 {
		s.IsTerminal = true
		s.Winner = winner
	} else {
		s.HandJustEnded = true
		s.ResetRewardFlag = true
	}
}

func (s *GameState) requestBet() {
	ladder := []int{1, 3, 6, 9, 12}

	if !s.WaitingForBet {
		s.PendingBet = 3
		s.WaitingForBet = true
		s.TrucoHolder = s.CurrentPlayer
		s.OriginalTurn = s.CurrentPlayer
		s.CurrentPlayer = 1 - s.CurrentPlayer
		return
	}

	for i, bet := range ladder {
		if bet == s.PendingBet && i < len(ladder)-1 {
			s.CurrentBet = s.PendingBet
			s.PendingBet = ladder[i+1]
			s.TrucoHolder = s.CurrentPlayer
			s.CurrentPlayer = 1 - s.CurrentPlayer
			return
		}
	}
}

func (s *GameState) acceptBet() {
	if s.WaitingForMaoDeOnze {
		s.WaitingForMaoDeOnze = false
		s.CurrentBet = 3
		s.CurrentPlayer = s.HandStarter
		return
	}

	if !s.WaitingForBet {
		return
	}
	s.CurrentBet = s.PendingBet
	s.PendingBet = 0
	s.WaitingForBet = false
	s.CurrentPlayer = s.OriginalTurn
}

func (s *GameState) determineHandWinner() int {
	if s.RoundWins[0] >= 2 {
		return 0
	}
	if s.RoundWins[1] >= 2 {
		return 1
	}

	w1, w2, w3 := s.RoundWinners[0], s.RoundWinners[1], s.RoundWinners[2]

	if w1 == -1 && w2 == -1 && w3 == -1 {
		return -1
	}

	if w1 == -1 {
		if w2 != -1 {
			return w2
		}
		if w3 != -1 {
			return w3
		}
		return -1
	}

	if w1 != -1 && w2 == -1 {
		return w1
	}

	if w1 != -1 && w2 != -1 && w3 == -1 {
		return w1
	}

	return -1
}

func (s *GameState) applyHandScores() {
	winner := s.determineHandWinner()
	if winner == -1 {
		s.Reward = 0.0
		s.HandJustEnded = true
		s.ResetRewardFlag = true
		return
	}

	s.Score[winner] += s.CurrentBet
	if winner == 0 {
		s.Reward = float64(s.CurrentBet)
	} else {
		s.Reward = -float64(s.CurrentBet)
	}

	s.HandJustEnded = true
	s.ResetRewardFlag = true

	if s.Score[winner] >= 12 {
		s.IsTerminal = true
		s.Winner = winner
	}
}

func (s *GameState) checkGameEnd() {
	if s.HandJustEnded {
		if !s.IsTerminal {
			s.dealNewHand()
		}
		return
	}

	if len(s.TableCards) > 0 {
		return
	}

	isHandOver := false

	if s.RoundWins[0] >= 2 || s.RoundWins[1] >= 2 {
		isHandOver = true
	} else if s.CurrentRound == 2 {
		w1, w2 := s.RoundWinners[0], s.RoundWinners[1]
		if w1 == -1 && w2 != -1 {
			isHandOver = true
		}
		if w1 != -1 && w2 == -1 {
			isHandOver = true
		}
	} else if s.CurrentRound >= 3 {
		isHandOver = true
	}

	if isHandOver {
		s.applyHandScores()
		if !s.IsTerminal {
			s.dealNewHand()
		}
	}
}

func (s *GameState) dealNewHand() {
	deck := NewDeck().Shuffle()
	viraIdx := len(deck) - 1
	s.Vira = deck[viraIdx]
	handCards := deck[:viraIdx]
	s.Hands[0], s.Hands[1] = handCards[:3], handCards[3:6]
	s.TableCards = []Card{}
	s.RoundWins = [2]int{0, 0}
	s.RoundHistory = [3][]Card{{}, {}, {}}
	s.RoundStarter = [3]int{-1, -1, -1}
	s.RoundWinners = [3]int{-1, -1, -1}
	s.CurrentRound = 0
	s.CurrentBet = 1
	s.PendingBet = 0
	s.WaitingForBet = false
	s.WaitingForMaoDeOnze = false
	s.TrucoHolder = -1
	s.OriginalTurn = -1
	s.HandJustEnded = false

	s.HandStarter = 1 - s.HandStarter
	s.CurrentPlayer = s.HandStarter

	if s.Score[0] == 11 && s.Score[1] < 11 {
		s.WaitingForMaoDeOnze = true
		s.CurrentPlayer = 0
	} else if s.Score[1] == 11 && s.Score[0] < 11 {
		s.WaitingForMaoDeOnze = true
		s.CurrentPlayer = 1
	}
}

func (s *GameState) updateLegalActions() {
	actions := []int{}
	hand := s.Hands[s.CurrentPlayer]

	if s.WaitingForMaoDeOnze {
		actions = append(actions, 4, 5)
	} else if s.WaitingForBet {
		actions = append(actions, 4, 5)
		if s.CurrentPlayer != s.TrucoHolder && s.PendingBet < 12 {
			actions = append(actions, 3)
		}
	} else {
		for i := range hand {
			actions = append(actions, i)
			if s.CurrentRound >= 1 {
				actions = append(actions, i+6)
			}
		}

		isMaoDeOnzeOrFerro := s.Score[0] == 11 || s.Score[1] == 11
		if !isMaoDeOnzeOrFerro && s.CurrentBet < 12 && (s.TrucoHolder == -1 || s.CurrentPlayer != s.TrucoHolder) {
			if len(hand) > 0 {
				actions = append(actions, 3)
			}
		}
	}

	if len(actions) == 0 && !s.IsTerminal {
		actions = append(actions, 5)
	}

	s.LegalActions = actions
}

func (s *GameState) Marshal() string {
	hand := s.Hands[s.CurrentPlayer]
	serializedHand := make([]string, len(hand))
	for i, c := range hand {
		serializedHand[i] = c.String()
	}

	tableCards := make([]string, len(s.TableCards))
	for i, c := range s.TableCards {
		tableCards[i] = c.String()
	}
	for len(tableCards) < 2 {
		tableCards = append(tableCards, "")
	}

	view := View{
		IsTerminal: s.IsTerminal, CurrentPlayer: s.CurrentPlayer, Score: s.Score,
		Hand: serializedHand, Vira: s.Vira.String(), TableCards: tableCards,
		CurrentBet: s.CurrentBet, WaitingForMaoDeOnze: s.WaitingForMaoDeOnze,
		LegalActions: s.LegalActions, Reward: s.Reward, Winner: s.Winner,
	}
	data, _ := json.Marshal(view)
	return string(data)
}
func main() {}
