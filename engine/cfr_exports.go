package main

/*
#include <stdlib.h>
*/
import "C"
import "encoding/json"

//export InitGameFull
func InitGameFull() *C.char {
	gameState = createNewGame()
	data, _ := json.Marshal(gameState)
	return C.CString(string(data))
}

//export InitGameFromScore
func InitGameFromScore(scoreP0 C.int, scoreP1 C.int, handStarter C.int) *C.char {
	gs := createNewGame()
	gs.Score = [2]int{int(scoreP0), int(scoreP1)}
	gs.HandStarter = int(handStarter)
	gs.CurrentPlayer = int(handStarter)

	if gs.Score[0] == 11 && gs.Score[1] < 11 {
		gs.WaitingForMaoDeOnze = true
		gs.CurrentPlayer = 0
		gs.LegalActions = []int{4, 5}
	} else if gs.Score[1] == 11 && gs.Score[0] < 11 {
		gs.WaitingForMaoDeOnze = true
		gs.CurrentPlayer = 1
		gs.LegalActions = []int{4, 5}
	} else {
		gs.updateLegalActions()
	}

	gameState = gs
	data, _ := json.Marshal(gs)
	return C.CString(string(data))
}
