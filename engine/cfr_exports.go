package main

/*
#include <stdlib.h>
*/
import "C"
import (
	"encoding/json"
	"fmt"
)

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

//export CFRTrain
func CFRTrain(numIterations C.int, resumePath *C.char) *C.char {
	if cfrTables == nil {
		cfrTables = newCFRTables()
	}

	if resumePath != nil {
		rp := C.GoString(resumePath)
		if rp != "" {
			if err := cfrTables.load(rp); err != nil {
				return C.CString(fmt.Sprintf(`{"error": "failed to load: %s"}`, err))
			}
			fmt.Printf("[CFR-Go] Resumed from %s (%d iterations, %d info sets)\n",
				rp, cfrTables.Iterations, len(cfrTables.RegretSum))
		}
	}

	cfrTables.train(int(numIterations))
	return C.CString(fmt.Sprintf(`{"iterations": %d, "info_sets": %d}`,
		cfrTables.Iterations, len(cfrTables.RegretSum)))
}

//export CFRSave
func CFRSave(path *C.char) *C.char {
	if cfrTables == nil {
		return C.CString(`{"error": "no CFR tables to save"}`)
	}
	goPath := C.GoString(path)
	if err := cfrTables.save(goPath); err != nil {
		return C.CString(fmt.Sprintf(`{"error": "save failed: %s"}`, err))
	}
	return C.CString(fmt.Sprintf(`{"saved": "%s", "iterations": %d, "info_sets": %d}`,
		goPath, cfrTables.Iterations, len(cfrTables.RegretSum)))
}

//export CFRLoad
func CFRLoad(path *C.char) *C.char {
	if cfrTables == nil {
		cfrTables = newCFRTables()
	}
	goPath := C.GoString(path)
	if err := cfrTables.load(goPath); err != nil {
		return C.CString(fmt.Sprintf(`{"error": "load failed: %s"}`, err))
	}
	return C.CString(fmt.Sprintf(`{"loaded": "%s", "iterations": %d, "info_sets": %d}`,
		goPath, cfrTables.Iterations, len(cfrTables.RegretSum)))
}
