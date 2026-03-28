package main

/*
#include <stdlib.h>
*/
import "C"
import "encoding/json"

// InitGameFull creates a new game and returns the full GameState JSON
// (not the View). This is needed by CFR training which requires access
// to both players' hands for tree traversal.
//
//export InitGameFull
func InitGameFull() *C.char {
	gameState = createNewGame()
	data, _ := json.Marshal(gameState)
	return C.CString(string(data))
}
