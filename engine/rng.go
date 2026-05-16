package main

// #include <stdlib.h>
import "C"

import (
	"encoding/json"
	"math/rand"
	"time"
)

// engineRng drives the live game stream: deck shuffles in InitGame /
// dealNewHand, and CFR sampling during training. Its state is what locks
// the per-hand deal under SeedEngine.
var engineRng *rand.Rand = rand.New(rand.NewSource(time.Now().UnixNano()))

// rolloutRng drives MCTS rollouts (RolloutFromState). It lives on a
// SEPARATE stream from engineRng so MCTS thinking does not perturb the
// deck-shuffle sequence — both streams are reproducible from the same
// SeedEngine seed, but consuming from one does not advance the other.
// This guarantees that hand N of game N has the same deal regardless of
// whether MCTS or any other agent is playing.
var rolloutRng *rand.Rand = rand.New(rand.NewSource(time.Now().UnixNano() ^ rolloutSeedMix))

// Arbitrary 64-bit constant (frac bits of sqrt(2), per SHA-2) used to
// derive an independent stream seed from the same master.
const rolloutSeedMix int64 = 0x6A09E667F3BCC908

// useRolloutRng is a single-threaded context flag set by StepFromState and
// RolloutFromState. While true, gameRng() returns rolloutRng so deck
// shuffles triggered by mid-search hand transitions stay on the search
// stream and do not perturb the live engineRng sequence.
var useRolloutRng bool

// gameRng returns the RNG appropriate for the current call context: the
// search stream during StepFromState / RolloutFromState, otherwise the
// live engine stream.
func gameRng() *rand.Rand {
	if useRolloutRng {
		return rolloutRng
	}
	return engineRng
}

//export SeedEngine
func SeedEngine(seed C.longlong) *C.char {
	engineRng = rand.New(rand.NewSource(int64(seed)))
	rolloutRng = rand.New(rand.NewSource(int64(seed) ^ rolloutSeedMix))
	resp, _ := json.Marshal(map[string]int64{"seed": int64(seed)})
	return C.CString(string(resp))
}
