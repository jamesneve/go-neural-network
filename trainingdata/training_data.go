package trainingdata

import (
	"math/rand"
	"time"
)

type TrainingData struct {
	TrainingInput []float64
	DesiredOutputs []float64
}

func ShuffleTrainingData(td []TrainingData) []TrainingData {
	rand.Seed(time.Now().UnixNano())

	res := make([]TrainingData, len(td))
	perm := rand.Perm(len(td))
	for i := range td {
		res[i] = td[perm[i]]
	}
	return res
}