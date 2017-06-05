package learn

import (
	"fmt"
	"github.com/jamesneve/go-neural-network/network"
	"github.com/jamesneve/go-neural-network/trainingdata"
)

type NetworkTrainer struct {
	net                    *network.Network
	costFunction           CostFunction
	regularizationFunction RegularizationFunction
	eta                    float64
	lambda                 float64
	trainingData           []trainingdata.TrainingData
	reportResults          bool
}

func NewNetworkTrainer(nw *network.Network, trainingData []trainingdata.TrainingData, costFunction CostFunction, regularizationFunction RegularizationFunction, eta, lambda float64, reportResults bool) NetworkTrainer {
	return NetworkTrainer{
		net:                    nw,
		costFunction:           costFunction,
		regularizationFunction: regularizationFunction,
		eta:           eta,
		lambda:        lambda,
		trainingData:  trainingData,
		reportResults: reportResults,
	}
}

func (t *NetworkTrainer) TrainByGradientDescent(epochs, miniBatchSize int, testData []trainingdata.TrainingData) {
	for i := 0; i < epochs; i++ {
		currentTrainingData := trainingdata.ShuffleTrainingData(t.trainingData)
		batchCount := len(t.trainingData) / miniBatchSize
		for j := 0; j < batchCount; j++ {
			batch := currentTrainingData[j*miniBatchSize : (j+1)*miniBatchSize]
			t.UpdateMiniBatch(batch)
		}

		if t.reportResults {
			correct := t.Evaluate(testData)
			fmt.Println("Epoch", i+1, ":", correct, "/", len(testData))
		}
	}
}

func (t *NetworkTrainer) UpdateMiniBatch(miniBatch []trainingdata.TrainingData) {
	nablaB, nablaW := t.initializeZeroBiasedWeights()

	for _, trainingDatum := range miniBatch {
		deltaNablaB, deltaNablaW := t.Backpropagation(trainingDatum.TrainingInput, trainingDatum.DesiredOutputs)
		for i := range nablaW {
			for j := range nablaW[i] {
				nablaB[i][j] += deltaNablaB[i][j]
				for k := range nablaW[i][j] {
					nablaW[i][j][k] += deltaNablaW[i][j][k]
				}
			}
		}
	}

	for i := range t.net.Layers {
		for j := range t.net.Layers[i].Neurons {
			t.net.Layers[i].Neurons[j].Bias = t.net.Layers[i].Neurons[j].Bias - ((t.eta / float64(len(miniBatch))) * nablaB[i][j])
			for k := range t.net.Layers[i].Neurons[j].InSynapses {
				newWeight := t.regularizationFunction.CalculateNewWeight(t.eta, t.lambda, t.net.Layers[i].Neurons[j].InSynapses[k].Weight, nablaW[i][j][k], float64(len(miniBatch)), float64(len(t.trainingData)))
				t.net.Layers[i].Neurons[j].InSynapses[k].Weight = newWeight
			}
		}
	}
}

func (t *NetworkTrainer) initializeZeroBiasedWeights() ([][]float64, [][][]float64) {
	b := make([][]float64, len(t.net.Layers))
	w := make([][][]float64, len(t.net.Layers))
	for i := range t.net.Layers {
		b[i] = make([]float64, len(t.net.Layers[i].Neurons))
		w[i] = make([][]float64, len(t.net.Layers[i].Neurons))
		for j := range t.net.Layers[i].Neurons {
			w[i][j] = make([]float64, len(t.net.Layers[i].Neurons[j].InSynapses))
		}
	}

	return b, w
}

func (t *NetworkTrainer) Backpropagation(in, ideal []float64) ([][]float64, [][][]float64) {
	// First, calculate the final output of the network for the training inputs
	t.net.CalculateOutput(in)

	// Make the array to calculate the error from each node
	nablaB, nablaW := t.initializeZeroBiasedWeights()

	// Make the array to hold the errors for the final layer
	last := len(t.net.Layers) - 1
	l := t.net.Layers[last]

	delta := t.costFunction.CalculateDelta(l, t.net.Out, ideal)

	nablaB[last] = delta
	for i, n2 := range l.Neurons {
		for j := range n2.InSynapses {
			nablaW[last][i][j] = delta[i] * t.net.Layers[last-1].Neurons[j].CalculateOutput()
		}
	}

	//// !! Backpropagation step
	for i := last - 1; i >= 0; i-- {
		delta = t.CalculateIntermediateDelta(i, delta)

		nablaB[i] = delta

		l := t.net.Layers[i]
		for j, n2 := range l.Neurons {
			for k := range n2.InSynapses {
				if i != 0 {
					nablaW[i][j][k] = delta[j] * t.net.Layers[i-1].Neurons[k].CalculateOutput()
				} else {
					nablaW[i][j][k] = delta[j] * t.net.InputNeurons[k].Input
				}
			}
		}
	}

	return nablaB, nablaW
}

func (t *NetworkTrainer) CalculateIntermediateDelta(layerNumber int, previousDelta []float64) []float64 {
	newDelta := make([]float64, len(t.net.Layers[layerNumber].Neurons))
	for i, neuron := range t.net.Layers[layerNumber].Neurons {
		sp := neuron.CalculateOutputDelta()
		var sum float64
		for i, s := range neuron.OutSynapses {
			sum += previousDelta[i] * s.Weight
		}
		newDelta[i] = sp * sum
	}

	return newDelta
}

func (t *NetworkTrainer) Evaluate(testData []trainingdata.TrainingData) int {
	correctResults := 0
	for _, td := range testData {
		output := t.net.CalculateOutput(td.TrainingInput)
		if t.ResultNumberFromOutput(output) == t.ResultNumberFromOutput(td.DesiredOutputs) {
			correctResults += 1
		}
	}

	return correctResults
}

func (t *NetworkTrainer) ResultNumberFromOutput(a []float64) int {
	var max float64
	var maxval int
	for i := range a {
		if a[i] > max {
			max = a[i]
			maxval = i
		}
	}
	return maxval
}
