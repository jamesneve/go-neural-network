package network

import (
	"math/rand"
	"time"
	"fmt"
	"github.com/jamesneve/neuralnetwork2/learn"
)

type Network struct {
	Entries []*Entry
	Layers []*Layer
	Out []float64
}

func NewNetwork(in int, layers []int) *Network {
	n := &Network{
		Entries: make([]*Entry, 0, in),
		Layers: make([]*Layer, 0, len(layers)),
	}
	n.init(in, layers)
	return n
}

func (n *Network) init(in int, layers []int) {
	n.initLayers(layers)
	n.initEntries(in)
	n.ConnectLayers()
	n.ConnectEntries()
}

func (n *Network) initLayers(layers []int) {
	for _, count := range layers {
		layer := NewLayer(count)
		n.Layers = append(n.Layers, layer)
	}
}

func (n *Network) initEntries(in int) {
	for ; in > 0; in-- {
		e := NewEntry()
		n.Entries = append(n.Entries, e)
	}
}

func (n *Network) ConnectLayers() {
	for i := len(n.Layers) - 1; i > 0; i-- {
		n.Layers[i - 1].ConnectTo(n.Layers[i])
	}
}

func (n *Network) ConnectEntries() {
	for _, e := range n.Entries {
		e.ConnectTo(*n.Layers[0])
	}
}

func (n *Network) setEntries(v *[]float64) {
	values := *v
	if len(values) != len(n.Entries) {
		panic("Values and inputs don't match")
	}

	for i, e := range n.Entries {
		e.Input = values[i]
	}
}

func (n *Network) sendEntries() {
	for _, e := range n.Entries {
		e.Signal()
	}
}

func (n *Network) calculateLayers() {
	for _, l := range n.Layers {
		l.CalculateNewOutputs()
	}
}

func (n *Network) generateOut() []float64 {
	outL := n.Layers[len(n.Layers) - 1]
	n.Out = make([]float64, len(outL.Neurons))

	for i, neuron := range outL.Neurons {
		n.Out[i] = neuron.Out
	}

	return n.Out
}

func (n *Network) CalculateOutput(entries []float64) []float64 {
	n.setEntries(&entries)
	n.sendEntries()
	n.calculateLayers()
	out := n.generateOut()

	return out
}

func (n *Network) RandomizeSynapses() {
	rand.Seed(time.Now().UnixNano())

	for _, l := range n.Layers {
		for _, n := range l.Neurons {
			for _, s := range n.InSynapses {
				s.Weight = 2 * (rand.Float64() - 0.5)
			}
			n.Bias = rand.NormFloat64()
		}
	}
}

func (n *Network) TrainByGradientDescent(trainingData []learn.TrainingData, epochs, miniBatchSize int, eta float64, testData []learn.TrainingData) {
	for i := 0; i < epochs; i++ {
		currentTrainingData := learn.ShuffleTrainingData(trainingData)
		batchCount := len(trainingData) / miniBatchSize
		for j := 0; j < batchCount; j++ {
			batch := currentTrainingData[j * miniBatchSize : (j + 1) * miniBatchSize]
			n.UpdateMiniBatch(batch, eta)
		}

		correct := n.Evaluate(testData)
		fmt.Println("Epoch", i + 1, ":", correct, "/", len(testData))
	}
}

func (n *Network) UpdateMiniBatch(miniBatch []learn.TrainingData, eta float64) {
	nablaB, nablaW := n.initializeZeroBiasedWeights()
	learningRate := eta / float64(len(miniBatch))

	for _, trainingDatum := range miniBatch {
		deltaNablaB, deltaNablaW := n.Backpropagation(trainingDatum.TrainingInput, trainingDatum.DesiredOutputs, eta)
		for i := range nablaW {
			for j := range nablaW[i] {
				nablaB[i][j] += deltaNablaB[i][j]
				for k := range nablaW[i][j] {
					nablaW[i][j][k] += deltaNablaW[i][j][k]
				}
			}
		}
	}
	fmt.Println(nablaB, nablaW)

	for i := range n.Layers {
		for j := range n.Layers[i].Neurons {
			n.Layers[i].Neurons[j].Bias = (n.Layers[i].Neurons[j].Bias - learningRate) * nablaB[i][j]
			for k := range n.Layers[i].Neurons[j].InSynapses {
				n.Layers[i].Neurons[j].InSynapses[k].Weight = (n.Layers[i].Neurons[j].InSynapses[k].Weight - learningRate) * nablaW[i][j][k]
			}
		}
	}

}

func (n *Network) initializeZeroBiasedWeights() ([][]float64, [][][]float64) {
	b := make([][]float64, len(n.Layers))
	w := make([][][]float64, len(n.Layers))
	for i := range n.Layers {
		b[i] = make([]float64, len(n.Layers[i].Neurons))
		w[i] = make([][]float64, len(n.Layers[i].Neurons))
		for j := range n.Layers[i].Neurons {
			w[i][j] = make([]float64, len(n.Layers[i].Neurons[j].InSynapses))
		}
	}

	return b, w
}

func (n *Network) Backpropagation(in, ideal []float64, speed float64) ([][]float64, [][][]float64) {
	// First, calculate the final output of the network for the training inputs
	n.CalculateOutput(in)

	// Make the array to calculate the error from each node
	nablaB, nablaW := n.initializeZeroBiasedWeights()

	// Make the array to hold the errors for the final layer
	last := len(n.Layers) - 1
	l := n.Layers[last]

	delta := n.CalculateFinalDelta(ideal)
	nablaB[last] = delta
	for i, n2 := range l.Neurons {
		for j := range n2.InSynapses {
			nablaW[last][i][j] = delta[i] * n.Layers[last - 1].Neurons[j].CalculateOutput()
		}
	}

	//// !! Backpropagation step
	for i := last - 1; i >= 0; i-- {
		delta = n.CalculateIntermediateDelta(i, delta)

		nablaB[i] = delta

		l := n.Layers[i]
		for j, n2 := range l.Neurons {
			for k := range n2.InSynapses {
				if i == 0 {
					nablaW[i][j][k] = delta[j] * n.Entries[k].Input
				} else {
					nablaW[i][j][k] = delta[j] * n.Layers[i - 1].Neurons[k].CalculateOutput()
				}
			}
		}
	}

	return nablaB, nablaW
}

func (n *Network) CalculateIntermediateDelta(layerNumber int, previousDelta []float64) []float64 {
	newDelta := make([]float64, len(n.Layers[layerNumber].Neurons))
	for i, neuron := range n.Layers[layerNumber].Neurons {
		sp := SigmoidPrime(neuron.CalculateWeightedInput())
		var sum float64
		for i, s := range neuron.OutSynapses {
			sum += previousDelta[i] * s.Weight
		}
		newDelta[i] = sp * sum
	}

	return newDelta
}

func (n *Network) CalculateFinalDelta(idealOutputs []float64) []float64 {
	last := len(n.Layers) - 1
	r := make([]float64, len(n.Layers[last].Neurons))
	for i, neuron := range n.Layers[last].Neurons {
		r[i] = (n.Out - idealOutputs[i]) * SigmoidPrime(neuron.CalculateWeightedInput())
	}
	return r
}

func (n *Network) Evaluate(testData []learn.TrainingData) int {
	correctResults := 0
	for _, t := range testData {
		output := n.CalculateOutput(t.TrainingInput)
		if n.ResultNumberFromOutput(output) == n.ResultNumberFromOutput(t.DesiredOutputs) {
			correctResults += 1
		}
	}

	return correctResults
}

func (n *Network) ResultNumberFromOutput(a []float64) int {
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