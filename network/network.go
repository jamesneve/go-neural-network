package network

import (
	"math/rand"
	"time"
	"math"
)

type Network struct {
	Entries []*InputNeuron
	Layers []*Layer
	Out []float64
}

func NewNetwork(in int, layers []int) *Network {
	n := &Network{
		Entries: make([]*InputNeuron, 0, in),
		Layers: make([]*Layer, 0, len(layers)),
	}
	n.init(in, layers)
	return n
}

func (n *Network) init(in int, layers []int) {
	n.initLayers(layers)
	n.initInputNeurons(in)
	n.ConnectLayers()
	n.ConnectInputNeurons()
	n.RandomizeWeightsAndBiases()
}

func (n *Network) initLayers(layers []int) {
	for _, count := range layers {
		layer := NewLayer(count)
		n.Layers = append(n.Layers, layer)
	}
}

func (n *Network) initInputNeurons(in int) {
	for ; in > 0; in-- {
		e := NewInputNeuron()
		n.Entries = append(n.Entries, e)
	}
}

func (n *Network) ConnectLayers() {
	for i := len(n.Layers) - 1; i > 0; i-- {
		n.Layers[i - 1].ConnectTo(n.Layers[i])
	}
}

func (n *Network) ConnectInputNeurons() {
	for _, e := range n.Entries {
		e.ConnectTo(*n.Layers[0])
	}
}

func (n *Network) setInputNeurons(v *[]float64) {
	values := *v
	if len(values) != len(n.Entries) {
		panic("Values and inputs don't match")
	}

	for i, e := range n.Entries {
		e.Input = values[i]
	}
}

func (n *Network) triggerInputNeurons() {
	for _, e := range n.Entries {
		e.Trigger()
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
	n.setInputNeurons(&entries)
	n.triggerInputNeurons()
	n.calculateLayers()
	return n.generateOut()
}

func (n *Network) RandomizeWeightsAndBiases() {
	rand.Seed(time.Now().UnixNano())

	for _, l := range n.Layers {
		for _, n := range l.Neurons {
			for _, s := range n.InSynapses {
				// Initial weight SD is 1/root(n) to help avoid early saturation
				s.Weight = rand.NormFloat64() / math.Sqrt(float64(len(n.InSynapses)))
			}
			n.Bias = rand.NormFloat64()
		}
	}
}