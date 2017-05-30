package network

import (
	"math"
)

type Neuron struct {
	InSynapses []*Synapse
	OutSynapses []*Synapse
	Bias float64
	Out float64
}

func (n *Neuron) SynapseTo(nTo *Neuron, weight float64) {
	NewSynapseFromTo(n, nTo, weight)
}

func (n *Neuron) CalculateWeightedInput() float64 {
	var sum float64

	for _, s := range n.InSynapses {
		sum += s.Out
	}

	sum += n.Bias

	return sum
}

func (n *Neuron) CalculateOutput() float64 {
	z := n.CalculateWeightedInput()

	out := Sigmoid(z)

	return out
}

func (n *Neuron) CalculateAndSignalOutput() {
	n.Out = n.CalculateOutput()

	for _, s := range n.OutSynapses {
		s.Signal(n.Out)
	}
}

func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func SigmoidPrime(z float64) float64 {
	return Sigmoid(z) * (1 - Sigmoid(z))
}