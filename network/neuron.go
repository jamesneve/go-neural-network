package network

import (
	"math"
)

type SigmoidNeuron struct {
	InSynapses []*Synapse
	OutSynapses []*Synapse
	Bias float64
	Out float64
}

func (n *SigmoidNeuron) CreateSynapseTo(nTo *SigmoidNeuron, weight float64) {
	NewSynapseFromTo(n, nTo, weight)
}

func (n *SigmoidNeuron) CalculateWeightedInput() float64 {
	var sum float64

	for _, s := range n.InSynapses {
		sum += s.Out
	}

	sum += n.Bias

	return sum
}

func (n *SigmoidNeuron) CalculateOutput() float64 {
	z := n.CalculateWeightedInput()

	out := Sigmoid(z)

	return out
}

func (n *SigmoidNeuron) CalculateAndSendOutput() {
	n.Out = n.CalculateOutput()

	for _, s := range n.OutSynapses {
		s.Trigger(n.Out)
	}
}

func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func SigmoidPrime(z float64) float64 {
	return Sigmoid(z) * (1 - Sigmoid(z))
}