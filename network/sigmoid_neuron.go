package network

import (
	"math"
)

type SigmoidNeuron struct {
	InSynapses  []*Synapse
	OutSynapses []*Synapse
	Bias        float64
	Out         float64
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

func (n *SigmoidNeuron) CalculateOutputDelta() float64 {
	z := n.CalculateWeightedInput()

	return n.sigmoidPrime(z)
}

func (n *SigmoidNeuron) CalculateOutput() float64 {
	z := n.CalculateWeightedInput()

	return n.sigmoid(z)
}

func (n *SigmoidNeuron) CalculateAndSendOutput() {
	n.Out = n.CalculateOutput()

	for _, s := range n.OutSynapses {
		s.Trigger(n.Out)
	}
}

func (n *SigmoidNeuron) sigmoidPrime(z float64) float64 {
	return n.sigmoid(z) * (1 - n.sigmoid(z))
}

func (n *SigmoidNeuron) sigmoid(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}
