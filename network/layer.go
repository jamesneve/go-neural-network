package network

type Layer struct {
	Neurons []*SigmoidNeuron
}

func NewLayer(neurons int) *Layer {
	l := &Layer{}
	l.init(neurons)
	return l
}

func (l *Layer) init(neurons int) {
	n := make([]*SigmoidNeuron, 0, neurons)

	for i := 0; i < neurons; i++ {
		n = append(n, new(SigmoidNeuron))
	}

	l.Neurons = n
}

func (l *Layer) ConnectTo(layer *Layer) {
	for _, n := range l.Neurons {
		for _, toN := range layer.Neurons {
			n.CreateSynapseTo(toN, 0)
		}
	}
}

func (l *Layer) CalculateNewOutputs() {
	for i := range l.Neurons {
		l.Neurons[i].CalculateAndSendOutput()
	}
}