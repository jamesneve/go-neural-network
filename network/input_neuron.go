package network

type InputNeuron struct {
	OutSynapses []*Synapse
	Input float64
}

func NewInputNeuron() *InputNeuron {
	return &InputNeuron{}
}

func (e *InputNeuron) ConnectTo(layer Layer) {
	for _, n := range layer.Neurons {
		e.CreateSynapseTo(n, 0)
	}
}

func (e *InputNeuron) CreateSynapseTo(nTo *SigmoidNeuron, weight float64) {
	syn := NewSynapse(weight)

	e.OutSynapses = append(e.OutSynapses, syn)
	nTo.InSynapses = append(nTo.InSynapses, syn)
}

func (e *InputNeuron) Trigger() {
	for _, s := range e.OutSynapses {
		s.Trigger(e.Input)
	}
}