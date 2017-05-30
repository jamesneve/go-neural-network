package network

type Entry struct {
	OutSynapses []*Synapse
	Input float64
}

func NewEntry() *Entry {
	return &Entry{}
}

func (e *Entry) ConnectTo(layer Layer) {
	for _, n := range layer.Neurons {
		e.SynapseTo(n, 0)
	}
}

func (e *Entry) SynapseTo(nTo *Neuron, weight float64) {
	syn := NewSynapse(weight)

	e.OutSynapses = append(e.OutSynapses, syn)
	nTo.InSynapses = append(nTo.InSynapses, syn)
}

func (e *Entry) Signal() {
	for _, s := range e.OutSynapses {
		s.Signal(e.Input)
	}
}