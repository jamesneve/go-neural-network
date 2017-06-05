package network

type Synapse struct {
	Weight float64
	In     float64
	Out    float64
}

func NewSynapse(weight float64) *Synapse {
	return &Synapse{Weight: weight}
}

func NewSynapseFromTo(from, to *SigmoidNeuron, weight float64) *Synapse {
	syn := NewSynapse(weight)

	from.OutSynapses = append(from.OutSynapses, syn)
	to.InSynapses = append(to.InSynapses, syn)

	return syn
}

func (s *Synapse) Trigger(value float64) {
	s.In = value
	s.Out = s.In * s.Weight
}
