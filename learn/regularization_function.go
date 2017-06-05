package learn

type RegularizationFunction interface {
	CalculateNewWeight(eta, lambda, weight, weightNabla, miniBatchLength, trainingDataLength float64) float64
}

// -------

var NoRegularization RegularizationFunction = newNoRegularization()

type noRegularization struct{}

func newNoRegularization() RegularizationFunction {
	r := noRegularization{}
	return RegularizationFunction(&r)
}

func (r *noRegularization) CalculateNewWeight(eta, lambda, weight, weightNabla, miniBatchLength, trainingDataLength float64) float64 {
	return weight - ((eta / miniBatchLength) * weightNabla)
}

// -------

var L1Regularization RegularizationFunction = newL1Regularization()

type l1Regularization struct{}

func newL1Regularization() RegularizationFunction {
	r := l1Regularization{}
	return RegularizationFunction(&r)
}

func (r *l1Regularization) CalculateNewWeight(eta, lambda, weight, weightNabla, miniBatchLength, trainingDataLength float64) float64 {
	var sign float64 = 1
	if weight < 0 {
		sign = -1
	}
	return weight - sign*eta*(lambda/trainingDataLength) - ((eta / miniBatchLength) * weightNabla)
}

// -------

var L2Regularization RegularizationFunction = newL2Regularization()

type l2Regularization struct{}

func newL2Regularization() RegularizationFunction {
	r := l2Regularization{}
	return RegularizationFunction(&r)
}

func (r *l2Regularization) CalculateNewWeight(eta, lambda, weight, weightNabla, miniBatchLength, trainingDataLength float64) float64 {
	return (1-eta*(lambda/trainingDataLength))*weight - ((eta / miniBatchLength) * weightNabla)
}
