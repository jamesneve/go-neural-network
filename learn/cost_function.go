package learn

type CostFunction interface {
	Cost(z float64) float64
	CostDerivative(z float64) float64
}
//
//// -------
//
//type quadraticCost struct {}
//
//func NewQuadraticCost() CostFunction {
//	q := quadraticCost{}
//	return CostFunction(&q)
//}
//
//func (q *quadraticCost) Cost(z float64) float64 {
//	return 1.0 / (1.0 + math.Exp(-z))
//}
//
//func (q *quadraticCost) CostDerivative(z float64) float64 {
//	return q.Cost(z) * (1 - q.Cost(z))
//}
//
//// -------
//
//type crossEntropy struct {}
//
//func NewCrossEntropy() CostFunction {
//	c := crossEntropy{}
//	return CostFunction(&c)
//}
//
//func (c *crossEntropy) Cost(z float64) float64 {
//
//}