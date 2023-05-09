package layers

import (
	"gonum.org/v1/gonum/mat"
)

type ReluLayer struct {
	n_inputs int
}

func (layer *ReluLayer) Initialize(n_inputs int) {
	layer.n_inputs = n_inputs
}

func (layer *ReluLayer) Pass(input []float64) []float64 {
	for i := 0; i < layer.n_inputs; i++ {
		if input[i] < 0 {
			input[i] = 0
		}
	}
	return input
}

func (layer *ReluLayer) Back(inputs []float64, outputs []float64, forwardGradients mat.Matrix) (mat.Matrix, mat.Matrix) {
	rows, _ := forwardGradients.Dims()
	for i := 0; i < rows; i++ {
		if inputs[i] < 0 {
			forwardGradients.(*mat.Dense).Set(i, 0, 0)
		}
	}
	return nil, forwardGradients
}

func (layer *ReluLayer) GetShape() mat.Matrix {
	return nil
}

func (layer *ReluLayer) ApplyShift(shift mat.Matrix, scale float64) {}

func (layer *ReluLayer) NumOutputs() int {
	return layer.n_inputs
}
