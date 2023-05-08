package layers

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type SigmoidLayer struct {
	n_inputs int
}

func (layer *SigmoidLayer) Initialize(n_inputs int, _ int) {
	layer.n_inputs = n_inputs
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func (layer *SigmoidLayer) Pass(input []float64) []float64 {
	for i := 0; i < layer.n_inputs; i++ {
		input[i] = sigmoid(input[i])
	}
	return input
}

func (layer *SigmoidLayer) Back(inputs []float64, outputs []float64, forwardGradients mat.Matrix) (mat.Matrix, mat.Matrix) {
	rows, _ := forwardGradients.Dims()
	newGradients := make([]float64, rows)
	for i := 0; i < rows; i++ {
		newGradients[i] = outputs[i] * (1 - outputs[i]) * forwardGradients.At(i, 0)
	}
	return nil, mat.NewDense(rows, 1, newGradients)
}

func (layer *SigmoidLayer) GetShape() mat.Matrix {
	return nil
}

func (layer *SigmoidLayer) ApplyShift(shift mat.Matrix, scale float64) {}
