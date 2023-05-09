package layers

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type SoftmaxLayer struct {
	n_inputs int
}

func (layer *SoftmaxLayer) Initialize(n_inputs int) {
	layer.n_inputs = n_inputs
}

func (layer *SoftmaxLayer) Pass(input []float64) []float64 {
	expInput := make([]float64, len(input))

	// Preserve the bias
	expInput[len(input)-1] = 1

	// Calculate e^x for all inputs, and sum them at the same time
	expInputSum := 0.0
	for i := 0; i < len(input)-1; i++ {
		expVal := math.Exp(input[i])
		expInput[i] = expVal
		expInputSum += expVal
	}

	// Normalize all outputs based on all their sums
	for i := 0; i < len(input)-1; i++ {
		expInput[i] /= expInputSum
	}

	return expInput
}

func (layer *SoftmaxLayer) Back(inputs []float64, outputs []float64, forwardGradients mat.Matrix) (mat.Matrix, mat.Matrix) {
	rows, _ := forwardGradients.Dims()
	newGradients := make([]float64, rows)
	for i := 0; i < rows; i++ {
		newGradients[i] = outputs[i] * (1 - outputs[i]) * forwardGradients.At(i, 0)
	}
	return nil, mat.NewDense(rows, 1, newGradients)
}

func (layer *SoftmaxLayer) GetShape() mat.Matrix {
	return nil
}

func (layer *SoftmaxLayer) ApplyShift(shift mat.Matrix, scale float64) {}

func (layer *SoftmaxLayer) NumOutputs() int {
	return layer.n_inputs
}
