package layers

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type SigmoidLayer struct {
	n_inputs int
}

func (layer SigmoidLayer) Initialize(n_inputs int, _ int) Layer {
	layer.n_inputs = n_inputs
	return layer
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func (layer SigmoidLayer) Pass(input []float64) []float64 {
	for i := 0; i < layer.n_inputs; i++ {
		input[i] = sigmoid(input[i])
	}
	return input
}

func (layer SigmoidLayer) Back(forwardGradients []float64) (shifts mat.Matrix, backwardsPass []float64) {
	return nil, nil
}
