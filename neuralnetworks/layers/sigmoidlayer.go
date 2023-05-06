package layers

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type SigmoidLayer struct {
	n_inputs int
}

func (layer *SigmoidLayer) Initialize(n_inputs int, _ int) { layer.n_inputs = n_inputs }

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func (layer *SigmoidLayer) Pass(input []float64) []float64 {
	output := make([]float64, layer.n_inputs+1)
	for i := 0; i < layer.n_inputs; i++ {
		output[i] = sigmoid(input[i])
	}
	output[layer.n_inputs] = 1
	return output
}

func (layer *SigmoidLayer) Back(forwardGradients mat.Vector) (shifts mat.Matrix, backwardsPass mat.Vector) {
	return nil, nil
}
