package layers

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type LinearLayer struct {
	weights   mat.Matrix
	n_inputs  int
	n_outputs int
}

func (layer LinearLayer) Initialize(numInputs int, numOutputs int) Layer {
	data := make([]float64, (numInputs+1)*numOutputs)
	for i := range data {
		data[i] = rand.NormFloat64()
	}
	layer.weights = mat.NewDense(numOutputs, numInputs+1, data)

	layer.n_inputs = numInputs + 1
	layer.n_outputs = numOutputs

	return layer
}

func (layer LinearLayer) Pass(input []float64) []float64 {
	output := mat.NewVecDense(layer.n_outputs, nil)
	output.MulVec(layer.weights, mat.NewVecDense(layer.n_inputs, input))

	outputSlice := make([]float64, layer.n_outputs+1)
	for i := 0; i < layer.n_outputs; i++ {
		outputSlice[i] = output.AtVec(i)
	}
	outputSlice[layer.n_outputs] = 1
	return outputSlice
}

func (layer LinearLayer) Back(forwardGradients []float64) (shifts mat.Matrix, backwardsPass []float64) {
	return nil, nil
}
