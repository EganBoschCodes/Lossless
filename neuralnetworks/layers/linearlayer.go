package layers

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

type LinearLayer struct {
	weights   mat.Matrix
	n_inputs  int
	n_outputs int
}

func (layer *LinearLayer) Initialize(numInputs int, numOutputs int) {
	data := make([]float64, numInputs*numOutputs)
	for i := range data {
		//data[i] = rand.NormFloat64()
		data[i] = float64(i)
	}
	layer.weights = mat.NewDense(numOutputs, numInputs, data)

	layer.n_inputs = numInputs
	layer.n_outputs = numOutputs
	fmt.Println(layer.weights)
}

func (layer *LinearLayer) Pass(input []float64) []float64 {
	output := mat.NewVecDense(layer.n_outputs, nil)
	output.MulVec(layer.weights, mat.NewVecDense(layer.n_inputs, input))

	outputSlice := make([]float64, layer.n_outputs+1)
	for i := 0; i < layer.n_outputs; i++ {
		outputSlice[i] = output.AtVec(i)
	}
	outputSlice[layer.n_outputs] = 1
	return outputSlice
}

func (layer *LinearLayer) Back(forwardGradients mat.Vector) (shifts mat.Matrix, backwardsPass mat.Vector) {
	return nil, nil
}
