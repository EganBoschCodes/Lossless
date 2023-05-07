package layers

import (
	"go-ml-library/utils"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type LinearLayer struct {
	weights   mat.Matrix
	n_inputs  int
	n_outputs int
}

func (layer *LinearLayer) Initialize(numInputs int, numOutputs int) {
	data := make([]float64, (numInputs+1)*numOutputs)
	for i := range data {
		data[i] = rand.NormFloat64()
	}
	layer.weights = mat.NewDense(numOutputs, numInputs+1, data)

	layer.n_inputs = numInputs + 1
	layer.n_outputs = numOutputs
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

func (layer *LinearLayer) Back(inputs []float64, outputs []float64, forwardGradients mat.Matrix) (mat.Matrix, mat.Matrix) {
	gradSize, _ := forwardGradients.Dims()
	shift := mat.NewDense(gradSize, len(inputs), nil)
	inputVec := mat.NewDense(1, len(inputs), inputs)

	utils.PrintMat("weights", layer.weights)

	shift.Mul(forwardGradients, inputVec)
	//utils.PrintMat("shift", shift)

	subweights := layer.weights.(*mat.Dense).Slice(0, gradSize, 0, len(inputs)-1).T()
	//utils.PrintMat("Subweights", subweights)
	//utils.PrintMat("forwardGradients", forwardGradients)

	newGradient := mat.NewDense(len(inputs)-1, 1, nil)
	newGradient.Mul(subweights, forwardGradients)

	//utils.PrintMat("NewGradient", newGradient)

	return shift, newGradient
}

func (layer *LinearLayer) ApplyShift(shift mat.Matrix, scale float64) {
	shift.(*mat.Dense).Scale(scale, shift)
	layer.weights.(*mat.Dense).Add(layer.weights, shift)
}
