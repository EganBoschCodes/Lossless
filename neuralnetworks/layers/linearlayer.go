package layers

import (
	"go-ml-library/utils"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type LinearLayer struct {
	weights mat.Matrix
	Outputs int
}

func (layer *LinearLayer) Initialize(numInputs int) {
	numOutputs := layer.Outputs
	data := make([]float64, (numInputs+1)*numOutputs)
	for i := range data {
		data[i] = rand.NormFloat64() / 10
	}
	layer.weights = mat.NewDense(numOutputs, numInputs+1, data)
}

func (layer *LinearLayer) Pass(input mat.Matrix) mat.Matrix {
	// Add the bias term
	ir, _ := input.Dims()
	input = input.(*mat.Dense).Grow(1, 0)
	input.(*mat.Dense).Set(ir, 0, 1)

	// Multiply by weights
	output := mat.NewDense(layer.Outputs, 1, nil)
	output.Mul(layer.weights, input)
	return output
}

func (layer *LinearLayer) Back(inputs mat.Matrix, _ mat.Matrix, forwardGradients mat.Matrix) (mat.Matrix, mat.Matrix) {
	inputSize, _ := inputs.Dims()
	inputSize += 1

	gradSize, _ := forwardGradients.Dims()
	shift := mat.NewDense(gradSize, inputSize, nil)

	inputSlice := inputs.(*mat.Dense).RawMatrix().Data
	inputSlice = append(inputSlice, 1.0)
	inputVec := mat.NewDense(1, inputSize, inputSlice)

	shift.Mul(forwardGradients, inputVec)

	subweights := layer.weights.(*mat.Dense).Slice(0, gradSize, 0, inputSize-1).T()

	newGradient := mat.NewDense(inputSize-1, 1, nil)
	newGradient.Mul(subweights, forwardGradients)

	return shift, newGradient
}

func (layer *LinearLayer) GetShape() mat.Matrix {
	return utils.DenseLike(layer.weights)
}

func (layer *LinearLayer) ApplyShift(shift mat.Matrix, scale float64) {
	shift.(*mat.Dense).Scale(scale, shift)
	layer.weights.(*mat.Dense).Add(layer.weights, shift)
}

func (layer *LinearLayer) NumOutputs() int {
	return layer.Outputs
}
