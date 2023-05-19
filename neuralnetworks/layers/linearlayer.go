package layers

import (
	"fmt"
	"math/rand"

	"github.com/EganBoschCodes/lossless/neuralnetworks/save"
	"github.com/EganBoschCodes/lossless/utils"

	"gonum.org/v1/gonum/mat"
)

type LinearLayer struct {
	Outputs int

	weights  mat.Matrix
	biases   mat.Matrix
	n_inputs int

	initialized bool
}

func (layer *LinearLayer) Initialize(numInputs int) {
	layer.n_inputs = numInputs
	if layer.weights != nil {
		return
	}

	if layer.Outputs == 0 {
		panic("You must specify how many Outputs a LinearLayer has!")
	}

	// Use Xavier Initialization on the weights
	fan_avg := (float64(numInputs) + float64(layer.Outputs)) / 2
	initialWeights := make([]float64, numInputs*layer.Outputs)
	for i := range initialWeights {
		initialWeights[i] = rand.NormFloat64() / fan_avg
	}

	initialBiases := make([]float64, layer.Outputs)
	for i := range initialBiases {
		initialBiases[i] = rand.NormFloat64() / fan_avg
	}

	layer.weights = mat.NewDense(layer.Outputs, numInputs, initialWeights)
	layer.biases = mat.NewDense(layer.Outputs, 1, initialBiases)

	layer.initialized = true
}

func (layer *LinearLayer) Pass(input mat.Matrix) mat.Matrix {
	// Multiply by weights
	output := mat.NewDense(layer.Outputs, 1, nil)
	output.Mul(layer.weights, input)

	// Add biases
	output.Add(output, layer.biases)
	return output
}

func (layer *LinearLayer) Back(inputs mat.Matrix, _ mat.Matrix, forwardGradients mat.Matrix) (ShiftType, mat.Matrix) {
	inputSize, _ := inputs.Dims()
	gradSize, _ := forwardGradients.Dims()

	shift := mat.NewDense(gradSize, inputSize, nil)
	shift.Mul(forwardGradients, inputs.T())

	newGradient := mat.NewDense(inputSize, 1, nil)
	newGradient.Mul(layer.weights.T(), forwardGradients)

	return &WeightShift{weightShift: shift, biasShift: forwardGradients}, newGradient
}

func (layer *LinearLayer) NumOutputs() int {
	return layer.Outputs
}

func (layer *LinearLayer) ToBytes() []byte {
	weightSlice, biasSlice := save.ToBytes(utils.GetSlice(layer.weights)), save.ToBytes(utils.GetSlice(layer.biases))
	saveBytes := save.ConstantsToBytes(layer.Outputs, len(weightSlice)/8)

	saveBytes = append(saveBytes, weightSlice...)
	saveBytes = append(saveBytes, biasSlice...)
	return saveBytes
}

func (layer *LinearLayer) FromBytes(bytes []byte) {
	constInts, weightsSlice := save.ConstantsFromBytes(bytes[:8]), save.FromBytes(bytes[8:])
	layer.Outputs = constInts[0]
	weightLength := constInts[1]

	layer.weights = mat.NewDense(layer.Outputs, weightLength/layer.Outputs, weightsSlice[:weightLength])
	layer.biases = mat.NewDense(layer.Outputs, 1, weightsSlice[weightLength:])
}

func (layer *LinearLayer) PrettyPrint() string {
	ret := fmt.Sprintf("Linear Layer\n%d Inputs -> %d Outputs\n\n", layer.n_inputs, layer.Outputs)
	return ret + fmt.Sprintf("weights =\n%s", utils.JSify(layer.weights))
}
