package layers

import (
	"fmt"
	"math/rand"

	"github.com/EganBoschCodes/lossless/neuralnetworks/save"
	"github.com/EganBoschCodes/lossless/utils"

	"gonum.org/v1/gonum/mat"
)

/*
Linear (or Dense) layer type
*/
type VariableLinearLayer struct {
	InputSize  int
	OutputSize int

	ConstantLengthInput bool
	inputLength         int

	weights *mat.Dense
	biases  *mat.Dense
}

func (layer *VariableLinearLayer) Initialize(numInputs int) {
	if layer.ConstantLengthInput {
		layer.inputLength = numInputs
	}

	if layer.weights != nil {
		return
	}
	if layer.InputSize == 0 {
		panic("You must specify how many inputs a VariableLinearLayer has!")
	}
	if layer.OutputSize == 0 {
		panic("You must specify how many outputs a VariableLinearLayer has!")
	}

	// Use Xavier Initialization on the weights
	fan_avg := (float64(layer.InputSize) + float64(layer.OutputSize)) / 2
	initialWeights := make([]float64, layer.InputSize*layer.OutputSize)
	for i := range initialWeights {
		initialWeights[i] = rand.NormFloat64() / fan_avg
	}

	initialBiases := make([]float64, layer.OutputSize)
	for i := range initialBiases {
		initialBiases[i] = rand.NormFloat64() / fan_avg
	}

	layer.weights = mat.NewDense(layer.OutputSize, layer.InputSize, initialWeights)
	layer.biases = mat.NewDense(layer.OutputSize, 1, initialBiases)
}

func (layer *VariableLinearLayer) Pass(input *mat.Dense) (*mat.Dense, CacheType) {
	inputSlice := utils.GetSlice(input)
	outputSlice := make([]float64, 0)

	for i := 0; i < len(inputSlice); i += layer.InputSize {
		// Get the input chunk
		inputChunk := utils.FromSlice(inputSlice[i : i+layer.InputSize])

		// Multiply by weights
		outputChunk := mat.NewDense(layer.OutputSize, 1, nil)
		outputChunk.Mul(layer.weights, inputChunk)

		// Add biases
		outputChunk.Add(outputChunk, layer.biases)
		outputSlice = append(outputSlice, utils.GetSlice(outputChunk)...)
	}

	return utils.FromSlice(outputSlice), &InputCache{Input: input}
}

func (layer *VariableLinearLayer) Back(cache CacheType, forwardGradients *mat.Dense) (ShiftType, *mat.Dense) {
	inputs := cache.(*InputCache).Input
	inputSlice, gradientSlice := utils.GetSlice(inputs), utils.GetSlice(forwardGradients)

	weightShift, biasShift := mat.NewDense(layer.OutputSize, layer.InputSize, nil), mat.NewDense(layer.OutputSize, 1, nil)
	backwardPass := make([]float64, 0)
	for i := 0; i < len(inputSlice)/layer.InputSize; i++ {
		// Get the input chunk
		inputChunk, gradientChunk := utils.FromSlice(inputSlice[i*layer.InputSize:(i+1)*layer.InputSize]), utils.FromSlice(gradientSlice[i*layer.OutputSize:(i+1)*layer.OutputSize])

		chunkWeightShift := mat.NewDense(layer.OutputSize, layer.InputSize, nil)
		chunkWeightShift.Mul(gradientChunk, inputChunk.T())

		weightShift.Add(weightShift, chunkWeightShift)
		biasShift.Add(biasShift, gradientChunk)

		newGradient := mat.NewDense(layer.InputSize, 1, nil)
		newGradient.Mul(layer.weights.T(), gradientChunk)

		backwardPass = append(backwardPass, utils.GetSlice(newGradient)...)
	}

	return &WeightShift{weightShift: weightShift, biasShift: biasShift}, utils.FromSlice(backwardPass)
}

func (layer *VariableLinearLayer) NumOutputs() int {
	if layer.ConstantLengthInput {
		return layer.inputLength / layer.InputSize * layer.OutputSize
	}
	return -1
}

func (layer *VariableLinearLayer) ToBytes() []byte {
	weightSlice, biasSlice := save.ToBytes(utils.GetSlice(layer.weights)), save.ToBytes(utils.GetSlice(layer.biases))
	saveBytes := save.ConstantsToBytes(layer.InputSize, layer.OutputSize, utils.BoolToInt(layer.ConstantLengthInput), len(weightSlice)/8)

	saveBytes = append(saveBytes, weightSlice...)
	saveBytes = append(saveBytes, biasSlice...)
	return saveBytes
}

func (layer *VariableLinearLayer) FromBytes(bytes []byte) {
	constInts, weightsSlice := save.ConstantsFromBytes(bytes[:16]), save.FromBytes(bytes[16:])
	layer.InputSize, layer.OutputSize, layer.ConstantLengthInput = constInts[0], constInts[1], constInts[2] != 0
	weightLength := constInts[3]

	layer.weights = mat.NewDense(layer.OutputSize, weightLength/layer.OutputSize, weightsSlice[:weightLength])
	layer.biases = mat.NewDense(layer.OutputSize, 1, weightsSlice[weightLength:])
}

func (layer *VariableLinearLayer) PrettyPrint() string {
	ret := fmt.Sprintf("Variable Linear Layer\n%d Input Size -> %d Output Size\n\n", layer.InputSize, layer.OutputSize)
	return ret + fmt.Sprintf("weights =\n%s\n\nbiases =\n%s", utils.JSify(layer.weights), utils.JSify(layer.biases))
}
