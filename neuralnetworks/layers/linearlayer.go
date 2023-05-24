package layers

import (
	"fmt"
	"math/rand"

	"github.com/EganBoschCodes/lossless/neuralnetworks/optimizers"
	"github.com/EganBoschCodes/lossless/neuralnetworks/save"
	"github.com/EganBoschCodes/lossless/utils"

	"gonum.org/v1/gonum/mat"
)

/*
Linear (or Dense) layer type
*/
type LinearLayer struct {
	Outputs int

	weights  *mat.Dense
	biases   *mat.Dense
	n_inputs int
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
}

func (layer *LinearLayer) Pass(input *mat.Dense) (*mat.Dense, CacheType) {
	// Multiply by weights
	output := mat.NewDense(layer.Outputs, 1, nil)
	output.Mul(layer.weights, input)

	// Add biases
	output.Add(output, layer.biases)
	return output, &InputCache{Input: input}
}

func (layer *LinearLayer) Back(cache CacheType, forwardGradients *mat.Dense) (ShiftType, *mat.Dense) {
	inputs := cache.(*InputCache).Input

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

/*
ShiftType used by LinearLayers
*/
type WeightShift struct {
	weightShift *mat.Dense
	biasShift   *mat.Dense
}

func (w *WeightShift) Apply(layer Layer, opt optimizers.Optimizer, scale float64) {
	w.weightShift, w.biasShift = opt.Rescale(w.weightShift), opt.Rescale(w.biasShift)
	w.weightShift.Scale(scale, w.weightShift)
	w.biasShift.Scale(scale, w.biasShift)

	layer.(*LinearLayer).weights.Add(layer.(*LinearLayer).weights, w.weightShift)
	layer.(*LinearLayer).biases.Add(layer.(*LinearLayer).biases, w.biasShift)
}

func (w *WeightShift) Combine(w2 ShiftType) ShiftType {
	w.weightShift.Add(w.weightShift, w2.(*WeightShift).weightShift)
	w.biasShift.Add(w.biasShift, w2.(*WeightShift).biasShift)

	return w
}

func (w *WeightShift) Scale(f float64) {
	w.weightShift.Scale(f, w.weightShift)
	w.biasShift.Scale(f, w.biasShift)
}

func (w *WeightShift) NumMatrices() int {
	return 2
}
