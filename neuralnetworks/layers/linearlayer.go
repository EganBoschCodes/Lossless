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
	n_inputs int
}

func (layer *LinearLayer) Initialize(numInputs int) {
	layer.n_inputs = numInputs
	if layer.weights != nil {
		return
	}

	if layer.Outputs == 0 {
		fmt.Println("You must specify how many Outputs a LinearLayer has!")
		panic(1)
	}

	// Use Xavier Initialization on the weights
	fan_avg := (float64(numInputs) + float64(layer.Outputs)) / 2
	data := make([]float64, (numInputs+1)*layer.Outputs)
	for i := range data {
		data[i] = rand.NormFloat64() / fan_avg
	}
	layer.weights = mat.NewDense(layer.Outputs, numInputs+1, data)
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

func (layer *LinearLayer) Back(inputs mat.Matrix, _ mat.Matrix, forwardGradients mat.Matrix) (ShiftType, mat.Matrix) {
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

	return &WeightShift{shift: shift}, newGradient
}

func (layer *LinearLayer) NumOutputs() int {
	return layer.Outputs
}

func (layer *LinearLayer) ToBytes() []byte {
	saveBytes := save.ConstantsToBytes(layer.Outputs)
	saveBytes = append(saveBytes, save.ToBytes(utils.GetSlice(layer.weights))...)
	return saveBytes
}

func (layer *LinearLayer) FromBytes(bytes []byte) {
	constInts, weightSlice := save.ConstantsFromBytes(bytes[:4]), save.FromBytes(bytes[4:])
	layer.Outputs = constInts[0]
	layer.weights = mat.NewDense(layer.Outputs, len(weightSlice)/layer.Outputs, weightSlice)
}

func (layer *LinearLayer) PrettyPrint() string {
	ret := fmt.Sprintf("Linear Layer\n%d Inputs -> %d Outputs\n\n", layer.n_inputs, layer.Outputs)
	return ret + fmt.Sprintf("weights =\n%s", utils.JSify(layer.weights))
}
