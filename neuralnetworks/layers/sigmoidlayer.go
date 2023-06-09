package layers

import (
	"fmt"
	"math"

	"github.com/EganBoschCodes/lossless/utils"

	"gonum.org/v1/gonum/mat"
)

type SigmoidLayer struct {
	GradientScale float64

	n_inputs int
}

func (layer *SigmoidLayer) Initialize(n_inputs int) {
	layer.n_inputs = n_inputs

	if layer.GradientScale == 0 {
		layer.GradientScale = 1
	}
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func (layer *SigmoidLayer) Pass(input *mat.Dense) (*mat.Dense, CacheType) {
	input.Apply(func(i int, j int, v float64) float64 { return sigmoid(v) }, input)
	return input, &OutputCache{Output: input}
}

func (layer *SigmoidLayer) Back(cache CacheType, forwardGradients *mat.Dense) (ShiftType, *mat.Dense) {
	outputSlice := utils.GetSlice(cache.(*OutputCache).Output)
	_, c := forwardGradients.Dims()
	forwardGradients.Apply(func(i, j int, v float64) float64 {
		val := outputSlice[i*c+j]
		return v * val * (1 - val) * layer.GradientScale
	}, forwardGradients)

	return &NilShift{}, forwardGradients
}

func (layer *SigmoidLayer) NumOutputs() int {
	return layer.n_inputs
}

func (layer *SigmoidLayer) ToBytes() []byte {
	return []byte{}
}

func (layer *SigmoidLayer) FromBytes(bytes []byte) {}

func (layer *SigmoidLayer) PrettyPrint() string {
	return fmt.Sprintln("Sigmoid Activation")
}
