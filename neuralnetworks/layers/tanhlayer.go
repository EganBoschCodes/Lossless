package layers

import (
	"fmt"
	"math"

	"github.com/EganBoschCodes/lossless/utils"

	"gonum.org/v1/gonum/mat"
)

type TanhLayer struct {
	GradientScale float64

	n_inputs int
}

func (layer *TanhLayer) Initialize(n_inputs int) {
	layer.n_inputs = n_inputs

	if layer.GradientScale == 0 {
		layer.GradientScale = 1
	}
}

func (layer *TanhLayer) Pass(input *mat.Dense) (*mat.Dense, CacheType) {
	input.Apply(func(i int, j int, v float64) float64 { return math.Tanh(v) }, input)
	return input, &OutputCache{Output: input}
}

func (layer *TanhLayer) Back(cache CacheType, forwardGradients *mat.Dense) (ShiftType, *mat.Dense) {
	outputSlice := utils.GetSlice(cache.(*OutputCache).Output)
	_, c := forwardGradients.Dims()
	forwardGradients.Apply(func(i int, j int, v float64) float64 {
		val := outputSlice[i*c+j]
		return v * (1 - val*val) * layer.GradientScale
	}, forwardGradients)
	return &NilShift{}, forwardGradients
}

func (layer *TanhLayer) NumOutputs() int {
	return layer.n_inputs
}

func (layer *TanhLayer) ToBytes() []byte {
	return []byte{}
}

func (layer *TanhLayer) FromBytes(bytes []byte) {}

func (layer *TanhLayer) PrettyPrint() string {
	return fmt.Sprintln("Tanh Activation")
}
