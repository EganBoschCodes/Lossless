package layers

import (
	"fmt"

	"github.com/EganBoschCodes/lossless/utils"

	"gonum.org/v1/gonum/mat"
)

type ReluLayer struct {
	n_inputs int
}

func (layer *ReluLayer) Initialize(n_inputs int) {
	layer.n_inputs = n_inputs
}

func (layer *ReluLayer) Pass(input *mat.Dense) (*mat.Dense, CacheType) {
	inputCache := utils.DenseLike(input)
	inputCache.Copy(input)
	output := utils.FastApply(input, func(i, j int, v float64) float64 {
		if v < 0 {
			return 0
		}
		return v
	})
	return output, &InputCache{Input: inputCache}
}

func (layer *ReluLayer) Back(cache CacheType, forwardGradients *mat.Dense) (ShiftType, *mat.Dense) {
	inputSlice := utils.GetSlice(cache.(*InputCache).Input)
	_, c := forwardGradients.Dims()
	forwardGradients.Apply(func(i, j int, v float64) float64 {
		val := inputSlice[i*c+j]
		if val <= 0.0 {
			return 0.1 * v
		}
		return v
	}, forwardGradients)
	return &NilShift{}, forwardGradients
}

func (layer *ReluLayer) NumOutputs() int {
	return layer.n_inputs
}

func (layer *ReluLayer) ToBytes() []byte {
	return make([]byte, 0)
}

func (layer *ReluLayer) FromBytes(bytes []byte) {}

func (layer *ReluLayer) PrettyPrint() string {
	return fmt.Sprintln("Relu Activation")
}
