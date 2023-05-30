package layers

import (
	"fmt"

	"github.com/EganBoschCodes/lossless/utils"

	"gonum.org/v1/gonum/mat"
)

type LanhLayer struct {
	GradientScale float64
	n_inputs      int
}

func (layer *LanhLayer) Initialize(n_inputs int) {
	layer.n_inputs = n_inputs
	if layer.GradientScale == 0 {
		layer.GradientScale = 1
	}
}

func (layer *LanhLayer) Pass(input *mat.Dense) (*mat.Dense, CacheType) {
	inputCache := utils.DenseLike(input)
	inputCache.Copy(input)
	output := utils.FastApply(input, func(i, j int, v float64) float64 {
		if v < -1 {
			return -1
		} else if v > 1 {
			return 1
		}
		return v
	})
	return output, &InputCache{Input: inputCache}
}

func (layer *LanhLayer) Back(cache CacheType, forwardGradients *mat.Dense) (ShiftType, *mat.Dense) {
	inputSlice := utils.GetSlice(cache.(*InputCache).Input)
	_, c := forwardGradients.Dims()
	forwardGradients.Apply(func(i, j int, v float64) float64 {
		val := inputSlice[i*c+j]
		if val <= -1.0 || val >= 1.0 {
			return 0.1 * v * layer.GradientScale
		}
		return v * layer.GradientScale
	}, forwardGradients)
	return &NilShift{}, forwardGradients
}

func (layer *LanhLayer) NumOutputs() int {
	return layer.n_inputs
}

func (layer *LanhLayer) ToBytes() []byte {
	return make([]byte, 0)
}

func (layer *LanhLayer) FromBytes(bytes []byte) {}

func (layer *LanhLayer) PrettyPrint() string {
	return fmt.Sprintln("Lanh Activation")
}
