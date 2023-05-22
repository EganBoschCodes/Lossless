package layers

import (
	"fmt"
	"math"

	"github.com/EganBoschCodes/lossless/utils"

	"gonum.org/v1/gonum/mat"
)

type SoftmaxLayer struct {
	n_inputs int
}

func (layer *SoftmaxLayer) Initialize(n_inputs int) {
	layer.n_inputs = n_inputs
}

func (layer *SoftmaxLayer) Pass(input mat.Matrix) (mat.Matrix, CacheType) {
	inputSlice := input.(*mat.Dense).RawMatrix().Data
	maxVal := utils.Reduce(inputSlice, math.Max)

	expSlice := utils.Map(inputSlice, func(a float64) float64 { return math.Exp(a - maxVal) })
	sumExps := utils.Reduce(expSlice, func(a float64, b float64) float64 { return a + b })
	expSlice = utils.Map(expSlice, func(a float64) float64 { return a / sumExps })

	r, c := input.Dims()
	output := mat.NewDense(r, c, expSlice)

	return output, &OutputCache{Output: output}
}

func (layer *SoftmaxLayer) Back(cache CacheType, forwardGradients mat.Matrix) (ShiftType, mat.Matrix) {
	outputs := cache.(*OutputCache).Output
	forwardGradients.(*mat.Dense).Apply(func(i, j int, v float64) float64 {
		val := outputs.At(i, j)
		return v * val * (1 - val)
	}, forwardGradients)

	return &NilShift{}, forwardGradients
}

func (layer *SoftmaxLayer) GetShape() mat.Matrix { return nil }

func (layer *SoftmaxLayer) NumOutputs() int {
	return layer.n_inputs
}

func (layer *SoftmaxLayer) ToBytes() []byte {
	return make([]byte, 0)
}

func (layer *SoftmaxLayer) FromBytes(bytes []byte) {}

func (layer *SoftmaxLayer) PrettyPrint() string {
	return fmt.Sprintln("Softmax Activation")
}
