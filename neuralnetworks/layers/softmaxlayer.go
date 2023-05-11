package layers

import (
	"fmt"
	"go-ml-library/utils"
	"math"

	"gonum.org/v1/gonum/mat"
)

type SoftmaxLayer struct {
	n_inputs int
}

func (layer *SoftmaxLayer) Initialize(n_inputs int) {
	layer.n_inputs = n_inputs
}

func (layer *SoftmaxLayer) Pass(input mat.Matrix) mat.Matrix {
	inputSlice := input.(*mat.Dense).RawMatrix().Data
	maxVal := utils.Reduce(inputSlice, math.Max)

	expSlice := utils.Map(inputSlice, func(a float64) float64 { return math.Exp(a - maxVal) })
	sumExps := utils.Reduce(expSlice, func(a float64, b float64) float64 { return a + b })
	expSlice = utils.Map(expSlice, func(a float64) float64 { return a / sumExps })

	r, c := input.Dims()

	return mat.NewDense(r, c, expSlice)
}

func (layer *SoftmaxLayer) Back(inputs mat.Matrix, outputs mat.Matrix, forwardGradients mat.Matrix) (ShiftType, mat.Matrix) {
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

func (layer *SoftmaxLayer) PrettyPrint() {
	fmt.Println("Softmax Activation")
}
