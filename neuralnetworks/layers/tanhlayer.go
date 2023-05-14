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

func (layer *TanhLayer) Pass(input mat.Matrix) mat.Matrix {
	input.(*mat.Dense).Apply(func(i int, j int, v float64) float64 { return math.Tanh(v) }, input)
	return input
}

func (layer *TanhLayer) Back(inputs mat.Matrix, outputs mat.Matrix, forwardGradients mat.Matrix) (ShiftType, mat.Matrix) {
	outputSlice := utils.GetSlice(outputs)
	_, c := forwardGradients.Dims()
	forwardGradients.(*mat.Dense).Apply(func(i int, j int, v float64) float64 {
		val := outputSlice[i*c+j]
		return v * (1 - val*val) * layer.GradientScale
	}, forwardGradients)
	return &NilShift{}, forwardGradients
}

func (layer *TanhLayer) GetShape() mat.Matrix { return nil }

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
