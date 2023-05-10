package layers

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type TanhLayer struct {
	n_inputs int
}

func (layer *TanhLayer) Initialize(n_inputs int) {
	layer.n_inputs = n_inputs
}

func (layer *TanhLayer) Pass(input mat.Matrix) mat.Matrix {
	input.(*mat.Dense).Apply(func(i int, j int, v float64) float64 { return math.Tanh(v) }, input)
	return input
}

func (layer *TanhLayer) Back(inputs mat.Matrix, outputs mat.Matrix, forwardGradients mat.Matrix) (ShiftType, mat.Matrix) {
	forwardGradients.(*mat.Dense).Apply(func(i, j int, v float64) float64 {
		val := outputs.At(i, j)
		return v * (1 - val*val)
	}, forwardGradients)
	return &NilShift{}, forwardGradients
}

func (layer *TanhLayer) GetShape() mat.Matrix { return nil }

func (layer *TanhLayer) NumOutputs() int {
	return layer.n_inputs
}
