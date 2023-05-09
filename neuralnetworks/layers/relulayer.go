package layers

import (
	"go-ml-library/utils"
	"math"

	"gonum.org/v1/gonum/mat"
)

type ReluLayer struct {
	n_inputs int
}

func (layer *ReluLayer) Initialize(n_inputs int) {
	layer.n_inputs = n_inputs
}

func (layer *ReluLayer) Pass(input mat.Matrix) mat.Matrix {
	r, c := input.Dims()
	rawData := input.(*mat.Dense).RawMatrix().Data
	return mat.NewDense(r, c, utils.Map(rawData, func(a float64) float64 { return math.Max(a, 0.0) }))
}

func (layer *ReluLayer) Back(inputs mat.Matrix, _ mat.Matrix, forwardGradients mat.Matrix) (mat.Matrix, mat.Matrix) {
	forwardGradients.(*mat.Dense).Apply(func(i, j int, v float64) float64 {
		val := inputs.At(i, j)
		if val < 0 {
			return 0
		}
		return v
	}, forwardGradients)
	return nil, forwardGradients
}

func (layer *ReluLayer) GetShape() mat.Matrix {
	return nil
}

func (layer *ReluLayer) ApplyShift(shift mat.Matrix, scale float64) {}

func (layer *ReluLayer) NumOutputs() int {
	return layer.n_inputs
}
