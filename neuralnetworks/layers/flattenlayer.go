package layers

import (
	"fmt"

	"github.com/EganBoschCodes/lossless/utils"

	"gonum.org/v1/gonum/mat"
)

type FlattenLayer struct {
	n_inputs int
}

func (layer *FlattenLayer) Initialize(n_inputs int) {
	layer.n_inputs = n_inputs
}

func (layer *FlattenLayer) Pass(input mat.Matrix) mat.Matrix {
	return mat.NewDense(layer.n_inputs, 1, utils.GetSlice(input))
}

func (layer *FlattenLayer) Back(_ mat.Matrix, _ mat.Matrix, forwardGradients mat.Matrix) (ShiftType, mat.Matrix) {
	return &NilShift{}, forwardGradients
}

func (layer *FlattenLayer) NumOutputs() int {
	return layer.n_inputs
}

func (layer *FlattenLayer) ToBytes() []byte {
	return make([]byte, 0)
}

func (layer *FlattenLayer) FromBytes(bytes []byte) {}

func (layer *FlattenLayer) PrettyPrint() string {
	return fmt.Sprintf("Flatten -> %dx1\n", layer.n_inputs)
}
