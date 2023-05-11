package layers

import (
	"go-ml-library/neuralnetworks/save"
	"go-ml-library/utils"

	"gonum.org/v1/gonum/mat"
)

type MaxPoolLayer struct {
	PoolShape Shape

	n_inputs int
}

func (layer *MaxPoolLayer) Initialize(n_inputs int) {
	layer.n_inputs = n_inputs
}

func (layer *MaxPoolLayer) Pass(input mat.Matrix) mat.Matrix {
	return utils.MaxPool(input, layer.PoolShape.Rows, layer.PoolShape.Cols)
}

func (layer *MaxPoolLayer) Back(inputs mat.Matrix, _ mat.Matrix, forwardGradients mat.Matrix) (ShiftType, mat.Matrix) {
	grownGradients := utils.UnMaxPool(forwardGradients, layer.PoolShape.Rows, layer.PoolShape.Cols)
	gradientMap := utils.MaxPoolMap(inputs, layer.PoolShape.Rows, layer.PoolShape.Cols)

	r, c := inputs.Dims()
	returnMatrix := mat.NewDense(r, c, nil)
	returnMatrix.MulElem(grownGradients, gradientMap)

	return &NilShift{}, returnMatrix
}

func (layer *MaxPoolLayer) NumOutputs() int {
	return layer.n_inputs / layer.PoolShape.Rows / layer.PoolShape.Cols
}

func (layer *MaxPoolLayer) ToBytes() []byte {
	saveBytes := save.ConstantsToBytes(layer.PoolShape.Rows, layer.PoolShape.Cols)
	return saveBytes
}

func (layer *MaxPoolLayer) FromBytes(bytes []byte) {
	constInts := save.ConstantsFromBytes(bytes)
	layer.PoolShape = Shape{Rows: constInts[0], Cols: constInts[1]}
}
