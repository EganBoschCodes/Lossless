package layers

import (
	"fmt"

	"github.com/EganBoschCodes/lossless/neuralnetworks/save"
	"github.com/EganBoschCodes/lossless/utils"

	"gonum.org/v1/gonum/mat"
)

type MaxPool2DLayer struct {
	PoolShape Shape

	n_inputs int
}

func (layer *MaxPool2DLayer) Initialize(n_inputs int) {
	if layer.PoolShape.Rows == 0 || layer.PoolShape.Cols == 0 {
		fmt.Println("You must specify the PoolShape for a MaxPoolLayer!")
		panic(1)
	}

	if n_inputs%(layer.PoolShape.Rows*layer.PoolShape.Cols) != 0 {
		fmt.Printf("%d outputs from the last layer can't be pooled by an %dx%d pool!\n", n_inputs, layer.PoolShape.Rows, layer.PoolShape.Cols)
		panic(1)
	}

	layer.n_inputs = n_inputs
}

func (layer *MaxPool2DLayer) Pass(input *mat.Dense) (*mat.Dense, CacheType) {
	return utils.MaxPool(input, layer.PoolShape.Rows, layer.PoolShape.Cols), &InputCache{Input: input}
}

func (layer *MaxPool2DLayer) Back(cache CacheType, forwardGradients *mat.Dense) (ShiftType, *mat.Dense) {
	inputs := cache.(*InputCache).Input

	grownGradients := utils.UnMaxPool(forwardGradients, layer.PoolShape.Rows, layer.PoolShape.Cols)
	gradientMap := utils.MaxPoolMap(inputs, layer.PoolShape.Rows, layer.PoolShape.Cols)

	r, c := inputs.Dims()
	returnMatrix := mat.NewDense(r, c, utils.FastDot(utils.GetSlice(grownGradients), utils.GetSlice(gradientMap)))

	return &NilShift{}, returnMatrix
}

func (layer *MaxPool2DLayer) NumOutputs() int {
	return layer.n_inputs / layer.PoolShape.Rows / layer.PoolShape.Cols
}

func (layer *MaxPool2DLayer) ToBytes() []byte {
	saveBytes := save.ConstantsToBytes(layer.PoolShape.Rows, layer.PoolShape.Cols)
	return saveBytes
}

func (layer *MaxPool2DLayer) FromBytes(bytes []byte) {
	constInts := save.ConstantsFromBytes(bytes)
	layer.PoolShape = Shape{Rows: constInts[0], Cols: constInts[1]}
}

func (layer *MaxPool2DLayer) PrettyPrint() string {
	return fmt.Sprintf("MaxPool (%dx%d)\n", layer.PoolShape.Rows, layer.PoolShape.Cols)
}
