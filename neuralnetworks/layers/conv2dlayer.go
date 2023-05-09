package layers

import (
	"fmt"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type Shape struct {
	Rows int
	Cols int
}

type Conv2DLayer struct {
	kernels     []mat.Matrix
	InputShape  Shape
	KernelShape Shape
	NumKernels  int

	outputShape Shape
}

func (layer *Conv2DLayer) Initialize(numInputs int) {
	if numInputs != layer.NumKernels*layer.InputShape.Rows*layer.InputShape.Cols {
		fmt.Printf("%d outputs from the last layer does not match the expected %d inputs! (%dx%dx%d)", numInputs, layer.NumKernels*layer.InputShape.Rows*layer.InputShape.Cols, layer.NumKernels, layer.InputShape.Rows, layer.InputShape.Cols)
		panic(1)
	}

	layer.outputShape = Shape{
		Rows: layer.InputShape.Rows - layer.KernelShape.Rows + 1,
		Cols: layer.InputShape.Cols - layer.KernelShape.Cols + 1,
	}

	layer.kernels = make([]mat.Matrix, layer.NumKernels)
	for i := range layer.kernels {
		randweights := make([]float64, layer.KernelShape.Rows*layer.KernelShape.Cols)
		for j := range randweights {
			randweights[j] = rand.NormFloat64()
		}
		layer.kernels[i] = mat.NewDense(layer.KernelShape.Rows, layer.KernelShape.Cols, randweights)
	}
}

func (layer *Conv2DLayer) NumOutputs() int {
	return layer.NumKernels * layer.outputShape.Rows * layer.outputShape.Cols
}
