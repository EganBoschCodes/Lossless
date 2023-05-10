package layers

import (
	"fmt"
	"go-ml-library/utils"
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

	inputMatrices int
	outputShape   Shape
}

func (layer *Conv2DLayer) Initialize(numInputs int) {
	if layer.NumKernels*layer.InputShape.Rows*layer.InputShape.Cols%numInputs != 0 {
		fmt.Printf("%d outputs from the last layer does not divide the expected %d inputs! (%dx%dx%d)", numInputs, layer.NumKernels*layer.InputShape.Rows*layer.InputShape.Cols, layer.NumKernels, layer.InputShape.Rows, layer.InputShape.Cols)
		panic(1)
	}

	layer.inputMatrices = layer.NumKernels * layer.InputShape.Rows * layer.InputShape.Cols / numInputs

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

func (layer *Conv2DLayer) Pass(input mat.Matrix) mat.Matrix {
	return nil
}

func (layer *Conv2DLayer) Back(inputs mat.Matrix, _ mat.Matrix, forwardGradients mat.Matrix) (mat.Matrix, mat.Matrix) {
	allShifts := make([]float64, 0)
	inputSlice := inputs.(*mat.Dense).RawMatrix().Data
	gradientSlice := forwardGradients.(*mat.Dense).RawMatrix().Data

	// Calculate the shifts for the local kernels
	for i := 0; i < layer.inputMatrices; i++ {
		inputMat := mat.NewDense(layer.InputShape.Rows, layer.InputShape.Cols, inputSlice[i*layer.InputShape.Rows*layer.InputShape.Cols:(i+1)*layer.InputShape.Rows*layer.InputShape.Cols])
		for k := 0; k < layer.NumKernels/layer.inputMatrices; k++ {
			j := i*layer.NumKernels/layer.inputMatrices + k
			correspondingGradient := mat.NewDense(layer.outputShape.Rows, layer.outputShape.Cols, gradientSlice[j*layer.outputShape.Rows*layer.outputShape.Cols:(j+1)*layer.outputShape.Rows*layer.outputShape.Cols])

			kernelShift := utils.ConvolveNoPadding(inputMat, correspondingGradient)
			allShifts = append(allShifts, kernelShift.(*mat.Dense).RawMatrix().Data...)
		}
	}

	// Finally shaping it into a storable matrix
	//allShiftsMat := mat.NewDense(layer.KernelShape.Rows*layer.NumKernels, layer.KernelShape.Cols, allShifts)

	// Calculate the gradients to pass back
	//for i, kernel := range layer.kernels {
	//rotatedKernel := mat.NewDense(layer.KernelShape.Rows, layer.KernelShape.Cols, utils.Reverse(kernel.(*mat.Dense).RawMatrix().Data))
	//}

	return nil, nil
}

func (layer *Conv2DLayer) NumOutputs() int {
	return layer.NumKernels * layer.outputShape.Rows * layer.outputShape.Cols
}
