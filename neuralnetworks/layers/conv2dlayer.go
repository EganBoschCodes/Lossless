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

	inputMatrices   int
	inputLen        int
	kernelsPerInput int
	outputShape     Shape
	outputLen       int
}

func (layer *Conv2DLayer) Initialize(numInputs int) {

	layer.inputMatrices = numInputs / (layer.InputShape.Rows * layer.InputShape.Cols)
	layer.kernelsPerInput = layer.NumKernels / layer.inputMatrices
	layer.inputLen = layer.InputShape.Rows * layer.InputShape.Cols

	if layer.NumKernels%layer.inputMatrices != 0 {
		fmt.Printf("%d outputs from the last layer does not divide the expected %d inputs! (%dx%dx%d)\n", numInputs, layer.NumKernels*layer.InputShape.Rows*layer.InputShape.Cols, layer.NumKernels, layer.InputShape.Rows, layer.InputShape.Cols)
		panic(1)
	}

	layer.outputShape = Shape{
		Rows: layer.InputShape.Rows - layer.KernelShape.Rows + 1,
		Cols: layer.InputShape.Cols - layer.KernelShape.Cols + 1,
	}
	layer.outputLen = layer.outputShape.Rows * layer.outputShape.Cols

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
	passingSlice := make([]float64, 0)
	inputSlice := input.(*mat.Dense).RawMatrix().Data

	for k, kernel := range layer.kernels {
		inputIndex := k / layer.kernelsPerInput
		correspondingInput := mat.NewDense(layer.InputShape.Rows, layer.InputShape.Cols, inputSlice[inputIndex*layer.inputLen:(inputIndex+1)*layer.inputLen])

		convolution := utils.ConvolveNoPadding(correspondingInput, kernel)
		passingSlice = append(passingSlice, convolution.(*mat.Dense).RawMatrix().Data...)
	}

	return mat.NewDense(layer.NumKernels*layer.outputShape.Rows, layer.outputShape.Cols, passingSlice)
}

func (layer *Conv2DLayer) Back(inputs mat.Matrix, _ mat.Matrix, forwardGradients mat.Matrix) (ShiftType, mat.Matrix) {
	allShifts := make([]mat.Matrix, layer.NumKernels)
	inputSlice := utils.GetSlice(inputs)
	gradientSlice := utils.GetSlice(forwardGradients)

	// Calculate the shifts for the local kernels
	for i := 0; i < layer.inputMatrices; i++ {
		inputMat := mat.NewDense(layer.InputShape.Rows, layer.InputShape.Cols, inputSlice[i*layer.inputLen:(i+1)*layer.inputLen])
		for k := 0; k < layer.kernelsPerInput; k++ {
			j := i*layer.kernelsPerInput + k
			correspondingGradient := mat.NewDense(layer.outputShape.Rows, layer.outputShape.Cols, gradientSlice[j*layer.outputLen:(j+1)*layer.outputLen])

			allShifts[j] = utils.ConvolveNoPadding(inputMat, correspondingGradient)
		}
	}

	// Calculate the gradients to pass back

	// Hacky way to avoid doing a whole lot of list appending later; I create a slice, then a bunch of
	// matrices spaced out along the slice. As the matrices get modified, so does the underlying slice,
	// then at the end I have a slice containing all the relevant data pre-concatenated.
	passbackSlice := make([]float64, layer.inputMatrices*layer.inputLen)
	/*passbackMatrices := make([]mat.Matrix, layer.inputMatrices)
	for i := range passbackMatrices {
		passbackMatrices[i] = mat.NewDense(layer.InputShape.Rows, layer.InputShape.Cols, passbackSlice[i*layer.inputLen:(i+1)*layer.inputLen])
	}

	for i, kernel := range layer.kernels {
		rotatedKernel := mat.NewDense(layer.KernelShape.Rows, layer.KernelShape.Cols, utils.Reverse(kernel.(*mat.Dense).RawMatrix().Data))
		inputIndex := i / layer.kernelsPerInput
		correspondingGradient := mat.NewDense(layer.outputShape.Rows, layer.outputShape.Cols, gradientSlice[i*layer.outputLen:(i+1)*layer.outputLen])

		passbackMatrices[inputIndex].(*mat.Dense).Add(passbackMatrices[inputIndex], utils.ConvolveWithPadding(correspondingGradient, rotatedKernel))
	}*/

	return &KernelShift{shifts: allShifts}, mat.NewDense(layer.inputMatrices*layer.InputShape.Rows, layer.InputShape.Cols, passbackSlice)
}

func (layer *Conv2DLayer) NumOutputs() int {
	return layer.NumKernels * layer.outputShape.Rows * layer.outputShape.Cols
}
