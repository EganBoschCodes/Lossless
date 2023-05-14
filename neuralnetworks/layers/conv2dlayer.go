package layers

import (
	"fmt"
	"math/rand"

	"github.com/EganBoschCodes/lossless/neuralnetworks/save"
	"github.com/EganBoschCodes/lossless/utils"

	"gonum.org/v1/gonum/mat"
)

type Shape struct {
	Rows int
	Cols int
}

type Conv2DLayer struct {
	InputShape  Shape
	KernelShape Shape
	NumKernels  int
	FirstLayer  bool

	kernels         []mat.Matrix
	inputMatrices   int
	inputLen        int
	kernelsPerInput int
	outputShape     Shape
	outputLen       int
}

func (layer *Conv2DLayer) Initialize(numInputs int) {
	if layer.InputShape.Rows == 0 || layer.InputShape.Cols == 0 {
		fmt.Println("You must specify the InputShape for a Conv2DLayer!")
		panic(1)
	}

	if layer.KernelShape.Rows == 0 || layer.KernelShape.Cols == 0 {
		fmt.Println("You must specify the KernelShape for a Conv2DLayer!")
		panic(1)
	}

	if layer.NumKernels == 0 {
		fmt.Println("You must specify the NumKernels for a Conv2DLayer!")
		panic(1)
	}

	// Computing useful constants for consistent use
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

	// If the layer has already had it's kernels initialized elsewhere (like from a save file) don't bother populating with randoms
	if layer.kernels != nil {
		return
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

	if layer.FirstLayer {
		return &KernelShift{shifts: allShifts}, nil
	}

	// Calculate the gradients to pass back

	// Hacky way to avoid doing a whole lot of list appending later; I create a slice, then a bunch of
	// matrices spaced out along the slice. As the matrices get modified, so does the underlying slice,
	// then at the end I have a slice containing all the relevant data pre-concatenated.
	passbackSlice := make([]float64, layer.inputMatrices*layer.inputLen)
	passbackMatrices := make([]mat.Matrix, layer.inputMatrices)
	for i := range passbackMatrices {
		passbackMatrices[i] = mat.NewDense(layer.InputShape.Rows, layer.InputShape.Cols, passbackSlice[i*layer.inputLen:(i+1)*layer.inputLen])
	}

	for i, kernel := range layer.kernels {
		rotatedKernel := mat.NewDense(layer.KernelShape.Rows, layer.KernelShape.Cols, utils.Reverse(kernel.(*mat.Dense).RawMatrix().Data))
		inputIndex := i / layer.kernelsPerInput
		correspondingGradient := mat.NewDense(layer.outputShape.Rows, layer.outputShape.Cols, gradientSlice[i*layer.outputLen:(i+1)*layer.outputLen])

		passbackMatrices[inputIndex].(*mat.Dense).Add(passbackMatrices[inputIndex], utils.ConvolveWithPadding(correspondingGradient, rotatedKernel))
	}

	return &KernelShift{shifts: allShifts}, mat.NewDense(layer.inputMatrices*layer.InputShape.Rows, layer.InputShape.Cols, passbackSlice)
}

func (layer *Conv2DLayer) NumOutputs() int {
	return layer.NumKernels * layer.outputShape.Rows * layer.outputShape.Cols
}

func (layer *Conv2DLayer) ToBytes() []byte {
	saveBytes := save.ConstantsToBytes(layer.InputShape.Rows, layer.InputShape.Cols, layer.KernelShape.Rows, layer.KernelShape.Cols, layer.NumKernels)
	for _, kernel := range layer.kernels {
		kernelSlice := utils.GetSlice(kernel)
		saveBytes = append(saveBytes, save.ToBytes(kernelSlice)...)
	}
	return saveBytes
}

func (layer *Conv2DLayer) FromBytes(bytes []byte) {
	constInts, kernelSlice := save.ConstantsFromBytes(bytes[:20]), save.FromBytes(bytes[20:])

	layer.InputShape = Shape{Rows: constInts[0], Cols: constInts[1]}
	layer.KernelShape = Shape{Rows: constInts[2], Cols: constInts[3]}
	layer.NumKernels = constInts[4]

	layer.kernels = make([]mat.Matrix, layer.NumKernels)
	kernelSize := layer.KernelShape.Rows * layer.KernelShape.Cols
	for i := range layer.kernels {
		layer.kernels[i] = mat.NewDense(layer.KernelShape.Rows, layer.KernelShape.Cols, kernelSlice[i*kernelSize:(i+1)*kernelSize])
	}
}

func (layer *Conv2DLayer) PrettyPrint() string {
	ret := fmt.Sprintf("Conv2D Layer\n%d kernels\n%dx%d input\n\n", layer.NumKernels, layer.InputShape.Rows, layer.InputShape.Cols)
	for i, kernel := range layer.kernels {
		ret += fmt.Sprintln("Kernel", i, "=")
		ret += fmt.Sprintln(utils.JSify(kernel))
	}
	return ret
}
