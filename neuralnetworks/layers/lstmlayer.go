package layers

import (
	"math"

	"github.com/EganBoschCodes/lossless/utils"
	"gonum.org/v1/gonum/mat"
)

type LSTMLayer struct {
	Outputs        int
	IntervalSize   int
	OutputSequence bool

	numInputs int
	numConcat int

	forgetGate    LinearLayer
	inputGate     LinearLayer
	candidateGate LinearLayer
	outputGate    LinearLayer
}

func (layer *LSTMLayer) Initialize(numInputs int) {
	if layer.Outputs == 0 {
		panic("Set how many outputs you want in your LSTM layer!")
	}
	if layer.IntervalSize == 0 {
		panic("Set how long the time series is being passed to your LSTM layer!")
	}
	if numInputs%layer.IntervalSize != 0 {
		panic("Your number of inputs to LSTM layer should be cleanly divided by IntervalSize!")
	}

	layer.numInputs = numInputs / layer.IntervalSize
	layer.numConcat = layer.numInputs + layer.Outputs

	if layer.forgetGate.initialized {
		return
	}

	layer.forgetGate = LinearLayer{Outputs: layer.Outputs}
	layer.forgetGate.Initialize(layer.numConcat)

	layer.inputGate = LinearLayer{Outputs: layer.Outputs}
	layer.inputGate.Initialize(layer.numConcat)

	layer.candidateGate = LinearLayer{Outputs: layer.Outputs}
	layer.candidateGate.Initialize(layer.numConcat)

	layer.outputGate = LinearLayer{Outputs: layer.Outputs}
	layer.outputGate.Initialize(layer.numConcat)
}

func (layer *LSTMLayer) Pass(input mat.Matrix) mat.Matrix {
	hiddenState, cellState := mat.NewDense(layer.Outputs, 1, nil), mat.NewDense(layer.Outputs, 1, nil)
	inputSlice := utils.GetSlice(input)

	hiddenStates := make([]float64, 0)

	for i := 0; i < len(inputSlice); i += layer.numInputs {
		concatInput := utils.FromSlice(append(utils.GetSlice(hiddenState), inputSlice[i:i+layer.numInputs]...))

		// Forget Gate
		forgetOutput, _ := layer.forgetGate.Pass(concatInput)
		forgetOutput.(*mat.Dense).Apply(func(i int, j int, v float64) float64 {
			return sigmoid(v)
		}, forgetOutput)

		cellState.MulElem(forgetOutput, cellState)

		// Input and Candidate Gate
		inputOutput, _ := layer.forgetGate.Pass(concatInput)
		inputOutput.(*mat.Dense).Apply(func(i int, j int, v float64) float64 {
			return sigmoid(v)
		}, inputOutput)

		candidateOutput, _ := layer.forgetGate.Pass(concatInput)
		candidateOutput.(*mat.Dense).Apply(func(i int, j int, v float64) float64 {
			return math.Tanh(v)
		}, candidateOutput)

		newMemories := utils.DenseLike(candidateOutput)
		newMemories.MulElem(inputOutput, candidateOutput)

		cellState.Add(newMemories, cellState)

		// Output Gate
		outputOutput, _ := layer.forgetGate.Pass(concatInput)
		outputOutput.(*mat.Dense).Apply(func(i int, j int, v float64) float64 {
			return sigmoid(v)
		}, outputOutput)

		tanhCellState := utils.DenseLike(cellState)
		tanhCellState.Apply(func(i int, j int, v float64) float64 {
			return math.Tanh(v)
		}, candidateOutput)

		hiddenState.MulElem(outputOutput, tanhCellState)
		hiddenStates = append(hiddenStates, utils.GetSlice(hiddenState)...)
	}

	if layer.OutputSequence {
		return utils.FromSlice(hiddenStates)
	}
	return hiddenState
}

/*func (layer *LSTMLayer) Back(combinedInput mat.Matrix, _ mat.Matrix, forwardGradients mat.Matrix) (shift ShiftType, backpass mat.Matrix) {
	var startingGradient *mat.Dense
	if layer.OutputSequence {
		gradientSlice := utils.GetSlice(forwardGradients)
		startingGradient = utils.FromSlice(gradientSlice[len(gradientSlice)-layer.Outputs:])
	} else {
		startingGradient = forwardGradients.(*mat.Dense)
	}

	inputSlice := utils.GetSlice(combinedInput)
	inputs = make([]*mat.Dense, layer.IntervalSize)
	for i := 0; i < len(inputSlice); i += layer.numInputs {

	}

}*/
