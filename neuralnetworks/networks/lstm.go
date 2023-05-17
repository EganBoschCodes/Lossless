package networks

import (
	"fmt"
	"math"

	"github.com/EganBoschCodes/lossless/neuralnetworks/layers"
	"github.com/EganBoschCodes/lossless/utils"
	"gonum.org/v1/gonum/mat"
)

type LSTM struct {
	ForgetGate    []layers.Layer
	InputGate     []layers.Layer
	CandidateGate []layers.Layer
	OutputGate    []layers.Layer

	BatchSize    int
	LearningRate float64

	numInputs    int
	numOutputs   int
	concatInputs int
}

func (network *LSTM) initializeGate(layers []layers.Layer) {
	lastOutput := network.concatInputs
	for _, layer := range layers {
		layer.Initialize(lastOutput)
		lastOutput = layer.NumOutputs()
	}

	if lastOutput != network.numOutputs {
		panic("Each gate needs to output the same number of values as the network!")
	}
}

func (network *LSTM) Initialize(numInputs int, numOutputs int, ForgetGate []layers.Layer, InputGate []layers.Layer, CandidateGate []layers.Layer, OutputGate []layers.Layer) {
	network.numInputs, network.numOutputs, network.concatInputs = numInputs, numOutputs, numInputs+numOutputs
	if network.BatchSize == 0 {
		network.BatchSize = 8
	}
	if network.LearningRate == 0 {
		network.LearningRate = 0.05
	}

	network.ForgetGate, network.InputGate, network.CandidateGate, network.OutputGate = ForgetGate, InputGate, CandidateGate, OutputGate

	// Forget Gate - A sigmoid NN that pointwise multiplies the cell state
	switch utils.LastOf(network.ForgetGate).(type) {
	case *layers.SigmoidLayer:
	default:
		network.ForgetGate = append(network.ForgetGate, &layers.SigmoidLayer{})
	}
	network.initializeGate(network.ForgetGate)

	// Input Gate - A sigmoid NN that pointwise multiplies with the output of the candidate gate
	switch utils.LastOf(network.InputGate).(type) {
	case *layers.SigmoidLayer:
	default:
		network.InputGate = append(network.InputGate, &layers.SigmoidLayer{})
	}
	network.initializeGate(network.InputGate)

	// Candidate Gate - A tanh NN that pointwise multiplies with the input gate, before being added to cell state.
	switch utils.LastOf(network.CandidateGate).(type) {
	case *layers.TanhLayer:
	default:
		network.CandidateGate = append(network.CandidateGate, &layers.TanhLayer{})
	}
	network.initializeGate(network.CandidateGate)

	// Output Gate - A sigmoid NN that, after being pointwise multiplied with the pointwise tanh of the modified cell state, constitutes the output
	switch utils.LastOf(network.OutputGate).(type) {
	case *layers.SigmoidLayer:
	default:
		network.OutputGate = append(network.OutputGate, &layers.SigmoidLayer{})
	}
	network.initializeGate(network.OutputGate)
}

func (network *LSTM) passThroughGate(input mat.Matrix, gate []layers.Layer) *mat.Dense {
	for _, layer := range gate {
		input = layer.Pass(input)
	}
	return input.(*mat.Dense)
}

func (network *LSTM) Evaluate(inputSeries [][]float64) []float64 {
	cellState, hiddenState := mat.NewDense(network.numOutputs, 1, nil), mat.NewDense(network.numOutputs, 1, nil)

	for _, input := range inputSeries {
		hiddenSlice := make([]float64, network.numOutputs)
		copy(hiddenSlice, utils.GetSlice(hiddenState))
		concatInput := append(hiddenSlice, input...)

		concatInputMat := utils.FromSlice(concatInput)

		// Forget Gate Passthrough
		forgetGateOutput := network.passThroughGate(concatInputMat, network.ForgetGate)
		cellState.MulElem(cellState, forgetGateOutput)

		// Input and Candidate Gate
		inputGateOutput := network.passThroughGate(concatInputMat, network.InputGate)
		candidateGateOutput := network.passThroughGate(concatInputMat, network.CandidateGate)
		joinedOutput := mat.NewDense(network.numOutputs, 1, nil)
		joinedOutput.MulElem(inputGateOutput, candidateGateOutput)

		cellState.Add(cellState, joinedOutput)

		// Output Gate
		hiddenState = network.passThroughGate(concatInputMat, network.OutputGate)
		hiddenState.Apply(func(i int, j int, v float64) float64 {
			return v * math.Tanh(cellState.At(i, j))
		}, hiddenState)
		fmt.Printf("%.2f\n", utils.GetSlice(hiddenState))
	}
	return utils.GetSlice(hiddenState)
}
