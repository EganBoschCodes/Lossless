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
	InterpretGate []layers.Layer

	BatchSize    int
	LearningRate float64

	numInputs    int
	numOutputs   int
	concatInputs int
}

func (network *LSTM) initializeGate(layers []layers.Layer, numInputs int, expectedOutputs int) {
	lastOutput := numInputs
	for _, layer := range layers {
		layer.Initialize(lastOutput)
		lastOutput = layer.NumOutputs()
	}

	if expectedOutputs > 0 && lastOutput != network.numOutputs {
		panic("Each gate needs to output the same number of values as the network!")
	}
}

func (network *LSTM) Initialize(numInputs int, numOutputs int, ForgetGate []layers.Layer, InputGate []layers.Layer, CandidateGate []layers.Layer, OutputGate []layers.Layer, InterpretGate []layers.Layer) {
	network.numInputs, network.numOutputs, network.concatInputs = numInputs, numOutputs, numInputs+numOutputs
	if network.BatchSize == 0 {
		network.BatchSize = 8
	}
	if network.LearningRate == 0 {
		network.LearningRate = 0.05
	}

	network.ForgetGate, network.InputGate, network.CandidateGate, network.OutputGate, network.InterpretGate = ForgetGate, InputGate, CandidateGate, OutputGate, InterpretGate

	// Forget Gate - A sigmoid NN that pointwise multiplies the cell state
	switch utils.LastOf(network.ForgetGate).(type) {
	case *layers.SigmoidLayer:
	default:
		network.ForgetGate = append(network.ForgetGate, &layers.SigmoidLayer{})
	}
	network.initializeGate(network.ForgetGate, network.numInputs, network.numOutputs)

	// Input Gate - A sigmoid NN that pointwise multiplies with the output of the candidate gate
	switch utils.LastOf(network.InputGate).(type) {
	case *layers.SigmoidLayer:
	default:
		network.InputGate = append(network.InputGate, &layers.SigmoidLayer{})
	}
	network.initializeGate(network.InputGate, network.numInputs, network.numOutputs)

	// Candidate Gate - A tanh NN that pointwise multiplies with the input gate, before being added to cell state.
	switch utils.LastOf(network.CandidateGate).(type) {
	case *layers.TanhLayer:
	default:
		network.CandidateGate = append(network.CandidateGate, &layers.TanhLayer{})
	}
	network.initializeGate(network.CandidateGate, network.numInputs, network.numOutputs)

	// Output Gate - A sigmoid NN that, after being pointwise multiplied with the pointwise tanh of the modified cell state, constitutes the output
	switch utils.LastOf(network.OutputGate).(type) {
	case *layers.SigmoidLayer:
	default:
		network.OutputGate = append(network.OutputGate, &layers.SigmoidLayer{})
	}
	network.initializeGate(network.OutputGate, network.numInputs, network.numOutputs)

	network.initializeGate(network.InterpretGate, network.numOutputs, -1)
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
	}

	// Interpret Gate
	return utils.GetSlice(network.passThroughGate(hiddenState, network.InterpretGate))
}

func (network *LSTM) passThroughGateWithCache(input mat.Matrix, gate []layers.Layer) (*mat.Dense, []mat.Matrix) {
	inputs := []mat.Matrix{input}
	for _, layer := range gate {
		input = layer.Pass(input)
		inputs = append(inputs, input)
	}
	return input.(*mat.Dense), inputs
}

func createNilShifts(length int) []layers.ShiftType {
	return utils.Map(make([]layers.ShiftType, length), func(_ layers.ShiftType) layers.ShiftType { return &layers.NilShift{} })
}

func getGateShifts(gate []layers.Layer, gateCache []mat.Matrix, forwardGradients mat.Matrix) (shifts []layers.ShiftType, startingGradients mat.Matrix) {
	shifts = make([]layers.ShiftType, len(gate))
	for i := len(gate); i >= 0; i++ {
		shifts[i], forwardGradients = gate[i].Back(gateCache[i], gateCache[i+1], forwardGradients)
	}
	return shifts, forwardGradients
}

func (network *LSTM) learn(inputSeries [][]float64, targets [][]float64, shiftChannel chan [][]layers.ShiftType) {
	cellStates, hiddenStates := []mat.Matrix{mat.NewDense(network.numOutputs, 1, nil)}, []mat.Matrix{mat.NewDense(network.numOutputs, 1, nil)}
	forgetGateInputCache, inputGateInputCache, candidateGateInputCache, outputGateInputCache, interpretGateInputCache := make([][]mat.Matrix, 0), make([][]mat.Matrix, 0), make([][]mat.Matrix, 0), make([][]mat.Matrix, 0), make([][]mat.Matrix, 0)
	forgetGateShifts, inputGateShifts, candidateGateShifts, outputGateShifts, interpretGateShifts := createNilShifts(len(network.ForgetGate)), createNilShifts(len(network.InputGate)), createNilShifts(len(network.CandidateGate)), createNilShifts(len(network.OutputGate)), createNilShifts(len(network.InterpretGate))

	// Forward Pass
	for _, input := range inputSeries {
		hiddenSlice := make([]float64, network.numOutputs)
		copy(hiddenSlice, utils.GetSlice(utils.LastOf(hiddenStates)))
		concatInput := append(hiddenSlice, input...)

		concatInputMat := utils.FromSlice(concatInput)

		// Forget Gate Passthrough
		forgetGateOutput, forgotCache := network.passThroughGateWithCache(concatInputMat, network.ForgetGate)
		forgetGateInputCache = append(forgetGateInputCache, forgotCache)
		var cellState *mat.Dense
		cellState.MulElem(cellState, forgetGateOutput)

		// Input and Candidate Gate
		inputGateOutput, inputCache := network.passThroughGateWithCache(concatInputMat, network.InputGate)
		inputGateInputCache = append(inputGateInputCache, inputCache)

		candidateGateOutput, candidateCache := network.passThroughGateWithCache(concatInputMat, network.CandidateGate)
		candidateGateInputCache = append(candidateGateInputCache, candidateCache)

		joinedOutput := mat.NewDense(network.numOutputs, 1, nil)
		joinedOutput.MulElem(inputGateOutput, candidateGateOutput)

		cellState.Add(cellState, joinedOutput)
		cellStates = append(cellStates, cellState)

		// Output Gate
		hiddenState, outputCache := network.passThroughGateWithCache(concatInputMat, network.OutputGate)
		outputGateInputCache = append(outputGateInputCache, outputCache)
		hiddenState.Apply(func(i int, j int, v float64) float64 {
			return v * math.Tanh(cellState.At(i, j))
		}, hiddenState)

		// Interpret Gate
		_, interpretCache := network.passThroughGateWithCache(hiddenState, network.InterpretGate)
		interpretGateInputCache = append(interpretGateInputCache, interpretCache)
		fmt.Printf("%.2f\n", utils.GetSlice(hiddenState))
	}

	cellStateGradient, hiddenStateGradient := mat.NewDense(network.numOutputs, 1, nil), mat.NewDense(network.numOutputs, 1, nil)
	for i := len(inputSeries) - 1; i >= 0; i++ {
		initialCellState, finalCellState := cellStates[i], cellStates[i+1]

		// Calculate the loss through the interpret layer
		currentFrameLossGradient := utils.FromSlice(utils.DoubleMap(targets[i], utils.GetSlice(utils.LastOf(interpretGateInputCache[i])), func(a float64, b float64) float64 { return a - b }))
		localInterpretGateShifts, interpretGatePassback := getGateShifts(network.InterpretGate, interpretGateInputCache[i], currentFrameLossGradient)

		// Average together all the interpret layer shifts
		interpretGateShifts = utils.DoubleMap(interpretGateShifts, localInterpretGateShifts, func(a layers.ShiftType, b layers.ShiftType) layers.ShiftType { return a.Combine(b) })

		// Combine the loss gradient calculated for this current frame with the one passed back from the layer ahead.
		hiddenStateGradient.Add(hiddenStateGradient, interpretGatePassback)

		var tanhFinalCellState *mat.Dense
		tanhFinalCellState.Apply(func(i int, j int, f float64) float64 { return math.Tanh(f) }, finalCellState)

		// Combine the loss gradient calculated for this frame with respect to cell with the one passed down from ahead.
		outputGateOutput := utils.LastOf(outputGateInputCache[i])
		var cellStateLocalGradient *mat.Dense
		cellStateLocalGradient.Copy(hiddenStateGradient)
		cellStateLocalGradient.Apply(func(i int, j int, v float64) float64 {
			tanhCell := tanhFinalCellState.At(i, j)
			return v * outputGateOutput.At(i, j) * (1 - tanhCell*tanhCell)
		}, cellStateLocalGradient)
		cellStateGradient.Add(cellStateGradient, cellStateLocalGradient)

		// Output Gate Gradient Calculation
		var outputGateGradient *mat.Dense
		outputGateGradient.MulElem(tanhFinalCellState, hiddenStateGradient)
		localOutputGateShifts, outputGatePassback := getGateShifts(network.OutputGate, outputGateInputCache[i], outputGateGradient)
		outputGateShifts = utils.DoubleMap(outputGateShifts, localOutputGateShifts, func(a layers.ShiftType, b layers.ShiftType) layers.ShiftType { return a.Combine(b) })

		// Input and Candidate Gate Gradient Calculation
		inputGateOutput, candidateGateOutput := utils.LastOf(inputGateInputCache[i]), utils.LastOf(candidateGateInputCache[i])
		var candidateGateGradient *mat.Dense
		candidateGateGradient.MulElem(cellStateGradient, inputGateOutput)
		var inputGateGradient *mat.Dense
		inputGateGradient.MulElem(cellStateGradient, candidateGateOutput)

		localCandidateGateShifts, candidateGatePassback := getGateShifts(network.CandidateGate, candidateGateInputCache[i], candidateGateGradient)
		localInputGateShifts, inputGatePassback := getGateShifts(network.InputGate, inputGateInputCache[i], inputGateGradient)

		candidateGateShifts = utils.DoubleMap(candidateGateShifts, localCandidateGateShifts, func(a layers.ShiftType, b layers.ShiftType) layers.ShiftType { return a.Combine(b) })
		inputGateShifts = utils.DoubleMap(inputGateShifts, localInputGateShifts, func(a layers.ShiftType, b layers.ShiftType) layers.ShiftType { return a.Combine(b) })

		// Forget Gate Gradient Calculation
		var forgetGateGradient *mat.Dense
		forgetGateGradient.MulElem(cellStateGradient, initialCellState)
		localForgetGateShifts, forgetGatePassback := getGateShifts(network.ForgetGate, forgetGateInputCache[i], forgetGateGradient)
		forgetGateShifts = utils.DoubleMap(forgetGateShifts, localForgetGateShifts, func(a layers.ShiftType, b layers.ShiftType) layers.ShiftType { return a.Combine(b) })

		var combinedPassback *mat.Dense
		combinedPassback.Add(outputGatePassback, candidateGatePassback)
		combinedPassback.Add(combinedPassback, inputGatePassback)
		combinedPassback.Add(combinedPassback, forgetGatePassback)

		hiddenStateGradient = utils.FromSlice(utils.GetSlice(combinedPassback)[:network.numOutputs]).(*mat.Dense)
	}

	shiftChannel <- [][]layers.ShiftType{forgetGateShifts, inputGateShifts, candidateGateShifts, outputGateShifts, interpretGateShifts}
}
