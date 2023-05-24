package layers

import (
	"fmt"
	"math"

	"github.com/EganBoschCodes/lossless/neuralnetworks/optimizers"
	"github.com/EganBoschCodes/lossless/neuralnetworks/save"
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
		panic(fmt.Sprintf("Your number of inputs to LSTM layer (%d) should be cleanly divided by IntervalSize (%d)!", numInputs, layer.IntervalSize))
	}

	layer.numInputs = numInputs / layer.IntervalSize
	layer.numConcat = layer.numInputs + layer.Outputs

	layer.forgetGate = LinearLayer{Outputs: layer.Outputs}
	layer.forgetGate.Initialize(layer.numConcat)

	layer.inputGate = LinearLayer{Outputs: layer.Outputs}
	layer.inputGate.Initialize(layer.numConcat)

	layer.candidateGate = LinearLayer{Outputs: layer.Outputs}
	layer.candidateGate.Initialize(layer.numConcat)

	layer.outputGate = LinearLayer{Outputs: layer.Outputs}
	layer.outputGate.Initialize(layer.numConcat)
}

func (layer *LSTMLayer) Pass(input *mat.Dense) (*mat.Dense, CacheType) {
	hiddenState, cellState := mat.NewDense(layer.Outputs, 1, nil), mat.NewDense(layer.Outputs, 1, nil)
	inputSlice := utils.GetSlice(input)

	hiddenStates := make([]float64, 0)
	inputs, cellStates, forgetOutputs, inputOutputs, candidateOutputs, outputOutputs := make([]*mat.Dense, 0), make([]*mat.Dense, 0), make([]*mat.Dense, 0), make([]*mat.Dense, 0), make([]*mat.Dense, 0), make([]*mat.Dense, 0)

	for i := 0; i < len(inputSlice); i += layer.numInputs {
		concatInput := utils.FromSlice(append(utils.GetSlice(hiddenState), inputSlice[i:i+layer.numInputs]...))
		inputs = append(inputs, concatInput)

		// Forget Gate
		forgetOutput, _ := layer.forgetGate.Pass(concatInput)
		forgetOutput.Apply(func(i int, j int, v float64) float64 {
			return sigmoid(v)
		}, forgetOutput)
		cellState.MulElem(forgetOutput, cellState)
		forgetOutputs = append(forgetOutputs, forgetOutput)

		// Input and Candidate Gate
		inputOutput, _ := layer.forgetGate.Pass(concatInput)
		inputOutput.Apply(func(i int, j int, v float64) float64 {
			return sigmoid(v)
		}, inputOutput)
		inputOutputs = append(inputOutputs, inputOutput)

		candidateOutput, _ := layer.forgetGate.Pass(concatInput)
		candidateOutput.Apply(func(i int, j int, v float64) float64 {
			return math.Tanh(v)
		}, candidateOutput)
		candidateOutputs = append(candidateOutputs, candidateOutput)

		newMemories := utils.DenseLike(candidateOutput)
		newMemories.MulElem(inputOutput, candidateOutput)

		cellState.Add(newMemories, cellState)

		// Output Gate
		outputOutput, _ := layer.forgetGate.Pass(concatInput)
		outputOutput.Apply(func(i int, j int, v float64) float64 {
			return sigmoid(v)
		}, outputOutput)
		outputOutputs = append(outputOutputs, outputOutput)

		tanhCellState := utils.DenseLike(cellState)
		tanhCellState.Apply(func(i int, j int, v float64) float64 {
			return math.Tanh(v)
		}, cellState)

		hiddenState.MulElem(outputOutput, tanhCellState)
		hiddenStates = append(hiddenStates, utils.GetSlice(hiddenState)...)
		cellStates = append(cellStates, cellState)
	}

	layerCache := &LSTMCache{
		Inputs:           inputs,
		CellStates:       cellStates,
		ForgetOutputs:    forgetOutputs,
		InputOutputs:     inputOutputs,
		CandidateOutputs: candidateOutputs,
		OutputOutputs:    outputOutputs,
	}

	if layer.OutputSequence {
		return utils.FromSlice(hiddenStates), layerCache
	}
	return hiddenState, layerCache
}

func (layer *LSTMLayer) Back(cache CacheType, frontalPass *mat.Dense) (shift ShiftType, backpass *mat.Dense) {
	lstmCache := cache.(*LSTMCache)
	inputs, cellStates, forgetOutputs, inputOutputs, candidateOutputs, outputOutputs := lstmCache.Inputs, lstmCache.CellStates, lstmCache.ForgetOutputs, lstmCache.InputOutputs, lstmCache.CandidateOutputs, lstmCache.OutputOutputs

	var forgetShift, inputShift, candidateShift, outputShift ShiftType
	forgetShift, inputShift, candidateShift, outputShift = &NilShift{}, &NilShift{}, &NilShift{}, &NilShift{}

	// If we output a sequence, we will have loss gradients for each individual input. However,
	// if we didn't, then we will say we have zero loss for every hidden state except the last.
	var forwardGradients []*mat.Dense
	if layer.OutputSequence {
		forwardGradients = utils.Map(utils.Cut(utils.GetSlice(frontalPass), layer.Outputs), utils.FromSlice)
	} else {
		forwardGradients = utils.Duplicate(mat.NewDense(layer.Outputs, 1, nil), layer.IntervalSize)
		forwardGradients[len(forwardGradients)-1] = frontalPass
	}

	backwardGradients := make([]float64, 0)

	cellStateGradient, hiddenStateGradient := mat.NewDense(layer.Outputs, 1, nil), mat.NewDense(layer.Outputs, 1, nil)
	for i := len(forwardGradients) - 1; i >= 0; i-- {
		// Combine the loss gradient calculated for this current frame with the one passed back from the layer ahead.
		hiddenStateGradient.Add(hiddenStateGradient, forwardGradients[i])

		// Combine the hidden state gradient into the cell state's gradient
		tanhCellState := utils.DenseLike(cellStates[i])
		tanhCellState.Apply(func(_ int, _ int, v float64) float64 {
			return math.Tanh(v)
		}, cellStates[i])

		localCellStateGradient := utils.DenseLike(hiddenStateGradient)
		localCellStateGradient.Apply(func(r, c int, v float64) float64 {
			tanhVal := tanhCellState.At(r, c)
			return v * outputOutputs[i].At(r, c) * (1 - tanhVal*tanhVal)
		}, hiddenStateGradient)
		cellStateGradient.Add(cellStateGradient, localCellStateGradient)

		// Output Gate Gradient Calculation
		outputGateGradient := utils.DenseLike(hiddenStateGradient)
		outputGateGradient.MulElem(tanhCellState, hiddenStateGradient)
		outputGateGradient.Apply(func(r, c int, v float64) float64 {
			outputVal := outputOutputs[i].At(r, c)
			return v * outputVal * (1 - outputVal)
		}, outputGateGradient)
		localOutputShift, outputPassback := layer.outputGate.Back(&InputCache{Input: inputs[i]}, outputGateGradient)
		outputShift = outputShift.Combine(localOutputShift)

		// Input and Candidate Gate Gradient Calculation
		inputGateGradient, candidateGateGradient := utils.DenseLike(cellStateGradient), utils.DenseLike(cellStateGradient)

		inputGateGradient.MulElem(cellStateGradient, candidateOutputs[i])
		inputGateGradient.Apply(func(r, c int, v float64) float64 {
			inputVal := inputOutputs[i].At(r, c)
			return v * inputVal * (1 - inputVal)
		}, inputGateGradient)

		candidateGateGradient.MulElem(cellStateGradient, inputOutputs[i])
		candidateGateGradient.Apply(func(r, c int, v float64) float64 {
			candidateVal := candidateOutputs[i].At(r, c)
			return v * (1 - candidateVal*candidateVal)
		}, candidateGateGradient)

		localInputShift, inputPassback := layer.inputGate.Back(&InputCache{Input: inputs[i]}, inputGateGradient)
		inputShift = inputShift.Combine(localInputShift)

		localCandidateShift, candidatePassback := layer.candidateGate.Back(&InputCache{Input: inputs[i]}, candidateGateGradient)
		candidateShift = candidateShift.Combine(localCandidateShift)

		// Forget Gate Gradient Calculation
		var initialCellState *mat.Dense
		if i == 0 {
			initialCellState = utils.DenseLike(cellStateGradient)
		} else {
			initialCellState = cellStates[i-1]
		}

		forgetGateGradient := utils.DenseLike(cellStateGradient)
		forgetGateGradient.MulElem(initialCellState, cellStateGradient)
		forgetGateGradient.Apply(func(r, c int, v float64) float64 {
			forgetVal := forgetOutputs[i].At(r, c)
			return v * forgetVal * (1 - forgetVal)
		}, forgetGateGradient)
		cellStateGradient.MulElem(cellStateGradient, forgetOutputs[i])

		localForgetShift, forgetPassback := layer.forgetGate.Back(&InputCache{Input: inputs[i]}, forgetGateGradient)
		forgetShift = forgetShift.Combine(localForgetShift)

		combinedPassback := utils.DenseLike(forgetPassback)
		combinedPassback.Add(forgetPassback, inputPassback)
		combinedPassback.Add(combinedPassback, candidatePassback)
		combinedPassback.Add(combinedPassback, outputPassback)

		hiddenStateGradient = utils.FromSlice(utils.GetSlice(combinedPassback)[:layer.Outputs])
		backwardGradients = append(utils.GetSlice(combinedPassback)[layer.Outputs:], backwardGradients...)
	}

	return &LSTMShift{
		forgetShift:    forgetShift,
		inputShift:     inputShift,
		candidateShift: candidateShift,
		outputShift:    outputShift,
	}, utils.FromSlice(backwardGradients)
}

func (layer *LSTMLayer) NumOutputs() int {
	if layer.OutputSequence {
		return layer.Outputs * layer.IntervalSize
	}
	return layer.Outputs
}

func (layer *LSTMLayer) ToBytes() []byte {
	forgetBytes := layer.forgetGate.ToBytes()
	inputBytes := layer.inputGate.ToBytes()
	candidateBytes := layer.candidateGate.ToBytes()
	outputBytes := layer.outputGate.ToBytes()

	saveBytes := save.ConstantsToBytes(layer.Outputs, layer.IntervalSize, utils.BoolToInt(layer.OutputSequence), len(forgetBytes))
	saveBytes = append(saveBytes, forgetBytes...)
	saveBytes = append(saveBytes, inputBytes...)
	saveBytes = append(saveBytes, candidateBytes...)
	saveBytes = append(saveBytes, outputBytes...)

	return saveBytes
}

func (layer *LSTMLayer) FromBytes(bytes []byte) {

	constInts, layersSlice := save.ConstantsFromBytes(bytes[:16]), bytes[16:]
	layer.Outputs, layer.IntervalSize, layer.OutputSequence = constInts[0], constInts[1], constInts[2] != 0
	gateSliceLength := constInts[3]

	layer.forgetGate.FromBytes(layersSlice[:gateSliceLength])
	layer.inputGate.FromBytes(layersSlice[gateSliceLength : gateSliceLength*2])
	layer.candidateGate.FromBytes(layersSlice[gateSliceLength*2 : gateSliceLength*3])
	layer.outputGate.FromBytes(layersSlice[gateSliceLength*3:])
}

func (layer *LSTMLayer) PrettyPrint() string {
	ret := fmt.Sprintf("LSTM Layer\n%d Inputs -> %d Outputs\n\n", layer.numInputs, layer.Outputs)
	ret += "\nForget Gate:\n" + layer.forgetGate.PrettyPrint()
	ret += "\nInput Gate:\n" + layer.inputGate.PrettyPrint()
	ret += "\nCandidate Gate:\n" + layer.candidateGate.PrettyPrint()
	ret += "\nOutput Gate:\n" + layer.outputGate.PrettyPrint()
	return ret
}

type LSTMShift struct {
	forgetShift    ShiftType
	inputShift     ShiftType
	candidateShift ShiftType
	outputShift    ShiftType
}

func (l *LSTMShift) Apply(layer Layer, opt optimizers.Optimizer, scale float64) {
	lstmLayer := layer.(*LSTMLayer)
	l.forgetShift.Apply(&lstmLayer.forgetGate, opt, scale)
	l.inputShift.Apply(&lstmLayer.inputGate, opt, scale)
	l.candidateShift.Apply(&lstmLayer.candidateGate, opt, scale)
	l.outputShift.Apply(&lstmLayer.outputGate, opt, scale)
}

func (l *LSTMShift) Combine(l2 ShiftType) ShiftType {
	lstm2 := l2.(*LSTMShift)
	l.forgetShift = l.forgetShift.Combine(lstm2.forgetShift)
	l.inputShift = l.inputShift.Combine(lstm2.inputShift)
	l.candidateShift = l.candidateShift.Combine(lstm2.candidateShift)
	l.outputShift = l.outputShift.Combine(lstm2.outputShift)

	return l
}

func (l *LSTMShift) NumMatrices() int {
	return 8
}
