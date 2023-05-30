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
	Outputs   int
	InputSize int

	OutputSequence      bool
	ConstantLengthInput bool

	numConcat       int
	numTotalOutputs int

	forgetGate    LinearLayer
	inputGate     LinearLayer
	candidateGate LinearLayer
	outputGate    LinearLayer

	initialHiddenState *mat.Dense
	initialCellState   *mat.Dense
}

func (layer *LSTMLayer) Initialize(totalInputs int) {
	if layer.Outputs == 0 {
		panic("Set how many outputs you want in your LSTM layer!")
	}
	if layer.InputSize == 0 {
		panic("Set how large each input chuck being passed to your LSTM layer is!")
	}
	if layer.initialCellState != nil {
		return
	}

	layer.numConcat = layer.InputSize + layer.Outputs
	if layer.ConstantLengthInput {
		layer.numTotalOutputs = totalInputs / layer.InputSize * layer.Outputs
	}

	layer.forgetGate = LinearLayer{Outputs: layer.Outputs}
	layer.forgetGate.Initialize(layer.numConcat)

	layer.inputGate = LinearLayer{Outputs: layer.Outputs}
	layer.inputGate.Initialize(layer.numConcat)

	layer.candidateGate = LinearLayer{Outputs: layer.Outputs}
	layer.candidateGate.Initialize(layer.numConcat)

	layer.outputGate = LinearLayer{Outputs: layer.Outputs}
	layer.outputGate.Initialize(layer.numConcat)

	layer.initialHiddenState, layer.initialCellState = mat.NewDense(layer.Outputs, 1, nil), mat.NewDense(layer.Outputs, 1, nil)
}

func (layer *LSTMLayer) Pass(input *mat.Dense) (*mat.Dense, CacheType) {
	hiddenState, cellState := mat.NewDense(layer.Outputs, 1, nil), mat.NewDense(layer.Outputs, 1, nil)
	hiddenState.Copy(layer.initialHiddenState)
	hiddenState.Copy(layer.initialCellState)

	inputSlice := utils.GetSlice(input)

	hiddenStates := make([]float64, 0)
	inputs, cellStates, forgetOutputs, inputOutputs, candidateOutputs, outputOutputs := make([]*mat.Dense, 0), make([]*mat.Dense, 0), make([]*mat.Dense, 0), make([]*mat.Dense, 0), make([]*mat.Dense, 0), make([]*mat.Dense, 0)

	for i := 0; i < len(inputSlice); i += layer.InputSize {
		concatInput := utils.FromSlice(append(utils.GetSlice(hiddenState), inputSlice[i:i+layer.InputSize]...))
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
		forwardGradients = utils.Duplicate(mat.NewDense(layer.Outputs, 1, nil), len(inputs))
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
		forgetShift:      forgetShift,
		inputShift:       inputShift,
		candidateShift:   candidateShift,
		outputShift:      outputShift,
		cellStateShift:   cellStateGradient,
		hiddenStateShift: hiddenStateGradient,
	}, utils.FromSlice(backwardGradients)
}

func (layer *LSTMLayer) NumOutputs() int {
	if layer.OutputSequence && !layer.ConstantLengthInput {
		return -1
	} else if layer.OutputSequence {
		return layer.numTotalOutputs
	}
	return layer.Outputs
}

func (layer *LSTMLayer) ToBytes() []byte {
	forgetBytes := layer.forgetGate.ToBytes()
	inputBytes := layer.inputGate.ToBytes()
	candidateBytes := layer.candidateGate.ToBytes()
	outputBytes := layer.outputGate.ToBytes()

	cellBytes, hiddenBytes := save.ToBytes(utils.GetSlice(layer.initialCellState)), save.ToBytes(utils.GetSlice(layer.initialHiddenState))

	saveBytes := save.ConstantsToBytes(layer.Outputs, layer.InputSize, utils.BoolToInt(layer.OutputSequence), utils.BoolToInt(layer.ConstantLengthInput), len(forgetBytes), len(cellBytes))
	saveBytes = append(saveBytes, forgetBytes...)
	saveBytes = append(saveBytes, inputBytes...)
	saveBytes = append(saveBytes, candidateBytes...)
	saveBytes = append(saveBytes, outputBytes...)

	saveBytes = append(saveBytes, cellBytes...)
	saveBytes = append(saveBytes, hiddenBytes...)

	return saveBytes
}

func (layer *LSTMLayer) FromBytes(bytes []byte) {

	constInts, bytes := save.ConstantsFromBytes(bytes[:24]), bytes[24:]
	layer.Outputs, layer.InputSize, layer.OutputSequence, layer.ConstantLengthInput = constInts[0], constInts[1], constInts[2] != 0, constInts[3] != 0
	gateSliceLength, stateSliceLength := constInts[4], constInts[5]

	layer.forgetGate.FromBytes(bytes[:gateSliceLength])
	layer.inputGate.FromBytes(bytes[gateSliceLength : gateSliceLength*2])
	layer.candidateGate.FromBytes(bytes[gateSliceLength*2 : gateSliceLength*3])
	layer.outputGate.FromBytes(bytes[gateSliceLength*3 : gateSliceLength*4])

	bytes = bytes[gateSliceLength*4:]
	layer.initialCellState, layer.initialHiddenState = utils.FromSlice(save.FromBytes(bytes[:stateSliceLength])), utils.FromSlice(save.FromBytes(bytes[stateSliceLength:]))
}

func (layer *LSTMLayer) PrettyPrint() string {
	ret := fmt.Sprintf("LSTM Layer\n%d Inputs -> %d Outputs\n\n", layer.InputSize, layer.Outputs)
	ret += "\nForget Gate:\n" + layer.forgetGate.PrettyPrint()
	ret += "\nInput Gate:\n" + layer.inputGate.PrettyPrint()
	ret += "\nCandidate Gate:\n" + layer.candidateGate.PrettyPrint()
	ret += "\nOutput Gate:\n" + layer.outputGate.PrettyPrint()
	return ret
}

/*
The shift type for LSTM Layers.
*/

type LSTMShift struct {
	forgetShift    ShiftType
	inputShift     ShiftType
	candidateShift ShiftType
	outputShift    ShiftType

	cellStateShift   *mat.Dense
	hiddenStateShift *mat.Dense
}

func (l *LSTMShift) Apply(layer Layer, scale float64) {
	lstmLayer := layer.(*LSTMLayer)
	l.forgetShift.Apply(&lstmLayer.forgetGate, scale)
	l.inputShift.Apply(&lstmLayer.inputGate, scale)
	l.candidateShift.Apply(&lstmLayer.candidateGate, scale)
	l.outputShift.Apply(&lstmLayer.outputGate, scale)

	l.cellStateShift.Scale(scale, l.cellStateShift)
	l.hiddenStateShift.Scale(scale, l.hiddenStateShift)
	layer.(*LSTMLayer).initialCellState.Add(layer.(*LSTMLayer).initialCellState, l.cellStateShift)
	layer.(*LSTMLayer).initialHiddenState.Add(layer.(*LSTMLayer).initialHiddenState, l.hiddenStateShift)
}

func (l *LSTMShift) Combine(l2 ShiftType) ShiftType {
	lstm2 := l2.(*LSTMShift)
	l.forgetShift = l.forgetShift.Combine(lstm2.forgetShift)
	l.inputShift = l.inputShift.Combine(lstm2.inputShift)
	l.candidateShift = l.candidateShift.Combine(lstm2.candidateShift)
	l.outputShift = l.outputShift.Combine(lstm2.outputShift)

	l.cellStateShift.Add(lstm2.cellStateShift, l.cellStateShift)
	l.hiddenStateShift.Add(lstm2.hiddenStateShift, l.hiddenStateShift)

	return l
}

func (l *LSTMShift) Optimize(opt optimizers.Optimizer, index int) {
	l.forgetShift.Optimize(opt, index)
	l.inputShift.Optimize(opt, index+2)
	l.candidateShift.Optimize(opt, index+4)
	l.outputShift.Optimize(opt, index+6)

	l.cellStateShift, l.hiddenStateShift = opt.Rescale(l.cellStateShift, index+8), opt.Rescale(l.hiddenStateShift, index+9)
}

func (l *LSTMShift) NumMatrices() int {
	return 10
}

func (l *LSTMShift) Scale(f float64) {
	l.forgetShift.Scale(f)
	l.inputShift.Scale(f)
	l.candidateShift.Scale(f)
	l.outputShift.Scale(f)
}
