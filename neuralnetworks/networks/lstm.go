package networks

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/EganBoschCodes/lossless/datasets"
	"github.com/EganBoschCodes/lossless/neuralnetworks/layers"
	"github.com/EganBoschCodes/lossless/neuralnetworks/optimizers"
	"github.com/EganBoschCodes/lossless/neuralnetworks/save"
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
	SubBatch     int
	LearningRate float64

	numInputs    int
	numOutputs   int
	concatInputs int

	Optimizer optimizers.Optimizer
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
	if network.SubBatch == 0 {
		network.SubBatch = 1
	}
	if network.LearningRate == 0 {
		network.LearningRate = 0.05
	}
	if network.Optimizer == nil {
		network.Optimizer = &optimizers.GradientDescent{}
	}

	network.ForgetGate, network.InputGate, network.CandidateGate, network.OutputGate, network.InterpretGate = ForgetGate, InputGate, CandidateGate, OutputGate, InterpretGate

	// Forget Gate - A sigmoid NN that pointwise multiplies the cell state
	switch utils.LastOf(network.ForgetGate).(type) {
	case *layers.SigmoidLayer:
	default:
		network.ForgetGate = append(network.ForgetGate, &layers.SigmoidLayer{})
	}
	network.initializeGate(network.ForgetGate, network.concatInputs, network.numOutputs)

	// Input Gate - A sigmoid NN that pointwise multiplies with the output of the candidate gate
	switch utils.LastOf(network.InputGate).(type) {
	case *layers.SigmoidLayer:
	default:
		network.InputGate = append(network.InputGate, &layers.SigmoidLayer{})
	}
	network.initializeGate(network.InputGate, network.concatInputs, network.numOutputs)

	// Candidate Gate - A tanh NN that pointwise multiplies with the input gate, before being added to cell state.
	switch utils.LastOf(network.CandidateGate).(type) {
	case *layers.TanhLayer:
	default:
		network.CandidateGate = append(network.CandidateGate, &layers.TanhLayer{})
	}
	network.initializeGate(network.CandidateGate, network.concatInputs, network.numOutputs)

	// Output Gate - A sigmoid NN that, after being pointwise multiplied with the pointwise tanh of the modified cell state, constitutes the output
	switch utils.LastOf(network.OutputGate).(type) {
	case *layers.SigmoidLayer:
	default:
		network.OutputGate = append(network.OutputGate, &layers.SigmoidLayer{})
	}
	network.initializeGate(network.OutputGate, network.concatInputs, network.numOutputs)

	network.initializeGate(network.InterpretGate, network.numOutputs, -1)
}

func (network *LSTM) passThroughGate(input *mat.Dense, gate []layers.Layer) *mat.Dense {
	for _, layer := range gate {
		input, _ = layer.Pass(input)
	}
	return input
}

func (network *LSTM) passThroughGateWithCache(input *mat.Dense, gate []layers.Layer) (*mat.Dense, []layers.CacheType) {
	caches := make([]layers.CacheType, len(gate))

	for i, layer := range gate {
		output, cache := layer.Pass(input)

		caches[i] = cache
		input = output
	}
	return input, caches
}

func (network *LSTM) Evaluate(inputSeries [][]float64) []float64 {
	cellState, hiddenState := mat.NewDense(network.numOutputs, 1, nil), mat.NewDense(network.numOutputs, 1, nil)

	for _, input := range inputSeries {
		hiddenSlice := make([]float64, network.numOutputs)
		copy(hiddenSlice, utils.GetSlice(hiddenState))
		concatInput := append(hiddenSlice, input...)

		concatInputMat := utils.FromSlice(concatInput)

		// Forget Gate Passthrough
		fmt.Println(concatInputMat.Dims())
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

func (network *LSTM) EvaluateAcrossInterval(inputSeries [][]float64) [][]float64 {
	cellState, hiddenState := mat.NewDense(network.numOutputs, 1, nil), mat.NewDense(network.numOutputs, 1, nil)

	outputs := make([][]float64, len(inputSeries))

	for i, input := range inputSeries {
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

		// Interpret Gate
		outputs[i] = utils.GetSlice(network.passThroughGate(hiddenState, network.InterpretGate))
	}

	return outputs
}

func createNilShifts(length int) []layers.ShiftType {
	return utils.Map(make([]layers.ShiftType, length), func(_ layers.ShiftType) layers.ShiftType { return &layers.NilShift{} })
}

func getGateShifts(gate []layers.Layer, gateCache []layers.CacheType, forwardGradients *mat.Dense) (shifts []layers.ShiftType, startingGradients *mat.Dense) {
	shifts = make([]layers.ShiftType, len(gate))
	for i := len(gate) - 1; i >= 0; i-- {
		shifts[i], forwardGradients = gate[i].Back(gateCache[i], forwardGradients)
	}
	return shifts, forwardGradients
}

type GateCache struct {
	output *mat.Dense
	caches []layers.CacheType
}

func (network *LSTM) learn(dataset []datasets.DataPoint, shiftChannel chan [][]layers.ShiftType) {
	inputSeries, targets := datasets.Split(dataset)

	cellStates, hiddenStates := []*mat.Dense{mat.NewDense(network.numOutputs, 1, nil)}, []*mat.Dense{mat.NewDense(network.numOutputs, 1, nil)}
	forgetGateCaches, inputGateCaches, candidateGateCaches, outputGateCaches, interpretGateCaches := make([]GateCache, 0), make([]GateCache, 0), make([]GateCache, 0), make([]GateCache, 0), make([]GateCache, 0)
	forgetGateShifts, inputGateShifts, candidateGateShifts, outputGateShifts, interpretGateShifts := createNilShifts(len(network.ForgetGate)), createNilShifts(len(network.InputGate)), createNilShifts(len(network.CandidateGate)), createNilShifts(len(network.OutputGate)), createNilShifts(len(network.InterpretGate))

	// Forward Pass
	for _, input := range inputSeries {
		hiddenSlice := make([]float64, network.numOutputs)
		copy(hiddenSlice, utils.GetSlice(utils.LastOf(hiddenStates)))
		concatInput := append(hiddenSlice, input...)

		concatInputMat := utils.FromSlice(concatInput)

		// Forget Gate Passthrough
		forgetGateOutput, forgotCache := network.passThroughGateWithCache(concatInputMat, network.ForgetGate)
		forgetGateCaches = append(forgetGateCaches, GateCache{output: forgetGateOutput, caches: forgotCache})
		cellState := mat.NewDense(network.numOutputs, 1, nil)
		cellState.MulElem(utils.LastOf(cellStates), forgetGateOutput)

		// Input and Candidate Gate
		inputGateOutput, inputCache := network.passThroughGateWithCache(concatInputMat, network.InputGate)
		inputGateCaches = append(inputGateCaches, GateCache{output: inputGateOutput, caches: inputCache})

		candidateGateOutput, candidateCache := network.passThroughGateWithCache(concatInputMat, network.CandidateGate)
		candidateGateCaches = append(candidateGateCaches, GateCache{output: candidateGateOutput, caches: candidateCache})

		joinedOutput := mat.NewDense(network.numOutputs, 1, nil)
		joinedOutput.MulElem(inputGateOutput, candidateGateOutput)

		cellState.Add(cellState, joinedOutput)
		cellStates = append(cellStates, cellState)

		// Output Gate
		hiddenState, outputCache := network.passThroughGateWithCache(concatInputMat, network.OutputGate)
		outputGateCaches = append(outputGateCaches, GateCache{output: hiddenState, caches: outputCache})
		hiddenState.Apply(func(i int, j int, v float64) float64 {
			return v * math.Tanh(cellState.At(i, j))
		}, hiddenState)

		// Interpret Gate
		interpretGateOutput, interpretCache := network.passThroughGateWithCache(hiddenState, network.InterpretGate)
		interpretGateCaches = append(interpretGateCaches, GateCache{output: interpretGateOutput, caches: interpretCache})
	}

	cellStateGradient, hiddenStateGradient := mat.NewDense(network.numOutputs, 1, nil), mat.NewDense(network.numOutputs, 1, nil)
	for i := len(inputSeries) - 1; i >= 0; i-- {
		initialCellState, finalCellState := cellStates[i], cellStates[i+1]

		// Calculate the loss through the interpret layer
		currentFrameLossGradient := utils.FromSlice(utils.DoubleMap(targets[i], utils.GetSlice(interpretGateCaches[i].output), func(a float64, b float64) float64 { return a - b }))
		localInterpretGateShifts, interpretGatePassback := getGateShifts(network.InterpretGate, interpretGateCaches[i].caches, currentFrameLossGradient)

		// Average together all the interpret layer shifts
		interpretGateShifts = utils.DoubleMap(interpretGateShifts, localInterpretGateShifts, func(a layers.ShiftType, b layers.ShiftType) layers.ShiftType { return a.Combine(b) })

		// Combine the loss gradient calculated for this current frame with the one passed back from the layer ahead.
		//hiddenStateGradient.Scale(0.5, hiddenStateGradient)
		hiddenStateGradient.Add(hiddenStateGradient, interpretGatePassback)

		tanhFinalCellState := mat.NewDense(network.numOutputs, 1, nil)
		tanhFinalCellState.Apply(func(i int, j int, f float64) float64 { return math.Tanh(f) }, finalCellState)

		// Combine the loss gradient calculated for this frame with respect to cell with the one passed down from ahead.
		outputGateOutput := outputGateCaches[i].output
		cellStateLocalGradient := mat.NewDense(network.numOutputs, 1, nil)
		cellStateLocalGradient.Copy(hiddenStateGradient)
		cellStateLocalGradient.Apply(func(i int, j int, v float64) float64 {
			tanhCell := tanhFinalCellState.At(i, j)
			return v * outputGateOutput.At(i, j) * (1 - tanhCell*tanhCell)
		}, cellStateLocalGradient)
		cellStateGradient.Add(cellStateGradient, cellStateLocalGradient)

		// Output Gate Gradient Calculation
		outputGateGradient := mat.NewDense(network.numOutputs, 1, nil)
		outputGateGradient.MulElem(tanhFinalCellState, hiddenStateGradient)
		localOutputGateShifts, outputGatePassback := getGateShifts(network.OutputGate, outputGateCaches[i].caches, outputGateGradient)
		outputGateShifts = utils.DoubleMap(outputGateShifts, localOutputGateShifts, func(a layers.ShiftType, b layers.ShiftType) layers.ShiftType { return a.Combine(b) })

		// Input and Candidate Gate Gradient Calculation
		inputGateOutput, candidateGateOutput := inputGateCaches[i].output, candidateGateCaches[i].output
		candidateGateGradient := mat.NewDense(network.numOutputs, 1, nil)
		candidateGateGradient.MulElem(cellStateGradient, inputGateOutput)
		inputGateGradient := mat.NewDense(network.numOutputs, 1, nil)
		inputGateGradient.MulElem(cellStateGradient, candidateGateOutput)

		localCandidateGateShifts, candidateGatePassback := getGateShifts(network.CandidateGate, candidateGateCaches[i].caches, candidateGateGradient)
		localInputGateShifts, inputGatePassback := getGateShifts(network.InputGate, inputGateCaches[i].caches, inputGateGradient)

		candidateGateShifts = utils.DoubleMap(candidateGateShifts, localCandidateGateShifts, func(a layers.ShiftType, b layers.ShiftType) layers.ShiftType { return a.Combine(b) })
		inputGateShifts = utils.DoubleMap(inputGateShifts, localInputGateShifts, func(a layers.ShiftType, b layers.ShiftType) layers.ShiftType { return a.Combine(b) })

		// Forget Gate Gradient Calculation
		forgetGateGradient := mat.NewDense(network.numOutputs, 1, nil)
		forgetGateGradient.MulElem(cellStateGradient, initialCellState)
		forgetGateOutput := forgetGateCaches[i].output
		cellStateGradient.MulElem(cellStateGradient, forgetGateOutput)
		localForgetGateShifts, forgetGatePassback := getGateShifts(network.ForgetGate, forgetGateCaches[i].caches, forgetGateGradient)
		forgetGateShifts = utils.DoubleMap(forgetGateShifts, localForgetGateShifts, func(a layers.ShiftType, b layers.ShiftType) layers.ShiftType { return a.Combine(b) })

		combinedPassback := mat.NewDense(network.concatInputs, 1, nil)
		combinedPassback.Add(outputGatePassback, candidateGatePassback)
		combinedPassback.Add(combinedPassback, inputGatePassback)
		combinedPassback.Add(combinedPassback, forgetGatePassback)

		hiddenStateGradient = utils.FromSlice(utils.GetSlice(combinedPassback)[:network.numOutputs])
	}

	shiftChannel <- [][]layers.ShiftType{forgetGateShifts, inputGateShifts, candidateGateShifts, outputGateShifts, interpretGateShifts}
}

func (network *LSTM) getLoss(dataset []datasets.DataPoint) float64 {
	inputs, targets := utils.Map(dataset, func(d datasets.DataPoint) []float64 { return d.Input }), utils.Map(dataset, func(d datasets.DataPoint) []float64 { return d.Output })
	guesses := network.EvaluateAcrossInterval(inputs)
	differences := utils.DoubleMap(targets, guesses, utils.Subtract)
	toMeanSquared := func(a float64) float64 { return 0.5 * a * a }
	differences = utils.Map(differences, func(a []float64) []float64 { return utils.Map(a, toMeanSquared) })

	return utils.Reduce(utils.Reduce(differences, utils.Add), func(a float64, b float64) float64 { return a + b })
}

func (network *LSTM) applyShiftsToGate(layers []layers.Layer, shifts []layers.ShiftType) {
	for i, shift := range shifts {
		shift.Apply(layers[i], network.LearningRate)
	}
}

func combineShifts(current []layers.ShiftType, next []layers.ShiftType) []layers.ShiftType {
	return utils.DoubleMap(current, next, func(a layers.ShiftType, b layers.ShiftType) layers.ShiftType { return a.Combine(b) })
}

func (network *LSTM) applyShifts(shifts [][]layers.ShiftType) {
	forgetGateShifts, inputGateShifts, candidateGateShifts, outputGateShifts, interpretGateShifts := shifts[0], shifts[1], shifts[2], shifts[3], shifts[4]

	if !network.Optimizer.Initialized() {
		numShifts := 0
		for _, shift := range forgetGateShifts {
			numShifts += shift.NumMatrices()
		}
		for _, shift := range inputGateShifts {
			numShifts += shift.NumMatrices()
		}
		for _, shift := range candidateGateShifts {
			numShifts += shift.NumMatrices()
		}
		for _, shift := range outputGateShifts {
			numShifts += shift.NumMatrices()
		}
		for _, shift := range interpretGateShifts {
			numShifts += shift.NumMatrices()
		}
		network.Optimizer.Initialize(numShifts)
	}
	network.applyShiftsToGate(network.ForgetGate, forgetGateShifts)
	network.applyShiftsToGate(network.InputGate, inputGateShifts)
	network.applyShiftsToGate(network.CandidateGate, candidateGateShifts)
	network.applyShiftsToGate(network.OutputGate, outputGateShifts)
	network.applyShiftsToGate(network.InterpretGate, interpretGateShifts)
}

func (network *LSTM) optimize(allShifts [][]layers.ShiftType, done chan [][]layers.ShiftType) {
	// Initialize the optimizer with the right amount of memory if not already done.
	if !network.Optimizer.Initialized() {
		numShifts := 0
		for _, gateShifts := range allShifts {
			for _, shift := range gateShifts {
				numShifts += shift.NumMatrices()
			}
		}
		network.Optimizer.Initialize(numShifts)
	}

	// Optimize the shifts!
	i := 0
	for _, gateShifts := range allShifts {
		for _, shift := range gateShifts {
			shift.Optimize(network.Optimizer, i)
			i += shift.NumMatrices()
		}
	}

	// Send back to the main thread
	done <- allShifts
}

func (network *LSTM) Train(trainingData []datasets.DataPoint, testingData []datasets.DataPoint, stepSize int, timespan time.Duration) {
	fmt.Printf("Beginning Loss (Training, Testing): %.2f, %.2f\n\n", network.getLoss(trainingData), network.getLoss(testingData))

	start := time.Now()
	trainingTime := time.Since(start)
	intervalsTrainedOn := 0

	for trainingTime < timespan {
		shiftChannel := make(chan [][]layers.ShiftType)

		// Start the training intervals
		for i := 0; i < network.BatchSize; i++ {
			intervalStart := rand.Intn(len(trainingData) - stepSize)
			go network.learn(trainingData[intervalStart:intervalStart+stepSize], shiftChannel)
		}

		// Capture the calculated shifts, compute a sub-average, then send to Optimizer
		optimizedShiftChannel := make(chan [][]layers.ShiftType)
		for item := 0; item < network.BatchSize/network.SubBatch; item++ {
			var subShifts [][]layers.ShiftType
			for i := 0; i < network.SubBatch; i++ {
				datapointShifts := <-shiftChannel
				if i == 0 {
					subShifts = datapointShifts
				} else {
					subShifts = utils.DoubleMap(subShifts, datapointShifts, combineShifts)
				}
			}
			go network.optimize(subShifts, optimizedShiftChannel)
		}

		combinedShifts := [][]layers.ShiftType{createNilShifts(len(network.ForgetGate)), createNilShifts(len(network.InputGate)), createNilShifts(len(network.CandidateGate)), createNilShifts(len(network.OutputGate)), createNilShifts(len(network.InterpretGate))}
		for i := 0; i < network.BatchSize/network.SubBatch; i++ {
			optimizedShifts := <-optimizedShiftChannel
			combinedShifts = utils.DoubleMap(combinedShifts, optimizedShifts, combineShifts)
		}

		network.applyShifts(combinedShifts)

		// Just let me know how much time is left
		trainingTime = time.Since(start)
		steps := math.Min(100, float64(trainingTime*1000/timespan)/10)
		progressBar := ""
		for i := 0; i < 20; i++ {
			if i < int(steps)/5 {
				progressBar = fmt.Sprint(progressBar, "▒")
				continue
			}
			progressBar = fmt.Sprint(progressBar, " ")
		}
		fmt.Printf("\rTraining Progress : -{%s}- (%.1f%%)  ", progressBar, steps)

		intervalsTrainedOn += network.BatchSize
	}

	fmt.Printf("\n\nIntervals Trained: %d\nFinal Loss (Training, Testing): %.2f, %.2f\n", intervalsTrainedOn, network.getLoss(trainingData), network.getLoss(testingData))
}

func getGateBytes(gate []layers.Layer) []byte {
	bytes := save.ConstantsToBytes(len(gate))
	for _, layer := range gate {
		layerBytes := layer.ToBytes()
		bytes = append(bytes, save.ConstantsToBytes(layers.LayerToIndex(layer), len(layerBytes))...)
		bytes = append(bytes, layerBytes...)
	}
	return bytes
}

func (network *LSTM) ToBytes() []byte {
	bytes := save.ConstantsToBytes(network.numInputs, network.numOutputs)

	bytes = append(bytes, getGateBytes(network.ForgetGate)...)
	bytes = append(bytes, getGateBytes(network.InputGate)...)
	bytes = append(bytes, getGateBytes(network.CandidateGate)...)
	bytes = append(bytes, getGateBytes(network.OutputGate)...)
	bytes = append(bytes, getGateBytes(network.InterpretGate)...)

	return bytes
}

func (network *LSTM) toGateFrom(bytes []byte) ([]layers.Layer, []byte) {
	numLayers, bytes := save.ConstantsFromBytes(bytes[:4])[0], bytes[4:]
	numOutputs := network.concatInputs

	gateLayers := make([]layers.Layer, numLayers)
	for i := range gateLayers {
		constants := save.ConstantsFromBytes(bytes[:8])
		layer := layers.IndexToLayer(constants[0])
		layer.FromBytes(bytes[8 : 8+constants[1]])
		bytes = bytes[8+constants[1]:]
		layer.Initialize(numOutputs)
		numOutputs = layer.NumOutputs()

		gateLayers[i] = layer
	}
	return gateLayers, bytes
}

func (network *LSTM) FromBytes(bytes []byte) {
	constants := save.ConstantsFromBytes(bytes[:8])
	network.numInputs, network.numOutputs, network.concatInputs = constants[0], constants[1], constants[0]+constants[1]

	bytes = bytes[8:]
	network.ForgetGate, bytes = network.toGateFrom(bytes)
	network.InputGate, bytes = network.toGateFrom(bytes)
	network.CandidateGate, bytes = network.toGateFrom(bytes)
	network.OutputGate, bytes = network.toGateFrom(bytes)
	network.InterpretGate, _ = network.toGateFrom(bytes)
}

func (network *LSTM) Save(dir string, name string) {
	if len(dir) > 0 {
		save.WriteBytesToFile(fmt.Sprintf("%s/%s.lsls", dir, name), network.ToBytes())
	} else {
		save.WriteBytesToFile(fmt.Sprintf("%s.lsls", name), network.ToBytes())
	}
}

func (network *LSTM) Open(dir string, name string) {
	var rawBytes []byte
	if len(dir) > 0 {
		rawBytes = save.ReadBytesFromFile(fmt.Sprintf("%s/%s.lsls", dir, name))
	} else {
		rawBytes = save.ReadBytesFromFile(fmt.Sprintf("%s.lsls", name))
	}
	network.FromBytes(rawBytes)
}
