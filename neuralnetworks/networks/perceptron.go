package networks

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/EganBoschCodes/lossless/datasets"
	"github.com/EganBoschCodes/lossless/neuralnetworks/layers"
	"github.com/EganBoschCodes/lossless/neuralnetworks/save"
	"github.com/EganBoschCodes/lossless/utils"

	"gonum.org/v1/gonum/mat"
)

// The baseline network type, this can be used for generic MLPs and CNNs.
type Perceptron struct {
	Layers       []layers.Layer
	BatchSize    int
	LearningRate float64

	numInputs int
}

// Takes in the number of inputs this network will accept, as well as a
// list of the layers constructing the network.
func (network *Perceptron) Initialize(numInputs int, ls ...layers.Layer) {
	network.numInputs = numInputs

	// Initialize all of the layers with the proper sizing.
	network.Layers = ls
	lastOutput := numInputs
	for index, layer := range ls {
		network.Layers[index].Initialize(lastOutput)
		lastOutput = layer.NumOutputs()
	}

	if network.BatchSize == 0 {
		network.BatchSize = 8
	}
	if network.LearningRate == 0 {
		network.LearningRate = 0.05
	}
}

// Takes in a single input and passes it through the network.
func (network *Perceptron) Evaluate(input []float64) []float64 {
	// Convert slice into matrix
	var inputMat mat.Matrix
	inputMat = mat.NewDense(len(input), 1, input)

	// Pass the input through all the layers
	for _, layer := range network.Layers {
		inputMat = layer.Pass(inputMat)
	}

	// Reconvert from matrix back to the underlying slice
	return inputMat.(*mat.Dense).RawMatrix().Data
}

/*
	learn (input []float64, target []float64, channel chan []mat.Matrix):
	---------------------------------------------------------------------
	Takes in an input, a target value, then calculates the weight shifts for all layers
	based on said input and target, and then passes the list of per-layer weight shifts
	to the channel so that we can add it to the batch's shift.
*/

func (network *Perceptron) learn(input []float64, target []float64, channel chan []layers.ShiftType) {
	// Done very similarly to Evaluate, but we just cache the inputs basically so we can use them to do backprop.
	inputCache := make([]mat.Matrix, 0)

	var inputMat mat.Matrix
	inputMat = mat.NewDense(len(input), 1, input)
	for _, layer := range network.Layers {
		inputCache = append(inputCache, inputMat)
		inputMat = layer.Pass(inputMat)
	}
	inputCache = append(inputCache, inputMat)

	// Now we start the gradient that we're gonna be passing back
	gradient := make([]float64, len(target))
	for i := range target {
		// Basic cross-entropy loss gradient.
		gradient[i] = (target[i] - inputMat.(*mat.Dense).At(i, 0))
	}
	var gradientMat mat.Matrix
	gradientMat = mat.NewDense(len(gradient), 1, gradient)

	// Get all the shifts for each layer
	shifts := make([]layers.ShiftType, len(network.Layers))
	for i := len(network.Layers) - 1; i >= 0; i-- {
		layer := network.Layers[i]
		shift, gradientTemp := layer.Back(inputCache[i], inputCache[i+1], gradientMat)
		gradientMat = gradientTemp
		shifts[i] = shift
	}

	channel <- shifts
}

/*
	getLoss(datapoint datasets.DataPoint, c chan float64)
	---------------------------------------------------------------------
	Mostly used just as a way to check if I know how to use channels, this
	helps me compare the loss across the dataset before and after I train it.
	This one just gets the loss of one datapoint, then passes it to the channel
	to be summed up.
*/

func (network *Perceptron) getLoss(datapoint datasets.DataPoint, lossChannel chan float64, correctChannel chan bool) {
	input, target := datapoint.Input, datapoint.Output
	output := network.Evaluate(input)

	loss := 0.0
	for i := range output {
		loss += 0.5 * (output[i] - target[i]) * (output[i] - target[i])
	}

	wasCorrect := utils.GetMaxIndex(output) == datasets.FromOneHot(target)

	lossChannel <- loss
	correctChannel <- wasCorrect
}

/*
	getTotalLoss(dataset []datasets.DataPoint) float64
	---------------------------------------------------------------------
	Like mentioned above, this takes the loss of the entire dataset for
	comparison.
*/

func (network *Perceptron) getTotalLoss(dataset []datasets.DataPoint) (float64, int) {
	loss := 0.0
	correctGuesses := 0

	sampleSize := len(dataset)

	lossChannel := make(chan float64)
	correctChannel := make(chan bool)
	for i := 0; i < sampleSize; i++ {
		datapoint := dataset[i]
		go network.getLoss(datapoint, lossChannel, correctChannel)
	}

	valuesRecieved := 0
	for valuesRecieved < sampleSize {
		loss += <-lossChannel
		if <-correctChannel {
			correctGuesses++
		}
		valuesRecieved++
	}

	return loss, correctGuesses
}

// Takes in a dataset and prints to Standard Output the loss and accuracy across the dataset.
func (network *Perceptron) TestOnAndLog(dataset []datasets.DataPoint) {
	network.testOnAndLogWithPrefix(dataset, "")
}

func (network *Perceptron) testOnAndLogWithPrefix(dataset []datasets.DataPoint, prefix string) {
	loss, correctGuesses := network.getTotalLoss(dataset)
	fmt.Printf("\n%sLoss: %.3f\n", prefix, loss)
	correctPercentage := float64(correctGuesses) / float64(len(dataset)) * 100
	fmt.Printf("Correct Guesses: %d/%d (%.2f%%)\n\n", correctGuesses, len(dataset), correctPercentage)
}

// Iterates across all the layers and gets a zero-matrix in the shape of
// the weights of each layer. We use this as a baseline to add the shifts
// of each datapoint from the batch into.
func (network *Perceptron) getEmptyShift() []layers.ShiftType {
	shifts := make([]layers.ShiftType, len(network.Layers))
	for i := range network.Layers {
		shifts[i] = &layers.NilShift{}
	}
	return shifts
}

// The main functionality! Accepts a training dataset, a validation dataset,
// and how long you wish to train for.
func (network *Perceptron) Train(dataset []datasets.DataPoint, testingData []datasets.DataPoint, timespan time.Duration) {
	// Get a baseline
	network.testOnAndLogWithPrefix(testingData, "Beginning ")

	// Start the tracking data
	start := time.Now()
	datapointIndex := 0
	epochs := 0

	trainingTime := time.Since(start)
	for trainingTime < timespan {

		// Prepare to capture the weight shifts from each datapoint in the batch
		shifts := network.getEmptyShift()
		shiftChannel := make(chan []layers.ShiftType)

		// Start the weight calculations with goroutines
		for item := 0; item < network.BatchSize; item++ {
			datapoint := dataset[datapointIndex]

			go network.learn(datapoint.Input, datapoint.Output, shiftChannel)

			datapointIndex++
			if datapointIndex >= len(dataset) {
				datapointIndex = 0
				rand.Shuffle(len(dataset), func(i, j int) { dataset[i], dataset[j] = dataset[j], dataset[i] })
				epochs++
			}
		}

		// Capture the calculated weight shifts as they finish and add to the shift
		for item := 0; item < network.BatchSize; item++ {
			datapointShifts := <-shiftChannel
			for i, layerShift := range datapointShifts {
				shifts[i] = shifts[i].Combine(layerShift)
			}
		}

		// Once all shifts have been added in, apply the averaged shifts to all layers
		for i, shift := range shifts {
			shift.Apply(network.Layers[i], network.LearningRate)
		}

		// Just let me know how much time is left
		trainingTime = time.Since(start)
		steps := float64(trainingTime*1000/timespan) / 10
		progressBar := ""
		for i := 0; i < 20; i++ {
			if i < int(steps)/5 {
				progressBar = fmt.Sprint(progressBar, "▒")
				continue
			}
			progressBar = fmt.Sprint(progressBar, " ")
		}
		fmt.Printf("\rTraining Progress : -{%s}- (%.1f%%)  ", progressBar, steps)
	}

	// Log how we did
	network.testOnAndLogWithPrefix(testingData, "Final ")
	fmt.Printf("Trained Epochs: %d, Trained Datapoints: %d", epochs, epochs*len(dataset)+datapointIndex)
}

// This is just for some sanity checking. This lets you see the datapoints
// your network guesses wrong on, cause sometimes it gets things wrong it
// shouldn't, and sometimes you cannot believe someone wrote a 4 like that
// (I'm looking at you, random MNIST contributor).
func (network *Perceptron) GetErrors(dataset []datasets.DataPoint) []datasets.DataPoint {
	errors := make([]datasets.DataPoint, 0)
	for _, datapoint := range dataset {
		wasCorrect := utils.GetMaxIndex(network.Evaluate(datapoint.Input)) == datasets.FromOneHot(datapoint.Output)
		if !wasCorrect {
			errors = append(errors, datapoint)
		}
	}

	return errors
}

// Compresses all the uniquely identifying information in your network
// (all the weights, and the layer structure) into a long array of bytes,
// that can be saved directly to a .lsls file.
func (network *Perceptron) ToBytes() []byte {
	bytes := save.ConstantsToBytes(network.numInputs)
	for _, layer := range network.Layers {
		layerBytes := layer.ToBytes()
		bytes = append(bytes, save.ConstantsToBytes(layers.LayerToIndex(layer), len(layerBytes))...)
		bytes = append(bytes, layerBytes...)
	}
	return bytes
}

// Essentially the reverse of ToBytes(), this takes the byte array that
// was put into .lsls file and rebuilds it into the network that was saved.
func (network *Perceptron) FromBytes(bytes []byte) {
	network.numInputs = save.ConstantsFromBytes(bytes[:4])[0]
	network.Layers = make([]layers.Layer, 0)

	lastOutput := network.numInputs
	i := 4
	for i < len(bytes) {
		layerData := save.ConstantsFromBytes(bytes[i : i+8])
		layer := layers.IndexToLayer(layerData[0])
		dataLength := layerData[1]
		i += 8

		layer.FromBytes(bytes[i : i+dataLength])
		i += dataLength

		layer.Initialize(lastOutput)
		lastOutput = layer.NumOutputs()

		network.Layers = append(network.Layers, layer)
	}
}

// Saves your Perceptron into a .lsls file, with the path [Project Directory]/{dir}/{name}.lsls.
func (network *Perceptron) Save(dir string, name string) {
	if len(dir) > 0 {
		save.WriteBytesToFile(fmt.Sprintf("%s/%s.lsls", dir, name), network.ToBytes())
	} else {
		save.WriteBytesToFile(fmt.Sprintf("%s.lsls", name), network.ToBytes())
	}
}

// Opens the .lsls file at path [Project Directory]/{dir}/{name}.lsls and populates the network
// with that saved information.
func (network *Perceptron) Open(dir string, name string) {
	var rawBytes []byte
	if len(dir) > 0 {
		rawBytes = save.ReadBytesFromFile(fmt.Sprintf("%s/%s.lsls", dir, name))
	} else {
		rawBytes = save.ReadBytesFromFile(fmt.Sprintf("%s.lsls", name))
	}
	network.FromBytes(rawBytes)
}

// Sometimes you want to take your network and move it to another language,
// raw embed it in the code. To that, I say gross. But I get it. Cause I
// did it. Anyways, this will write all the relevant info to recreate the
// network in a human "readable" form (as if a matrix with dimensions in the
// hundreds can ever be human readable).
func (network *Perceptron) PrettyPrint(dir string, name string) {
	outputString := ""
	for i, layer := range network.Layers {
		outputString += layer.PrettyPrint()
		if i < len(network.Layers)-1 {
			outputString += "\n---------------------------------\n\n"
		}
	}
	if len(dir) > 0 {
		save.WriteStringToFile(fmt.Sprintf("%s/%s.txt", dir, name), outputString)
	} else {
		save.WriteStringToFile(fmt.Sprintf("%s.txt", name), outputString)
	}

}
