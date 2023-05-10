package networks

import (
	"fmt"
	"go-ml-library/datasets"
	"go-ml-library/neuralnetworks/layers"
	"go-ml-library/utils"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

type Perceptron struct {
	Layers        []layers.Layer
	BATCH_SIZE    int
	LEARNING_RATE float64
}

func (network *Perceptron) Initialize(numInputs int, ls []layers.Layer) {
	// Initialize all of the layers with the proper sizing.
	network.Layers = ls
	lastOutput := numInputs
	for index, layer := range ls {
		network.Layers[index].Initialize(lastOutput)
		lastOutput = layer.NumOutputs()
	}

	network.BATCH_SIZE = 16
	network.LEARNING_RATE = 1.0
}

/*
	Evaluate (input []float64):
	---------------------------------------------------------------------
	Pretty much just here for testing or usage post training, this just
	takes an input and outputs what the network thinks it is.
*/

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

	sampleSize := utils.Min(len(dataset), 3000)

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

/*
	getEmptyShift() []mat.Matrix
	---------------------------------------------------------------------
	Iterates across all the layers and gets a zero-matrix in the shape of
	the weights of each layer. We use this as a baseline to add the shifts
	of each datapoint from the batch into.
*/

func (network *Perceptron) getEmptyShift() []layers.ShiftType {
	shifts := make([]layers.ShiftType, len(network.Layers))
	for i := range network.Layers {
		shifts[i] = &layers.NilShift{}
	}
	return shifts
}

/*
	Train(dataset []datasets.DataPoint, timespan time.Duration)
	---------------------------------------------------------------------
	The main functionality! This just takes in a dataset and how long you
	want to train, then goes about doing so.
*/

func (network *Perceptron) Train(dataset []datasets.DataPoint, timespan time.Duration) {
	// Get a baseline
	loss, correctGuesses := network.getTotalLoss(dataset)
	fmt.Printf("Beginning Loss: %.3f\n", loss)
	correctPercentage := float64(correctGuesses) / float64(utils.Min(len(dataset), 3000)) * 100
	fmt.Printf("Correct Guesses: %d/%d (%.2f%%)\n\n", correctGuesses, utils.Min(len(dataset), 3000), correctPercentage)

	// Start the tracking data
	start := time.Now()
	datapointIndex := 0
	epochs := 0

	for time.Since(start) < timespan {

		steps := float64(time.Since(start)*1000/timespan) / 10
		progressBar := ""
		for i := 0; i < 20; i++ {
			if i < int(steps)/5 {
				progressBar = fmt.Sprint(progressBar, "▒")
				continue
			}
			progressBar = fmt.Sprint(progressBar, " ")
		}
		fmt.Printf("\rTraining Progress : -{%s}- (%.1f%%)  ", progressBar, steps)

		// Prepare to capture the weight shifts from each datapoint in the batch
		shifts := network.getEmptyShift()
		shiftChannel := make(chan []layers.ShiftType)

		// Start the weight calculations with goroutines
		for item := 0; item < network.BATCH_SIZE; item++ {
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
		for item := 0; item < network.BATCH_SIZE; item++ {
			datapointShifts := <-shiftChannel
			for i, layerShift := range datapointShifts {
				shifts[i] = shifts[i].Combine(layerShift)
			}
		}

		// Once all shifts have been added in, apply the averaged shifts to all layers
		for i, shift := range shifts {
			shift.Apply(network.Layers[i], network.LEARNING_RATE)
		}
	}

	fmt.Println("\rTraining Progress : -{▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒}- (100.0%)")

	// Log how we did
	loss, correctGuesses = network.getTotalLoss(dataset)
	fmt.Printf("\nEnding Loss: %.3f\n", loss)

	correctPercentage = float64(correctGuesses) / float64(utils.Min(len(dataset), 3000)) * 100
	fmt.Printf("Correct Guesses: %d/%d (%.2f%%)\n", correctGuesses, utils.Min(len(dataset), 3000), correctPercentage)
	fmt.Println("Trained Epochs:", epochs, ", Trained Datapoints:", epochs*len(dataset)+datapointIndex)
}
