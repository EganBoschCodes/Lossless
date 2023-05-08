package networks

import (
	"fmt"
	"go-ml-library/datasets"
	"go-ml-library/neuralnetworks/layers"
	"time"

	"gonum.org/v1/gonum/mat"
)

type Perceptron struct {
	Layers        []layers.Layer
	BATCH_SIZE    int
	LEARNING_RATE float64
}

func (network *Perceptron) Initialize(sizes []int, ls []layers.Layer) {
	i := 0

	// Initialize all of the layers with the proper sizing.
	// Since only linear layers have different numbers of inputs and outputs, I don't move the counter for any non-linear layers.
	network.Layers = ls
	for index, layer := range ls {
		switch layer.(type) {
		case *layers.LinearLayer:
			network.Layers[index].Initialize(sizes[i], sizes[i+1])
			i += 1
		default:
			network.Layers[index].Initialize(sizes[i], sizes[i])
		}
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
	// Add the "Bias" before passing to the first layer
	input = append(input, 1)

	// Pass the input through all the layers
	for _, layer := range network.Layers {
		input = layer.Pass(input)
	}

	// Return the return value, minus the bias.
	return input[:len(input)-1]
}

/*
	learn (input []float64, target []float64, channel chan []mat.Matrix):
	---------------------------------------------------------------------
	Takes in an input, a target value, then calculates the weight shifts for all layers
	based on said input and target, and then passes the list of per-layer weight shifts
	to the channel so that we can add it to the batch's shift.
*/

func (network *Perceptron) learn(input []float64, target []float64, channel chan []mat.Matrix) {
	// Done very similarly to Evaluate, but we just cache the inputs basically so we can use them to do backprop.
	inputCache := make([][]float64, 0)

	input = append(input, 1)
	for _, layer := range network.Layers {
		inputCache = append(inputCache, input)
		input = layer.Pass(input)
	}
	inputCache = append(inputCache, input)

	// Now we start the gradient that we're gonna be passing back
	gradient := make([]float64, len(target))
	for i := range target {
		// Basic cross-entropy loss gradient.
		gradient[i] = (target[i] - input[i])
	}
	gradientMat := mat.NewDense(len(gradient), 1, gradient)

	// Get all the shifts for each layer
	shifts := make([]mat.Matrix, len(network.Layers))
	for i := len(network.Layers) - 1; i >= 0; i-- {
		layer := network.Layers[i]
		shift, gradientTemp := layer.Back(inputCache[i], inputCache[i+1], gradientMat)

		gradientMat = mat.DenseCopyOf(gradientTemp)
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

func (network *Perceptron) getLoss(datapoint datasets.DataPoint, c chan float64) {
	input, target := datapoint.Input, datapoint.Output
	output := network.Evaluate(input)

	loss := 0.0
	for i := range output {
		loss += 0.5 * (output[i] - target[i]) * (output[i] - target[i])
	}
	c <- loss
}

/*
	getTotalLoss(dataset []datasets.DataPoint) float64
	---------------------------------------------------------------------
	Like mentioned above, this takes the loss of the entire dataset for
	comparison.
*/

func (network *Perceptron) getTotalLoss(dataset []datasets.DataPoint) float64 {
	loss := 0.0

	c := make(chan float64)
	for _, datapoint := range dataset {
		go network.getLoss(datapoint, c)
	}

	valuesRecieved := 0
	for valuesRecieved < len(dataset) {
		loss += <-c
		valuesRecieved++
	}

	return loss
}

/*
	getEmptyShift() []mat.Matrix
	---------------------------------------------------------------------
	Iterates across all the layers and gets a zero-matrix in the shape of
	the weights of each layer. We use this as a baseline to add the shifts
	of each datapoint from the batch into.
*/

func (network *Perceptron) getEmptyShift() []mat.Matrix {
	shifts := make([]mat.Matrix, len(network.Layers))
	for i, layer := range network.Layers {
		shifts[i] = layer.GetShape()
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
	fmt.Printf("Beginning Loss: %.3f\n", network.getTotalLoss(dataset))

	// Start the tracking data
	start := time.Now()
	datapointIndex := 0
	epochs := 0

	for time.Since(start) < timespan {
		// Prepare to capture the weight shifts from each datapoint in the batch
		shifts := network.getEmptyShift()
		shiftChannel := make(chan []mat.Matrix)

		// Start the weight calculations with goroutines
		for item := 0; item < network.BATCH_SIZE; item++ {
			datapoint := dataset[datapointIndex]

			go network.learn(datapoint.Input, datapoint.Output, shiftChannel)

			datapointIndex++
			if datapointIndex >= len(dataset) {
				datapointIndex = 0
				epochs++
			}
		}

		// Capture the calculated weight shifts as they finish and add to the shift
		for item := 0; item < network.BATCH_SIZE; item++ {
			datapointShifts := <-shiftChannel
			for i, layerShift := range datapointShifts {
				if layerShift == nil {
					continue
				}
				shifts[i].(*mat.Dense).Add(layerShift, shifts[i])
			}
		}

		// Once all shifts have been added in, apply the averaged shifts to all layers
		for i, shift := range shifts {
			network.Layers[i].ApplyShift(shift, network.LEARNING_RATE)
		}
	}

	// Log how we did
	fmt.Printf("Ending Loss: %.6f\n", network.getTotalLoss(dataset))
	fmt.Println("Trained Epochs:", epochs, ", Trained Datapoints:", epochs*len(dataset)+datapointIndex)
}
