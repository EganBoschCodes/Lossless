package networks

import (
	"fmt"
	"go-ml-library/datasets"
	"go-ml-library/neuralnetworks/layers"
	"time"

	"gonum.org/v1/gonum/mat"
)

type Perceptron struct {
	Layers []layers.Layer
}

func (network *Perceptron) Initialize(sizes []int, ls []layers.Layer) {
	i := 0

	// Initialize all of the layers with the proper sizing.
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
}

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

func (network *Perceptron) learn(input []float64, target []float64, channel chan []mat.Matrix) {
	// Done very similarly to Evaluate, but we just cache the inputs basically so we can use them to do backprop.
	inputCache := make([][]float64, 0)

	input = append(input, 1)
	for _, layer := range network.Layers {
		inputCache = append(inputCache, input)
		input = layer.Pass(input)
	}
	inputCache = append(inputCache, input)

	//fmt.Printf("Output: %.5f\n", input[:len(input)-1])

	// Now we start the gradient that we're gonna be passing back
	gradient := make([]float64, len(target))
	for i := range target {
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

func (network *Perceptron) getLoss(datapoint datasets.DataPoint, c chan float64) {
	input, target := datapoint.Input, datapoint.Output
	output := network.Evaluate(input)

	loss := 0.0
	for i := range output {
		loss += 0.5 * (output[i] - target[i]) * (output[i] - target[i])
	}
	c <- loss
}

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

func (network *Perceptron) getEmptyShift() []mat.Matrix {
	shifts := make([]mat.Matrix, len(network.Layers))
	for i, layer := range network.Layers {
		shifts[i] = layer.GetShape()
	}
	return shifts
}

func (network *Perceptron) Train(dataset []datasets.DataPoint, timespan time.Duration) {
	fmt.Printf("Beginning Loss: %.3f\n", network.getTotalLoss(dataset))

	start := time.Now()
	i := 0

	BATCH_SIZE := 16
	LEARNING_RATE := 1.0

	epochs := 0

	for time.Since(start) < timespan {
		shifts := network.getEmptyShift()

		shiftChannel := make(chan []mat.Matrix)

		for item := 0; item < BATCH_SIZE; item++ {
			datapoint := dataset[i]

			go network.learn(datapoint.Input, datapoint.Output, shiftChannel)

			i++
			if i >= len(dataset) {
				i = 0
				epochs++
			}
		}

		for item := 0; item < BATCH_SIZE; item++ {
			datapointShifts := <-shiftChannel
			for i, layerShift := range datapointShifts {
				if layerShift == nil {
					continue
				}
				shifts[i].(*mat.Dense).Add(layerShift, shifts[i])
			}
		}

		for i, shift := range shifts {
			network.Layers[i].ApplyShift(shift, LEARNING_RATE)
		}
	}

	fmt.Printf("Ending Loss: %.6f\n", network.getTotalLoss(dataset))
	fmt.Println("Trained Epochs:", epochs, ", Trained Datapoints:", epochs*len(dataset)+i)
}
