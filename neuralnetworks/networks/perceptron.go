package networks

import (
	"fmt"
	"go-ml-library/neuralnetworks/layers"

	"gonum.org/v1/gonum/mat"
)

type Perceptron struct {
	Layers []layers.Layer
}

func (network *Perceptron) Initialize(sizes []int, layers []layers.Layer) {
	// Basic input sanitization
	if len(sizes) != len(layers)+1 {
		fmt.Println("There needs to be exactly one more layer size value than layer type!")
		return
	}

	// Initialize all of the layers with the proper sizing.
	network.Layers = layers
	for i := range layers {
		network.Layers[i].Initialize(sizes[i], sizes[i+1])
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

func (network *Perceptron) Learn(input []float64, target []float64) []mat.Matrix {
	// Done very similarly to Evaluate, but we just cache the inputs basically so we can use them to do backprop.
	inputCache := make([][]float64, 0)

	input = append(input, 1)
	for _, layer := range network.Layers {
		inputCache = append(inputCache, input)
		input = layer.Pass(input)
	}
	inputCache = append(inputCache, input)

	fmt.Printf("Output: %.5f\n", input[:len(input)-1])

	gradient := make([]float64, len(target))
	for i := range target {
		gradient[i] = (target[i] - input[i])
	}
	gradientMat := mat.NewDense(len(gradient), 1, gradient)

	shifts := make([]mat.Matrix, len(network.Layers))
	for i := len(network.Layers) - 1; i >= 0; i-- {
		layer := network.Layers[i]
		shift, backPass := layer.Back(inputCache[i], inputCache[i+1], gradientMat)
		gradientMat = mat.DenseCopyOf(backPass)
		shifts[i] = shift
	}

	for i := range shifts {
		network.Layers[i].ApplyShift(shifts[i], 10)
	}

	return shifts
}
