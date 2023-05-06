package networks

import (
	"fmt"
	"go-ml-library/neuralnetworks/layers"
)

type Perceptron struct {
	Layers []layers.Layer
}

func (network *Perceptron) Initialize(sizes []int, layers []layers.Layer) {
	if len(sizes) != len(layers)+1 {
		fmt.Println("There needs to be exactly one more layer size value than layer type!")
		return
	}

	network.Layers = layers
	for i, layer := range layers {
		network.Layers[i] = layer.Initialize(sizes[i], sizes[i+1])
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
