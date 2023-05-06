package main

import (
	"fmt"
	"go-ml-library/neuralnetworks/layers"
	"go-ml-library/neuralnetworks/networks"
)

func main() {
	network := networks.Perceptron{}
	network.Initialize([]int{5, 2, 2}, []layers.Layer{layers.LinearLayer{}, layers.SigmoidLayer{}})

	input := []float64{1, 2, 3, 4, 5}
	fmt.Println(network.Evaluate(input))

	//layer := layers.LinearLayer{}.Initialize(3, 4)

	//sig := layers.SigmoidLayer{}.Initialize(4, 4)

	//input := []float64{1, 2, 3}
	//output := layer.Pass(input)

	//output = sig.Pass(output)

	//fmt.Println(output)
}
