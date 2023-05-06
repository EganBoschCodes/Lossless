package main

import (
	"fmt"
	"go-ml-library/neuralnetworks/layers"
)

func main() {
	layer := layers.LinearLayer{}
	layer.Initialize(3, 4)

	input := []float64{1, 2, 3}
	output := layer.Pass(input)

	fmt.Println(output)
}
