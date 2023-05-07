package main

import (
	"go-ml-library/neuralnetworks/layers"
	"go-ml-library/neuralnetworks/networks"
)

func main() {
	network := networks.Perceptron{}
	network.Initialize([]int{2, 2, 2}, []layers.Layer{&layers.LinearLayer{}, &layers.SigmoidLayer{}})

	input := []float64{-1, 1}
	output := []float64{1, 0}
	for i := 0; i < 1000; i++ {
		network.Learn(input, output)
	}

	//fmt.Println(network.Evaluate(input))

	/*
		a := mat.NewDense(3, 1, []float64{1, 2, 3})
		utils.PrintMat("a", a)
		b := mat.NewDense(1, 2, []float64{1, 2})
		utils.PrintMat("b", b)

		m := mat.NewDense(3, 2, nil)
		m.Mul(a, b)

		utils.PrintMat("m", m)
	*/
}
