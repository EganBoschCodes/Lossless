package main

import (
	"go-ml-library/datasets"
	"go-ml-library/neuralnetworks/layers"
	"go-ml-library/neuralnetworks/networks"
	"time"
)

func main() {
	network := networks.Perceptron{}
	network.Initialize([]int{2, 5, 3}, []layers.Layer{&layers.LinearLayer{}, &layers.SigmoidLayer{}, &layers.LinearLayer{}, &layers.SoftmaxLayer{}})

	dataset := datasets.GetSpiralDataset()
	datasets.NormalizeInputs(dataset)

	network.Train(dataset, time.Second*10)
}
