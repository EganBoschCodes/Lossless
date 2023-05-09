package main

import (
	"go-ml-library/datasets"
	"go-ml-library/datasets/mnist"
	"go-ml-library/neuralnetworks/layers"
	"go-ml-library/neuralnetworks/networks"
	"time"
)

func main() {
	network := networks.Perceptron{}
	network.Initialize(784,
		[]layers.Layer{
			&layers.LinearLayer{Outputs: 128},
			&layers.TanhLayer{},
			&layers.LinearLayer{Outputs: 10},
			&layers.SoftmaxLayer{},
		})

	network.BATCH_SIZE = 32
	network.LEARNING_RATE = 0.03

	dataset := mnist.GetMNISTTrain()
	datasets.NormalizeInputs(dataset)

	network.Train(dataset, time.Second*30)
}
