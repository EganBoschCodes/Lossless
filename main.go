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
			&layers.ReluLayer{},
			&layers.SigmoidLayer{},
			&layers.LinearLayer{Outputs: 10},
			&layers.ReluLayer{},
			&layers.SoftmaxLayer{},
		})

	network.BATCH_SIZE = 32
<<<<<<< HEAD
	network.LEARNING_RATE = 0.08
=======
	network.LEARNING_RATE = 0.1
>>>>>>> tmp

	dataset := mnist.GetMNISTTrain()
	datasets.NormalizeInputs(dataset)

	network.Train(dataset, time.Second*60)
}
