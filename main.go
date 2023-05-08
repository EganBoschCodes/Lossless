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
	network.Initialize([]int{717, 128, 10}, []layers.Layer{&layers.LinearLayer{}, &layers.SigmoidLayer{}, &layers.LinearLayer{}, &layers.SoftmaxLayer{}})

	network.BATCH_SIZE = 12
	network.LEARNING_RATE = 0.1

	dataset := mnist.GetMNISTTrain()
	datasets.NormalizeInputs(dataset)

	network.Train(dataset, time.Second*30)

	//for i := 0; i < 20; i++ {
	//datasets.IsCorrect(network.Evaluate(dataset[i].Input), dataset[i].Output)
	//}
}
