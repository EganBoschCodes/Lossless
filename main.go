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
	network.Initialize(717,
		[]layers.Layer{
			&layers.LinearLayer{Outputs: 128},
			&layers.ReluLayer{},
			&layers.SigmoidLayer{},
			&layers.LinearLayer{Outputs: 10},
			&layers.ReluLayer{},
			&layers.SoftmaxLayer{},
		})

	network.BATCH_SIZE = 32
	network.LEARNING_RATE = 0.1

	dataset := mnist.GetMNISTTrain()
	datasets.NormalizeInputs(dataset)

	network.Train(dataset, time.Second*60)

	//for i := 0; i < 20; i++ {
	//datasets.IsCorrect(network.Evaluate(dataset[i].Input), dataset[i].Output)
	//}

	/*layer := layers.ReluLayer{}
	layer.Initialize(3)

	fmt.Println(layer.Pass([]float64{1, 0, -1}))*/
}
