package main

import (
	"github.com/EganBoschCodes/lossless/datasets/mnist"
	"github.com/EganBoschCodes/lossless/neuralnetworks/networks"
)

func main() {
	network := networks.Perceptron{}
	network.Open("savednetworks", "NewBest")

	trainingData := mnist.GetMNISTTrain()
	testData := mnist.GetMNISTTest()

	network.TestOnAndLog(trainingData)
	network.TestOnAndLog(testData)

	/*network.Initialize(784,
		&layers.Conv2DLayer{
			InputShape:  layers.Shape{Rows: 28, Cols: 28},
			KernelShape: layers.Shape{Rows: 3, Cols: 3},
			NumKernels:  6,
			FirstLayer:  true,
		},
		&layers.MaxPoolLayer{
			PoolShape: layers.Shape{Rows: 2, Cols: 2},
		},
		&layers.TanhLayer{},
		&layers.Conv2DLayer{
			InputShape:  layers.Shape{Rows: 13, Cols: 13},
			KernelShape: layers.Shape{Rows: 3, Cols: 3},
			NumKernels:  12,
		},
		&layers.TanhLayer{},
		&layers.FlattenLayer{},
		&layers.LinearLayer{Outputs: 100},
		&layers.TanhLayer{},
		&layers.LinearLayer{Outputs: 10},
		&layers.SoftmaxLayer{},
	)

	network.BatchSize = 32
	network.LearningRate = 0.02

	trainingData := mnist.GetMNISTTrain()
	testData := mnist.GetMNISTTest()

	network.Train(trainingData, testData, time.Second*180)*/

	//network.Save("savednetworks", "NewBest")
}
