package main

import (
	"go-ml-library/datasets/mnist"
	"go-ml-library/neuralnetworks/layers"
	"go-ml-library/neuralnetworks/networks"
	"time"
)

func main() {
	network := networks.Perceptron{}
	network.Initialize(784,
		[]layers.Layer{
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
			&layers.FlattenLayer{},
			&layers.TanhLayer{},
			&layers.LinearLayer{Outputs: 128},
			&layers.TanhLayer{},
			&layers.LinearLayer{Outputs: 10},
			&layers.SoftmaxLayer{},
		})

	network.BATCH_SIZE = 32
	network.LEARNING_RATE = 0.02

	trainingData := mnist.GetMNISTTrain()
	testData := mnist.GetMNISTTest()

	network.Train(trainingData, testData, time.Second*900)
}
