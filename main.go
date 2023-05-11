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

	network.BatchSize = 32
	network.LearningRate = 0.02

	trainingData := mnist.GetMNISTTrain()
	testData := mnist.GetMNISTTest()

	network.Train(trainingData, testData, time.Second*20)

	network.Save("MNIST_Classifier")

	/*
		errors := network.GetErrors(testData)
		for i := 0; i < 20; i++ {
			mnist.PrintLetter(errors[i])
			fmt.Printf("Guess:  %.2f\n\n\n", network.Evaluate(errors[i].Input))
		}*/

}
