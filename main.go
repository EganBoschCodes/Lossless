package main

import (
	"lossless/datasets"
	"lossless/datasets/mnist"
	"lossless/neuralnetworks/layers"
	"lossless/neuralnetworks/networks"
	"time"
)

func TrainPerceptron() {
	network := networks.Perceptron{}
	network.Initialize(784,
		&layers.Conv2DLayer{
			InputShape:  layers.Shape{Rows: 28, Cols: 28},
			KernelShape: layers.Shape{Rows: 3, Cols: 3},
			NumKernels:  2,
			FirstLayer:  true,
		},
		&layers.MaxPoolLayer{
			PoolShape: layers.Shape{Rows: 2, Cols: 2},
		},
		&layers.TanhLayer{},
		&layers.Conv2DLayer{
			InputShape:  layers.Shape{Rows: 13, Cols: 13},
			KernelShape: layers.Shape{Rows: 3, Cols: 3},
			NumKernels:  4,
		},
		&layers.TanhLayer{},
		&layers.Conv2DLayer{
			InputShape:  layers.Shape{Rows: 11, Cols: 11},
			KernelShape: layers.Shape{Rows: 3, Cols: 3},
			NumKernels:  4,
		},
		&layers.TanhLayer{},
		&layers.FlattenLayer{},
		&layers.LinearLayer{Outputs: 32},
		&layers.TanhLayer{},
		&layers.LinearLayer{Outputs: 10},
		&layers.SoftmaxLayer{},
	)

	network.BatchSize = 32
	network.LearningRate = 0.02

	trainingData := mnist.GetMNISTTrain()
	testData := mnist.GetMNISTTest()

	network.Train(trainingData, testData, time.Second*60)

	network.Save("savednetworks", "MNIST_Tiny_2Max")

	/*errors := network.GetErrors(testData)
	for i := 0; i < 20; i++ {
		mnist.PrintLetter(errors[i])
		fmt.Printf("Guess:  %.2f\n\n\n", network.Evaluate(errors[i].Input))
	}*/
}

func main() {

	//TrainPerceptron()
	network := networks.Perceptron{}
	network.Initialize(2,
		&layers.LinearLayer{Outputs: 7},
		&layers.TanhLayer{},
		&layers.LinearLayer{Outputs: 3},
		&layers.SoftmaxLayer{},
	)

	dataset := datasets.GetSpiralDataset()
	trainingData, testData := dataset[:80], dataset[80:100]

	network.Train(trainingData, testData, 10*time.Second)

	/*testData := mnist.GetMNISTTest()

	errors := network.GetErrors(testData)
	for i := 0; i < 20; i++ {
		mnist.PrintLetter(errors[i])
		fmt.Printf("Guess:  %.2f\n\n\n", network.Evaluate(errors[i].Input))
	}*/
}
