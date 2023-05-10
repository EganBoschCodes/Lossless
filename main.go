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
			&layers.Conv2DLayer{
				InputShape:  layers.Shape{Rows: 28, Cols: 28},
				KernelShape: layers.Shape{Rows: 3, Cols: 3},
				NumKernels:  4,
			},
			&layers.FlattenLayer{},
			&layers.TanhLayer{},
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

	network.Train(dataset, time.Second*180)

	/*conv := layers.Conv2DLayer{
		InputShape:  layers.Shape{Rows: 2, Cols: 2},
		KernelShape: layers.Shape{Rows: 1, Cols: 1},
		NumKernels:  4,
	}
	conv.Initialize(8)

	utils.PrintMat("output", conv.Pass(mat.NewDense(2, 4, []float64{1, 2, 3, 4, 5, 6, 7, 8})))*/
}
