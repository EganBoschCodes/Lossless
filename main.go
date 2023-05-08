package main

import (
	"go-ml-library/utils"

	"gonum.org/v1/gonum/mat"
)

func main() {
	/*network := networks.Perceptron{}
	network.Initialize([]int{717, 128, 10}, []layers.Layer{&layers.LinearLayer{}, &layers.SigmoidLayer{}, &layers.LinearLayer{}, &layers.SoftmaxLayer{}})

	network.BATCH_SIZE = 32
	network.LEARNING_RATE = 0.05

	dataset := mnist.GetMNISTTrain()
	datasets.NormalizeInputs(dataset)

	network.Train(dataset, time.Second*60)*/

	m := mat.NewDense(12, 16, []float64{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
	utils.PrintMat("m", m)
	k := mat.NewDense(2, 2, []float64{1, 1, 1, 1})
	utils.PrintMat("k", k)

	output := utils.ConvolveWithPadding(m, k)

	utils.PrintMat("output", output)
	//for i := 0; i < 20; i++ {
	//datasets.IsCorrect(network.Evaluate(dataset[i].Input), dataset[i].Output)
	//}
}
