package mnist

import (
	"fmt"
	"lossless/datasets"
	"lossless/utils"
)

func mnistify(rawMNIST [][]float64) []datasets.DataPoint {
	dataset := make([]datasets.DataPoint, 0)

	for _, row := range rawMNIST {
		if len(row) == 0 {
			continue
		}
		input, output := utils.Map(row[1:], func(a float64) float64 { return a / 255.0 }), datasets.ToOneHot(row[0], 10)
		dataset = append(dataset, datasets.DataPoint{Input: input, Output: output})
	}

	return dataset
}

func GetMNISTTrain() []datasets.DataPoint {
	return mnistify(datasets.Open("datasets/mnist/mnist_train.csv"))
}

func GetMNISTTest() []datasets.DataPoint {
	return mnistify(datasets.Open("datasets/mnist/mnist_test.csv"))
}

func toASCII(values []float64) string {
	colors := []string{"  ", "░░", "▒▒", "▓▓", "██"}
	stringVal := ""

	for i, val := range values {
		if i%28 == 0 && i != 0 {
			stringVal = fmt.Sprint(stringVal, "\n")
		}
		index := int(val * 4.99)
		stringVal = fmt.Sprint(stringVal, colors[index])
	}

	return stringVal
}

func PrintLetter(letter datasets.DataPoint) {
	fmt.Println("Printing Digit:", datasets.FromOneHot(letter.Output))
	fmt.Println(toASCII(letter.Input))
	fmt.Printf("Output: %.2f\n", letter.Output)
}
