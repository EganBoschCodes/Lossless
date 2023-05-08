package mnist

import (
	"fmt"
	"go-ml-library/datasets"
)

func GetMNISTTrain() []datasets.DataPoint {
	rawMNIST := datasets.Open("datasets/mnist/mnist_train.csv")
	dataset := make([]datasets.DataPoint, 0)

	for _, row := range rawMNIST {
		if len(row) == 0 {
			continue
		}
		input, output := row[1:], datasets.ToOneHot(row[0], 10)
		dataset = append(dataset, datasets.DataPoint{Input: input, Output: output})
	}

	return dataset
}

func toASCII(values []float64) string {
	colors := []string{" ", "░", "▒", "▓", "█"}
	stringVal := ""

	for i, val := range values {
		if i%28 == 0 && i != 0 {
			stringVal = fmt.Sprint(stringVal, "\n")
		}
		index := int(val / 256.0 * 5.0)
		stringVal = fmt.Sprint(stringVal, colors[index])
	}

	return stringVal
}

func PrintLetter(letter datasets.DataPoint) {
	fmt.Println("Printing Digit:", datasets.FromOneHot(letter.Output))
	fmt.Println(toASCII(letter.Input))
	fmt.Println("Output:", letter.Output)
}
