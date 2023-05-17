package datasets

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/EganBoschCodes/lossless/utils"
)

func IsCorrect(output []float64, target []float64) {
	fmt.Printf("Output: %.2f\nTarget: %.2f\nCorrect: %t\n\n", output, target, utils.GetMaxIndex(output) == FromOneHot(target))
}

func NormalizeInputs(dataset []DataPoint) {
	rand.Shuffle(len(dataset), func(i, j int) { dataset[i], dataset[j] = dataset[j], dataset[i] })

	sampleSize := len(dataset)
	means := GetMean(dataset, sampleSize)

	numInputs := len(dataset[0].Input)
	stddevs := make([]float64, 0)
	for i := 0; i < numInputs; i++ {
		stddevs = append(stddevs, 0)
		for j := 0; j < utils.Min(sampleSize, len(dataset)); j++ {
			datapoint := dataset[j]
			diff := datapoint.Input[i] - means[i]
			stddevs[i] += diff * diff
		}
		stddevs[i] = math.Sqrt(stddevs[i]) / float64(utils.Min(sampleSize, len(dataset)))
	}

	for i, datapoint := range dataset {
		newInput := make([]float64, 0)
		for j := 0; j < numInputs; j++ {
			if stddevs[j] == 0 {
				newInput = append(newInput, 0)
				continue
			}
			newInput = append(newInput, (datapoint.Input[j]-means[j])/stddevs[j])
		}
		dataset[i].Input = newInput
	}
}

func GetMean(dataset []DataPoint, sampleSize int) []float64 {
	numSamples := utils.Min(sampleSize, len(dataset))

	means := make([]float64, 0)
	numInputs := len(dataset[0].Input)
	for i := 0; i < numInputs; i++ {
		means = append(means, 0)
		for j := 0; j < numSamples; j++ {
			datapoint := dataset[j]
			means[i] += datapoint.Input[i]
		}
		means[i] /= float64(numSamples)
	}
	return means
}

func Apply(dataset []DataPoint, f func([]float64) []float64) {
	for i, datapoint := range dataset {
		dataset[i] = DataPoint{Input: f(datapoint.Input), Output: datapoint.Output}
	}
}

func GetInputs(dataset []DataPoint) (inputs [][]float64) {
	return utils.Map(dataset, func(d DataPoint) []float64 { return d.Input })
}

func GetOutputs(dataset []DataPoint) (outputs [][]float64) {
	return utils.Map(dataset, func(d DataPoint) []float64 { return d.Output })
}

func Split(dataset []DataPoint) (inputs [][]float64, outputs [][]float64) {
	return GetInputs(dataset), GetOutputs(dataset)
}

func ToOneHot(value int, maxValue int) []float64 {
	vec := make([]float64, maxValue)
	vec[value] = 1
	return vec
}

func FromOneHot(output []float64) int {
	for i, val := range output {
		if val != 0 {
			return i
		}
	}
	return -1
}
