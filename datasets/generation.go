package datasets

import (
	"lossless/utils"
	"math"
	"math/rand"
)

type DataPoint struct {
	Input  []float64
	Output []float64
}

func GetSpiralDataset() []DataPoint {
	points := make([]DataPoint, 0)

	for r := 0.2; r < 3; r += 0.05 {
		p1 := DataPoint{Input: []float64{r * math.Sin(r), r * math.Cos(r)}, Output: []float64{1, 0, 0}}
		p2 := DataPoint{Input: []float64{r * math.Sin(r+2.049), r * math.Cos(r+2.049)}, Output: []float64{0, 1, 0}}
		p3 := DataPoint{Input: []float64{r * math.Sin(r-2.049), r * math.Cos(r-2.049)}, Output: []float64{0, 0, 1}}

		points = append(points, p1, p2, p3)
	}

	rand.Shuffle(len(points), func(i, j int) { points[i], points[j] = points[j], points[i] })

	return points
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
