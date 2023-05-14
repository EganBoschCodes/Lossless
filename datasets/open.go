package datasets

import (
	"os"
	"strconv"
	"strings"
)

func Open(address string) [][]float64 {
	rawData, err := os.ReadFile(address)
	if err != nil {
		panic(err)
	}

	entries := strings.Split(string(rawData), "\r\n")
	data := make([][]float64, 0)

	for i, entry := range entries {
		if i == 0 {
			continue
		}

		stringEntry := strings.Split(entry, ",")
		datapoint := make([]float64, 0)

		for i := range stringEntry {
			val, err := strconv.ParseFloat(stringEntry[i], 32)
			if err != nil {
				continue
			}
			datapoint = append(datapoint, float64(val))
		}

		data = append(data, datapoint)
	}

	return data
}

func SimpleSplit(data [][]float64, targetStart int) []DataPoint {
	datapoints := make([]DataPoint, 0)
	for _, datapoint := range data {
		input, target := make([]float64, 0), make([]float64, 0)
		for i := range datapoint {
			if i < targetStart {
				input = append(input, datapoint[i])
			} else {
				target = append(target, datapoint[i])
			}
		}
		datapoints = append(datapoints, DataPoint{Input: input, Output: target})
	}
	return datapoints
}

func ToOneHot(value float64, maxValue int) []float64 {
	vec := make([]float64, maxValue)
	vec[int(value)] = 1
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
