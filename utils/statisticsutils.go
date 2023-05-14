package utils

import "math"

func GetDistribution(values []float64) (float64, float64) {
	mean := 0.0
	for _, val := range values {
		mean += val
	}
	mean /= float64(len(values))

	variance := 0.0
	for _, val := range values {
		variance += (val - mean) * (val - mean)
	}
	variance /= float64(len(values))

	return mean, math.Sqrt(variance)
}
