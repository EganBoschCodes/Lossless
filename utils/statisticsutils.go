package utils

import "math"

func GetDistribution(values []float64) (mean float64, stddev float64) {
	length := float64(len(values))

	mean = Reduce(values, func(a float64, b float64) float64 { return a + b }) / length
	variance := Reduce(Map(values, func(a float64) float64 { return (a - mean) * (a - mean) }), func(a float64, b float64) float64 { return a + b }) / length

	return mean, math.Sqrt(variance)
}

func Normalize(values []float64) (normalizedValues []float64, mean float64, stddev float64) {
	mean, stddev = GetDistribution(values)

	// Don't wanna divide by zero
	if stddev == 0 {
		return make([]float64, len(values)), mean, 0
	}

	return Map(values, func(f float64) float64 { return (f - mean) / stddev }), mean, stddev
}
