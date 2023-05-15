package utils

import (
	"math"
)

func Min(a int, b int) int {
	if a < b {
		return a
	}
	return b
}

func Max(a int, b int) int {
	if a > b {
		return a
	}
	return b
}

func GetMaxIndex(values []float64) int {
	maxVal := math.Inf(-1)
	maxInd := -1
	for i, val := range values {
		if val > maxVal {
			maxVal = val
			maxInd = i
		}
	}
	return maxInd
}
