package utils

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

func PrintMat(name string, m mat.Matrix) {
	fmt.Println("mat ", name, " =")
	fmt.Printf("%.2f\n", mat.Formatted(m, mat.Prefix(""), mat.Squeeze()))
}

func Min(a int, b int) int {
	if a < b {
		return a
	}
	return b
}

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
