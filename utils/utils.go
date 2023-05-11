package utils

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

func PrintMat(name string, m mat.Matrix) {
	fmt.Println("mat ", name, " =")
	fmt.Printf("%.7f\n", mat.Formatted(m, mat.Prefix(""), mat.Squeeze()))
}

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

func Reduce(vals []float64, reduction func(float64, float64) float64) float64 {
	ret := vals[0]
	for i := 1; i < len(vals); i++ {
		ret = reduction(ret, vals[i])
	}
	return ret
}

func Map(vals []float64, mapfunc func(float64) float64) []float64 {
	mappedVals := make([]float64, len(vals))
	for i, val := range vals {
		mappedVals[i] = mapfunc(val)
	}
	return mappedVals
}

func Reverse(vals []float64) []float64 {
	rev := make([]float64, len(vals))
	for i, val := range vals {
		rev[len(vals)-1-i] = val
	}
	return rev
}

func JSify(m mat.Matrix) string {
	retString := "["
	mr, mc := m.Dims()
	for r := 0; r < mr; r++ {
		retString += "["
		for c := 0; c < mc; c++ {
			retString += fmt.Sprintf("%.8f", m.At(r, c))
			if c < mc-1 {
				retString += ", "
			}
		}
		retString += "]"
		if r < mr-1 {
			retString += ",\n"
		}
	}
	retString += "]"
	return retString
}
