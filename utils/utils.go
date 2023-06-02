package utils

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

func Abs(a int) int {
	if a < 0 {
		return -a
	}
	return a
}

// Essentially getting a column of a matrix
func getParallelIndex[T any](index int, values ...[]T) []T {
	par := make([]T, len(values))
	for i := range values {
		par[i] = values[i][index]
	}
	return par
}

func GetMaxIndex[T int | float64 | float32](values ...[]T) int {
	maxVals := getParallelIndex(0, values...)
	maxInd := 0
	for i := range values[0] {
		currentVals := getParallelIndex(i, values...)
		for currentDepth := range values {
			if currentVals[currentDepth] > maxVals[currentDepth] {
				maxVals = currentVals
				maxInd = i
				break
			} else if currentVals[currentDepth] == maxVals[currentDepth] {
				continue
			}
			break
		}

	}
	return maxInd
}

func IntToBool(i int) bool {
	return i != 0
}

func BoolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}
