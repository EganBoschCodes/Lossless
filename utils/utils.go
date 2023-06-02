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

func GetMaxIndex[T int | float64 | float32](values []T) int {
	maxVal := values[0]
	maxInd := 0
	for i, val := range values {
		if val > maxVal {
			maxVal = val
			maxInd = i
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
