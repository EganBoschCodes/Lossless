package utils

func Reduce[T any](vals []T, reduction func(T, T) T) T {
	ret := vals[0]
	for i := 1; i < len(vals); i++ {
		ret = reduction(ret, vals[i])
	}
	return ret
}

func Map[T any, U any](vals []T, mapfunc func(T) U) []U {
	mappedVals := make([]U, len(vals))
	for i, val := range vals {
		mappedVals[i] = mapfunc(val)
	}
	return mappedVals
}

func Map2D[T any, U any](vals [][]T, mapfunc func(T) U) [][]U {
	mappedVals := make([][]U, len(vals))
	for i, val := range vals {
		mappedVals[i] = Map(val, mapfunc)
	}
	return mappedVals
}

func DoubleMap[T any, U any, V any](first []T, second []U, mapfunc func(T, U) V) []V {
	if len(first) != len(second) {
		panic("Arrays are not the same length in DoubleMap call!")
	}

	mappedVals := make([]V, len(first))
	for i := range first {
		mappedVals[i] = mapfunc(first[i], second[i])
	}
	return mappedVals
}

func Reverse[T any](vals []T) []T {
	rev := make([]T, len(vals))
	for i, val := range vals {
		rev[len(vals)-1-i] = val
	}
	return rev
}

func Flatten[T any](vals [][]T) []T {
	flat := make([]T, 0)
	for _, val := range vals {
		flat = append(flat, val...)
	}
	return flat
}

func Filter[T any](vals []T, f func(T) bool) []T {
	filteredVals := make([]T, 0)
	for _, val := range vals {
		if f(val) {
			filteredVals = append(filteredVals, val)
		}
	}
	return filteredVals
}

func All[T any](vals []T, f func(T) bool) bool {
	for _, val := range vals {
		if !f(val) {
			return false
		}
	}
	return true
}

func Any[T any](vals []T, f func(T) bool) bool {
	for _, val := range vals {
		if f(val) {
			return true
		}
	}
	return false
}

func Count[T any](vals []T, f func(T) bool) int {
	num := 0
	for _, val := range vals {
		if f(val) {
			num++
		}
	}
	return num
}

func Find[T comparable](vals []T, val T) int {
	for i, v := range vals {
		if v == val {
			return i
		}
	}
	return -1
}

func FindWithCompare[T any, U any](vals []T, val U, f func(T, U) bool) int {
	for i, v := range vals {
		if f(v, val) {
			return i
		}
	}
	return -1
}

func LastOf[T any](vals []T) T {
	return vals[len(vals)-1]
}
