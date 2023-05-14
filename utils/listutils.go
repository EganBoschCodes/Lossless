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
