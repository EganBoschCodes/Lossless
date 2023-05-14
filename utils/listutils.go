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

func Reverse[T any](vals []T) []T {
	rev := make([]T, len(vals))
	for i, val := range vals {
		rev[len(vals)-1-i] = val
	}
	return rev
}
