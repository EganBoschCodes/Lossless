package utils

import "fmt"

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

func MapWithIndex[T any, U any](vals []T, mapfunc func(int, T) U) []U {
	mappedVals := make([]U, len(vals))
	for i, val := range vals {
		mappedVals[i] = mapfunc(i, val)
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

func ForEach[T any](vals []T, foreach func(int, T)) {
	for i, val := range vals {
		foreach(i, val)
	}
}

// Returns the last item in a list.
func LastOf[T any](vals []T) T {
	return vals[len(vals)-1]
}

// Pointwise difference of two lists of floats.
func Subtract(a []float64, b []float64) []float64 {
	return DoubleMap(a, b, func(va float64, vb float64) float64 { return va - vb })
}

// Pointwise sum of two lists of floats.
func Add(a []float64, b []float64) []float64 {
	return DoubleMap(a, b, func(va float64, vb float64) float64 { return va + vb })
}

// Takes in a long single slice and cuts it into intervals of the given size. Used internally to take a giant slice containing a list of inputs and cut and map them into the individual input matrices, for example.
func Cut[T any](vals []T, intervalSize int) [][]T {
	if len(vals)%intervalSize != 0 {
		panic(fmt.Sprintf("A list of length %d can't be cut into intervals of %d!", len(vals), intervalSize))
	}

	cutVals := make([][]T, len(vals)/intervalSize)
	for i := range cutVals {
		cutVals[i] = vals[i*intervalSize : (i+1)*intervalSize]
	}
	return cutVals
}

func Duplicate[T any](val T, length int) []T {
	vals := make([]T, length)
	for i := range vals {
		vals[i] = val
	}
	return vals
}
