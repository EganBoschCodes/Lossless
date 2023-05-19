package utils

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

func GetSlice(m mat.Matrix) []float64 {
	return m.(*mat.Dense).RawMatrix().Data
}

func FromSlice(f []float64) mat.Matrix {
	return mat.NewDense(len(f), 1, f)
}

func DenseLike(m mat.Matrix) *mat.Dense {
	r, c := m.Dims()
	return mat.NewDense(r, c, nil)
}

/*
	Standard Matrix Convolution:
	----------------------------------------------------------------------------------
	Used for convolutional neural networks, this will convolve a matrix with a kernel.
	It will produce an output matrix with dimensions equal to that of the data matrix
	minus the dimensions of the kernel, plus one in each direction.

	Channeled and goroutined once again to promote performance.
*/

func convolveInner(data mat.Matrix, kernel mat.Matrix, output mat.Matrix, rstart int, rend int, cstart int, cend int, notifier chan int) {
	kr, kc := kernel.Dims()

	for r := rstart; r < rend; r++ {
		for c := cstart; c < cend; c++ {
			val := 0.0
			for or := 0; or < kr; or++ {
				for oc := 0; oc < kc; oc++ {
					val += data.At(r+or, c+oc) * kernel.At(or, oc)
				}
			}
			output.(*mat.Dense).Set(r, c, val)
		}
	}
	notifier <- 1
}

func ConvolveNoPadding(data mat.Matrix, kernel mat.Matrix) mat.Matrix {
	dr, dc := data.Dims()
	kr, kc := kernel.Dims()

	output := mat.NewDense(dr-kr+1, dc-kc+1, nil)
	or, oc := output.Dims()

	outputCounter := 0
	outputChannel := make(chan int)

	subdivisions := Max(1, or/7)
	for i := 0; i < subdivisions; i++ {
		for j := 0; j < subdivisions; j++ {
			go convolveInner(data, kernel, output, or*i/subdivisions, or*(i+1)/subdivisions, oc*j/subdivisions, oc*(j+1)/subdivisions, outputChannel)
		}
	}

	for outputCounter < subdivisions*subdivisions {
		outputCounter += <-outputChannel
	}

	return output
}

/*
	Padded Matrix Convolution:
	----------------------------------------------------------------------------------
	Used to get the backpropagation weights in a convolutional layer, this essentially
	just puts a layer of zeros around the data matrix of thickness equal to the dimensions
	of the kernel minus 1. So a 2x2 data matrix and a 2x2 kernel produce a 3x3 output.
*/

func convolveInnerPadding(data mat.Matrix, kernel mat.Matrix, output mat.Matrix, rstart int, rend int, cstart int, cend int, notifier chan int) {
	kr, kc := kernel.Dims()
	dr, dc := data.Dims()

	for r := rstart; r < rend; r++ {
		for c := cstart; c < cend; c++ {
			val := 0.0
			for or := 0; or < kr; or++ {
				for oc := 0; oc < kc; oc++ {
					if r-kr+1+or >= 0 && c-kc+1+oc >= 0 && r-kr+1+or < dr && c-kc+1+oc < dc {
						val += data.At(r-kr+1+or, c-kc+1+oc) * kernel.At(or, oc)
					}
				}
			}
			output.(*mat.Dense).Set(r, c, val)
		}
	}
	notifier <- 1
}

func ConvolveWithPadding(data mat.Matrix, kernel mat.Matrix) mat.Matrix {
	dr, dc := data.Dims()
	kr, kc := kernel.Dims()

	output := mat.NewDense(dr+kr-1, dc+kc-1, nil)
	or, oc := output.Dims()

	outputCounter := 0
	outputChannel := make(chan int)

	subdivisions := Max(1, or/7)
	for i := 0; i < subdivisions; i++ {
		for j := 0; j < subdivisions; j++ {
			go convolveInnerPadding(data, kernel, output, or*i/subdivisions, or*(i+1)/subdivisions, oc*j/subdivisions, oc*(j+1)/subdivisions, outputChannel)
		}
	}

	for outputCounter < subdivisions*subdivisions {
		outputCounter += <-outputChannel
	}

	return output
}

/*
	MaxPooling:
	----------------------------------------------------------------------------------
	Used as a kind of dimensionality reduction for convolutional neural nets, as the
	convolution layers usually multiply the amount of values passed to the next layer
	instead of reducing them.
*/

func MaxPool(data mat.Matrix, width int, height int) mat.Matrix {
	dr, dc := data.Dims()

	if dr%width != 0 || dc%height != 0 {
		fmt.Printf("A matrix of dimensions %d by %d cannot be max-pooled by a sample size of %d by %d.\n", dr, dc, width, height)
		return nil
	}

	output := mat.NewDense(dr/width, dc/height, nil)
	or, oc := output.Dims()
	for r := 0; r < or; r++ {
		for c := 0; c < oc; c++ {
			max := math.Inf(-1)
			for i := 0; i < width; i++ {
				for j := 0; j < height; j++ {
					max = math.Max(max, data.At(r*width+i, c*height+j))
				}
			}
			output.Set(r, c, max)
		}
	}

	return output
}

/*
	MaxPoolMap:
	----------------------------------------------------------------------------------
	Used again as a tool for backpropagation, this returns a matrix in the shape of the
	data input with 1's where the maximum value in each block maintains its value, and
	0's everywhere else.
*/

func MaxPoolMap(data mat.Matrix, width int, height int) mat.Matrix {
	dr, dc := data.Dims()

	if dr%width != 0 || dc%height != 0 {
		fmt.Printf("A matrix of dimensions %d by %d cannot be max-pooled by a sample size of %d by %d.\n", dr, dc, width, height)
		return nil
	}

	output := mat.NewDense(dr, dc, nil)

	for r := 0; r < dr; r += width {
		for c := 0; c < dc; c += height {
			max := math.Inf(-1)
			maxr, maxc := -1, -1
			for i := 0; i < width; i++ {
				for j := 0; j < height; j++ {
					output.Set(r+i, c+j, 0.1)
					if data.At(r+i, c+j) > max {
						max = data.At(r+i, c+j)
						maxr, maxc = i, j
					}
				}
			}
			output.Set(r+maxr, c+maxc, 1)
		}
	}

	return output
}

func UnMaxPool(data mat.Matrix, width int, height int) mat.Matrix {
	dr, dc := data.Dims()

	ret := mat.NewDense(dr*width, dc*height, nil)
	for r := 0; r < dr*width; r++ {
		for c := 0; c < dc*height; c++ {
			ret.Set(r, c, data.At(r/width, c/height))
		}
	}

	return ret
}

func PrintMat(name string, m mat.Matrix) {
	fmt.Println("mat ", name, " =")
	fmt.Printf("%.7f\n", mat.Formatted(m, mat.Prefix(""), mat.Squeeze()))
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
