package utils

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func PrintMat(name string, m mat.Matrix) {
	fmt.Println("mat ", name, " =")
	fmt.Printf("%.2f\n", mat.Formatted(m, mat.Prefix(""), mat.Squeeze()))
}
