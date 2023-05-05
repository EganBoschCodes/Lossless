package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func main() {
	zero := mat.NewDense(3, 5, nil)
	fmt.Println(zero)
}
