package optimizers

import "gonum.org/v1/gonum/mat"

type Optimizer interface {
	Rescale(*mat.Dense) *mat.Dense

	Initialize(int)
	Initialized() bool
}
