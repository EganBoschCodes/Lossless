package optimizers

import "gonum.org/v1/gonum/mat"

type GradientDescent struct{}

func (g *GradientDescent) Rescale(shifts *mat.Dense) *mat.Dense {
	return shifts
}

func (g *GradientDescent) Initialize(_ int)  {}
func (g *GradientDescent) Initialized() bool { return true }
