package optimizers

import "gonum.org/v1/gonum/mat"

type GradientDescent struct {
	size int
}

func (g *GradientDescent) Rescale(shifts *mat.Dense, _ int) *mat.Dense {
	return shifts
}

func (g *GradientDescent) Initialize(n int)  { g.size = n }
func (g *GradientDescent) Initialized() bool { return true }
func (g *GradientDescent) Size() int         { return g.size }
