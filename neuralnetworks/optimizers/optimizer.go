package optimizers

import "github.com/EganBoschCodes/lossless/neuralnetworks/layers"

type Optimizer interface {
	Rescale([]layers.ShiftType) []layers.ShiftType
}

const (
	GradientDescent Optimizer = GradientDescentOptimizer{}
)
