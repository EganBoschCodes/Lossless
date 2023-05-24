package optimizers

import "github.com/EganBoschCodes/lossless/neuralnetworks/layers"

type GradientDescent struct{}

func (g *GradientDescent) Rescale(shifts []layers.ShiftType) []layers.ShiftType {
	return shifts
}
