package layers

import (
	"gonum.org/v1/gonum/mat"
)

/*
	LAYER - The basic interface for all inner layers of an ANN.
	-----------------------------------------------------------
	Initialize (numInputs int, numOutputs int): Tells the layer how many inputs and how many outputs to expect.
	Pass (input mat.Vector) (output mat.Vector): Passes the input through the layer to get an output.
	Back (forwardGradients mat.Vector) (shifts mat.Matrix, backwardsPass mat.Vector): Takes the partial derivatives from the layers in front, calculates the gradient for itself, and passes it back to the last layer.
*/

type Layer interface {
	Initialize(int)
	Pass(mat.Matrix) mat.Matrix
	Back(mat.Matrix, mat.Matrix, mat.Matrix) (ShiftType, mat.Matrix)
	NumOutputs() int
}

type ShiftType interface {
	Apply(Layer, float64)
	Combine(ShiftType) ShiftType
}

type NilShift struct{}

func (n *NilShift) Apply(_ Layer, _ float64) {}
func (n *NilShift) Combine(other ShiftType) ShiftType {
	return other
}

type WeightShift struct {
	shift mat.Matrix
}

func (w *WeightShift) Apply(layer Layer, scale float64) {
	w.shift.(*mat.Dense).Scale(scale, w.shift)
	//utils.PrintMat("wshift", w.shift)
	layer.(*LinearLayer).weights.(*mat.Dense).Add(layer.(*LinearLayer).weights, w.shift)
}

func (w *WeightShift) Combine(w2 ShiftType) ShiftType {
	w.shift.(*mat.Dense).Add(w.shift, w2.(*WeightShift).shift)
	return w
}

type KernelShift struct {
	shifts []mat.Matrix
}

func (k *KernelShift) Apply(layer Layer, scale float64) {
	//for i, shift := range k.shifts {
	//shift.(*mat.Dense).Scale(scale, shift)
	//layer.(*Conv2DLayer).kernels[i].(*mat.Dense).Add(layer.(*Conv2DLayer).kernels[i], shift)
	//}
}

func (k *KernelShift) Combine(k2 ShiftType) ShiftType {
	for i := range k.shifts {
		k.shifts[i].(*mat.Dense).Add(k.shifts[i], k2.(*KernelShift).shifts[i])
	}
	return k
}
