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

	ToBytes() []byte
	FromBytes([]byte)
	PrettyPrint() string
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
	weightShift mat.Matrix
	biasShift   mat.Matrix
}

func (w *WeightShift) Apply(layer Layer, scale float64) {
	w.weightShift.(*mat.Dense).Scale(scale, w.weightShift)
	w.biasShift.(*mat.Dense).Scale(scale, w.biasShift)

	layer.(*LinearLayer).weights.(*mat.Dense).Add(layer.(*LinearLayer).weights, w.weightShift)
	layer.(*LinearLayer).biases.(*mat.Dense).Add(layer.(*LinearLayer).biases, w.biasShift)
}

func (w *WeightShift) Combine(w2 ShiftType) ShiftType {
	w.weightShift.(*mat.Dense).Add(w.weightShift, w2.(*WeightShift).weightShift)
	w.biasShift.(*mat.Dense).Add(w.biasShift, w2.(*WeightShift).biasShift)

	return w
}

type KernelShift struct {
	shifts []mat.Matrix
}

func (k *KernelShift) Apply(layer Layer, scale float64) {
	for i, shift := range k.shifts {
		shift.(*mat.Dense).Scale(scale, shift)
		layer.(*Conv2DLayer).kernels[i].(*mat.Dense).Add(layer.(*Conv2DLayer).kernels[i], shift)
	}
}

func (k *KernelShift) Combine(k2 ShiftType) ShiftType {
	for i := range k.shifts {
		k.shifts[i].(*mat.Dense).Add(k.shifts[i], k2.(*KernelShift).shifts[i])
	}
	return k
}

func IndexToLayer(index int) Layer {
	switch index {
	case 0:
		return &LinearLayer{}
	case 1:
		return &ReluLayer{}
	case 2:
		return &SigmoidLayer{}
	case 3:
		return &TanhLayer{}
	case 4:
		return &SoftmaxLayer{}
	case 5:
		return &Conv2DLayer{}
	case 6:
		return &MaxPool2DLayer{}
	case 7:
		return &FlattenLayer{}
	default:
		return nil
	}
}

func LayerToIndex(layer Layer) int {
	switch layer.(type) {
	case *LinearLayer:
		return 0
	case *ReluLayer:
		return 1
	case *SigmoidLayer:
		return 2
	case *TanhLayer:
		return 3
	case *SoftmaxLayer:
		return 4
	case *Conv2DLayer:
		return 5
	case *MaxPool2DLayer:
		return 6
	case *FlattenLayer:
		return 7
	default:
		return -1
	}
}
