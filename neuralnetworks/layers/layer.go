package layers

import (
	"gonum.org/v1/gonum/mat"
)

/*
	LAYER - The basic interface for all inner layers of an ANN.
	-----------------------------------------------------------
	Initialize (numInputs int, numOutputs int): Tells the layer how many inputs and how many outputs to expect.
	Pass (input mat.Vector) (output mat.Vector): Passes the input through the layer to get an output.
	Back (forwardGradients mat.Vector) (shifts *mat.Dense, backwardsPass mat.Vector): Takes the partial derivatives from the layers in front, calculates the gradient for itself, and passes it back to the last layer.
*/

type Layer interface {
	Initialize(int)
	Pass(*mat.Dense) (*mat.Dense, CacheType)
	Back(CacheType, *mat.Dense) (ShiftType, *mat.Dense)
	NumOutputs() int

	ToBytes() []byte
	FromBytes([]byte)
	PrettyPrint() string
}

type Shape struct {
	Rows int
	Cols int
}

type CacheType interface{}

type InputCache struct {
	Input *mat.Dense
}

type OutputCache struct {
	Output *mat.Dense
}

type BatchNormCache struct {
	Normed *mat.Dense
}

type LSTMCache struct {
	Inputs           []*mat.Dense
	HiddenStates     []*mat.Dense
	CellStates       []*mat.Dense
	ForgetOutputs    []*mat.Dense
	InputOutputs     []*mat.Dense
	CandidateOutputs []*mat.Dense
	OutputOutputs    []*mat.Dense
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
	weightShift *mat.Dense
	biasShift   *mat.Dense
}

func (w *WeightShift) Apply(layer Layer, scale float64) {
	w.weightShift.Scale(scale, w.weightShift)
	w.biasShift.Scale(scale, w.biasShift)

	layer.(*LinearLayer).weights.Add(layer.(*LinearLayer).weights, w.weightShift)
	layer.(*LinearLayer).biases.Add(layer.(*LinearLayer).biases, w.biasShift)
}

func (w *WeightShift) Combine(w2 ShiftType) ShiftType {
	w.weightShift.Add(w.weightShift, w2.(*WeightShift).weightShift)
	w.biasShift.Add(w.biasShift, w2.(*WeightShift).biasShift)

	return w
}

type KernelShift struct {
	shifts []*mat.Dense
}

func (k *KernelShift) Apply(layer Layer, scale float64) {
	for i, shift := range k.shifts {
		shift.Scale(scale, shift)
		layer.(*Conv2DLayer).kernels[i].Add(layer.(*Conv2DLayer).kernels[i], shift)
	}
}

func (k *KernelShift) Combine(k2 ShiftType) ShiftType {
	for i := range k.shifts {
		k.shifts[i].Add(k.shifts[i], k2.(*KernelShift).shifts[i])
	}
	return k
}

type BatchNormShift struct {
	meanShift   *mat.Dense
	stddevShift *mat.Dense
}

func (k *BatchNormShift) Apply(rawlayer Layer, scale float64) {
	layer := rawlayer.(*BatchnormLayer)
	if len(layer.cache) >= layer.BatchSize {
		layer.popCache()
	}

	k.meanShift.Scale(scale, k.meanShift)
	layer.trainedMeans.Add(layer.trainedMeans, k.meanShift)

	k.stddevShift.Scale(scale, k.stddevShift)
	layer.trainedStddevs.Add(layer.trainedStddevs, k.stddevShift)
}

func (k *BatchNormShift) Combine(k2 ShiftType) ShiftType {
	k.meanShift.Add(k.meanShift, k2.(*BatchNormShift).meanShift)
	k.stddevShift.Add(k.stddevShift, k2.(*BatchNormShift).stddevShift)

	return k
}

type LSTMShift struct {
	forgetShift    ShiftType
	inputShift     ShiftType
	candidateShift ShiftType
	outputShift    ShiftType
}

func (l *LSTMShift) Apply(layer Layer, scale float64) {
	lstmLayer := layer.(*LSTMLayer)
	l.forgetShift.Apply(&lstmLayer.forgetGate, scale)
	l.inputShift.Apply(&lstmLayer.inputGate, scale)
	l.candidateShift.Apply(&lstmLayer.candidateGate, scale)
	l.outputShift.Apply(&lstmLayer.outputGate, scale)
}

func (l *LSTMShift) Combine(l2 ShiftType) ShiftType {
	lstm2 := l2.(*LSTMShift)
	l.forgetShift = l.forgetShift.Combine(lstm2.forgetShift)
	l.inputShift = l.inputShift.Combine(lstm2.inputShift)
	l.candidateShift = l.candidateShift.Combine(lstm2.candidateShift)
	l.outputShift = l.outputShift.Combine(lstm2.outputShift)

	return l
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
	case 8:
		return &LSTMLayer{}
	case 9:
		return &BatchnormLayer{}
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
	case *LSTMLayer:
		return 8
	case *BatchnormLayer:
		return 9
	default:
		return -1
	}
}
