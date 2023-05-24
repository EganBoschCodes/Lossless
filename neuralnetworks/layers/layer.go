package layers

import (
	"github.com/EganBoschCodes/lossless/neuralnetworks/optimizers"
	"gonum.org/v1/gonum/mat"
)

/*
LAYER - The basic interface for all inner layers of an ANN.

Initialize (numInputs int): Tells the layer how many inputs to expect, and sets up everything accordingly.

Pass (*mat.Dense) (*mat.Dense, CacheType): Passes the input through the layer to get an output, and cache necessary information to do backprop.

Back (cache CacheType, forwardGradients *mat.Dense) (shift ShiftType, backwardsGradients *mat.Dense): Takes the partial derivatives from the layers in front, calculates the gradient for itself, and passes it back to the last layer.
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

/*
This is an interface for allowing layers to designate
their own types of caches. For example, on Tanh layers,
it is best to cache the output of the layer for backprop
as it avoids recalculating tanh values, but for linear
layers it is better to cache the inputs.
*/
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

/*
This is an interface for carrying all the different
gradient steps that will be applied after backprop.
The default NilShift is defined here, but most layer
specific shift types are defined in their own files.
*/
type ShiftType interface {
	Apply(Layer, optimizers.Optimizer, float64)
	Combine(ShiftType) ShiftType
	NumMatrices() int
}

type NilShift struct{}

func (n *NilShift) Apply(_ Layer, _ optimizers.Optimizer, _ float64) {}
func (n *NilShift) Combine(other ShiftType) ShiftType {
	return other
}
func (n *NilShift) NumMatrices() int { return 0 }

/*
This allows for mapping between layer types and ints
for the sake of saving to files and reconstructing.
*/
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
