package networks

import "github.com/EganBoschCodes/lossless/neuralnetworks/layers"

type LSTM struct {
	ForgetLayers    []layers.Layer
	InputLayers     []layers.Layer
	CandidateLayers []layers.Layer
	OutputLayers    []layers.Layer

	BatchSize    int
	LearningRate float64

	numInputs int
}
