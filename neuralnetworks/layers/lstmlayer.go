package layers

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type LSTMLayer struct {
	NumOutputs     int
	IntervalSize   int
	OutputSequence bool

	numInputs int
	numConcat int

	forgetGate    mat.Matrix
	inputGate     mat.Matrix
	candidateGate mat.Matrix
	outputGate    mat.Matrix
}

func (layer *LSTMLayer) Initialize(numInputs int) {
	if layer.NumOutputs == 0 {
		panic("Set how many outputs you want in your LSTM layer!")
	}
	if layer.IntervalSize == 0 {
		panic("Set how long the time series is being passed to your LSTM layer!")
	}
	if numInputs%layer.IntervalSize != 0 {
		panic("Your number of inputs to LSTM layer should be cleanly divided by IntervalSize!")
	}

	layer.numInputs = numInputs / layer.IntervalSize
	layer.numConcat = layer.numInputs + layer.NumOutputs

	if layer.forgetGate != nil {
		return
	}

	// Use Xavier Initialization on the weights
	fan_avg := (float64(layer.numInputs) + float64(layer.NumOutputs)) / 2
	forgetSlice, inputSlice, candidateSlice, outputSlice := make([]float64, (layer.numConcat+1)*layer.NumOutputs), make([]float64, (layer.numConcat+1)*layer.NumOutputs), make([]float64, (layer.numConcat+1)*layer.NumOutputs), make([]float64, (layer.numConcat+1)*layer.NumOutputs)
	for i := range forgetSlice {
		forgetSlice[i], inputSlice[i], candidateSlice[i], outputSlice[i] = rand.NormFloat64()/fan_avg, rand.NormFloat64()/fan_avg, rand.NormFloat64()/fan_avg, rand.NormFloat64()/fan_avg
	}
	layer.forgetGate, layer.inputGate, layer.candidateGate, layer.outputGate = mat.NewDense(layer.NumOutputs, layer.numConcat+1, forgetSlice), mat.NewDense(layer.NumOutputs, layer.numConcat+1, inputSlice), mat.NewDense(layer.NumOutputs, layer.numConcat+1, candidateSlice), mat.NewDense(layer.NumOutputs, layer.numConcat+1, outputSlice)
}

func (layer *LSTMLayer) Pass(input mat.Matrix) {

}
