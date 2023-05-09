package layers

import "gonum.org/v1/gonum/mat"

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
	Back(mat.Matrix, mat.Matrix, mat.Matrix) (mat.Matrix, mat.Matrix)
	GetShape() mat.Matrix
	ApplyShift(mat.Matrix, float64)
	NumOutputs() int
}
