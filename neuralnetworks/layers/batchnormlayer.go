package layers

import (
	"fmt"

	"github.com/EganBoschCodes/lossless/neuralnetworks/optimizers"
	"github.com/EganBoschCodes/lossless/neuralnetworks/save"
	"github.com/EganBoschCodes/lossless/utils"

	"gonum.org/v1/gonum/mat"
)

type BatchnormLayer struct {
	BatchSize     int
	GradientScale float64

	means          *mat.Dense
	stddevs        *mat.Dense
	trainedMeans   *mat.Dense
	trainedStddevs *mat.Dense

	cache           []*mat.Dense
	numCachesPopped int
	n_inputs        int

	appendChannel chan *mat.Dense
}

func (layer *BatchnormLayer) Initialize(n_inputs int) {
	layer.n_inputs = n_inputs
	if layer.GradientScale == 0 {
		layer.GradientScale = 1
	}
	if layer.BatchSize == 0 {
		layer.BatchSize = 30
	}

	layer.appendChannel = make(chan *mat.Dense)
	go layer.recordCache()

	if layer.means != nil {
		return
	}

	layer.means, layer.trainedMeans = mat.NewDense(n_inputs, 1, nil), mat.NewDense(n_inputs, 1, nil)
	layer.stddevs, layer.trainedStddevs = utils.FromSlice(utils.Duplicate(1.0, n_inputs)), utils.FromSlice(utils.Duplicate(1.0, n_inputs))
}

func (layer *BatchnormLayer) recordCache() {
	val := <-layer.appendChannel
	for {
		layer.cache = append(layer.cache, val)
		val = <-layer.appendChannel
	}
}

func (layer *BatchnormLayer) Pass(input *mat.Dense) (*mat.Dense, CacheType) {
	// Check for means and standard deviations of values coming in.
	layer.appendChannel <- input

	// Normalize, then rescale.
	_, cols := input.Dims()
	normed, output := utils.DenseLike(input), utils.DenseLike(input)
	normed.Apply(func(i int, j int, v float64) float64 {
		return (v - layer.means.At(i*cols+j, 0)) / layer.stddevs.At(i*cols+j, 0)
	}, input)
	output.Apply(func(i int, j int, v float64) float64 {
		return v*layer.trainedStddevs.At(i*cols+j, 0) + layer.trainedMeans.At(i*cols+j, 0)
	}, normed)

	return output, &BatchNormCache{Normed: normed}
}

// When we've accumulated a sufficiently large cache, we update our tracked means and stddevs.
func (layer *BatchnormLayer) popCache() {
	values := make([][]float64, layer.n_inputs)
	for _, stored := range layer.cache {
		if stored != nil {
			for i, val := range utils.GetSlice(stored) {
				values[i] = append(values[i], val)
			}
		}
	}

	meanSlice, stddevSlice := make([]float64, layer.n_inputs), make([]float64, layer.n_inputs)

	utils.ForEach(values, func(i int, vals []float64) {
		mean, stddev := utils.GetDistribution(vals)
		meanSlice[i], stddevSlice[i] = mean/float64(layer.numCachesPopped+1), stddev/float64(layer.numCachesPopped+1)
	})

	layer.means.Scale(float64(layer.numCachesPopped)/float64(layer.numCachesPopped+1), layer.means)
	layer.stddevs.Scale(float64(layer.numCachesPopped)/float64(layer.numCachesPopped+1), layer.stddevs)

	layer.means.Add(layer.means, utils.FromSlice(meanSlice))
	layer.stddevs.Add(layer.stddevs, utils.FromSlice(stddevSlice))

	layer.cache = make([]*mat.Dense, 0)
	layer.numCachesPopped++
}

func (layer *BatchnormLayer) Back(cache CacheType, forwardGradients *mat.Dense) (ShiftType, *mat.Dense) {
	meanShifts := utils.DenseLike(layer.means)
	normedSlice := utils.GetSlice(cache.(*BatchNormCache).Normed)

	// Calculate the Local Gradients
	_, cols := forwardGradients.Dims()
	stddevShifts := utils.FromSlice(utils.MapWithIndex(utils.GetSlice(forwardGradients), func(i int, v float64) float64 {
		return v * normedSlice[i] * layer.GradientScale
	}))
	meanShifts.Copy(forwardGradients)

	// Rescale the gradients to pass back
	forwardGradients.Apply(func(i, j int, v float64) float64 {
		return v * layer.trainedStddevs.At(i*cols+j, 0) / layer.stddevs.At(i*cols+j, 0)
	}, forwardGradients)

	return &BatchNormShift{meanShift: meanShifts, stddevShift: stddevShifts}, forwardGradients
}

func (layer *BatchnormLayer) NumOutputs() int {
	return layer.n_inputs
}

func (layer *BatchnormLayer) ToBytes() []byte {
	bytes := save.ConstantsToBytes(layer.n_inputs, layer.numCachesPopped, layer.BatchSize)
	bytes = append(bytes, save.ToBytes(utils.GetSlice(layer.means))...)
	bytes = append(bytes, save.ToBytes(utils.GetSlice(layer.stddevs))...)
	bytes = append(bytes, save.ToBytes(utils.GetSlice(layer.trainedMeans))...)
	bytes = append(bytes, save.ToBytes(utils.GetSlice(layer.trainedStddevs))...)
	return bytes
}

func (layer *BatchnormLayer) FromBytes(bytes []byte) {
	constants := save.ConstantsFromBytes(bytes[:12])
	layer.n_inputs, layer.numCachesPopped, layer.BatchSize, bytes = constants[0], constants[1], constants[2], bytes[12:]
	layer.means, layer.stddevs = utils.FromSlice(save.FromBytes(bytes[:layer.n_inputs*8])), utils.FromSlice(save.FromBytes(bytes[layer.n_inputs*8:layer.n_inputs*16]))
	layer.trainedMeans, layer.trainedStddevs = utils.FromSlice(save.FromBytes(bytes[layer.n_inputs*16:layer.n_inputs*24])), utils.FromSlice(save.FromBytes(bytes[layer.n_inputs*24:]))
}

func (layer *BatchnormLayer) PrettyPrint() string {
	return fmt.Sprintf("Batchnorm Layer\nMeans: %.4f,\nStdDevs: %.4f\nTrained Means: %.4f\nTrained StdDevs: %.4f\n", utils.GetSlice(layer.means), utils.GetSlice(layer.stddevs), utils.GetSlice(layer.trainedMeans), utils.GetSlice(layer.trainedStddevs))
}

type BatchNormShift struct {
	meanShift   *mat.Dense
	stddevShift *mat.Dense
}

func (b *BatchNormShift) Apply(rawlayer Layer, opt optimizers.Optimizer, scale float64) {
	layer := rawlayer.(*BatchnormLayer)
	if len(layer.cache) >= layer.BatchSize {
		layer.popCache()
	}

	b.meanShift, b.stddevShift = opt.Rescale(b.meanShift), opt.Rescale(b.stddevShift)
	b.meanShift.Scale(scale, b.meanShift)
	layer.trainedMeans.Add(layer.trainedMeans, b.meanShift)

	b.stddevShift.Scale(scale, b.stddevShift)
	layer.trainedStddevs.Add(layer.trainedStddevs, b.stddevShift)
}

func (b *BatchNormShift) Combine(b2 ShiftType) ShiftType {
	b.meanShift.Add(b.meanShift, b2.(*BatchNormShift).meanShift)
	b.stddevShift.Add(b.stddevShift, b2.(*BatchNormShift).stddevShift)

	return b
}

func (b *BatchNormShift) NumMatrices() int {
	return 2
}

func (b *BatchNormShift) Scale(f float64) {
	b.meanShift.Scale(f, b.meanShift)
	b.stddevShift.Scale(f, b.stddevShift)
}
