package optimizers

import (
	"math"

	"github.com/EganBoschCodes/lossless/utils"
	"gonum.org/v1/gonum/mat"
)

type RMSProp struct {
	Gamma float64

	cache []*mat.Dense

	counter     int
	initialized bool
}

func (rms *RMSProp) Initialize(n int) {
	rms.cache = make([]*mat.Dense, n)

	rms.initialized = true
}

func (rms *RMSProp) Initialized() bool {
	return rms.initialized
}

func (rms *RMSProp) Rescale(shift *mat.Dense) *mat.Dense {
	if rms.cache[rms.counter] == nil {
		rms.cache[rms.counter] = utils.DenseLike(shift)
		rms.cache[rms.counter].Copy(shift)
		rms.cache[rms.counter].Apply(func(_, _ int, v float64) float64 {
			return v * v
		}, rms.cache[rms.counter])
	}

	cache := rms.cache[rms.counter]
	cache.Apply(func(i, j int, v float64) float64 {
		shiftVal := shift.At(i, j)
		return rms.Gamma*v + (1-rms.Gamma)*shiftVal*shiftVal
	}, cache)

	shift.Apply(func(i, j int, v float64) float64 {
		return 1 / math.Sqrt(cache.At(i, j)+1e-7) * v
	}, shift)

	rms.counter = (rms.counter + 1) % len(rms.cache)
	return shift
}
