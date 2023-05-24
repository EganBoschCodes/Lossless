package optimizers

import (
	"math"

	"github.com/EganBoschCodes/lossless/utils"
	"gonum.org/v1/gonum/mat"
)

type AdaGrad struct {
	cache []*mat.Dense

	counter      int
	cachesPopped int
	initialized  bool
}

func (ada *AdaGrad) Initialize(n int) {
	ada.cache = make([]*mat.Dense, n)
	ada.cachesPopped = 1
}

func (ada *AdaGrad) Initialized() bool {
	return ada.initialized
}

func (ada *AdaGrad) Rescale(shift *mat.Dense) *mat.Dense {
	if ada.cache[ada.counter] == nil {
		ada.cache[ada.counter] = utils.DenseLike(shift)
	}

	cache := ada.cache[ada.counter]
	cache.Apply(func(i, j int, v float64) float64 {
		shiftVal := shift.At(i, j)
		return v + shiftVal*shiftVal
	}, cache)

	shift.Apply(func(i, j int, v float64) float64 {
		return 1 / math.Sqrt(cache.At(i, j)/float64(ada.cachesPopped)+1e-10) * v
	}, shift)

	ada.counter++
	if ada.counter >= len(ada.cache) {
		ada.counter = 0
		ada.cachesPopped++
	}

	return shift
}
