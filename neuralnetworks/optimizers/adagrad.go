package optimizers

import (
	"math"
	"sync"

	"github.com/EganBoschCodes/lossless/utils"
	"gonum.org/v1/gonum/mat"
)

type AdaGrad struct {
	Epsilon float64

	cache   []*mat.Dense
	mutexes []sync.Mutex

	cachesPopped int
	initialized  bool
}

func (ada *AdaGrad) Initialize(n int) {
	ada.cache = make([]*mat.Dense, n)
	ada.mutexes = make([]sync.Mutex, n)
	ada.cachesPopped = 1

	if ada.Epsilon == 0 {
		ada.Epsilon = 1e-8
	}

	ada.initialized = true
}

func (ada *AdaGrad) Initialized() bool {
	return ada.initialized
}

func (ada *AdaGrad) Size() int {
	return len(ada.cache)
}

func (ada *AdaGrad) Rescale(shift *mat.Dense, index int) *mat.Dense {
	ada.mutexes[index].Lock()
	if ada.cache[index] == nil {
		ada.cache[index] = utils.DenseLike(shift)
	}

	cache := ada.cache[index]
	cache = utils.FastApply(cache, func(i, j int, v float64) float64 {
		shiftVal := shift.At(i, j)
		return v + shiftVal*shiftVal
	})
	ada.mutexes[index].Unlock()

	shift = utils.FastApply(shift, func(i, j int, v float64) float64 {
		return v / (math.Sqrt(cache.At(i, j)/float64(ada.cachesPopped)) + ada.Epsilon)
	})

	if index == len(ada.cache) {
		ada.cachesPopped++
	}

	return shift
}
