package optimizers

import (
	"math"
	"sync"

	"github.com/EganBoschCodes/lossless/utils"
	"gonum.org/v1/gonum/mat"
)

type RMSProp struct {
	Gamma   float64
	Epsilon float64

	cache   []*mat.Dense
	mutexes []sync.Mutex

	initialized bool
}

func (rms *RMSProp) Initialize(n int) {
	rms.cache = make([]*mat.Dense, n)
	rms.mutexes = make([]sync.Mutex, n)

	if rms.Epsilon == 0 {
		rms.Epsilon = 1e-8
	}

	rms.initialized = true
}

func (rms *RMSProp) Initialized() bool {
	return rms.initialized
}

func (rms *RMSProp) Size() int {
	return len(rms.cache)
}

func (rms *RMSProp) Rescale(shift *mat.Dense, index int) *mat.Dense {
	rms.mutexes[index].Lock()
	if rms.cache[index] == nil {
		rms.cache[index] = utils.DenseLike(shift)
		rms.cache[index].Copy(shift)
		rms.cache[index].Apply(func(_, _ int, v float64) float64 {
			return v * v
		}, rms.cache[index])
	}

	cache := rms.cache[index]
	rms.cache[index] = utils.FastApply(cache, func(i, j int, v float64) float64 {
		shiftVal := shift.At(i, j)
		return rms.Gamma*v + (1-rms.Gamma)*shiftVal*shiftVal
	})
	rms.mutexes[index].Unlock()

	shift = utils.FastApply(shift, func(i, j int, v float64) float64 {
		return v / math.Sqrt(cache.At(i, j)+rms.Epsilon)
	})
	return shift
}
