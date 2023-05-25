package optimizers

import (
	"sync"

	"github.com/EganBoschCodes/lossless/utils"
	"gonum.org/v1/gonum/mat"
)

type Momentum struct {
	Gamma float64

	cache   []*mat.Dense
	mutexes []sync.Mutex

	initialized bool
	size        int
}

func (mom *Momentum) Initialize(n int) {
	mom.cache = make([]*mat.Dense, n)
	mom.mutexes = make([]sync.Mutex, n)

	mom.initialized = true
	mom.size = n
}

func (mom *Momentum) Initialized() bool {
	return mom.initialized
}

func (mom *Momentum) Rescale(shift *mat.Dense, index int) *mat.Dense {
	mom.mutexes[index].Lock()
	if mom.cache[index] == nil {
		mom.cache[index] = utils.DenseLike(shift)
		mom.cache[index].Copy(shift)
	}

	cache := mom.cache[index]
	cache.Scale(mom.Gamma, cache)
	//utils.PrintMat("cache", cache)
	shift.Scale(1-mom.Gamma, shift)
	cache.Add(cache, shift)
	shift.Copy(cache)
	mom.mutexes[index].Unlock()
	/*cache.Apply(func(i, j int, v float64) float64 {
		shiftVal := shift.At(i, j)
		return mom.Gamma*v + (1-mom.Gamma)*shiftVal
	}, cache)*/

	//utils.PrintMat("cache", cache)
	return shift
}

func (mom *Momentum) Size() int {
	return mom.size
}
