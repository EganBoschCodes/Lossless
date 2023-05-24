package optimizers

import (
	"github.com/EganBoschCodes/lossless/utils"
	"gonum.org/v1/gonum/mat"
)

type Momentum struct {
	Gamma float64

	cache []*mat.Dense

	counter     int
	initialized bool
}

func (mom *Momentum) Initialize(n int) {
	mom.cache = make([]*mat.Dense, n)
}

func (mom *Momentum) Initialized() bool {
	return mom.initialized
}

func (mom *Momentum) Rescale(shift *mat.Dense) *mat.Dense {
	if mom.cache[mom.counter] == nil {
		mom.cache[mom.counter] = utils.DenseLike(shift)
	}

	cache := mom.cache[mom.counter]
	cache.Apply(func(i, j int, v float64) float64 {
		shiftVal := shift.At(i, j)
		return mom.Gamma*v + (1-mom.Gamma)*shiftVal
	}, cache)

	mom.counter = (mom.counter + 1) % len(mom.cache)
	return cache
}
