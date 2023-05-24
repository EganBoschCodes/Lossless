package optimizers

import (
	"math"

	"github.com/EganBoschCodes/lossless/utils"
	"gonum.org/v1/gonum/mat"
)

type Adam struct {
	Gamma   float64
	Rho     float64
	Epsilon float64

	cache       []*mat.Dense
	cacheSquare []*mat.Dense

	counter     int
	t           int
	initialized bool
}

func (adam *Adam) Initialize(n int) {
	adam.cache = make([]*mat.Dense, n)
	adam.cacheSquare = make([]*mat.Dense, n)

	if adam.Epsilon == 0 {
		adam.Epsilon = 1e-7
	}
	adam.t = 1

	adam.initialized = true
}

func (adam *Adam) Initialized() bool {
	return adam.initialized
}

func (adam *Adam) Rescale(shift *mat.Dense) *mat.Dense {
	if adam.cache[adam.counter] == nil {
		adam.cache[adam.counter] = utils.DenseLike(shift)
		adam.cacheSquare[adam.counter] = utils.DenseLike(shift)
	}

	cache, cacheSquare := adam.cache[adam.counter], adam.cacheSquare[adam.counter]
	cache.Apply(func(i, j int, v float64) float64 {
		return adam.Rho*v + (1-adam.Rho)*shift.At(i, j)
	}, cache)
	cacheSquare.Apply(func(i, j int, v float64) float64 {
		return adam.Gamma*v + (1-adam.Gamma)*shift.At(i, j)*shift.At(i, j)
	}, cacheSquare)

	newShift := utils.DenseLike(shift)
	newShift.Apply(func(i, j int, _ float64) float64 {
		m, g := cache.At(i, j)/(1-math.Pow(adam.Rho, float64(adam.t))), cacheSquare.At(i, j)/(1-math.Pow(adam.Gamma, float64(adam.t)))
		return m / (math.Sqrt(g) + adam.Epsilon)
	}, newShift)

	adam.counter++
	if adam.counter >= len(adam.cache) {
		adam.counter = 0
		adam.t++
	}

	return newShift
}
