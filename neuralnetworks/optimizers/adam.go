package optimizers

import (
	"math"
	"sync"

	"github.com/EganBoschCodes/lossless/utils"
	"gonum.org/v1/gonum/mat"
)

type Adam struct {
	Beta2   float64
	Beta1   float64
	Epsilon float64

	cache          []*mat.Dense
	cacheSquare    []*mat.Dense
	cacheMultiples []*mat.Dense
	mutexes        []sync.Mutex

	size        int
	initialized bool
}

func (adam *Adam) Initialize(n int) {
	adam.cache = make([]*mat.Dense, n)
	adam.cacheSquare = make([]*mat.Dense, n)
	adam.cacheMultiples = make([]*mat.Dense, n)
	adam.mutexes = make([]sync.Mutex, n)

	if adam.Epsilon == 0 {
		adam.Epsilon = 1e-7
	}
	if adam.Beta1 == 0 {
		adam.Beta1 = 0.9
	}
	if adam.Beta2 == 0 {
		adam.Beta1 = 0.999
	}
	//adam.t = 1

	adam.initialized = true
	adam.size = n
}

func (adam *Adam) Initialized() bool {
	return adam.initialized
}

func (adam *Adam) Size() int {
	return adam.size
}

func (adam *Adam) Rescale(shift *mat.Dense, index int) *mat.Dense {
	adam.mutexes[index].Lock()
	if adam.cache[index] == nil {
		adam.cache[index] = utils.DenseLike(shift)
		adam.cache[index].Copy(shift)
		adam.cacheSquare[index] = utils.DenseLike(shift)
		adam.cacheSquare[index].MulElem(shift, shift)
		adam.cacheMultiples[index] = utils.DenseLike(shift)
	}

	//cache, cacheSquare, cacheMultiple := adam.cache[index], adam.cacheSquare[index], adam.cacheMultiples[index]
	adam.cache[index].Scale(adam.Beta1, adam.cache[index])
	adam.cacheSquare[index].Scale(adam.Beta2, adam.cacheSquare[index])

	shift.Scale(1-adam.Beta1, shift)
	adam.cache[index].Add(shift, adam.cache[index])

	adam.cacheSquare[index] = utils.FastApply(adam.cacheSquare[index], func(i, j int, v float64) float64 {
		return adam.Beta2*v + (1-adam.Beta2)*shift.At(i, j)*shift.At(i, j)
	})
	adam.cacheMultiples[index] = utils.FastApply(adam.cacheMultiples[index], func(i, j int, g float64) float64 {
		return 1 / (math.Sqrt(g) + adam.Epsilon)
	})

	shift.MulElem(adam.cache[index], adam.cacheMultiples[index])
	adam.mutexes[index].Unlock()

	return shift
}
