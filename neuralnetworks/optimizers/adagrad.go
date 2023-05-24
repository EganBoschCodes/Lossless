package optimizers

import "gonum.org/v1/gonum/mat"

type AdaGrad struct {
	cache []*mat.Dense

	counter     int
	initialized bool
}

func (ada *AdaGrad) Initialize(n int) {
	ada.cache = make([]*mat.Dense, n)
}

func (ada *AdaGrad) Initialized() bool {
	return ada.initialized
}
