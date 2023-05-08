package datasets

import (
	"fmt"
	"go-ml-library/utils"
	"math"
)

func IsCorrect(output []float64, target []float64) {
	fmt.Printf("Output: %.2f\nTarget: %.2f\nCorrect: %t\n\n", output, target, utils.Reduce(output, math.Max) == output[FromOneHot(target)])
}
