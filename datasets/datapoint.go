package datasets

type DataPoint struct {
	Input  []float64
	Output []float64
}

func (dp *DataPoint) ToBytes() []byte {
	return nil
}
