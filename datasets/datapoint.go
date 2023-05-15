package datasets

import (
	"fmt"

	"github.com/EganBoschCodes/lossless/neuralnetworks/save"
)

type DataPoint struct {
	Input  []float64
	Output []float64
}

func (dp *DataPoint) ToBytes() []byte {
	return append(save.ToBytes(dp.Input), save.ToBytes(dp.Output)...)
}

func DataPointFromBytes(bytes []byte, inputLength int) DataPoint {
	values := save.FromBytes(bytes)
	return DataPoint{Input: values[:inputLength], Output: values[inputLength:]}
}

func SaveDataset(dataset []DataPoint, dir string, name string) {
	bytes := save.ConstantsToBytes(len(dataset[0].Input), len(dataset[0].Output))
	for _, dp := range dataset {
		bytes = append(bytes, dp.ToBytes()...)
	}

	if len(dir) > 0 {
		save.WriteBytesToFile(fmt.Sprintf("%s/%s.dtst", dir, name), bytes)
	} else {
		save.WriteBytesToFile(fmt.Sprintf("%s.dtst", name), bytes)
	}
}

func OpenDataset(dir string, name string) []DataPoint {
	var rawBytes []byte
	if len(dir) > 0 {
		rawBytes = save.ReadBytesFromFile(fmt.Sprintf("%s/%s.dtst", dir, name))
	} else {
		rawBytes = save.ReadBytesFromFile(fmt.Sprintf("%s.dtst", name))
	}

	metadataRaw, datapointsRaw := rawBytes[:8], rawBytes[8:]
	metadata := save.ConstantsFromBytes(metadataRaw)
	inputLength, stride := metadata[0], (metadata[0]+metadata[1])*8

	dataset := make([]DataPoint, len(datapointsRaw)/stride)
	for i := 0; i < len(datapointsRaw); i += stride {
		dataset[i/stride] = DataPointFromBytes(datapointsRaw[i:i+stride], inputLength)
	}

	return dataset
}
