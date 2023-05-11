package save

import (
	"encoding/binary"
	"math"
)

func ToBytes(slice []float64) []byte {
	byteSlice := make([]byte, 8*len(slice))
	for i, f := range slice {
		binary.LittleEndian.PutUint64(byteSlice[i*8:(i+1)*8], math.Float64bits(f))
	}
	return byteSlice
}

func FromBytes(byteSlice []byte) []float64 {
	floatSlice := make([]float64, len(byteSlice)/8)
	for i := range floatSlice {
		bits := binary.LittleEndian.Uint64(byteSlice[i*8 : (i+1)*8])
		floatSlice[i] = math.Float64frombits(bits)
	}
	return floatSlice
}

func ConstantsToBytes(constants ...int) []byte {
	bytes := make([]byte, 4*len(constants))
	for i, c := range constants {
		binary.LittleEndian.PutUint32(bytes[i*4:(i+1)*4], uint32(c))
	}
	return bytes
}

func ConstantsFromBytes(bytes []byte) []int {
	ints := make([]int, len(bytes)/4)
	for i := range ints {
		ints[i] = int(binary.LittleEndian.Uint32(bytes[i*4 : (i+1)*4]))
	}
	return ints
}
