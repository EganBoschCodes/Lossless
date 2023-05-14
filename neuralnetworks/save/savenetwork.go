package save

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"strings"
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

func recursivelyCreateFolders(path []string) {
	cwd := path[0]
	for i := range path {
		if _, err := os.Stat(cwd); errors.Is(err, os.ErrNotExist) {
			os.Mkdir(cwd, os.ModePerm)
		}

		if i < len(path)-1 {
			cwd += "/" + path[i+1]
		}
	}
}

func WriteBytesToFile(path string, bytes []byte) {
	pathSteps := strings.Split(path, "/")
	if len(pathSteps) > 1 {
		recursivelyCreateFolders(pathSteps[:len(pathSteps)-1])
	}

	f, _ := os.Create(path)
	defer f.Close()

	_, err := f.Write(bytes)
	if err != nil {
		fmt.Println(err)
	}
}

func WriteStringToFile(path string, str string) {
	pathSteps := strings.Split(path, "/")
	if len(pathSteps) > 1 {
		recursivelyCreateFolders(pathSteps[:len(pathSteps)-1])
	}

	f, _ := os.Create(path)
	defer f.Close()

	f.WriteString(str)
}

func ReadBytesFromFile(path string) []byte {
	file, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	// Get the file size
	stat, err := file.Stat()
	if err != nil {
		panic(err)
	}

	// Read the file into a byte slice
	bytes := make([]byte, stat.Size())
	_, err = bufio.NewReader(file).Read(bytes)
	if err != nil && err != io.EOF {
		panic(err)
	}

	return bytes
}
