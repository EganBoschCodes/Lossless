package datasets

import (
	"fmt"
	"os"
	"strings"

	"github.com/EganBoschCodes/lossless/utils"
)

type DataFrame struct {
	headers []string
	values  [][]FrameEntry
}

type ReadOptions struct {
	Headers bool
}

func ReadCSV(path string, options ReadOptions) DataFrame {
	bytes, err := os.ReadFile(path)
	frame := DataFrame{}

	if err != nil {
		fmt.Printf("Error opening the file at %s!\n\n", path)
		panic(err)
	}

	rawRows := strings.Split(string(bytes), "\r\n")

	if options.Headers {
		var headerRow string
		headerRow, rawRows = rawRows[0], rawRows[1:]
		frame.headers = strings.Split(headerRow, ",")
	} else {
		numCols := len(strings.Split(rawRows[0], ","))
		frame.headers = make([]string, numCols)
		for i := range frame.headers {
			frame.headers[i] = fmt.Sprintf("Column %d", i)
		}
	}

	rawEntries := utils.Map(rawRows, func(s string))

	return frame
}

func (frame *DataFrame) Rows() int {
	return len(frame.values)
}

func (frame *DataFrame) Cols() int {
	if frame.Rows() == 0 {
		return 0
	}
	return len(frame.values[0])
}

func (frame *DataFrame) Dims() (int, int) {
	return frame.Rows(), frame.Cols()
}

func (frame *DataFrame) GetNthCol(col int) []FrameEntry {
	column := make([]FrameEntry, frame.Rows())
	for i, row := range frame.values {
		column[i] = row[col]
	}
	return column
}

func (frame *DataFrame) GetCol(title string) []FrameEntry {
	for i, str := range frame.headers {
		if str == title {
			return frame.GetNthCol(i)
		}
	}
	panic("There is no column titled \"%s\" in this dataframe!\n\n")
}
