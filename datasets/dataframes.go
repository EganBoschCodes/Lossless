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

	if len(bytes) == 0 {
		panic(fmt.Sprintf("The file at \"%s\" is empty!"))
	}

	rawRows := strings.Split(string(bytes), "\r\n")
	rawRows = utils.Filter(rawRows, func(s string) bool { return len(s) > 0 })

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

	if !utils.All(rawRows, func(s string) bool { return len(strings.Split(s, ",")) == len(frame.headers) }) {
		panic("Not all values are populated!")
	}

	rawEntries := utils.Map(rawRows, func(s string) []string { return strings.Split(s, ",") })
	frame.values = utils.Map2D(rawEntries, CreateEntry)

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

func (frame *DataFrame) OverwriteNthColumn(newColumn []FrameEntry, col int) {
	for i, row := range frame.values {
		row[col] = newColumn[i]
	}
}

func (frame *DataFrame) CategorizeNthColumn(col int) (options []FrameEntry) {
	newColumn, options := Categorize(frame.GetNthCol(col))
	frame.OverwriteNthColumn(newColumn, col)
	return options
}

func (frame *DataFrame) NumericallyCategorizeNthColumn(col int) {
	newColumn := NumericallyCategorize(frame.GetNthCol(col))
	frame.OverwriteNthColumn(newColumn, col)
}

func (frame *DataFrame) PrintSummary() {
	numEntities := utils.Min(10, len(frame.values))
	displayEntries := frame.values[:numEntities]

	numColumns := utils.Min(7, len(displayEntries[0]))
	displayEntries = utils.Map(displayEntries, func(row []FrameEntry) []FrameEntry { return row[:numColumns] })

	// Find out how wide we need to make each column
	displayLengths := utils.Map2D(displayEntries, func(entry FrameEntry) int { return len(entry.DisplayValue()) })
	columnWidths := utils.Map(frame.headers[:numColumns], func(s string) int { return len(s) })
	for _, row := range displayLengths {
		columnWidths = utils.DoubleMap(columnWidths, row, utils.Max)
	}
	columnWidths = utils.Map(columnWidths, func(a int) int { return a + 2 })
	totalWidth := utils.Reduce(columnWidths, func(a int, b int) int { return a + b })

	// Print headers
	for i, header := range frame.headers[:numColumns] {
		fmt.Print(utils.CenterPad(header, columnWidths[i]))
		if i < numColumns-1 {
			fmt.Print("|")
		} else if numColumns < len(frame.values[0]) {
			fmt.Printf(" (%d more columns...)\n", len(frame.values[0])-numColumns)
		} else {
			fmt.Print("\n")
		}
	}

	// Print a horizontal bar
	fmt.Println(string(utils.Map(make([]byte, totalWidth+len(columnWidths)-1), func(a byte) byte { return 45 })))

	// Print all the entries
	for _, row := range displayEntries {
		for i, entry := range row {
			fmt.Print(utils.CenterPad(entry.DisplayValue(), columnWidths[i]))
			if i < len(row)-1 {
				fmt.Print("|")
			} else {
				fmt.Print("\n")
			}
		}
	}

	if numEntities < len(frame.values) {
		fmt.Printf("\n%s\n", utils.CenterPad(fmt.Sprintf("(%d more rows...)", len(frame.values)-numEntities), totalWidth))
	}

}
