package datasets

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"

	"github.com/EganBoschCodes/lossless/utils"
)

type DataFrame struct {
	headers []string
	values  [][]FrameEntry
}

// Reads a csv at the given path into a DataFrame. The headers argument
// is a boolean representing if the first row in the dataset is just the
// headers for the columns. This should usually be set to true.
func ReadCSV(path string, headers bool) DataFrame {
	bytes, err := os.ReadFile(path)
	frame := DataFrame{}

	if err != nil {
		fmt.Printf("Error opening the file at %s!\n\n", path)
		panic(err)
	}

	if len(bytes) == 0 {
		panic(fmt.Sprintf("The file at \"%s\" is empty!", path))
	}

	rawRows := strings.Split(string(bytes), "\n")
	rawRows = utils.Map(rawRows, strings.TrimSpace)
	rawRows = utils.Filter(rawRows, func(s string) bool { return len(s) > 0 })

	if headers {
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

func (frame *DataFrame) GetCol(title string) []FrameEntry {
	index := utils.Find(frame.headers, title)
	if index < 0 {
		panic("There is no column titled \"%s\" in this dataframe!\n\n")
	}
	return frame.GetNthCol(index)
}

func (frame *DataFrame) GetNthCol(col int) []FrameEntry {
	if col < 0 || col >= len(frame.values[0]) {
		panic(fmt.Sprintf("%d is out of bounds for this dataframe! (Needs to be 0-%d)", col, len(frame.values[0])-1))
	}
	column := make([]FrameEntry, frame.Rows())
	for i, row := range frame.values {
		column[i] = row[col]
	}
	return column
}

func (frame *DataFrame) OverwriteColumn(newColumn []FrameEntry, title string) {
	index := utils.Find(frame.headers, title)
	if index < 0 {
		panic("There is no column titled \"%s\" in this dataframe!\n\n")
	}
	frame.OverwriteNthColumn(newColumn, index)
}

func (frame *DataFrame) OverwriteNthColumn(newColumn []FrameEntry, col int) {
	if len(frame.values) != len(newColumn) {
		panic("Lengths of columns do not match up!")
	}
	if col < 0 || col >= len(frame.values[0]) {
		panic(fmt.Sprintf("%d is out of bounds for this dataframe! (Needs to be 0-%d)", col, len(frame.values[0])-1))
	}

	for i, row := range frame.values {
		row[col] = newColumn[i]
	}
}

func (frame *DataFrame) CategorizeColumn(title string) (options []FrameEntry) {
	index := utils.Find(frame.headers, title)
	if index < 0 {
		panic("There is no column titled \"%s\" in this dataframe!\n\n")
	}
	return frame.CategorizeNthColumn(index)
}

func (frame *DataFrame) CategorizeNthColumn(col int) (options []FrameEntry) {
	if len(frame.values) == 0 {
		panic("Cannot categorize an empty column!")
	}
	if col < 0 || col >= len(frame.values[0]) {
		panic(fmt.Sprintf("%d is out of bounds for this dataframe! (Needs to be 0-%d)", col, len(frame.values[0])-1))
	}

	newColumn, options := Categorize(frame.GetNthCol(col))
	frame.OverwriteNthColumn(newColumn, col)
	return options
}

func (frame *DataFrame) CategorizeColumnSlice(colSlice string) (options [][]FrameEntry) {
	isSelected := utils.ParseSlice(colSlice)
	for i := 0; i < len(frame.headers); i++ {
		if !isSelected(i) {
			continue
		}
		newColumn, colOptions := Categorize(frame.GetNthCol(i))
		frame.OverwriteNthColumn(newColumn, i)
		options = append(options, colOptions)
	}

	return options
}

func (frame *DataFrame) NumericallyCategorizeColumn(title string) {
	index := utils.Find(frame.headers, title)
	if index < 0 {
		panic("There is no column titled \"%s\" in this dataframe!\n\n")
	}
	frame.NumericallyCategorizeNthColumn(index)
}

func (frame *DataFrame) NumericallyCategorizeNthColumn(col int) {
	if len(frame.values) == 0 {
		panic("Cannot numerically categorize an empty column!")
	}
	if col < 0 || col >= len(frame.values[0]) {
		panic(fmt.Sprintf("%d is out of bounds for this dataframe! (Needs to be 0-%d)", col, len(frame.values[0])-1))
	}

	newColumn := NumericallyCategorize(frame.GetNthCol(col))
	frame.OverwriteNthColumn(newColumn, col)
}

func (frame *DataFrame) NumericallyCategorizeColumnSlice(colSlice string) {
	isSelected := utils.ParseSlice(colSlice)
	for i := 0; i < len(frame.headers); i++ {
		if !isSelected(i) {
			continue
		}
		newColumn := NumericallyCategorize(frame.GetNthCol(i))
		frame.OverwriteNthColumn(newColumn, i)
	}
}

func (frame *DataFrame) NormalizeColumn(title string) (float64, float64) {
	index := utils.Find(frame.headers, title)
	if index < 0 {
		panic("There is no column titled \"%s\" in this dataframe!\n\n")
	}
	return frame.NormalizeNthColumn(index)
}

func (frame *DataFrame) NormalizeNthColumn(col int) (float64, float64) {
	column := frame.GetNthCol(col)

	if len(column) == 0 {
		panic("Can't normalize an empty column!")
	}
	if col < 0 || col >= len(frame.values[0]) {
		panic(fmt.Sprintf("%d is out of bounds for this dataframe! (Needs to be 0-%d)", col, len(frame.values[0])-1))
	}

	switch column[0].(type) {
	case *StringEntry:
		panic("Cannot normalize a string column (and you shouldn't want to)")
	case *VectorEntry:
		panic("Cannot normalize a vector column (and you shouldn't want to)")
	}

	rawColumn := utils.Map(column, func(f FrameEntry) float64 { return f.(*NumberEntry).Value })
	normedValues, mean, stddev := utils.Normalize(rawColumn)
	normalizedColumn := utils.Map(normedValues, func(f float64) FrameEntry { return &NumberEntry{Value: f} })

	frame.OverwriteNthColumn(normalizedColumn, col)
	return mean, stddev
}

func (frame *DataFrame) NormalizeColumnSlice(colSlice string) (means []float64, stddevs []float64) {
	means, stddevs = make([]float64, 0), make([]float64, 0)
	isSelected := utils.ParseSlice(colSlice)
	for i := 0; i < len(frame.headers); i++ {
		if !isSelected(i) {
			continue
		}
		mean, stddev := frame.NormalizeNthColumn(i)
		means = append(means, mean)
		stddevs = append(stddevs, stddev)
	}
	return means, stddevs
}

func (frame *DataFrame) ClampColumn(title string, newMin float64, newMax float64) (float64, float64) {
	index := utils.Find(frame.headers, title)
	if index < 0 {
		panic("There is no column titled \"%s\" in this dataframe!\n\n")
	}
	return frame.ClampNthColumn(index, newMin, newMax)
}

func (frame *DataFrame) ClampNthColumn(col int, newMin float64, newMax float64) (float64, float64) {
	column := frame.GetNthCol(col)

	if len(column) == 0 {
		panic("Can't normalize an empty column!")
	}
	if col < 0 || col >= len(frame.values[0]) {
		panic(fmt.Sprintf("%d is out of bounds for this dataframe! (Needs to be 0-%d)", col, len(frame.values[0])-1))
	}

	switch column[0].(type) {
	case *StringEntry:
		panic("Cannot clamp a string column (and you shouldn't want to)")
	case *VectorEntry:
		panic("Cannot clamp a vector column (and you shouldn't want to)")
	}

	columnNums := utils.Map(column, func(f FrameEntry) float64 { return f.(*NumberEntry).Value })
	min, max := utils.Reduce(columnNums, func(a float64, b float64) float64 { return math.Min(a, b) }), utils.Reduce(columnNums, func(a float64, b float64) float64 { return math.Max(a, b) })

	clampedValues := utils.Map(columnNums, func(f float64) float64 { return (f-min)/(max-min)*(newMax-newMin) + newMin })
	clampedColumn := utils.Map(clampedValues, func(f float64) FrameEntry { return &NumberEntry{Value: f} })

	frame.OverwriteNthColumn(clampedColumn, col)
	return min, max
}

func (frame *DataFrame) ClampColumnSlice(colSlice string, newMin float64, newMax float64) (mins []float64, maxes []float64) {
	mins, maxes = make([]float64, 0), make([]float64, 0)
	isSelected := utils.ParseSlice(colSlice)
	for i := 0; i < len(frame.headers); i++ {
		if !isSelected(i) {
			continue
		}
		min, max := frame.ClampNthColumn(i, newMin, newMax)
		mins = append(mins, min)
		maxes = append(maxes, max)
	}
	return mins, maxes
}

func (frame *DataFrame) MapFloatColumn(title string, lambda func(int, float64) float64) {
	index := utils.Find(frame.headers, title)
	if index < 0 {
		panic("There is no column titled \"%s\" in this dataframe!\n\n")
	}
	frame.MapNthFloatColumn(index, lambda)
}

func (frame *DataFrame) MapNthFloatColumn(col int, lambda func(int, float64) float64) {
	if len(frame.values) == 0 {
		panic("Cannot map an empty column!")
	}
	if col < 0 || col >= len(frame.values[0]) {
		panic(fmt.Sprintf("%d is out of bounds for this dataframe! (Needs to be 0-%d)", col, len(frame.values[0])-1))
	}

	switch frame.values[0][col].(type) {
	case *StringEntry:
		panic("Cannot apply a float function to a string column!.")
	case *VectorEntry:
		panic("Cannot map a vector column, only float or string columns.")
	}

	for i, row := range frame.values {
		row[col] = &NumberEntry{Value: lambda(i, row[col].(*NumberEntry).Value)}
	}
}

func (frame *DataFrame) MapFloatColumnSlice(colSlice string, lambda func(int, float64) float64) {
	isSelected := utils.ParseSlice(colSlice)
	for i := 0; i < frame.Cols(); i++ {
		if !isSelected(i) {
			continue
		}
		frame.MapNthFloatColumn(i, lambda)
	}
}

func (frame *DataFrame) MapStringColumn(title string, lambda func(string) string) {
	index := utils.Find(frame.headers, title)
	if index < 0 {
		panic("There is no column titled \"%s\" in this dataframe!\n\n")
	}
	frame.MapNthStringColumn(index, lambda)
}

func (frame *DataFrame) MapNthStringColumn(col int, lambda func(string) string) {
	if len(frame.values) == 0 {
		panic("Cannot map an empty column!")
	}
	if col < 0 || col >= len(frame.values[0]) {
		panic(fmt.Sprintf("%d is out of bounds for this dataframe! (Needs to be 0-%d)", col, len(frame.values[0])-1))
	}

	switch frame.values[0][col].(type) {
	case *NumberEntry:
		panic("Cannot apply a string function to a float column!.")
	case *VectorEntry:
		panic("Cannot map a vector column, only float or string columns.")
	}

	for _, row := range frame.values {
		row[col] = &StringEntry{Value: lambda(row[col].(*StringEntry).Value)}
	}
}

func (frame *DataFrame) MapStringColumnSlice(colSlice string, lambda func(string) string) {
	isSelected := utils.ParseSlice(colSlice)
	for i := 0; i < frame.Cols(); i++ {
		if !isSelected(i) {
			continue
		}
		frame.MapNthStringColumn(i, lambda)
	}
}

func (frame *DataFrame) ShuffleRows() {
	rand.Shuffle(len(frame.values), func(i, j int) { frame.values[i], frame.values[j] = frame.values[j], frame.values[i] })
}

func (frame *DataFrame) DeleteRows(sliceString string) {
	selected := utils.ParseSlice(sliceString)
	newValues := make([][]FrameEntry, 0)
	for i, row := range frame.values {
		if !selected(i) {
			newValues = append(newValues, row)
		}
	}
	frame.values = newValues
}

func (frame *DataFrame) SelectRowSlice(sliceString string) (selectedFrame DataFrame) {
	selected := utils.ParseSlice(sliceString)
	newValues := make([][]FrameEntry, 0)
	for i, row := range frame.values {
		if selected(i) {
			newValues = append(newValues, row)
		}
	}

	selectedFrame = DataFrame{headers: frame.headers, values: newValues}
	return selectedFrame
}

func (frame *DataFrame) SelectRowsMatching(header string, entry FrameEntry) (selectedFrame DataFrame) {
	newValues := make([][]FrameEntry, 0)
	col := frame.GetCol(header)
	for i, row := range frame.values {
		if col[i].Equals(entry) {
			newValues = append(newValues, row)
		}
	}

	selectedFrame = DataFrame{headers: frame.headers, values: newValues}
	return selectedFrame
}

func (frame *DataFrame) DeleteColumnSlice(sliceString string) {
	selected := utils.ParseSlice(sliceString)

	newHeaders := make([]string, 0)
	for i, val := range frame.headers {
		if !selected(i) {
			newHeaders = append(newHeaders, val)
		}
	}

	newValues := make([][]FrameEntry, len(frame.values))
	for i, row := range frame.values {
		newValues[i] = make([]FrameEntry, 0)
		for j, val := range row {
			if !selected(j) {
				newValues[i] = append(newValues[i], val)
			}
		}
	}

	frame.values, frame.headers = newValues, newHeaders
}

func (frame *DataFrame) DeleteColumns(headers ...string) {
	newHeaders := make([]string, 0)
	for _, val := range frame.headers {
		if utils.Find(headers, val) == -1 {
			newHeaders = append(newHeaders, val)
		}
	}

	newValues := make([][]FrameEntry, len(frame.values))
	for i, row := range frame.values {
		newValues[i] = make([]FrameEntry, 0)
		for j, val := range row {
			if utils.Find(headers, frame.headers[j]) == -1 {
				newValues[i] = append(newValues[i], val)
			}
		}
	}

	frame.values, frame.headers = newValues, newHeaders
}

func (frame *DataFrame) AddColumn(header string, values []float64) {
	if len(values) != len(frame.values) {
		panic("New column needs to have the same number of rows as the frame!")
	}
	frame.headers = append(frame.headers, header)
	for i := range frame.values {
		frame.values[i] = append(frame.values[i], &NumberEntry{values[i]})
	}
}

func (frame *DataFrame) SelectColumns(sliceString string) (selectedFrame DataFrame) {
	selected := utils.ParseSlice(sliceString)
	newHeaders := make([]string, 0)
	for i, header := range frame.headers {
		if selected(i) {
			newHeaders = append(newHeaders, header)
		}
	}

	newValues := make([][]FrameEntry, len(frame.values))
	for rownum, row := range frame.values {
		newValues[rownum] = make([]FrameEntry, 0)
		for i, val := range row {
			if selected(i) {
				newValues[rownum] = append(newValues[rownum], val)
			}
		}
	}

	selectedFrame = DataFrame{headers: newHeaders, values: newValues}
	return selectedFrame
}

func (frame *DataFrame) ToDataset(inputSlice string, outputSlice string) []DataPoint {
	isInput, isOutput := utils.ParseSlice(inputSlice), utils.ParseSlice(outputSlice)

	dataset := make([]DataPoint, len(frame.values))
	for i, row := range frame.values {
		input, output := make([]float64, 0), make([]float64, 0)
		for col, val := range row {
			if isInput(col) {
				input = val.MergeInto(input)
			} else if isOutput(col) {
				output = val.MergeInto(output)
			}
		}
		dataset[i] = DataPoint{Input: input, Output: output}
	}

	return dataset
}

func (frame *DataFrame) ToSequentialDataset(inputSlice string, outputSlice string, intervalLength int) []DataPoint {
	isInput, isOutput := utils.ParseSlice(inputSlice), utils.ParseSlice(outputSlice)

	dataset := make([]DataPoint, len(frame.values))
	for i := range frame.values[:len(frame.values)-intervalLength-1] {
		nextRow := frame.values[i+1]
		input, output := make([]float64, 0), make([]float64, 0)
		for _, row := range frame.values[i : i+intervalLength] {
			for col := range row {
				if isInput(col) {
					input = row[col].MergeInto(input)
				}
			}
		}
		for col := range nextRow {
			if isOutput(col) {
				output = nextRow[col].MergeInto(output)
			}
		}

		dataset[i] = DataPoint{Input: input, Output: output}
	}

	return dataset
}

func (frame *DataFrame) PrintSummary() {
	numEntities := utils.Min(10, len(frame.values))
	displayEntries := frame.values[:numEntities]

	numColumns := utils.Min(12, len(displayEntries[0]))
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
