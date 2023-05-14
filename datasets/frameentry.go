package datasets

import (
	"fmt"
	"strconv"

	"github.com/EganBoschCodes/lossless/utils"
)

type FrameEntry interface {
	FullValue() string
	DisplayValue() string
	Equals(FrameEntry) bool
}

type StringEntry struct {
	Value string
}

func (str *StringEntry) FullValue() string {
	return str.Value
}

func (str *StringEntry) DisplayValue() string {
	if len(str.Value) <= 12 {
		return str.Value
	}
	return str.Value[:9] + "..."
}

func (str *StringEntry) Equals(other FrameEntry) bool {
	switch o := other.(type) {
	case *StringEntry:
		return str.Value == o.Value
	default:
		return false
	}
}

type NumberEntry struct {
	Value float64
}

func (flt *NumberEntry) FullValue() string {
	return fmt.Sprintf("%f", flt.Value)
}

func (flt *NumberEntry) DisplayValue() string {
	return fmt.Sprintf("%.3f", flt.Value)
}

func (flt *NumberEntry) Equals(other FrameEntry) bool {
	switch o := other.(type) {
	case *NumberEntry:
		return flt.Value == o.Value
	default:
		return false
	}
}

type VectorEntry struct {
	Value []float64
}

func (vec *VectorEntry) FullValue() string {
	return fmt.Sprintf("%.1f", vec.Value)
}

func (vec *VectorEntry) DisplayValue() string {
	return fmt.Sprintf("%.1f", vec.Value)
}

func (vec *VectorEntry) Equals(other FrameEntry) bool {
	switch o := other.(type) {
	case *VectorEntry:
		if len(vec.Value) != len(o.Value) {
			return false
		}
		for i := range vec.Value {
			if vec.Value[i] != o.Value[i] {
				return false
			}
		}
		return true
	default:
		return false
	}
}

func CreateEntry(rawValue string) FrameEntry {
	val, err := strconv.ParseFloat(rawValue, 64)
	if err != nil {
		return &StringEntry{Value: rawValue}
	} else {
		return &NumberEntry{Value: val}
	}
}

func Categorize(entries []FrameEntry) (newColumn []FrameEntry, options []FrameEntry) {
	options = make([]FrameEntry, 0)
	for _, entry := range entries {
		if utils.FindWithCompare(options, entry, func(a FrameEntry, b FrameEntry) bool { return a.Equals(b) }) == -1 {
			options = append(options, entry)
		}
	}

	newColumn = make([]FrameEntry, len(entries))
	for i, entry := range entries {
		optionNumber := utils.FindWithCompare(options, entry, func(a FrameEntry, b FrameEntry) bool { return a.Equals(b) })
		newColumn[i] = &VectorEntry{Value: ToOneHot(optionNumber, len(options))}
	}

	return newColumn, options
}

func NumericallyCategorize(entries []FrameEntry) (newColumn []FrameEntry) {
	maxIndex := 0
	for _, entry := range entries {
		maxIndex = utils.Max(maxIndex, int(entry.(*NumberEntry).Value))
	}

	newColumn = make([]FrameEntry, len(entries))
	for i, entry := range entries {
		newColumn[i] = &VectorEntry{Value: ToOneHot(int(entry.(*NumberEntry).Value), maxIndex+1)}
	}

	return newColumn
}
