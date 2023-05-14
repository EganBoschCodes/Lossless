package datasets

import (
	"fmt"
	"strconv"
)

type FrameEntry interface {
	DisplayValue() string
	Equals(FrameEntry) bool
}

type StringEntry struct {
	Value string
}

func (str *StringEntry) DisplayValue() string {
	if len(str.Value) <= 12 {
		return str.Value
	}
	return str.Value[:9] + "..."
}

func (str *StringEntry) Equals(other FrameEntry) bool {
	switch other.(type) {
	case *StringEntry:
		return str.Value == other.(*StringEntry).Value
	default:
		return false
	}
}

type NumberEntry struct {
	Value float64
}

func (flt *NumberEntry) DisplayValue() string {
	return fmt.Sprintf("%.3f", flt.Value)
}

func (flt *NumberEntry) Equals(other FrameEntry) bool {
	switch other.(type) {
	case *NumberEntry:
		return flt.Value == other.(*NumberEntry).Value
	default:
		return false
	}
}

type VectorEntry struct {
	Value []float64
}

func (vec *VectorEntry) DisplayValue() string {
	return fmt.Sprintf("%.1f", vec.Value)
}

func (vec *VectorEntry) Equals(other FrameEntry) bool {
	switch other.(type) {
	case *VectorEntry:
		if len(vec.Value) != len(other.(*VectorEntry).Value) {
			return false
		}
		for i := range vec.Value {
			if vec.Value[i] != other.(*VectorEntry).Value[i] {
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
