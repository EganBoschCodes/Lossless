package datasets

import "fmt"

type FrameEntry interface {
	Value() string
	Equals(FrameEntry) bool
}

type StringEntry struct {
	Val string
}

func (str *StringEntry) Value() string {
	return str.Val
}

func (str *StringEntry) Equals(other FrameEntry) bool {
	switch other.(type) {
	case *StringEntry:
		return str.Val == other.(*StringEntry).Val
	default:
		return false
	}
}

type NumberEntry struct {
	Val float64
}

func (flt *NumberEntry) Value() string {
	return fmt.Sprintf("%.3E", flt.Val)
}

func (flt *NumberEntry) Equals(other FrameEntry) bool {
	switch other.(type) {
	case *NumberEntry:
		return flt.Val == other.(*NumberEntry).Val
	default:
		return false
	}
}

type VectorEntry struct {
	Val []float64
}

func (vec *VectorEntry) Value() string {
	return fmt.Sprintf("%.3E", vec.Val)
}

func (vec *VectorEntry) Equals(other FrameEntry) bool {
	switch other.(type) {
	case *VectorEntry:
		if len(vec.Val) != len(other.(*VectorEntry).Val) {
			return false
		}
		for i := range vec.Val {
			if vec.Val[i] != other.(*VectorEntry).Val[i] {
				return false
			}
		}
		return true
	default:
		return false
	}
}
