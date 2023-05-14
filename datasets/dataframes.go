package datasets

type DataFrame struct {
	headers []string
	values  [][]FrameEntry
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

func (frame *DataFrame) getCol(int) []FrameEntry {
	column := make([]FrameEntry, 0)
	return column
}
