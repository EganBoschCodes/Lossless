package utils

func CenterPad(str string, width int) string {
	additional := width - len(str)
	rightPad := additional / 2
	leftPad := additional - rightPad

	output := make([]byte, width)
	for i := range output {
		shiftI := i - leftPad
		if shiftI < 0 || shiftI >= len(str) {
			output[i] = 32
			continue
		}
		output[i] = str[shiftI]
	}

	return string(output)
}
