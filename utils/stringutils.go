package utils

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
)

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

type interval struct {
	min    int
	max    int
	stride int
}

func (in *interval) In(i int) bool {
	return i >= in.min && i < in.max && (in.stride == 1 || (i-in.min)%in.stride == 0)
}

func parseDefault(str string, def int) int {
	if len(str) == 0 {
		return def
	}

	val, err := strconv.ParseInt(str, 10, 32)
	if err != nil {
		panic(fmt.Sprintf("\"%s\" is not a valid integer in ParseSlice call!", str))
	}
	return int(val)
}

func parseInterval(str string) interval {
	strSplit := strings.Split(str, ":")
	switch len(strSplit) {
	case 0:
		panic("Slice Interval has no data! (Did you double comma?)")
	case 1:
		val, err := strconv.ParseInt(strSplit[0], 10, 32)
		if err != nil {
			panic(fmt.Sprintf("\"%s\" is not a valid integer!", strSplit[0]))
		}
		return interval{min: int(val), max: int(val + 1), stride: 1}
	case 2:
		minval, maxval := parseDefault(strSplit[0], 0), parseDefault(strSplit[1], 0xFFFFFFFF)
		return interval{min: minval, max: maxval, stride: 1}
	case 3:
		minval, maxval := parseDefault(strSplit[0], 0), parseDefault(strSplit[1], 0xFFFFFFFF)
		stride := parseDefault(strSplit[2], 1)
		return interval{min: minval, max: maxval, stride: stride}
	default:
		panic("Too many arguments in creating interval!")
	}
}

func ParseSlice(str string) func(int) bool {
	if len(str) == 0 || str[0] != "["[0] || str[len(str)-1] != "]"[0] {
		panic("Please encapsulate calls to ParseSlice with square brackets for consistency with Go syntax.")
	}

	intervals := Map(Map(strings.Split(str[1:len(str)-1], ","), strings.TrimSpace), parseInterval)

	return func(i int) bool {
		return Any(Map(intervals, func(in interval) bool { return in.In(i) }), func(b bool) bool { return b })
	}
}

func StartsWith(str string, prefix string) bool {
	if len(prefix) > len(str) {
		return false
	}
	return str[:len(prefix)] == prefix
}

type StringLengthInterface struct {
	strings []string
}

func (s StringLengthInterface) Len() int {
	return len(s.strings)
}

func (s StringLengthInterface) Less(i, j int) bool {
	return len(s.strings[i]) > len(s.strings[j])
}

func (s StringLengthInterface) Swap(i, j int) {
	s.strings[i], s.strings[j] = s.strings[j], s.strings[i]
}

func SortByDecreasingLength(strings []string) {
	stringInterface := StringLengthInterface{strings: strings}
	sort.Sort(stringInterface)
}

func SplitAny(s string, seps string) []string {
	splitter := func(r rune) bool {
		return strings.ContainsRune(seps, r)
	}
	return strings.FieldsFunc(s, splitter)
}
