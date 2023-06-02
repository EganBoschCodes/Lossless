package nlp

import (
	"fmt"

	"github.com/EganBoschCodes/lossless/utils"
)

type tokenPair struct {
	left  int
	right int
}

func tokenPairEquals(t tokenPair, t2 tokenPair) bool {
	return t.left == t2.left && t.right == t2.right
}

func toTokenPairs(word string, tokenMap *map[string]int, tokens *[]string) []tokenPair {
	bytePairs := make([]tokenPair, 0)
	if len(word) == 1 {
		b := string(word[0])

		_, mapHasB := (*tokenMap)[b]
		if !mapHasB {
			(*tokenMap)[b] = len(*tokens)
			*tokens = append(*tokens, b)
		}
	}
	for i := 0; i < len(word)-1; i++ {
		// Check that this byte pair won't contain any standalones.
		b1, b2 := string(word[i]), string(word[i+1])

		// Assign id's to unique byte
		_, mapHasB1 := (*tokenMap)[b1]
		_, mapHasB2 := (*tokenMap)[b2]
		if !mapHasB1 {
			(*tokenMap)[b1] = len(*tokens)
			*tokens = append(*tokens, b1)
		}
		if !mapHasB2 {
			(*tokenMap)[b2] = len(*tokens)
			*tokens = append(*tokens, b2)
		}

		// Record byte pair
		bytePairs = append(bytePairs, tokenPair{left: (*tokenMap)[b1], right: (*tokenMap)[b2]})
	}

	return bytePairs
}

func GenerateTokens(s string, vocabSize int, standalones []string) (tokens []string, tokenMap map[string]int) {
	tokens = standalones
	tokenMap = make(map[string]int)
	for i, str := range tokens {
		tokenMap[str] = i
	}

	splitOnStandalones := utils.SplitAny(s, utils.Reduce(standalones, func(s1 string, s2 string) string { return s1 + s2 }))
	bytePairs := make([][]tokenPair, 0)

	for _, word := range splitOnStandalones {
		bytePairs = append(bytePairs, toTokenPairs(word, &tokenMap, &tokens))
	}

	for len(tokens) < vocabSize && utils.Reduce(utils.Map(bytePairs, func(v []tokenPair) int { return len(v) }), utils.Max) > 0 {
		pairs, counts := utils.CountOccurancesWithCompare(utils.Flatten(bytePairs), tokenPairEquals)
		lengths := utils.Map(pairs, func(b tokenPair) int { return -(len(tokens[b.left]) + len(tokens[b.right])) })

		// Find the most often occuring byte sequence, but prioritize shorter ones
		maxOccurance := utils.GetMaxIndex(counts, lengths)
		maxPair := pairs[maxOccurance]
		newToken, newTokenID := tokens[maxPair.left]+tokens[maxPair.right], len(tokens)

		tokenMap[newToken] = newTokenID
		tokens = append(tokens, newToken)

		bytePairs = utils.Map(bytePairs, func(wordPairs []tokenPair) []tokenPair {
			newWordPairs := make([]tokenPair, 0)

			for i := 0; i < len(wordPairs); i++ {
				if tokenPairEquals(wordPairs[i], maxPair) {
					if i == 0 && len(wordPairs) > 1 {
						wordPairs[i+1].left = newTokenID
					}
					continue
				}

				if i < len(wordPairs)-1 && tokenPairEquals(wordPairs[i+1], maxPair) {
					wordPairs[i].right = newTokenID
					if i < len(wordPairs)-2 {
						wordPairs[i+2].left = newTokenID
					}
				}
				newWordPairs = append(newWordPairs, wordPairs[i])
			}

			return newWordPairs
		})
	}

	usedTokens, usedTokenMap := make([]string, 0), make(map[string]int)
	tokenized := Tokenize(s, tokenMap)
	for _, token := range tokenized {
		_, hasToken := usedTokenMap[tokens[token]]
		if !hasToken {
			usedTokenMap[tokens[token]] = len(usedTokens)
			usedTokens = append(usedTokens, tokens[token])
		}
	}

	return usedTokens, usedTokenMap
}

func Tokenize(s string, mappings map[string]int) []int {
	tokens, tokenized := make([]string, 0), make([]int, 0)
	for k := range mappings {
		tokens = append(tokens, k)
	}

	// Make sure we always go for the biggest tokens first
	utils.SortByDecreasingLength(tokens)

	for len(s) > 0 {
		broken := false
		for _, token := range tokens {
			if utils.StartsWith(s, token) {
				s = s[len(token):]
				tokenized = append(tokenized, mappings[token])
				broken = true
				break
			}
		}

		if !broken {
			panic(fmt.Sprintf("Could not find a token to match the start of \"%s\"!", s))
		}
	}

	return tokenized
}
