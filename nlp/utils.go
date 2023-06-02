package nlp

import (
	"math"

	"github.com/EganBoschCodes/lossless/neuralnetworks/save"
)

func CosineSimilarity(first []float64, second []float64) float64 {
	return dot(first, second) / (math.Sqrt(dot(first, first) * dot(second, second)))
}

func SaveEmbeddings(path string, tokens []string, embeddings [][]float64) {
	saveBytes := save.ConstantsToBytes(len(embeddings[0]))
	for i := range tokens {
		token, embedding := tokens[i], embeddings[i]
		saveBytes = append(saveBytes, save.ConstantsToBytes(len(token))...)

		saveBytes = append(saveBytes, []byte(token)...)
		saveBytes = append(saveBytes, save.ToBytes(embedding)...)
	}

	save.WriteBytesToFile(path, saveBytes)
}

func OpenEmbeddings(path string) (tokens []string, embeddings [][]float64) {
	bytes := save.ReadBytesFromFile(path)
	dimensions, bytes := save.ConstantsFromBytes(bytes[:4])[0], bytes[4:]

	tokens, embeddings = make([]string, 0), make([][]float64, 0)
	for len(bytes) > 0 {
		tokenLength := save.ConstantsFromBytes(bytes[:4])[0]
		bytes = bytes[4:]

		token := string(bytes[:tokenLength])
		embedding := save.FromBytes(bytes[tokenLength : tokenLength+8*dimensions])
		tokens, embeddings = append(tokens, token), append(embeddings, embedding)

		bytes = bytes[tokenLength+8*dimensions:]
	}
	return tokens, embeddings
}

func GetMapping(tokens []string) map[string]int {
	mapping := make(map[string]int)
	for i, token := range tokens {
		mapping[token] = i
	}
	return mapping
}
