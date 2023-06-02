package nlp

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/EganBoschCodes/lossless/utils"
)

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func dot(a []float64, b []float64) float64 {
	return utils.Reduce(utils.DoubleMap(a, b, func(c float64, d float64) float64 { return c * d }), func(c float64, d float64) float64 { return c + d })
}

func randomEmbedding(length int) []float64 {
	embedding := make([]float64, length)
	for i := range embedding {
		embedding[i] = rand.NormFloat64()
	}
	return embedding
}

type embeddingTrainer struct {
	embedding []float64
	context   []float64
	target    []float64
}

func initEmbeddingTrainer(dims int, targets int) embeddingTrainer {
	return embeddingTrainer{
		embedding: randomEmbedding(dims),
		context:   randomEmbedding(dims),
		target:    make([]float64, targets),
	}
}

func GetEmbeddings(tokenized []int, mappings map[string]int, contextSize int, dimensions int, numEpochs int, learningRate float64) [][]float64 {
	numTokens := len(mappings)

	tokens, counts := utils.CountOccurances(tokenized)
	indexedTokenCounts := make([]int, numTokens)
	for i, token := range tokens {
		indexedTokenCounts[token] = counts[i]
	}

	// Initialize embedding trainers
	embeddings := utils.Map(make([]byte, numTokens), func(_ byte) embeddingTrainer { return initEmbeddingTrainer(dimensions, numTokens) })

	// Set the targets for the embedding trainers
	for i := range tokenized {
		for j := -contextSize; j <= contextSize; j++ {
			if i+j < 0 || i+j >= len(tokenized) || j == 0 {
				continue
			}

			embeddings[tokenized[i]].target[tokenized[i+j]] += 1.0 / float64(indexedTokenCounts[tokenized[i]])
		}
	}
	for i := range embeddings {
		embeddings[i].target = utils.Map(embeddings[i].target, math.Tanh)
	}

	fmt.Print("Starting Embedding Training!\n\n")

	// Training time
	for epoch := 0; epoch < numEpochs; epoch++ {
		loss := 0.0

		embeddingShifts, contextShifts := make([][]float64, len(embeddings)), make([][]float64, len(embeddings))
		for i := range embeddingShifts {
			embeddingShifts[i] = make([]float64, dimensions)
			contextShifts[i] = make([]float64, dimensions)
		}

		for i, outer := range embeddings {
			for j, inner := range embeddings {
				if i == j {
					continue
				}
				prediction := sigmoid(dot(outer.embedding, inner.context))
				gradient := outer.target[j] - prediction
				loss += 0.5 * gradient * gradient

				for k := range outer.embedding {
					embeddingShifts[i][k] += learningRate * inner.context[k] * prediction * (1 - prediction) * gradient
					contextShifts[j][k] += learningRate * outer.embedding[k] * prediction * (1 - prediction) * gradient
				}
			}
		}

		// Add the shifts, and normalize
		for i := range embeddingShifts {
			embeddings[i].embedding = utils.Add(embeddings[i].embedding, embeddingShifts[i])
			embeddings[i].context = utils.Add(embeddings[i].context, contextShifts[i])

			if epoch%100 == 0 {
				embeddingSize, contextSize := math.Pow(dot(embeddings[i].embedding, embeddings[i].embedding)+1, 0.25), math.Pow(dot(embeddings[i].context, embeddings[i].context)+1, 0.25)
				for j := range embeddings[i].embedding {
					embeddings[i].embedding[j] /= embeddingSize
					embeddings[i].context[j] /= contextSize
				}
			}

		}

		if epoch%100 == 0 {
			for i := range embeddings {
				embeddingSize, contextSize := dot(embeddings[i].embedding, embeddings[i].embedding), dot(embeddings[i].context, embeddings[i].context)
				for j := range embeddings[i].embedding {
					if embeddingSize > 1 {
						embeddings[i].embedding[j] /= embeddingSize/10 + 1
					}
					if contextSize > 1 {
						embeddings[i].context[j] /= contextSize/10 + 1
					}
				}
			}
		}

		fmt.Printf("\rTraining Epoch: %d, Loss: %.2f", epoch+1, loss)
	}

	for i := range embeddings {
		embeddingSize, contextSize := math.Pow(dot(embeddings[i].embedding, embeddings[i].embedding)+1, 0.25), math.Pow(dot(embeddings[i].context, embeddings[i].context)+1, 0.25)
		for j := range embeddings[i].embedding {
			if embeddingSize > 1 {
				embeddings[i].embedding[j] /= embeddingSize
			}
			if contextSize > 1 {
				embeddings[i].context[j] /= contextSize
			}
		}
	}

	fmt.Println("\n\nEmbeddings finished!")

	return utils.Map(embeddings, func(e embeddingTrainer) []float64 { return e.embedding })
}

/*func EmbeddingSpace(str string, mappings map[string]int, contextSize int, dimensions []int) networks.Sequential {
	numTokens := 0
	for _, i := range mappings {
		numTokens = utils.Max(numTokens, i)
	}
	numTokens++

	sentences := make([]string, 0)
	for i := utils.Find([]byte(str), '.'); i > 0; i = utils.Find([]byte(str), '.') {
		sentences = append(sentences, str[:i+1])
		str = str[i+1:]
	}

	fmt.Println(sentences)

	layerStack := make([]layers.Layer, 0)
	for _, dim := range dimensions {
		layerStack = append(layerStack, &layers.LinearLayer{Outputs: dim})
		layerStack = append(layerStack, &layers.TanhLayer{GradientScale: 2.0})
	}

	for i := len(dimensions) - 2; i >= 0; i-- {
		dim := dimensions[i]
		layerStack = append(layerStack, &layers.LinearLayer{Outputs: dim})
		layerStack = append(layerStack, &layers.TanhLayer{GradientScale: 2.0})
	}
	layerStack = append(layerStack, &layers.LinearLayer{Outputs: numTokens})
	layerStack = append(layerStack, &layers.SoftmaxLayer{})

	embeddingNetwork := networks.Sequential{}
	embeddingNetwork.Initialize(numTokens, layerStack...)

	embeddingNetwork.BatchSize = 128
	embeddingNetwork.SubBatch = 16
	embeddingNetwork.LearningRate = 1
	embeddingNetwork.Optimizer = &optimizers.RMSProp{Gamma: 0.9, Epsilon: 0.1}

	embeddingDataset := make([]datasets.DataPoint, 0)
	for _, sentence := range sentences {
		tokenized := Tokenize(sentence, mappings)
		cleanTokenized := make([]int, 0)
		for _, val := range tokenized {
			if val != 1 {
				cleanTokenized = append(cleanTokenized, val)
			}
		}
		for i := range cleanTokenized {
			for j := -contextSize; j <= contextSize; j++ {
				if i+j < 0 || i+j >= len(cleanTokenized) || j == 0 {
					continue
				}

				for k := 0; k < utils.Abs(i-j); k++ {
					embeddingDataset = append(embeddingDataset, datasets.DataPoint{Input: datasets.ToOneHot(cleanTokenized[i], numTokens), Output: datasets.ToOneHot(cleanTokenized[i+j], numTokens)})
				}
			}
		}
	}

	fmt.Println("Starting Embedding Training:\n-----------------------------")
	embeddingNetwork.Train(embeddingDataset, embeddingDataset, 20*time.Second)

	justEmbeddings := networks.Sequential{}
	justEmbeddings.Initialize(numTokens, embeddingNetwork.Layers[:2*len(dimensions)]...)
	return justEmbeddings
}*/
