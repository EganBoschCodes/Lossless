package nlp

import (
	"fmt"
	"time"

	"github.com/EganBoschCodes/lossless/datasets"
	"github.com/EganBoschCodes/lossless/neuralnetworks/layers"
	"github.com/EganBoschCodes/lossless/neuralnetworks/networks"
	"github.com/EganBoschCodes/lossless/neuralnetworks/optimizers"
	"github.com/EganBoschCodes/lossless/utils"
)

func EmbeddingSpace(str string, mappings map[string]int, dimensions int, contextSize int) networks.Sequential {
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

	embeddingNetwork := networks.Sequential{}
	embeddingNetwork.Initialize(numTokens,
		&layers.LinearLayer{Outputs: (dimensions + numTokens) / 2},
		&layers.TanhLayer{},
		&layers.LinearLayer{Outputs: dimensions},
		&layers.TanhLayer{},
		&layers.LinearLayer{Outputs: (dimensions + numTokens) / 2},
		&layers.TanhLayer{},
		&layers.LinearLayer{Outputs: numTokens},
		&layers.SoftmaxLayer{},
	)
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
				embeddingDataset = append(embeddingDataset, datasets.DataPoint{Input: datasets.ToOneHot(cleanTokenized[i], numTokens), Output: datasets.ToOneHot(cleanTokenized[i+j], numTokens)})
				if i-j == 1 || j-i == 1 {
					embeddingDataset = append(embeddingDataset, datasets.DataPoint{Input: datasets.ToOneHot(cleanTokenized[i], numTokens), Output: datasets.ToOneHot(cleanTokenized[i+j], numTokens)})
				}
			}
		}
	}

	fmt.Println("Starting Embedding Training:\n-----------------------------")
	embeddingNetwork.Train(embeddingDataset, embeddingDataset, 20*time.Second)

	justEmbeddings := networks.Sequential{}
	justEmbeddings.Initialize(numTokens, embeddingNetwork.Layers[:4]...)
	return justEmbeddings
}
