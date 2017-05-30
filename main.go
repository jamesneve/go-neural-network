package main

import (
	"fmt"
	"github.com/jamesneve/neuralnetwork2/network"
	"github.com/jamesneve/neuralnetwork2/mnist"
	"github.com/jamesneve/neuralnetwork2/learn"
	"math/rand"
	"time"
)

func main() {
	fmt.Println("Neural Network 2")

	n := network.NewNetwork(784, []int{30, 10})
	n.RandomizeWeightsAndBiases()

	mnistData := mnist.NewMnistData()
	trainingData := mnistData.MakeTrainingData()
	testData := mnistData.MakeTestData()

	rand.Seed(time.Now().UnixNano())

	fmt.Println("Training MNIST Dataset")
	n.TrainByGradientDescent(trainingData, 10, 10, 3.0, testData)

	fmt.Println("Running test data")

	correctResults := n.Evaluate(testData)
	fmt.Println("Correct: ", correctResults, " / ", len(testData))
}