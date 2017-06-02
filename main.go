package main

import (
	"fmt"
	"github.com/jamesneve/go-neural-network/network"
	"github.com/jamesneve/go-neural-network/trainingdata"
	"github.com/jamesneve/go-neural-network/learn"
)

func main() {
	fmt.Println("Go Neural Network")
	fmt.Println("Default: SGD / L2 Regression / Cross-Entropy")

	n := network.NewNetwork(784, []int{30, 10})
	n.RandomizeWeightsAndBiases()

	mnistData := trainingdata.NewMnistData()
	trainingData := mnistData.MakeTrainingData()
	testData := mnistData.MakeTestData()

	nt := learn.NewNetworkTrainer(n)

	fmt.Println("Training MNIST Dataset")
	nt.TrainByGradientDescent(trainingData, 10, 10, 3.0, testData)
}