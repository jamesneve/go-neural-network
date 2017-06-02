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

	mnistData := trainingdata.NewMnistData()
	trainingData := mnistData.MakeTrainingData()
	testData := mnistData.MakeTestData()

	nt := learn.NewNetworkTrainer(n, trainingData, learn.CrossEntropy, learn.L2Regularization, 0.1, 5.0)

	fmt.Println("Training MNIST Dataset")
	nt.TrainByGradientDescent(30, 10, testData)
}