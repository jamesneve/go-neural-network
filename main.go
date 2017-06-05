package main

import (
	"errors"
	"flag"
	"fmt"
	"github.com/jamesneve/go-neural-network/learn"
	"github.com/jamesneve/go-neural-network/network"
	"github.com/jamesneve/go-neural-network/trainingdata"
	"strconv"
	"strings"
)

func main() {
	fmt.Println("Training MNIST dataset")

	layers := flag.String("layers", "30,10", "The pattern of layers in the network, starting from the first hidden layer, described as integers separated by commas")
	costFunction := flag.String("cost-function", "cross-entropy", "The cost function used. Options: cross-entropy, quadratic")
	regularization := flag.String("regularization", "l2", "The type of regularization. Options: none, l1, l2")
	learningRate := flag.Float64("learning-rate", 0.1, "The speed of gradient descent")
	lambda := flag.Float64("lambda", 5.0, "The lambda value for L2 Regularization. Doesn't do anything when using other modes")
	reportResults := flag.Bool("report-results", true, "Whether or not to print results of individual epochs as they're completed")
	epochs := flag.Int("epochs", 30, "The number of epochs to train for")
	miniBatchSize := flag.Int("mini-batch-size", 10, "The mini-batch size for SGD")

	flag.Parse()
	parsedLayers, err := parseLayers(*layers)
	if err != nil {
		fmt.Println(err.Error())
		return
	}

	parsedCostFunction := parseCostFunction(*costFunction)
	parsedRegularization := parseRegularization(*regularization)

	n := network.NewNetwork(784, *parsedLayers)

	mnistData := trainingdata.NewMnistData()
	trainingData := mnistData.MakeTrainingData()
	testData := mnistData.MakeTestData()

	nt := learn.NewNetworkTrainer(n, trainingData, parsedCostFunction, parsedRegularization, *learningRate, *lambda, *reportResults)
	nt.TrainByGradientDescent(*epochs, *miniBatchSize, testData)

	correct := nt.Evaluate(testData)
	fmt.Println("Final accuracy :", correct, "/", len(testData))
}

func parseLayers(l string) (*[]int, error) {
	stringLayers := strings.Split(l, ",")
	intLayers := make([]int, 0, len(stringLayers))
	for _, s := range stringLayers {
		i, err := strconv.ParseInt(s, 10, 32)
		if err != nil {
			return nil, errors.New("The layers must be a comma-separated list of integers")
		}

		intLayers = append(intLayers, int(i))
	}

	if intLayers[len(intLayers)-1] != 10 {
		return nil, errors.New("For training MNIST, the final (output) layer must be 10")
	}

	return &intLayers, nil
}

func parseCostFunction(e string) learn.CostFunction {
	switch e {
	case "cross-entropy":
		return learn.CrossEntropy
	case "quadratic":
		return learn.QuadraticCost
	}

	return learn.CrossEntropy
}

func parseRegularization(e string) learn.RegularizationFunction {
	switch e {
	case "none":
		return learn.NoRegularization
	case "l1":
		return learn.L1Regularization
	case "l2":
		return learn.L2Regularization
	}

	return learn.L2Regularization
}
