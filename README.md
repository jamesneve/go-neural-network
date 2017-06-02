# Go Neural Network

A neural network written in Go, trained by backpropagation.
MNIST data set and loader included. Running the main method trains the network on the MNIST dataset over 10 epochs.
Depending on the hyper parameters, it gets about 97% accuracy on MNIST, and can be pushed beyond that if you're prepared
to wait for more hidden neurons.
 
# Usage

Command line interface coming soon. For now, running the main method just trains the whole thing on the MNIST
dataset with some reasonable choice of hyper-parameters.

Otherwise, you can call it from within your own code like so:

```go
// Initialize the network, here with 784 inputs, one hidden layer of 30 neurons and an output layer of 10 neurons
// Random weights, with mean 0, SD 1/root(n)
n := network.NewNetwork(784, []int{30, 10})

// Training data and test date are []float64 arrays
// The MNIST dataset and a loader are built in
mnistData := trainingdata.NewMnistData()
trainingData := mnistData.MakeTrainingData()
testData := mnistData.MakeTestData()

// Train the network by SGD 
// With training data, cost function, regularization function, learning rate and lambda values...
nt := learn.NewNetworkTrainer(n, trainingData, learn.CrossEntropy, learn.L2Regularization, 0.1, 5.0)

// ... for 30 epochs, with a mini-batch size of 10, evaluating on the test data
nt.TrainByGradientDescent(30, 10, testData)
```

A few choices of regularization
```go
learn.NoRegularization
learn.L1Regularization
learn.L2Regularization
```

And of cost function
```go
learn.QuadraticCost
learn.CrossEntropy
```

# Credit

* Michael Nielsen's excellent book: http://neuralnetworksanddeeplearning.com/index.html