# Go Neural Network

A neural network written in Go, trained by backpropagation.
MNIST data set and loader included. Running the main method trains the network on the MNIST dataset over 10 epochs.
Depending on the hyper parameters, it gets about 97% accuracy on MNIST, and can be pushed beyond that if you're prepared
to wait for more hidden neurons.
 
# Usage

## Command Line

You can test it out by calling it from the command line. Command line calls can only be used to train the network with
the provided MNIST dataset. For now, there's no way to provide your own dataset from the command line.

```bash
~> go-neural-network --help
Usage of go-neural-network:
 -cost-function string
    The cost function used. Options: cross-entropy, quadratic (default "cross-entropy")
 -epochs int
    The number of epochs to train for (default 30)
 -lambda float
    The lambda value for L2 Regularization. Doesn't do anything when using other modes (default 5)
 -layers string
    The pattern of layers in the network, starting from the first hidden layer, described as integers separated by commas (default "30,10")
 -learning-rate float
    The speed of gradient descent (default 0.1)
 -mini-batch-size int
    The mini-batch size for SGD (default 10)
 -regularization string
    The type of regularization. Options: none, l1, l2 (default "l2")
 -report-results
    Whether or not to print results of individual epochs as they're completed (default true)
```

For example:

```bash
~> go-neural-network -epochs=2
Training MNIST dataset
Epoch 1 : 9216 / 10000
Epoch 2 : 9317 / 10000
Final accuracy : 9317 / 10000
```

## Code

Calling it from within code is the only way to train datasets besides MNIST.

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