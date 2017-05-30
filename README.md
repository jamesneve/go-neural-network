# Go Neural Network

A neural network written in Go, trained by backpropagation.
MNIST data set and loader included. Running the main method trains the network on the MNIST dataset over 30 epochs.
Depending a bit on the initial randomised weights and biases, the network gets 95-97% accuracy.

This was mostly just intended as an experiment for myself to help me understand the structure of neural networks and
the backpropagation algorithm. It takes a lot from existing implementations, and I don't claim any particular credit. 
I only publish because I couldn't find any other direct implementations in Go of the methods in Michael Nielsen's 
excellent book, and I find his Python arrays aren't always intuitive for someone not familiar with the language, so
maybe it'll help someone.

# Credit

* The structure of the network and division into structs was heavily inspired by NOX73's go-neural: https://github.com/NOX73/go-neural
* Training methods are Go versions of the Python code in Michael Nielsen's book: http://neuralnetworksanddeeplearning.com/index.html