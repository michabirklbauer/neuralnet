#!/usr/bin/env python3

# NEURAL NETWORK IMPLEMENTATION
# 2022 (c) Micha Johannes Birklbauer
# https://github.com/michabirklbauer/
# micha.birklbauer@gmail.com

import math
import numpy as np
from typing import Tuple
from typing import List

class LayerInitializer:
    """
    Functions for layer weight initialization.
    """

    # He normal initialization
    @staticmethod
    def he_normal(size: Tuple[int], fan_in: int) -> np.array:
        """
        HE NORMAL INITIALIZATION
        Draws samples from a truncated normal distribution centered at 0 mean
        with stddev = sqrt(2 / fan_in) where fan_in is the number of input
        units per unit in the layer.
        Parameters:
            - size: Tuple[int] (rows, columns)
                    shape of the initialized weight matrix
            - fan_in: int
                    number of input units per unit in the layer
        Returns:
            - np.array (rows, columns)
                    He normal initialized weight matrix
        Ref:
            https://arxiv.org/abs/1502.01852
        """
        return np.random.normal(0, math.sqrt(2 / fan_in), size = size)

    # Glorot / Xavier normal initialization
    @staticmethod
    def glorot_normal(size: Tuple[int], fan_in: int, fan_out: int) -> np.array:
        """
        GLOROT / XAVIER NORMAL INITIALIZATION
        Draws samples from a truncated normal distribution centered at 0 mean
        with stddev = sqrt(2 / (fan_in + fan_out)) where fan_in is the number of
        input units per unit in the layer and fan_out is the number of output
        units per unit in the layer.
        Parameters:
            - size: Tuple[int] (rows, columns)
                    shape of the initialized weight matrix
            - fan_in: int
                    number of input units per unit in the layer
            - fan_out: int
                    number of output units per unit in the layer
        Returns:
            - np.array (rows, columns)
                    Glorot normal initialized weight matrix
        Ref:
            http://proceedings.mlr.press/v9/glorot10a.html
        """
        return np.random.normal(0, math.sqrt(2 / (fan_in + fan_out)), size = size)

    # Bias initialization
    @staticmethod
    def bias(size: Tuple[int]):
        """
        BIAS INITIALIZATION
        Initializes the bias vector / matrix with zeros.
        Parameters:
            - size: Tuple[int] (rows, columns)
                    shape of the initialized bias vector / matrix
        Returns:
            - np.array (rows, columns)
                    Zero initialized bias vector / matrix
        Ref:
            https://cs231n.github.io/neural-networks-2/
        """
        return np.zeros(shape = size)

class ActivationFunctions:
    """
    Layer activation functions.
    """

    # Rectified Linear Units
    @staticmethod
    def relu(x: np.array, derivative: bool = False) -> np.array:
        """
        RECTIFIED LINEAR UNITS
        ReLU activation function.
        Parameters:
            - x: np.array
                    input matrix to apply activation function to
            - derivative: bool
                    if set to 'True' returns the derivative instead
                    DEFAULT: False
        Returns:
            - np.array (same shape as x)
                    activated x / derivative of x
        Ref:
            https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
        """
        if not derivative:
            return np.maximum(x, 0)
        else:
            return np.where(x > 0, 1, 0)

    # Sigmoid activation function
    @staticmethod
    def sigmoid(x: np.array, derivative: bool = False) -> np.array:
        """
        SIGMOID / LOGISTIC FUNCTION
        Sigmoid activation function.
        Parameters:
            - x: np.array
                    input matrix to apply activation function to
            - derivative: bool
                    if set to 'True' returns the derivative instead
                    DEFAULT: False
        Returns:
            - np.array (same shape as x)
                    activated x / derivative of x
        Refs:
            https://en.wikipedia.org/wiki/Sigmoid_function
            https://en.wikipedia.org/wiki/Activation_function
        """
        def f_sigmoid(x: np.array) -> np.array:
            return 1 / (1 + np.exp(-x))

        if not derivative:
            return f_sigmoid(x)
        else:
            return f_sigmoid(x) * (1 - f_sigmoid(x))

    # Softmax activation function
    @staticmethod
    def softmax(x: np.array, derivative: bool = False) -> np.array:
        """
        SOFTMAX FUNCTION
        Stable softmax activation function.
        Parameters:
            - x: np.array
                    input matrix to apply activation function to
        Returns:
            - np.array (same shape as x)
                    activated x
        Refs:
            https://en.wikipedia.org/wiki/Softmax_function
            https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
        """
        if not derivative:
            n = np.exp(x - np.max(x)) # stable softmax
            d = np.sum(n, axis = 0)
            return n / d
        else:
            raise NotImplementedError("Softmax derivative not implemented!")
            # https://stackoverflow.com/questions/54976533/derivative-of-softmax-function-in-python
            # xr = x.reshape((-1, 1))
            # return np.diagflat(x) - np.dot(xr, xr.T)

class LossFunctions:
    """
    Loss functions for neural net fitting.
    """

    # binary cross entropy loss
    @staticmethod
    def binary_cross_entropy(y_true: np.array, y_predicted: np.array) -> np.array:
        """
        BINARY CROSS ENTROPY LOSS
        Cross entropy loss for binary-class classification.
        L[BCE] = - p(i) * log(q(i)) - (1 - p(i)) * log(1 - q(i))
        where
            - p(i) is the true label
            - q(i) is the predicted sigmoid probability
        Parameters:
            - y_true: np.array (1, sample_size)
                    true label vector
            - y_predicted: np.array (1, sample_size)
                    the sigmoid probability
        Returns:
            - np.array (sample_size,)
                    loss for every given sample
        Ref:
            https://en.wikipedia.org/wiki/Cross_entropy
        """
        losses = []
        for i in range(y_true.shape[1]):
            ## stable BCE
            losses.append(float(-1 * (y_true[:, i] * np.log(y_predicted[:, i] + 1e-7) + (1 - y_true[:, i]) * np.log(1 - y_predicted[:, i] + 1e-7))))
            ## unstable BCE
            # losses.append(float(-1 * (y_true[:, i] * np.log(y_predicted[:, i]) + (1 - y_true[:, i]) * np.log(1 - y_predicted[:, i]))))
        return np.array(losses)

    # categorical cross entropy loss
    @staticmethod
    def categorical_cross_entropy(y_true: np.array, y_predicted: np.array) -> np.array:
        """
        CATEGORICAL CROSS ENTROPY LOSS
        Cross entropy loss for binary- and multi-class class classification.
        L[CCE] = - sum[from i = 0 to n]( p(i) * log(q(i)) )
        where
            - p(i) is the true label
            - q(i) is the predicted softmax probability
            - n is the number of classes
        Parameters:
            - y_true: np.array (n_classes, sample_size)
                    one-hot encoded true label vector
            - y_predicted: np.array (n_classes, sample_size)
                    the softmax probabilities
        Returns:
            - np.array (sample_size,)
                    loss for every given sample
        Ref:
            https://en.wikipedia.org/wiki/Cross_entropy
        """
        losses = []
        for i in range(y_true.shape[1]):
            ## stable CCE
            # losses.append(float(-1 * np.sum(y_true[:, i] * np.log(y_predicted[:, i] + 1e-7))))
            ## unstable CCE
            losses.append(float(-1 * np.sum(y_true[:, i] * np.log(y_predicted[:, i]))))

        return np.array(losses)

class NeuralNetwork:
    """
    Implementation of a classic feed-forward neural network that is trained via
    backpropagation. Adopts a Keras-like interface for convenient usage (see
    https://michabirklbauer.github.io/neuralnet for examples).
    """

    # constructor
    def __init__(self, input_size: int):
        """
        CONSTRUCTOR
        Initializes the neural network model.
        Parameters:
            - input_size: int
                    nr. of features in the training data
        Returns:
            - None
        Example usage:
            NN = NeuralNetwork(data.shape[1])
        """
        self.input_size = input_size
        self.architecture = []
        self.layers = []

    # adding layers
    def add_layer(self, units: int, activation: str = "relu", initialization: str = None) -> None:
        """
        LAYER MANAGEMENT
        Construct the neural network architecture by adding different layers.
        Parameters:
            - units: int
                    nr. of units in the layer
            - activation: str, one of ("relu", "sigmoid", "softmax")
                    activation function of the layer
                    DEFAULT: "relu"
            - initialization: str, one of ("he", "glorot")
                    weight initialization to use
                    DEFAULT: None, "relu" layers are 'he normal' initialized,
                                   all other layers are 'glorot normal'
                                   initialized
        Returns:
            - None
        Example usage:
            NN = NeuralNetwork(data.shape[1])
            NN.add_layer(16, "relu", "glorot")
            NN.add_layer(8)
            NN.add_layer(1, "sigmoid")
        """
        if initialization == None:
            if activation == "relu":
                layer_init = "he"
            else:
                layer_init = "glorot"
        else:
            layer_init = initialization

        self.architecture.append({"units": units, "activation": activation, "init": layer_init})

    # compiling model
    def compile(self, loss: str = "categorical crossentropy") -> None:
        """
        MODEL INITIALIZATION
        Initializes all parameters of the neural network architecture and
        prepares the model for training.
        Parameters:
            - loss: str, one of ("binary crossentropy", "categorical crossentropy")
                    the loss function that should be used for training
                    DEFAULT: "categorical crossentropy"
        Returns:
            - None
        Example usage:
            NN = NeuralNetwork(data.shape[1])
            NN.add_layer(16, "relu", "glorot")
            NN.add_layer(8)
            NN.add_layer(1, "sigmoid")
            NN.compile("binary crossentropy")
        """
        self.loss = loss

        # initialize all layer weights and biases
        for i in range(len(self.architecture)):
            units = self.architecture[i]["units"]
            activation = self.architecture[i]["activation"]
            init = self.architecture[i]["init"]

            units_previous_layer = self.input_size
            if i > 0:
                units_previous_layer = self.architecture[i - 1]["units"]
            units_next_layer = 0
            if i < len(self.architecture) - 1:
                units_next_layer = self.architecture[i + 1]["units"]

            if init  == "he":
                W = LayerInitializer.he_normal((units, units_previous_layer), fan_in = units_previous_layer)
                b = LayerInitializer.bias((units, 1))
            elif init == "glorot":
                W = LayerInitializer.glorot_normal((units, units_previous_layer), fan_in = units_previous_layer, fan_out = units_next_layer)
                b = LayerInitializer.bias((units, 1))
            else:
                raise NotImplementedError("Layer initialization '" + init + "' not implemented!")

            self.layers.append({"W": W, "b": b, "activation": activation})

    # forward propagation
    def __forward_propagation(self, data: np.array) -> None:
        """
        FORWARD PROPAGATION (INTERNAL)
        Internal function calculating the forward pass of A(Wx + b).
            - The result of 'Wx + b' (L) is stored in self.layers[layer]["L"]
            - The result of 'Activation(L)' (A) is stored in self.layers[layer]["A"]
        Parameters:
            - data: np.array
                    input data for the forward pass
        Returns:
            - None, "L" and "A" are set in the layer dictionary, to retrieve the
                    last layer output call 'self.layers[-1]["A"]'
        """

        for i in range(len(self.layers)):

            if i == 0:
                A = data
            else:
                A = self.layers[i - 1]["A"]

            # Wx + b where x is the input data for the first layer and otherwise
            # the output (A) of the previous layer
            self.layers[i]["L"] = self.layers[i]["W"].dot(A) + self.layers[i]["b"]
            if self.layers[i]["activation"] == "relu":
                self.layers[i]["A"] = ActivationFunctions.relu(self.layers[i]["L"])
            elif self.layers[i]["activation"] == "sigmoid":
                self.layers[i]["A"] = ActivationFunctions.sigmoid(self.layers[i]["L"])
            elif self.layers[i]["activation"] == "softmax":
                self.layers[i]["A"] = ActivationFunctions.softmax(self.layers[i]["L"])
            else:
                raise NotImplementedError("Activation function '" + layer["activation"] + "' not implemented!")

    # back propagation
    def __back_propagation(self, data: np.array, target: np.array, learning_rate: float = 0.1) -> float:
        """
        BACK PROPAGATION (INTERNAL)
        Internal function for learning layer weights and biases using gradient
        descent and back propagation.
        Parameters:
            - data: np.array
                    input data
            - target: np.array
                    class labels of the input data
            - learning_rate: float
                    learning rate / how far in the direction of the gradient to
                    go
                    DEFAULT: 0.1
        Returns:
            - float
                    loss of the current forward pass
        """
        # forward pass
        self.__forward_propagation(data)

        output = self.layers[-1]["A"]
        batch_size = data.shape[1]
        loss = 0

        # calculate loss of the current forward pass
        if self.loss == "categorical crossentropy":
            losses = LossFunctions.categorical_cross_entropy(y_true = target, y_predicted = output)
            # reduction by sum over batch size
            loss = float(np.sum(losses) / batch_size)
        elif self.loss == "binary crossentropy":
            losses = LossFunctions.binary_cross_entropy(y_true = target, y_predicted = output)
            # reduction by sum over batch size
            loss = float(np.sum(losses) / batch_size)
        else:
            raise NotImplementedError("Loss function '" + self.loss + "' not implemented!")

        # calculate and back pass the derivate of the loss w.r.t the output
        # activation function
        # this implementation suppports CCE + Softmax and BCE + Sigmoid in the
        # output layer
        if self.loss == "categorical crossentropy" and self.layers[-1]["activation"] == "softmax":
            # for categorical cross entropy loss the derivative of softmax simplifies to
            # P(i) - Y(i)
            # where P(i) is the softmax output and Y(i) is the true label
            # https://www.ics.uci.edu/~pjsadows/notes.pdf
            # https://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function
            previous_layer_activation = data.T if len(self.layers) == 1 else self.layers[len(self.layers) - 2]["A"].T
            dL = self.layers[-1]["A"] - target
            dW = dL.dot(previous_layer_activation) / batch_size
            db = np.reshape(np.sum(dL, axis = 1), (-1, 1)) / batch_size

            # parameter tracking
            previous_dL = np.copy(dL)
            previous_W = np.copy(self.layers[-1]["W"])

            # update
            self.layers[-1]["W"] -= learning_rate * dW
            self.layers[-1]["b"] -= learning_rate * db
        elif self.loss == "binary crossentropy" and self.layers[-1]["activation"] == "sigmoid":
            # for binary cross entropy loss the derivative of the loss function is
            # L' = -1 * (Y(i) / P(i) - (1 - Y(i)) / (1 - P(i)))
            # where P(i) is the sigmoid output and Y(i) is the true label
            # and we multiply that with the derivative of the sigmoid function [1]
            # https://math.stackexchange.com/questions/2503428/derivative-of-binary-cross-entropy-why-are-my-signs-not-right
            previous_layer_activation = data.T if len(self.layers) == 1 else self.layers[len(self.layers) - 2]["A"].T
            # [1]
            # A = np.clip(self.layers[-1]["A"], 1e-7, 1 - 1e-7)
            # derivative_loss = -1 * np.divide(target, A) + np.divide(1 - target, 1 - A)
            # dL = derivative_loss * ActivationFunctions.sigmoid(self.layers[-1]["L"], derivative = True)
            # alternatively we can directly simplify the derivative of the binary cross entropy loss
            # with sigmoid activation function to
            # P(i) - Y(i)
            # where P(i) is the sigmoid output and Y(i) is the true label
            # done in [2]
            # https://math.stackexchange.com/questions/4227931/what-is-the-derivative-of-binary-cross-entropy-loss-w-r-t-to-input-of-sigmoid-fu
            # [2]
            dL = (self.layers[-1]["A"] - target) / batch_size
            dW = dL.dot(previous_layer_activation) / batch_size
            db = np.reshape(np.sum(dL, axis = 1), (-1, 1)) / batch_size

            # parameter tracking
            previous_dL = np.copy(dL)
            previous_W = np.copy(self.layers[-1]["W"])

            # update
            self.layers[-1]["W"] -= learning_rate * dW
            self.layers[-1]["b"] -= learning_rate * db
        else:
            raise NotImplementedError("The combination of '" + self.loss + " loss' and '" + self.layers[i]["activation"] + " activation' is not implemented!")

        # back propagation through the remaining hidden layers
        for i in reversed(range(len(self.layers) - 1)):

            if i == 0:
                if self.layers[i]["activation"] == "relu":
                    dL = previous_W.T.dot(previous_dL) * ActivationFunctions.relu(self.layers[i]["L"], derivative = True)
                    dW = dL.dot(data.T) / batch_size
                    db = np.reshape(np.sum(dL, axis = 1), (-1, 1)) / batch_size
                elif self.layers[i]["activation"] == "sigmoid":
                    dL = previous_W.T.dot(previous_dL) * ActivationFunctions.sigmoid(self.layers[i]["L"], derivative = True)
                    dW = dL.dot(data.T) / batch_size
                    db = np.reshape(np.sum(dL, axis = 1), (-1, 1)) / batch_size
                else:
                    raise NotImplementedError("Activation function '" + self.layers[i]["activation"] + "' not implemented for hidden layers!")

                # parameter tracking
                previous_dL = np.copy(dL)
                previous_W = np.copy(self.layers[i]["W"])

                #update
                self.layers[i]["W"] -= learning_rate * dW
                self.layers[i]["b"] -= learning_rate * db
            else:
                if self.layers[i]["activation"] == "relu":
                    dL = previous_W.T.dot(previous_dL) * ActivationFunctions.relu(self.layers[i]["L"], derivative = True)
                    dW = dL.dot(self.layers[i - 1]["A"].T) / batch_size
                    db = np.reshape(np.sum(dL, axis = 1), (-1, 1)) / batch_size
                elif self.layers[i]["activation"] == "sigmoid":
                    dL = previous_W.T.dot(previous_dL) * ActivationFunctions.sigmoid(self.layers[i]["L"], derivative = True)
                    dW = dL.dot(self.layers[i - 1]["A"].T) / batch_size
                    db = np.reshape(np.sum(dL, axis = 1), (-1, 1)) / batch_size
                else:
                    raise NotImplementedError("Activation function '" + self.layers[i]["activation"] + "' not implemented for hidden layers!")

                # parameter tracking
                previous_dL = np.copy(dL)
                previous_W = np.copy(self.layers[i]["W"])

                #update
                self.layers[i]["W"] -= learning_rate * dW
                self.layers[i]["b"] -= learning_rate * db

        return loss

    # neural network architecture summary
    def summary(self) -> None:
        """
        MODEL SUMMARY
        Print a summary of the neural network architecture.
        Parameters:
            - None
        Returns:
            - None, prints a summary of the neural network architecture to
                    stdout
        Example usage:
            NN.summary()
        """
        print("---- Model Summary ----")
        for i, layer in enumerate(self.layers):
            print("Layer " + str(i + 1) + ": " + layer["activation"])
            if "L" in layer:
                print("W: " + str(layer["W"].shape) + " " +
                      "b: " + str(layer["b"].shape) + " " +
                      "L: " + str(layer["L"].shape) + " " +
                      "A: " + str(layer["A"].shape))
            else:
                print("W: " + str(layer["W"].shape) + " " +
                      "b: " + str(layer["b"].shape))
            print("Trainable parameters: " + str(
                layer["W"].shape[0] * layer["W"].shape[1] +
                layer["b"].shape[0] * layer["b"].shape[1]))

    # train neural network on data
    def fit(self, X: np.array, y: np.array, epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.1, verbose: int = 1) -> List[float]:
        """
        TRAIN MODEL
        Train the neural network.
        Parameters:
            - X: np.array (samples, features)
                    input data to train on
            - y: np.array (samples, labels) or (labels,)
                    labels of the input data
            - epochs: int
                    how many iterations to train
                    DEFAULT: 100
            - batch_size: int
                    how many samples to use per backward pass
                    DEFAULT: 32
            - learning_rate: float
                    learning rate / how far in the direction of the gradient to
                    go
                    DEFAULT: 0.1
            - verbose: int, one of (0, 1) / bool
                    print information for every epoch
                    DEFAULT: 1 (True)
        Returns:
            - List[float]
                    loss history over all epochs
        Example usage:
            NN.fit(data_train, labels_train)
        """
        # reshaping inputs
        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))

        data = X.T
        target = y.T
        sample_size = data.shape[1]

        history = []

        # train network
        for i in range(epochs):
            if verbose:
                print("Training epoch " + str(i + 1) + "...")
            # generate random batches of size batch_size
            idx = np.random.choice(sample_size, sample_size, replace = False)
            batches = np.array_split(idx, math.ceil(sample_size / batch_size))
            batch_losses = []
            for batch in batches:
                current_data = data[:, batch]
                current_target = target[:, batch]
                batch_loss = self.__back_propagation(current_data, current_target, learning_rate = learning_rate)
                batch_losses.append(batch_loss)
            history.append(np.mean(batch_losses))
            if verbose:
                print("Current loss: ", np.mean(batch_losses))
                print("Epoch " + str(i + 1) + " done!")

        print("Training finished after epoch " + str(epochs) + " with a loss of " + str(history[-1]) + ".")

        return history

    # predict data with fitted neural network
    def predict(self, X: np.array) -> np.array:
        """
        GENERATE PREDICTIONS
        Predict labels for the given input data.
        Parameters:
            - X: np.array (samples, features) or (features,)
                    input data to predict
        Returns:
            - np.array
                    predictions
        Example usage:
            NN.predict(data_test)
        """
        if X.ndim == 1:
            X = np.reshape(X, (1, -1))

        self.__forward_propagation(X.T)

        return self.layers[-1]["A"].T

if __name__ == "__main__":
    pass

    """
    #### Multi-class Classification ####

    import pandas as pd
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.model_selection import train_test_split

    data = pd.read_csv("multiclass_train.csv")
    train, test = train_test_split(data, test_size = 0.3)
    train_data = train.loc[:, train.columns != "label"].to_numpy() / 255
    train_target = train["label"].to_numpy()
    test_data = test.loc[:, test.columns != "label"].to_numpy() / 255
    test_target = test["label"].to_numpy()

    one_hot = OneHotEncoder(sparse = False, categories = "auto")
    train_target = one_hot.fit_transform(train_target.reshape(-1, 1))
    test_target = one_hot.transform(test_target.reshape(-1, 1))

    NN = NeuralNetwork(input_size = train_data.shape[1])
    NN.add_layer(32, "relu")
    NN.add_layer(16, "relu")
    NN.add_layer(10, "softmax")
    NN.compile(loss = "categorical crossentropy")
    NN.summary()

    hist = NN.fit(train_data, train_target, epochs = 30, batch_size = 16, learning_rate = 0.05)

    train_predictions = np.argmax(NN.predict(train_data), axis = 1)
    print("Training accuracy: ", accuracy_score(train["label"].to_numpy(), train_predictions))
    test_predictions = np.argmax(NN.predict(test_data), axis = 1)
    print("Test accuracy: ", accuracy_score(test["label"].to_numpy(), test_predictions))

    #### Binary-class Classification ####

    import pandas as pd
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.model_selection import train_test_split

    data = pd.read_csv("binaryclass_train.csv", header = None)
    data["label"] = data[1].apply(lambda x: 1 if x == "M" else 0)
    train, test = train_test_split(data, test_size = 0.3)
    train_data = train.loc[:, ~train.columns.isin([0, 1, "label"])].to_numpy()
    train_target = train["label"].to_numpy()
    test_data = test.loc[:, ~test.columns.isin([0, 1, "label"])].to_numpy()
    test_target = test["label"].to_numpy()

    NN = NeuralNetwork(input_size = train_data.shape[1])
    NN.add_layer(16, "relu")
    NN.add_layer(16, "relu")
    NN.add_layer(1, "sigmoid")
    NN.compile(loss = "binary crossentropy")
    NN.summary()

    hist = NN.fit(train_data, train_target, epochs = 1000, batch_size = 32, learning_rate = 0.01)

    train_predictions = np.round(NN.predict(train_data))
    print("Training accuracy: ", accuracy_score(train["label"].to_numpy(), train_predictions))
    test_predictions = np.round(NN.predict(test_data))
    print("Test accuracy: ", accuracy_score(test["label"].to_numpy(), test_predictions))

    """
