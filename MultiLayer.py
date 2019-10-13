from math import log
import matplotlib.pyplot as plt
import numpy as np
from numpy import random


class MultiLayer:
    def __init__(self, x, y, epoch, alpha, batch_size):
        self.X = self.normalize(x)
        self.W = np.random.rand(self.X.shape[1], 100) / 100
        self.W2 = np.random.rand(100, 5) / 100
        self.y = y
        self.b = np.zeros((1, 100))
        self.b2 = np.zeros((1, 5))
        self.epoch = epoch
        self.alpha = alpha
        self.batch_size = batch_size
        self.loss = []
        self.output = None

    def normalize(self, value):
        return value / 255

    def feedforward(self, bX):
        hidden_layer = np.maximum(0, np.dot(bX, self.W) + self.b)  # note, ReLU activation
        scores = np.dot(hidden_layer, self.W2) + self.b2
        return hidden_layer, scores

    # Return the output of the softmax function for the matrix of output.
    def softmax(self, outputs):
        # normalizing and obtaining probs
        return np.exp(outputs) / np.sum(np.exp(outputs), axis=1, keepdims=True)

    def cross_entropy_loss(self,bX, bY, probs):
        # computing the loss which is cross-entropy loss
        data_loss = np.sum(-np.log(probs[range(bX.shape[0]), bY])) / bX.shape[0]
        return data_loss

    def backpropagation(self, bX, bY, probs, hidden_layer):

        d_outputs = probs
        d_outputs[range(bX.shape[0]), bY] -= 1
        d_outputs /= bX.shape[0]

        dW2 = np.dot(hidden_layer.T, d_outputs)
        db2 = np.sum(d_outputs, axis=0, keepdims=True)

        #backprop to hidden layer
        dhidden = np.dot(d_outputs, self.W2.T)

        dhidden[hidden_layer <= 0] = 0

        dW = np.dot(bX.T, dhidden)
        db = np.sum(dhidden, axis=0, keepdims=True)

        # updating the parameters
        self.W += -self.alpha * dW
        self.b += -self.alpha * db

        self.W2 += -self.alpha * dW2
        self.b2 += -self.alpha * db2

    def train(self):
        X = self.X
        for i in np.arange(0, self.epoch):
            epoch_loss_values = []
            # train the data using mini-batch
            for current_batch in np.arange(0, X.shape[0], self.batch_size):
                bX = X[current_batch:current_batch + self.batch_size]
                batch_labels = self.y[current_batch:current_batch + self.batch_size]

                hidden_layer, self.outputs = self.feedforward(bX)
                probs = self.softmax(self.outputs) # getting a probability of each class

                loss = self.cross_entropy_loss(bX, batch_labels, probs) # calculate the loss using cross-entropy
                epoch_loss_values.append(loss)

                # minimize the loss func. using gradient descent
                self.backpropagation(bX, batch_labels, probs, hidden_layer)
            if i % 10 == 0: print("iteration %d: loss %f" % (i, float(np.mean(epoch_loss_values))))
            self.loss.append(np.mean(epoch_loss_values))

        model = [self.W, self.W2, self.b, self.b2]
        np.save("../model/slnn_model", model)

    def test(self, X, y):
        # evaluate training set accuracy
        hidden_layer = np.maximum(0, np.dot(X, self.W) + self.b)
        scores = np.dot(hidden_layer, self.W2) + self.b2
        predicted_class = np.argmax(scores, axis=1)
        print('accuracy: %.2f' % (float(np.mean(predicted_class == y)*100)))

        return np.mean(predicted_class == y)*100

    def plot_loss(self):
        plt.plot(np.arange(0, self.epoch), self.loss)
        plt.title("epoch: " + str(self.epoch) + " alpha: " + str(self.alpha) + " batch: " + str(self.batch_size))
        plt.xlabel("epoch")
        plt.ylabel("loss value")
        plt.savefig('loss.png', bbox_inches='tight')
        plt.show()

    def load_model(self, model_path):
        model = np.load(model_path)
        self.W = model[0]
        self.W2 = model[1]
        self.b = model[2]
        self.b2 = model[3]
