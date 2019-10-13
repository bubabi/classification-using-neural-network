from math import log
import matplotlib.pyplot as plt
import numpy as np
from numpy import random


class SingleLayer:
    def __init__(self, x, y, epoch, alpha, batch_size):
        self.X = self.normalize(x)
        self.weights = np.random.rand(self.X.shape[1], 5) / 500
        self.y = y
        self.b = np.zeros((1, 5))
        self.epoch = epoch
        self.alpha = alpha
        self.batch_size = batch_size
        self.loss = []
        self.output = None

    def normalize(self, value):
        return value / 255

    # "relu": means reLU activation func., "tanh":means tanh activation func.,
    # else run the sigmoid activation func.
    def feedforward(self, bX, a_func):
        if a_func == "dot":
            return np.dot(bX, self.weights) + self.b
        elif a_func == "relu":
            return np.maximum(0, np.dot(bX, self.weights) + self.b)
        elif a_func == "tanh":
            return (np.exp(np.dot(bX, self.weights) + self.b) - np.exp(-np.dot(bX, self.weights) + self.b)) \
                   / (np.exp(np.dot(bX, self.weights) + self.b) + np.exp(-np.dot(bX, self.weights) + self.b))
        else:
            return self.sigmoid(bX)

    # Return the output of the softmax function.
    def softmax(self, outputs):
        # normalizing and obtaining probs
        return np.exp(outputs) / np.sum(np.exp(outputs), axis=1, keepdims=True)

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def vectorize_y(self, correct_class):
        res = np.zeros(5, dtype=int)
        res[correct_class] = 1
        return res

    def cross_entropy_loss(self,bX, bY, probs):
        # computing the loss which is cross-entropy loss
        data_loss = np.sum(-np.log(probs[range(bX.shape[0]), bY])) / bX.shape[0]
        return data_loss

    def sum_squared_error(self, output, labels):
        errors = output - labels
        loss = np.sum(errors ** 2)
        return errors, loss

    # Return the gradient of the cost function with respect to W and b
    def backpropagation(self, bX, bY, probs):

        d_outputs = probs
        d_outputs[range(bX.shape[0]), bY] -= 1
        d_outputs /= bX.shape[0]

        dW = np.dot(bX.T, d_outputs)
        db = np.sum(d_outputs, axis=0, keepdims=True)

        # then updating the parameters
        self.weights += -self.alpha * dW
        self.b += -self.alpha * db

    def train(self):
        X = self.X
        for i in np.arange(0, self.epoch):
            epoch_loss_values = []
            # train the data using mini-batch
            for current_batch in np.arange(0, X.shape[0], self.batch_size):
                bX = X[current_batch:current_batch + self.batch_size]
                batch_labels = self.y[current_batch:current_batch + self.batch_size]

                self.outputs = self.feedforward(bX, "dot") # dot means softmax
                probs = self.softmax(self.outputs) # getting a probability of each class

                loss = self.cross_entropy_loss(bX, batch_labels, probs) # calculate the loss using cross-entropy
                epoch_loss_values.append(loss)

                self.backpropagation(bX, batch_labels, probs) # minimize the loss func. using gradient descent
            if i % 10 == 0: print("iteration %d: loss %f" % (i, float(np.mean(epoch_loss_values))))
            self.loss.append(np.mean(epoch_loss_values))

        model = [self.weights, self.b]
        np.save("../model/slnn_model", model)

    def train_sigmoid(self):
        X = self.X
        for i in np.arange(0, self.epoch):
            epoch_loss_values = []
            # train the data using mini-batch
            for current_batch in np.arange(0, X.shape[0], self.batch_size):
                bX = X[current_batch:current_batch + self.batch_size]
                batch_labels = self.y[current_batch:current_batch + self.batch_size]

                p_outputs = self.sigmoid(np.dot(bX, self.weights))
                batch_label_vectors = []
                for true_val in batch_labels:
                    batch_label_vectors.append(self.vectorize_y(true_val))

                errors, loss = self.sum_squared_error(p_outputs, batch_label_vectors)

                epoch_loss_values.append(loss)

                dW = bX.T.dot(errors) / bX.shape[0]
                db = np.sum(errors, axis=0, keepdims=True)

                self.weights += -self.alpha * dW
                #self.b += -self.alpha * db

            #if i % 10 == 0: print("iteration %d: loss %f" % (i, float(np.mean(epoch_loss_values))))
            self.loss.append(np.mean(epoch_loss_values))

        model = [self.weights, self.b]
        np.save("slnn_model", model)

    def test(self, X, y):
        #print(self.weights)
        outputs = np.dot(X, self.weights) + self.b
        predicted_class = np.argmax(outputs, axis=1)
        print('accuracy: %.2f' % (np.mean(predicted_class == y)))

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
        self.weights = model[0]
        self.b = model[1]

    def visualize_params(self):
        fig = plt.figure(figsize=(8, 4))
        plt.title("Visualized the learned parameters")
        plt.axis('off')
        ax = []
        titles = ["(a) daisy",
                  "(b) dandelion",
                  "(c) rose",
                  "(d) sunflower",
                  "(e) tulip"]

        for i in range(1, 6):
            #params = self.weights[:, i-1]
            params2 = self.weights.T[i-1][0:768]

            img = np.reshape(params2, (32, 24))
            ax.append(fig.add_subplot(1, 5, i))
            ax[-1].set_xlabel(titles[i-1])  # set title
            plt.imshow(img, cmap='gray')

        plt.savefig('params.png', bbox_inches='tight')
        plt.show()
