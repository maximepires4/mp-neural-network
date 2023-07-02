import numpy as np
from . import utils


class FFNN:
    def __init__(self, m):
        self.data = []
        self.m = m

    def train(self, loss, input, output, learning_rate, max_iter):
        W1, b1, W2, b2 = utils.init_params()
        X = np.copy(input)
        Y = np.copy(output)
        #Y = utils.one_hot(output)

        for i in range(max_iter):
            Z1, A1, Z2, A2 = utils.forward_prop(W1, b1, W2, b2, X)
            dW1, db1, dW2, db2 = utils.backward_prop(
                Z1, A1, Z2, A2, W1, W2, X, Y, self.m)
            W1, b1, W2, b2 = utils.update_params(
                W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
            if i % 10 == 0:
                print("Iteration: ", i)
                predictions = utils.get_predictions(A2)
                print(utils.get_accuracy(predictions, Y))
        return W1, b1, W2, b2
