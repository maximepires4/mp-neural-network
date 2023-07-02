import sys
import getopt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle, gzip

GRAPH_MIN = -1
GRAPH_MAX = 6


def arrayB(array):
    # Calculate a new array with a column of ones based on the input array
    arrayB = np.zeros((array.shape[0], array.shape[1]+1))
    for i in range(array.shape[0]):
        arrayB[i, 0] = 1
        arrayB[i, 1:] = array[i, :]

    return arrayB


def doForwardPropagation(Xb, Y, v, w):
    # Forward propagation algorithm

    Xbb = np.dot(Xb, v)
    F = sigmoid(Xbb)

    Fb = arrayB(F)

    Fbb = np.dot(Fb, w)
    G = sigmoid(Fbb)

    error = np.sum((Y - G)**2)

    return F, Fb, G, error


def resolveForwardPropagation(Xb, Y, v, w):
    # Forward propagation algorithm, this function is simplified when we only need to resolve for some given points

    Xbb = np.dot(Xb, v)
    F = sigmoid(Xbb)

    Fb = np.zeros((F.shape[0], F.shape[1]+1))
    for i in range(F.shape[0]):
        Fb[i, 0] = 1
        Fb[i, 1:] = F[i, :]

    Fbb = np.dot(Fb, w)
    G = sigmoid(Fbb)

    return G


def sigmoid(x):
    # sigmoid function
    return 1 / (1 + np.exp(-x))


def doBackPropagation(Xb, F, Fb, G, v, w, alpha1, alpha2, Y):
    # Back propagation algorithm

    delta_w = np.dot(Fb.T, (Y - G)*G*(1-G))
    delta_v = np.dot(Xb.T, np.dot((Y - G)*G*(1-G), w[1:, :].T)*F*(1-F))

    w = w + alpha1*delta_w
    v = v + alpha2*delta_v

    return v, w


def ffnn(data_in, v, w, Y, print_error, print_output_variable):
    # Feedforward neural network algorithm

    # Initialize variables
    alpha1 = 0.001
    alpha2 = alpha1
    epsilon = 0.0001
    error = np.power(10, 5)
    error_old = error
    deltaError = error
    itera = 0

    Xb = arrayB(data_in[['x1', 'x2']].to_numpy())

    error_history = []

    # Iterate until the error is acceptable
    while np.abs(deltaError) > epsilon:
        itera = itera + 1
        F, Fb, G, error = doForwardPropagation(Xb, Y, v, w)
        v, w = doBackPropagation(Xb, F, Fb, G, v, w, alpha1, alpha2, Y)

        # Keep track of the error
        deltaError = error - error_old
        error_old = error
        error_history.append(error)

    if print_error:
        print(f"Error: {error}")
        print(f"Iterations: {itera}")
        plt.figure(2)
        plt.plot(error_history)
        plt.xlabel('Iteration')
        plt.ylabel('Error')

    if print_output_variable:
        print(f"G: {G}")
        print(f"v: {v}")
        print(f"w: {w}")

    return v, w


def main(argv):
    opts, args = getopt.getopt(argv, "hev", ["error", "verbose"])
    print_error = False
    print_output_variable = False

    for opt, arg in opts:
        if opt == '-h':
            print('ffnn.py -e -v')
            sys.exit()
        elif opt in ("-e", "--error"):
            print_error = True
        elif opt in ("-v", "--verbose"):
            print_output_variable = True

    name_file = './data.txt'

    # Import data
    columns = ['x1', 'x2', 'y']
    data_in = pd.read_csv(name_file,
                          names=columns,
                          sep=' ')

    # Prepare plots
    fig, axs = plt.subplots(2)

    def applyPlotStyle(ax, title):
        ax.set(xlabel='x', ylabel='y', title=title)
        ax.set_xlim(GRAPH_MIN, GRAPH_MAX)
        ax.set_ylim(GRAPH_MIN, GRAPH_MAX)
        ax.set_aspect('equal')

    applyPlotStyle(axs[0], 'Given Data')
    applyPlotStyle(axs[1], 'AI Frontier')

    # Plot imported data
    x1 = np.asarray(data_in[data_in['y'] == 0]['x1'])
    y1 = np.asarray(data_in[data_in['y'] == 0]['x2'])

    x2 = np.asarray(data_in[data_in['y'] == 1]['x1'])
    y2 = np.asarray(data_in[data_in['y'] == 1]['x2'])

    x3 = np.asarray(data_in[data_in['y'] == 2]['x1'])
    y3 = np.asarray(data_in[data_in['y'] == 2]['x2'])

    axs[0].plot(x1, y1, 'ro')
    axs[0].plot(x2, y2, 'bo')
    axs[0].plot(x3, y3, 'yo')

    # Initialize neurons number and weights
    K = 4
    v = np.random.rand(2+1, K)  # 2+1 inputs / K neurons
    w = np.random.rand(K+1, 3)  # K+1 neurons / 3 outputs

    # Initialize the output array
    Y = np.zeros((data_in.shape[0], 3))
    for i in range(data_in.shape[0]):
        Y[i, int(data_in['y'][i])] = 1

    v, w = ffnn(data_in, v, w, Y, print_error, print_output_variable)

    # Create an array representing all the points in the graph
    x = np.arange(GRAPH_MIN, GRAPH_MAX+0.1, 0.3)
    y = np.arange(GRAPH_MIN, GRAPH_MAX+0.1, 0.3)
    xx, yy = np.meshgrid(x, y)
    M = np.column_stack((xx.ravel(), yy.ravel()))

    Mb = arrayB(M)

    # Resolve with the trained weights
    G = resolveForwardPropagation(
        Mb, Y, v, w)

    # Plot the frontier
    for i in range(G.shape[0]):
        if G[i, 0] > G[i, 1] and G[i, 0] > G[i, 2]:
            axs[1].plot(M[i, 0], M[i, 1], 'rs')
        elif G[i, 1] > G[i, 0] and G[i, 1] > G[i, 2]:
            axs[1].plot(M[i, 0], M[i, 1], 'bs')
        elif G[i, 2] > G[i, 0] and G[i, 2] > G[i, 1]:
            axs[1].plot(M[i, 0], M[i, 1], 'ys')
        else:
            axs[1].plot(M[i, 0], M[i, 1], 'ks')

    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
