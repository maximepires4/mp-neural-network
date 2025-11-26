import numpy as np

from example.dataset import load_mnist
from mpneuralnetwork.activations import ReLU
from mpneuralnetwork.layers import BatchNormalization2D, Convolutional, Dense, Flatten, MaxPooling2D
from mpneuralnetwork.losses import CategoricalCrossEntropy
from mpneuralnetwork.model import Model
from mpneuralnetwork.optimizers import Adam

if __name__ == "__main__":
    print("--- Benchmark: 2D CNN (Convolutional + Pooling) ---")
    seed = 69
    np.random.seed(seed)

    (X_train, y_train), _, _ = load_mnist()

    N_SAMPLES = 500
    X_train = X_train[:N_SAMPLES].reshape(-1, 1, 28, 28)
    y_train = y_train[:N_SAMPLES]

    print(f"Training on {X_train.shape[0]} images of shape {X_train.shape[1:]}")

    network = [
        Convolutional(output_depth=16, kernel_size=3, input_shape=(1, 28, 28)),
        BatchNormalization2D(),
        ReLU(),
        MaxPooling2D(pool_size=2, strides=2),
        Convolutional(output_depth=32, kernel_size=3),
        ReLU(),
        Flatten(),
        Dense(128),
        ReLU(),
        Dense(10),
    ]

    model = Model(network, CategoricalCrossEntropy(), Adam(learning_rate=0.005))

    model.train(X_train, y_train, epochs=3, batch_size=32, auto_evaluation=0.0, model_checkpoint=False)
