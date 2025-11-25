import gzip
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


def get_file(url, path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, path)


def load_mnist(conv=False):
    base_path = Path("data/mnist")
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        "x_train": ("train-images-idx3-ubyte.gz", 16),
        "y_train": ("train-labels-idx1-ubyte.gz", 8),
        "x_test": ("t10k-images-idx3-ubyte.gz", 16),
        "y_test": ("t10k-labels-idx1-ubyte.gz", 8),
    }

    data = {}
    for key, (filename, offset) in files.items():
        p = base_path / filename
        get_file(base_url + filename, p)
        with gzip.open(p, "rb") as f_in:
            data[key] = np.frombuffer(f_in.read(), dtype=np.uint8, offset=offset)

    if conv:
        X_train = data["x_train"].reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
        X_test = data["x_test"].reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
    else:
        X_train = data["x_train"].reshape(-1, 784).astype(np.float32) / 255.0
        X_test = data["x_test"].reshape(-1, 784).astype(np.float32) / 255.0

    y_train = np.eye(10)[data["y_train"]]
    y_test = np.eye(10)[data["y_test"]]

    return (X_train[:50000], y_train[:50000]), (X_train[50000:], y_train[50000:]), (X_test, y_test)


def load_california():
    base_path = Path("data/california")
    url = "https://download.mlcc.google.com/mledu-datasets/california_housing_"

    get_file(f"{url}train.csv", base_path / "train.csv")
    get_file(f"{url}test.csv", base_path / "test.csv")

    df_tr = pd.read_csv(base_path / "train.csv").sample(frac=1, random_state=69)
    df_te = pd.read_csv(base_path / "test.csv")

    def split(df):
        return df.drop(columns=["median_house_value"]).values, df["median_house_value"].values.reshape(-1, 1) / 100000.0

    X, y = split(df_tr)
    X_te, y_te = split(df_te)

    idx = int(len(X) * 0.8)

    mean = X[:idx].mean(axis=0)
    std = X[:idx].std(axis=0)
    std[std == 0] = 1.0

    X = (X - mean) / std
    X_te = (X_te - mean) / std

    return (X[:idx], y[:idx]), (X[idx:], y[idx:]), (X_te, y_te)
