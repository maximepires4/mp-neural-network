import dill as pickle
from pathlib import Path

import numpy as np

class NeuralNetHandler():

    def __init__(self, path: Path = Path("output/model.pkl")):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
    
    def predict(self, image):
        img = np.array(image).reshape((784,1))
        return self.model.predict(img)