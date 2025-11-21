import gzip

import dill as pickle  # Utilise dill pour correspondre à create_model.py
import numpy as np


def load_data():
    """Charge le jeu de données MNIST."""
    with gzip.open("data/mnist.pkl.gz", "rb") as f:
        f.seek(0)
        training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
        return (training_data, validation_data, test_data)


_, _, test_data = load_data()

test_inputs = test_data[0]
test_labels = test_data[1]

with open("output/model.pkl", "rb") as f:
    model = pickle.load(f)

correct_predictions = 0

total_samples = len(test_inputs)
for i in range(total_samples):
    prediction_vector = model.predict(test_inputs[i])

    predicted_digit = np.argmax(prediction_vector)

    true_digit = test_labels[i]

    if predicted_digit == true_digit:
        correct_predictions += 1

accuracy = (correct_predictions / total_samples) * 100
print(f"Échantillons de test : {total_samples}")
print(f"Prédictions correctes : {correct_predictions}")
print(f"Précision (Accuracy) : {accuracy:.2f}%")
