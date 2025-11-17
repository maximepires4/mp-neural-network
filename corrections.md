# Corrections pour la bibliothèque de Deep Learning

Voici les corrections à apporter à votre projet.

## 1. Correction de la fonction de perte `CrossEntropy`

Le fichier `src/mpneuralnetwork/losses.py` contient une implémentation incorrecte de la fonction de perte `CrossEntropy` et de sa dérivée.

**Erreurs actuelles :**

1.  La méthode `direct` (passe avant) utilise `np.mean` et ne somme pas correctement les termes de l'entropie croisée.
2.  La méthode `prime` (passe arrière) a une formule incorrecte. La formule correcte pour la dérivée de la `CrossEntropy` avec une couche `Softmax` en sortie est `prediction - attendu` (`output - output_expected`).

**Fichier à modifier :** `src/mpneuralnetwork/losses.py`

**Code à remplacer :**

```python
class CrossEntropy(Loss):
    def direct(self, output, output_expected):
        epsilon = 1e-9
        return np.mean(-output_expected * np.log(output + epsilon))

    def prime(self, output, output_expected):
        return (output_expected - output) / output.size
```

**Nouveau code :**

```python
class CrossEntropy(Loss):
    def direct(self, output, output_expected):
        epsilon = 1e-9
        return -np.sum(output_expected * np.log(output + epsilon))

    def prime(self, output, output_expected):
        return output - output_expected
```

## 2. Correction de la fonction d'activation `PReLU`

La fonction d'activation `PReLU` dans `src/mpneuralnetwork/activations.py` a une dérivée incorrecte qui ne fonctionne pas avec les tableaux NumPy.

**Erreur actuelle :**

La condition `if x < 0` dans l'expression lambda ne peut pas s'appliquer sur un tableau NumPy entier.

**Fichier à modifier :** `src/mpneuralnetwork/activations.py`

**Code à remplacer :**

```python
class PReLU(Activation):
    def __init__(self, alpha=0.01):
        super().__init__(
            lambda x: np.maximum(alpha * x, x), lambda x: alpha if x < 0 else 1
        )
```

**Nouveau code :**

```python
class PReLU(Activation):
    def __init__(self, alpha=0.01):
        super().__init__(
            lambda x: np.maximum(alpha * x, x), lambda x: np.where(x < 0, alpha, 1)
        )
```

## 3. Vectorisation (Changement majeur d'architecture)

Le problème le plus important de votre bibliothèque est l'absence de **vectorisation**. Actuellement, vous traitez les données échantillon par échantillon dans une boucle, ce qui est extrêmement lent. Les bibliothèques comme NumPy sont optimisées pour effectuer des opérations sur des matrices entières (des "batchs" de données) en une seule fois.

L'objectif est de modifier les passes `forward` et `backward` pour qu'elles traitent un `(batch_size, n_features)` au lieu d'un seul `(n_features,)`.

### 3.1. Modifier la boucle d'entraînement (`model.py`)

La boucle `for x, y in zip(...)` doit être supprimée. Vous devez passer le batch entier de données à travers les couches.

**Fichier à modifier :** `src/mpneuralnetwork/model.py`

**Logique actuelle (à changer) :**

```python
# ...
for x, y in zip(
    input_copy[batch * batch_size : (batch + 1) * batch_size],
    output_copy[batch * batch_size : (batch + 1) * batch_size],
):
    y_hat = x
    for layer in self.layers:
        y_hat = layer.forward(y_hat)
    # ... puis backward pour un seul échantillon
# ...
```

**Nouvelle logique (principe) :**

```python
# ...
# Récupérer le batch entier
batch_X = input_copy[batch * batch_size : (batch + 1) * batch_size]
batch_Y = output_copy[batch * batch_size : (batch + 1) * batch_size]

# Passe avant (Forward) sur tout le batch
y_hat = batch_X
for layer in self.layers:
    y_hat = layer.forward(y_hat)

# Calcul de l'erreur et du gradient sur tout le batch
error += self.loss.direct(batch_Y, y_hat)
grad = self.loss.prime(batch_Y, y_hat)

# Passe arrière (Backward) sur tout le batch
for layer in reversed(self.layers):
    grad = layer.backward(grad)
# ...
```

### 3.2. Modifier les couches (`layers.py`)

Les couches doivent être adaptées pour gérer des entrées qui sont des matrices.

**Fichier à modifier :** `src/mpneuralnetwork/layers.py`

#### Exemple pour la couche `Dense` :

La forme des entrées, des poids et des biais doit être cohérente. La convention est souvent d'avoir des entrées de forme `(batch_size, input_size)`.

**Modification de `forward` :**

Le produit matriciel doit être dans le bon ordre.

```python
# forward dans la classe Dense
def forward(self, input):
    self.input = input # Forme: (batch_size, input_size)
    # La transposée des poids est utilisée si les poids sont (output_size, input_size)
    output = self.input @ self.weights.T + self.biases 
    return output
```

**Modification de `backward` :**

Les gradients doivent aussi être calculés sur le batch.

```python
# backward dans la classe Dense
def backward(self, output_gradient): # Forme: (batch_size, output_size)
    # Gradient par rapport aux poids
    self.weights_gradient += self.input.T @ output_gradient
    # Gradient par rapport aux biais (sommer sur le batch)
    self.output_gradient += np.sum(output_gradient, axis=0, keepdims=True)

    # Gradient à propager à la couche précédente
    return output_gradient @ self.weights
```

Ces changements sont fondamentaux et doivent être appliqués à toutes les couches et fonctions d'activation pour que la bibliothèque soit performante.

## 4. Ajout de Tests Unitaires

Les erreurs de calcul dans les fonctions de perte et d'activation montrent la nécessité d'une suite de tests. Les tests unitaires sont cruciaux pour garantir que les opérations mathématiques de votre bibliothèque sont correctes.

### 4.1. Mettre en place un Framework de Test

Je vous recommande d'utiliser `pytest`, un framework de test populaire en Python.

1.  **Installez pytest :**
    ```bash
    pip install pytest
    ```
2.  **Créez un répertoire `tests/`** à la racine de votre projet.
3.  **Écrivez vos tests** dans des fichiers nommés `test_*.py`.

### 4.2. Exemple de Test simple

Vous pouvez commencer par des tests simples, comme vérifier les dimensions des sorties de vos couches.

**Fichier : `tests/test_layers.py`**
```python
import numpy as np
from mpneuralnetwork.layers import Dense

def test_dense_forward_shape():
    # Arrange
    batch_size = 10
    input_size = 5
    output_size = 3
    layer = Dense(input_size, output_size)
    input_data = np.random.randn(batch_size, input_size)

    # Act
    output = layer.forward(input_data)

    # Assert
    assert output.shape == (batch_size, output_size)
```

### 4.3. Vérification du Gradient (Gradient Checking)

C'est la technique la plus importante pour valider votre passe `backward`. L'idée est de comparer le gradient que vous avez calculé analytiquement (dans votre code `backward`) avec un gradient calculé numériquement.

Le gradient numérique est une approximation calculée en utilisant la définition de la dérivée :
`f'(x) ≈ (f(x + ε) - f(x - ε)) / (2 * ε)` où `ε` est une très petite valeur.

Vous pouvez écrire une fonction qui calcule ce gradient numérique pour chaque poids de votre réseau et le compare au gradient calculé par votre passe `backward`. S'ils sont très proches, votre implémentation de la backpropagation est probablement correcte. C'est une technique essentielle pour débugger les réseaux de neurones.
