# **Guide de Qualité du Code**

Ce document résume les outils utilisés dans mp-neural-network pour maintenir un code propre, sûr et robuste.

## **Périmètre d'Action**

| Outil | Périmètre | Pourquoi ? |
| :---- | :---- | :---- |
| **Ruff** | **Tout le projet** | Le style et les imports doivent être propres partout (tests inclus). |
| **Mypy** | **src/** | Le typage strict est vital pour la librairie, mais optionnel pour les scripts de tests. |
| **Coverage** | **src/** | On veut savoir si la librairie est bien testée, pas si les tests se testent eux-mêmes. |

## **Commandes Manuelles**

### **1. Nettoyer le code (Ruff)**

```
# Agit sur src, tests, examples
ruff check . --fix
ruff format .
```

### **2. Vérifier les Types (Mypy)**

```
# Grâce à la config pyproject.toml, il cible automatiquement src/mpneuralnetwork
mypy
```

### **4. Lancer les Tests (Coverage)**

```
# Coverage est configuré pour ignorer les fichiers de tests dans le rapport
coverage run -m pytest
coverage report -m
```

## **Installation**

```
pip install -e .[dev]
pre-commit install
```
