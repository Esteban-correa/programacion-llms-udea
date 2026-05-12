import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def detectar_sobreajuste(X, y):

    # Modelo sin límite de profundidad
    model1 = DecisionTreeClassifier(max_depth=None, random_state=42)

    # Modelo limitado
    model2 = DecisionTreeClassifier(max_depth=2, random_state=42)

    # Entrenar modelos
    model1.fit(X, y)
    model2.fit(X, y)

    # Accuracy sobre los mismos datos
    acc1 = accuracy_score(y, model1.predict(X))
    acc2 = accuracy_score(y, model2.predict(X))

    # Detectar sobreajuste
    return (acc1 - acc2) > 0.15