import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression


def entrenar_con_seleccion(X, y, k_features):

    # Seleccionar características
    selector = SelectKBest(score_func=f_regression, k=k_features)
    X_selected = selector.fit_transform(X, y)

    # Entrenar modelo
    modelo = LinearRegression()
    modelo.fit(X_selected, y)

    # Retornar selector y modelo entrenados
    return selector, modelo