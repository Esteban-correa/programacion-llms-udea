import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import VarianceThreshold

def generar_caso_de_uso_limpiar_y_codificar():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función limpiar_y_codificar.
    """

    # ---------------------------------------------------------
    # 1. Dimensiones aleatorias
    # ---------------------------------------------------------
    n_rows = random.randint(10, 50)
    n_cat = random.randint(1, 3)
    n_num = random.randint(1, 3)

    # ---------------------------------------------------------
    # 2. Crear columnas categóricas
    # ---------------------------------------------------------
    categorias = ['A', 'B', 'C']
    data_cat = {
        f"cat_{i}": np.random.choice(categorias, size=n_rows)
        for i in range(n_cat)
    }

    # ---------------------------------------------------------
    # 3. Crear columnas numéricas (incluyendo algunas constantes)
    # ---------------------------------------------------------
    data_num = {}
    for i in range(n_num):
        if random.random() < 0.3:
            # columna constante (varianza 0)
            valor = random.randint(1, 5)
            data_num[f"num_{i}"] = np.full(n_rows, valor)
        else:
            data_num[f"num_{i}"] = np.random.randn(n_rows)

    # ---------------------------------------------------------
    # 4. Crear DataFrame
    # ---------------------------------------------------------
    df = pd.DataFrame({**data_cat, **data_num})

    # ---------------------------------------------------------
    # 5. INPUT
    # ---------------------------------------------------------
    input_data = {
        "df": df.copy()
    }

    # ---------------------------------------------------------
    # 6. OUTPUT esperado (ground truth)
    # ---------------------------------------------------------

    # A. Separar columnas categóricas y numéricas
    cat_cols = df.select_dtypes(include=['object']).columns
    num_cols = df.select_dtypes(exclude=['object']).columns

    # B. OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    if len(cat_cols) > 0:
        X_cat = encoder.fit_transform(df[cat_cols])
    else:
        X_cat = np.empty((n_rows, 0))

    # C. Numéricas
    X_num = df[num_cols].to_numpy()

    # D. Concatenar
    X = np.hstack([X_cat, X_num])

    # E. VarianceThreshold
    selector = VarianceThreshold(threshold=0.0)
    X_filtered = selector.fit_transform(X)

    output_data = X_filtered

    # ---------------------------------------------------------
    return input_data, output_data


# ---------------------------------------------------------
# Ejemplo de prueba
# ---------------------------------------------------------
if __name__ == "__main__":
    entrada, salida = generar_caso_de_uso_limpiar_y_codificar()

    print("=== INPUT ===")
    print("DataFrame shape:", entrada["df"].shape)
    print(entrada["df"].head())

    print("\n=== OUTPUT ===")
    print("Shape resultado:", salida.shape)
    print("Primeras filas:\n", salida[:5])