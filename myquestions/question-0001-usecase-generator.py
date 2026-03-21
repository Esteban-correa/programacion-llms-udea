import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import random

def generar_caso_de_uso_detectar_anomalias():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función detectar_anomalias.
    """

    # ---------------------------------------------------------
    # 1. Generar dimensiones aleatorias
    # ---------------------------------------------------------
    n_rows = random.randint(20, 100)
    n_features = random.randint(2, 5)

    # ---------------------------------------------------------
    # 2. Crear datos normales
    # ---------------------------------------------------------
    data = np.random.randn(n_rows, n_features)

    # Introducir anomalías (valores extremos)
    n_outliers = max(1, int(0.1 * n_rows))
    indices = np.random.choice(n_rows, n_outliers, replace=False)

    data[indices] += np.random.uniform(8, 15)  # valores extremos

    # Crear DataFrame
    columns = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(data, columns=columns)

    # ---------------------------------------------------------
    # 3. Parámetro contamination aleatorio
    # ---------------------------------------------------------
    contamination = round(random.uniform(0.01, 0.2), 2)

    # ---------------------------------------------------------
    # 4. INPUT
    # ---------------------------------------------------------
    input_data = {
        "df": df.copy(),
        "contaminacion": contamination
    }

    # ---------------------------------------------------------
    # 5. OUTPUT esperado (ground truth)
    # ---------------------------------------------------------
    modelo = IsolationForest(contamination=contamination, random_state=42)
    modelo.fit(df)
    predicciones = modelo.predict(df)

    output_data = predicciones

    # ---------------------------------------------------------
    return input_data, output_data


# ---------------------------------------------------------
# Ejemplo de prueba
# ---------------------------------------------------------
if __name__ == "__main__":
    entrada, salida = generar_caso_de_uso_detectar_anomalias()

    print("=== INPUT ===")
    print("Contaminación:", entrada["contaminacion"])
    print("DataFrame shape:", entrada["df"].shape)

    print("\n=== OUTPUT ===")
    print("Predicciones:", salida[:10])