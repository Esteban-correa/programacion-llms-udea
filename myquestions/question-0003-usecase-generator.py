import numpy as np
import random
from sklearn.cluster import FeatureAgglomeration

def generar_caso_de_uso_comprimir_sensores_correlacionados():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función comprimir_sensores_correlacionados.
    """

    # ---------------------------------------------------------
    # 1. Dimensiones aleatorias
    # ---------------------------------------------------------
    n_samples = random.randint(20, 100)
    n_features = random.randint(4, 10)

    # n_clusters < n_features (obligatorio)
    n_clusters = random.randint(2, n_features - 1)

    # ---------------------------------------------------------
    # 2. Generar datos correlacionados
    # ---------------------------------------------------------
    # Base latente
    base = np.random.randn(n_samples, 1)

    # Crear features correlacionadas
    X = np.hstack([
        base + np.random.normal(0, 0.1, size=(n_samples, 1))
        for _ in range(n_features)
    ])

    # ---------------------------------------------------------
    # 3. INPUT
    # ---------------------------------------------------------
    input_data = {
        "X": X.copy(),
        "n_clusters": n_clusters
    }

    # ---------------------------------------------------------
    # 4. OUTPUT esperado (ground truth)
    # ---------------------------------------------------------
    modelo = FeatureAgglomeration(n_clusters=n_clusters)
    modelo.fit(X)
    X_transformado = modelo.transform(X)

    output_data = X_transformado

    # ---------------------------------------------------------
    return input_data, output_data


# ---------------------------------------------------------
# Ejemplo de prueba
# ---------------------------------------------------------
if __name__ == "__main__":
    entrada, salida = generar_caso_de_uso_comprimir_sensores_correlacionados()

    print("=== INPUT ===")
    print("Shape X:", entrada["X"].shape)
    print("n_clusters:", entrada["n_clusters"])

    print("\n=== OUTPUT ===")
    print("Shape transformado:", salida.shape)
    print("Primeras filas:\n", salida[:5])