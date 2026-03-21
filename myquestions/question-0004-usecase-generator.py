import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer

def generar_caso_de_uso_preparar_texto_soporte():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función preparar_texto_soporte.
    """

    # ---------------------------------------------------------
    # 1. Generar datos de texto
    # ---------------------------------------------------------
    vocabulario = [
        "error", "sistema", "falla", "usuario", "conexion",
        "servidor", "red", "acceso", "lento", "respuesta",
        "soporte", "ticket", "problema", "datos", "actualizar"
    ]

    n_rows = random.randint(5, 20)

    textos = []
    for _ in range(n_rows):
        longitud = random.randint(3, 10)
        frase = " ".join(random.choices(vocabulario, k=longitud))
        textos.append(frase)

    col_texto = "mensaje"
    df = pd.DataFrame({col_texto: textos})

    # ---------------------------------------------------------
    # 2. Parámetro max_palabras
    # ---------------------------------------------------------
    max_palabras = random.randint(5, len(vocabulario))

    # ---------------------------------------------------------
    # 3. INPUT
    # ---------------------------------------------------------
    input_data = {
        "df": df.copy(),
        "col_texto": col_texto,
        "max_palabras": max_palabras
    }

    # ---------------------------------------------------------
    # 4. OUTPUT esperado (ground truth)
    # ---------------------------------------------------------

    # A. Convertir a minúsculas
    texto_procesado = df[col_texto].str.lower()

    # B. CountVectorizer
    vectorizer = CountVectorizer(max_features=max_palabras)
    X_sparse = vectorizer.fit_transform(texto_procesado)

    # C. Convertir a denso
    X_dense = X_sparse.toarray()

    output_data = X_dense

    # ---------------------------------------------------------
    return input_data, output_data


# ---------------------------------------------------------
# Ejemplo de prueba
# ---------------------------------------------------------
if __name__ == "__main__":
    entrada, salida = generar_caso_de_uso_preparar_texto_soporte()

    print("=== INPUT ===")
    print("Columna texto:", entrada["col_texto"])
    print("Max palabras:", entrada["max_palabras"])
    print(entrada["df"].head())

    print("\n=== OUTPUT ===")
    print("Shape:", salida.shape)
    print("Primeras filas:\n", salida[:5])