import pandas as pd
import numpy as np


def procesar_especificaciones(df):

    df_out = df.copy()

    # Extraer color y talla
    df_out['color'] = df_out['specs'].str.extract(r'Color:\s*([A-Za-z]+)')
    df_out['talla'] = df_out['specs'].str.extract(r'Talla:\s*([A-Za-z]+)')

    # Eliminar columna original
    df_out = df_out.drop(columns=['specs'])

    # Eliminar filas con nulos
    df_out = df_out.dropna(subset=['color', 'talla'])

    # Crear tabla dinámica
    if not df_out.empty:

        output_expected = pd.pivot_table(
            df_out,
            index='color',
            columns='talla',
            aggfunc='size',
            fill_value=0
        )

    else:
        output_expected = pd.DataFrame()

    return output_expected