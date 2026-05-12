import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer


def transformar_cuantiles(df):

    transformer = QuantileTransformer(output_distribution="uniform")

    X_transformed = transformer.fit_transform(df)

    return X_transformed