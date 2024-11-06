#Source: https://stackoverflow.com/questions/58558412/typeerror-type-tuple-cannot-be-instantiated-use-tuple-instead
import pandas as pd
import numpy as np
from typing import Tuple

def split_data(self, df: pd.DataFrame, split_quantile: float) -> Tuple(pd.DataFrame, pd.DataFrame):
    '''Split data sets into two parts - train and test data sets.'''
    df = df.sort_values(by='datein').reset_index(drop=True)
    quantile = int(np.quantile(df.index, split_quantile))
    return (
        df[df.index <= quantile].reset_index(drop=True),
        df[df.index > quantile].reset_index(drop=True)
    )