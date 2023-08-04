import itertools
from tqdm.notebook import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import igraph as ig
import community
from unidecode import unidecode


def explode_df(df : pd.DataFrame, col1 : str, col2 : str) -> pd.DataFrame:
    """
    Explode the non-atomic entries in col1 and col2 of df.
    """
    return (
        df[[col1, col2]]
        .dropna()
        .applymap(lambda x: x.split(';'))
        .explode(col1, ignore_index=False)
        .explode(col2, ignore_index=False)
    )