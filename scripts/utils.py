import itertools
from tqdm.notebook import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import igraph as ig
import community
from unidecode import unidecode
from matplotlib import patches
from matplotlib.colors import ListedColormap
from wordcloud import WordCloud


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

def get_graph_stats(G, directed=False):
    """
    Calculate the network statistics of a networkx graph.
    """
    N = G.order()
    L = G.size()
    density = (2 * L) / (N * (N - 1)) * 100
    stat_dict = {}
    if G.name != '':
        print(G.name)
    print(f'Total number of nodes: {N}')
    print(f'Total number of edges: {L}')
    print()
    print('Minimum degree: {}'.format(min(dict(G.degree).values())))
    print('Maximum degree: {}'.format(max(dict(G.degree).values())))
    print(f'Average degree: {L/N:.2f}')
    print()
    if directed:
        density /= 2
        print('Minimum in degree: {}'.format(min(dict(G.in_degree).values())))
        print('Maximum in degree: {}'.format(max(dict(G.in_degree).values())))
        print('Average in degree: {:.2f}'.format(
            np.mean(list(dict(G.in_degree).values()))
        ))
        print()
        print('Minimum out degree: {}'.format(
            min(dict(G.out_degree).values())
        ))
        print('Maximum out degree: {}'.format(
            max(dict(G.out_degree).values())
        ))
        print('Average out degree: {:.2f}'.format(
            np.mean(list(dict(G.out_degree).values()))
        ))
        print()
    print('Network density: {:.2f}%'.format(density))

def get_central_nodes_as_series(G : nx.Graph,
                                centrality_fn : callable,
                                n : int=10) -> pd.Series:
    """
    Calculate centrality measures for nodes in G using centrality_fn, and
    return the top n nodes and their score.
    """
    return pd.Series(centrality_fn(G)).sort_values(ascending=False).head(n)