# services/hvg_graph.py

import networkx as nx
import numpy as np


def build_hvg(data):
    """
    Vytvoří Horizontal Visibility Graph (HVG) z časové řady `data`.
    Vrcholy jsou indexy časové řady (0, 1, ..., n-1).

    Podmínka viditelnosti:
    Mezi i < j existuje hrana, pokud pro každý k v (i, j):
        data[k] < data[i] a data[k] < data[j]
    """
    G = nx.Graph()
    n = len(data)
    G.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            if all(data[k] < data[i] and data[k] < data[j] for k in range(i + 1, j)):
                G.add_edge(i, j)

    return G


def build_configuration_graph_from_hvg(G, seed=42):
    """
    Vytvoří jednoduchý konfigurační graf (null model)
    se stejnou stupňovou posloupností jako HVG graf `G`.

    Používá NetworkX `configuration_model`, převede na obyčejný graf
    a odstraní případné self-loops.
    """
    degrees = [d for _, d in G.degree()]

    # multigraf s danou stupňovou posloupností
    H_multi = nx.configuration_model(degrees, seed=seed)

    # převedeme na obyčejný graf (sloučení paralelních hran)
    H = nx.Graph(H_multi)

    # odstraníme smyčky (self-loops)
    H.remove_edges_from(nx.selfloop_edges(H))

    return H
