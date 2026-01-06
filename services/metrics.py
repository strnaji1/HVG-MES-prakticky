# services/metrics.py

import numpy as np
import networkx as nx


class SmallWorldAnalyzer:
    """
    Pomocná třída pro výpočet a interpretaci small-world indexu σ.
    Teoretická hranice: σ > 1 => small-world.
    """
    def __init__(self, C, L, C_rand, L_rand):
        self.C = C
        self.L = L
        self.C_rand = C_rand
        self.L_rand = L_rand
        self.sigma = self._compute_sigma()

    def _compute_sigma(self):
        if (
            self.C is None or self.L is None or
            self.C_rand in (None, 0) or
            self.L_rand is None
        ):
            return None
        try:
            return (self.C / self.C_rand) / (self.L / self.L_rand)
        except Exception:
            return None

    def interpretation(self, atol=0.05):
        """
        Vrátí (typ, zpráva) podle hodnoty σ:
        - 'success'  -> small-world
        - 'info'     -> podobné náhodnému grafu (σ ≈ 1)
        - 'warning'  -> není small-world
        """
        if self.sigma is None or np.isnan(self.sigma):
            return (
                "info",
                "Small-world index σ nelze spolehlivě spočítat "
                "(chybí některá z metrik nebo došlo k numerické chybě)."
            )

        s = self.sigma
        if s > 1 + atol:
            return (
                "success",
                "Síť má **small-world vlastnosti** "
                "(σ > 1 – vyšší clustering než náhodný graf při podobné délce cest)."
            )
        elif abs(s - 1.0) <= atol:
            return (
                "info",
                "Síť je **velmi podobná náhodnému grafu** "
                "(σ ≈ 1 – žádné výrazné small-world chování)."
            )
        else:
            return (
                "warning",
                "Síť **pravděpodobně není small-world** "
                "(σ < 1 – kombinace clusteringu a délky cest neodpovídá small-world síti)."
            )


def _compute_random_graph_baseline(n_nodes: int, avg_deg: float):
    """
    Jednoduchý ER-like odhad pro náhodný graf:
    L_rand ≈ log(N) / log(k)
    C_rand ≈ k / N
    Vrací (L_rand, C_rand) nebo (None, None), pokud nelze spočítat.
    """
    if n_nodes <= 1 or avg_deg <= 1:
        return None, None
    try:
        L_rand = np.log(n_nodes) / np.log(avg_deg)
        C_rand = avg_deg / n_nodes
        return float(L_rand), float(C_rand)
    except Exception:
        return None, None


def compute_basic_metrics(G: nx.Graph):
    """
    Spočítá základní metriky pro graf G a vrátí je jako dict.

    Klíče:
      - n_nodes
      - n_edges
      - degrees
      - avg_degree
      - C           (average clustering)
      - is_connected
      - L           (average shortest path length nebo None)
      - diameter    (graph diameter nebo None)
      - assortativity
      - L_rand, C_rand
      - sigma       (small-world index podle SmallWorldAnalyzer)
      - analyzer    (instance SmallWorldAnalyzer pro případnou interpretaci)
    """
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    degrees = [d for _, d in G.degree()]
    avg_degree = float(np.mean(degrees)) if len(degrees) > 0 else 0.0

    # Clustering
    try:
        C = nx.average_clustering(G)
    except Exception:
        C = float("nan")

    # Souvislost, průměrná délka cesty, průměr grafu
    is_conn = nx.is_connected(G) if n_nodes > 0 else False
    L = None
    diam = None
    if is_conn and n_nodes > 1:
        try:
            L = nx.average_shortest_path_length(G)
        except Exception:
            L = None
        try:
            diam = nx.diameter(G)
        except Exception:
            diam = None

    # Assortativita stupňů
    try:
        assort = nx.degree_assortativity_coefficient(G)
    except Exception:
        assort = None

    # Náhodný graf – baseline
    L_rand, C_rand = _compute_random_graph_baseline(n_nodes, avg_degree)

    # Small-world index
    analyzer = SmallWorldAnalyzer(C, L, C_rand, L_rand)
    sigma = analyzer.sigma

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "degrees": degrees,
        "avg_degree": avg_degree,
        "C": C,
        "is_connected": is_conn,
        "L": L,
        "diameter": diam,
        "assortativity": assort,
        "L_rand": L_rand,
        "C_rand": C_rand,
        "sigma": sigma,
        "analyzer": analyzer,
    }
