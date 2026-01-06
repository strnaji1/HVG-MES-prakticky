import numpy as np
import networkx as nx

# Zkusíme importovat powerlaw – pokud není, jen nastavíme flag
try:
    import powerlaw
    HAS_POWERLAW = True
except ImportError:
    HAS_POWERLAW = False


# =========================
#  HVG – pomocné funkce
# =========================

def build_hvg(data):
    """
    Vytvoří Horizontal Visibility Graph (HVG) z časové řady `data`.
    Vrcholy jsou indexy časové řady 0..n-1, hrana (i, j) existuje,
    pokud pro všechny k mezi i a j platí:
        data[k] < data[i] a data[k] < data[j].
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
    se stejnou stupňovou posloupností jako HVG graf G.
    """
    degrees = [d for _, d in G.degree()]
    H_multi = nx.configuration_model(degrees, seed=seed)
    H = nx.Graph(H_multi)          # převedeme na jednoduchý graf
    H.remove_edges_from(nx.selfloop_edges(H))  # odstraníme smyčky
    return H


# =========================
#  Entropie / statistika
# =========================

def shannon_entropy(x, bins="auto"):
    """
    Jednoduchý odhad Shannonovy entropie z histogramu.
    Používá np.histogram s density=True.
    """
    if len(x) == 0:
        return np.nan
    hist, _ = np.histogram(x, bins=bins, density=True)
    hist = hist[hist > 0]
    if len(hist) == 0:
        return np.nan
    return -np.sum(hist * np.log2(hist))


# =========================
#  Small-world + metriky grafu
# =========================

def compute_graph_basic_metrics(G):
    """
    Spočítá základní metriky grafu G a vrátí je jako slovník:
        - n_nodes, n_edges
        - degrees (list)
        - avg_deg
        - C (average clustering)
        - is_conn (bool)
        - L (average shortest path length nebo None)
        - diameter (průměr grafu nebo None)
        - assort (degree assortativity nebo None)
    """
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    degrees = [d for _, d in G.degree()]
    avg_deg = float(np.mean(degrees)) if len(degrees) > 0 else 0.0

    # Clustering
    try:
        C = nx.average_clustering(G)
    except Exception:
        C = float("nan")

    # Souvislost, délky cest, průměr
    is_conn = nx.is_connected(G) if n_nodes > 0 else False
    L = None
    diameter = None
    if is_conn and n_nodes > 1:
        try:
            L = nx.average_shortest_path_length(G)
        except Exception:
            L = None
        try:
            diameter = nx.diameter(G)
        except Exception:
            diameter = None

    # Assortativita stupňů
    try:
        assort = nx.degree_assortativity_coefficient(G)
    except Exception:
        assort = None

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "degrees": degrees,
        "avg_deg": avg_deg,
        "C": C,
        "is_conn": is_conn,
        "L": L,
        "diameter": diameter,
        "assort": assort,
    }


def estimate_er_random_graph_metrics(n_nodes, avg_deg):
    """
    Odhad teoretických metrik pro náhodný graf G(N, p) se
    stejným průměrným stupněm avg_deg:
        L_rand ≈ log(N) / log(k)
        C_rand ≈ k / N
    Vrací (L_rand, C_rand), případně (None, None).
    """
    if n_nodes <= 1 or avg_deg <= 1:
        return None, None
    try:
        L_rand = np.log(n_nodes) / np.log(avg_deg)
        C_rand = avg_deg / n_nodes
        return L_rand, C_rand
    except Exception:
        return None, None


class SmallWorldAnalyzer:
    """
    Pomocná třída pro výpočet a interpretaci small-world indexu σ.
    Teoretická hranice: σ > 1 => small-world.

    σ = (C / C_rand) / (L / L_rand)
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
        Vrátí dvojici (typ, zpráva) podle hodnoty σ:
        - typ: 'success'  -> small-world
               'info'     -> podobné náhodnému grafu (σ ≈ 1)
               'warning'  -> není small-world
        - zpráva: lidská interpretace
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


# =========================
#  Volitelný wrapper na power-law fit
# =========================

def fit_powerlaw_to_degrees(degrees, discrete=True, xmin=None):
    """
    Pomocná funkce pro power-law fit stupňového rozdělení.
    Používá balík `powerlaw`, pokud je k dispozici.

    Parameters
    ----------
    degrees : sekvence stupňů (např. list[int])
    discrete : bool – použít discrete=True ve fitu
    xmin : volitelné k_min, pokud chceme zadat ručně (jinak se odhadne)

    Returns
    -------
    dict nebo None:
        {
          "alpha": ...,
          "xmin": ...,
          "R": ...,
          "p": ...,
          "fit": powerlaw.Fit objekt
        }
    nebo None pokud:
        - powerlaw není nainstalován
        - dat je málo
        - dojde k chybě
    """
    if not HAS_POWERLAW:
        return None

    # filtrujeme jen stupně >= 1
    degs = np.array([d for d in degrees if d > 0])
    if len(degs) < 10:
        return None

    try:
        fit_kwargs = dict(discrete=discrete, verbose=False)
        if xmin is not None:
            fit_kwargs["xmin"] = xmin

        fit = powerlaw.Fit(degs, **fit_kwargs)
        alpha = fit.power_law.alpha
        xmin_est = fit.power_law.xmin

        # porovnání power-law vs. exponenciální rozdělení
        R, p = fit.distribution_compare("power_law", "exponential")

        return {
            "alpha": alpha,
            "xmin": xmin_est,
            "R": R,
            "p": p,
            "fit": fit,
        }
    except Exception:
        return None
