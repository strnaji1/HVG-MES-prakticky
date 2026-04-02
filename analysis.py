import numpy as np
import networkx as nx
from app import compute_graph_metrics
from services.metrics import compute_basic_metrics, SmallWorldAnalyzer
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


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

def compute_configuration_model_metrics(G, seed=42):
    G_conf = build_configuration_graph_from_hvg(G, seed=seed)
    metrics = compute_graph_metrics(G_conf)
    return G_conf, metrics

def compute_graph_layout(G, layout_type="spring", seed=42):
    if layout_type == "planar":
        try:
            is_planar, _ = nx.check_planarity(G)
            if is_planar:
                return nx.planar_layout(G)
        except Exception:
            pass

    return nx.spring_layout(G, seed=seed)

def unpack_graph_metrics(metrics):
    return (
        metrics["n_nodes"],
        metrics["n_edges"],
        metrics["degrees"],
        metrics["avg_deg"],
        metrics["C"],
        metrics["is_conn"],
        metrics["L"],
        metrics["diam"],
        metrics["assort"],
        metrics["L_rand"],
        metrics["C_rand"],
        metrics["sigma"],
    )
    


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

def prepare_network_traces(
    G,
    pos,
    node_color="skyblue",
    node_size=10,
    edge_color="#888",
    edge_width=1,
    show_labels=False,
    hover_texts=None,
    text_color="black",
):
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=edge_width, color=edge_color),
        hoverinfo="none",
    )

    node_x, node_y = [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    if show_labels:
        node_mode = "markers+text"
        node_text = [str(n) for n in G.nodes()]
        text_position = "bottom center"
    else:
        node_mode = "markers"
        node_text = None
        text_position = None

    if hover_texts is None:
        hover_texts = [f"Vrchol: {n}" for n in G.nodes()]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode=node_mode,
        text=node_text,
        textposition=text_position,
        hoverinfo="text",
        hovertext=hover_texts,
        marker=dict(size=node_size, color=node_color, line_width=1),
        textfont=dict(size=10, color=text_color),
    )

    return edge_trace, node_trace
def compute_degree_distribution_metrics(degrees):
    degrees = list(degrees)

    if len(degrees) == 0:
        return {
            "degrees": [],
            "unique_deg": np.array([]),
            "counts": np.array([]),
            "pk": np.array([]),
            "entropy_deg": 0.0,
            "entropy_deg_norm": 0.0,
            "mean_degree": 0.0,
            "median_degree": 0.0,
            "max_degree": 0,
            "min_degree": 0,
            "degree_range": 0,
            "peak_degree": None,
            "cdf": np.array([]),
        }

    unique_deg, counts = np.unique(degrees, return_counts=True)
    pk = counts / counts.sum()

    entropy_deg = -np.sum(pk * np.log(pk)) if len(pk) > 0 else 0.0

    if len(unique_deg) > 1:
        entropy_deg_norm = entropy_deg / np.log(len(unique_deg))
    else:
        entropy_deg_norm = 0.0

    cdf = np.cumsum(pk)

    peak_degree = unique_deg[np.argmax(pk)] if len(pk) > 0 else None

    return {
        "degrees": degrees,
        "unique_deg": unique_deg,
        "counts": counts,
        "pk": pk,
        "entropy_deg": float(entropy_deg),
        "entropy_deg_norm": float(entropy_deg_norm),
        "mean_degree": float(np.mean(degrees)),
        "median_degree": float(np.median(degrees)),
        "max_degree": int(np.max(degrees)),
        "min_degree": int(np.min(degrees)),
        "degree_range": int(np.max(degrees) - np.min(degrees)),
        "peak_degree": int(peak_degree) if peak_degree is not None else None,
        "cdf": cdf,
    }


def classify_entropy_level(entropy_deg_norm):
    if entropy_deg_norm < 0.2:
        return (
            "velmi nízká",
            "Vrcholy mají velmi podobné stupně a stupňové rozdělení je silně koncentrované. "
            "Síť působí velmi uspořádaně a strukturálně omezeně."
        )
    elif entropy_deg_norm < 0.4:
        return (
            "nízká",
            "Vrcholy mají spíše podobné stupně a rozdělení není příliš rozptýlené. "
            "Síť vykazuje výraznější pravidelnost a nižší variabilitu."
        )
    elif entropy_deg_norm < 0.6:
        return (
            "střední",
            "Stupňové rozdělení je středně rozptýlené. "
            "Síť kombinuje určitou pravidelnost i variabilitu."
        )
    elif entropy_deg_norm < 0.8:
        return (
            "vysoká",
            "Vrcholy mají rozmanitější stupně a stupňové rozdělení je výrazněji rozptýlené. "
            "Síť působí komplexněji a méně pravidelně."
        )
    else:
        return (
            "velmi vysoká",
            "Vrcholy mají velmi různorodé stupně a rozdělení je silně rozptýlené. "
            "Síť vykazuje vysokou variabilitu a vysokou míru strukturální různorodosti."
        )

def create_degree_histogram_figure(degrees, title="Histogram stupňů"):
    df_deg = pd.DataFrame({"degree": degrees})

    max_degree = max(degrees) if len(degrees) > 0 else 1

    fig = px.histogram(
        df_deg,
        x="degree",
        nbins=max_degree + 1,
        title=title,
        labels={"degree": "Stupeň"},
        opacity=0.7,
    )
    fig.update_layout(yaxis_title="Počet vrcholů")
    return fig


def create_degree_pdf_figure(unique_deg, pk, title="PDF stupňového rozdělení P(k)"):
    df_pdf = pd.DataFrame({"degree": unique_deg, "pk": pk})

    fig = px.line(
        df_pdf,
        x="degree",
        y="pk",
        markers=True,
        title=title,
        labels={"degree": "Stupeň k", "pk": "P(k)"},
    )

    fig.update_layout(
        xaxis_title="Stupeň k",
        yaxis_title="Pravděpodobnost P(k)",
    )
    return fig


def create_degree_cdf_figure(unique_deg, cdf, title="CDF stupňového rozdělení F(k)"):
    df_cdf = pd.DataFrame({"degree": unique_deg, "cdf": cdf})

    fig = px.line(
        df_cdf,
        x="degree",
        y="cdf",
        markers=True,
        title=title,
        labels={"degree": "Stupeň k", "cdf": "F(k) = P(K ≤ k)"},
    )

    fig.update_layout(
        xaxis_title="Stupeň k",
        yaxis_title="Kumulativní pravděpodobnost F(k)",
        yaxis_range=[0, 1.05],
    )
    return fig


def create_arc_diagram_figure(G, values, title="Arc Diagram HVG", node_color="skyblue"):
    n = len(values)
    node_x_line = np.arange(n)
    node_y_line = np.zeros(n)

    fig = go.Figure()

    for i, j in G.edges():
        r = (j - i) / 2
        mid = i + r
        theta = np.linspace(0, np.pi, 100)
        x_arc = mid + r * np.cos(theta)
        y_arc = r * np.sin(theta)

        fig.add_trace(
            go.Scatter(
                x=x_arc,
                y=y_arc,
                mode="lines",
                line=dict(color="gray", width=1),
                hoverinfo="none",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=node_x_line,
            y=node_y_line,
            mode="markers",
            marker=dict(size=8, color=node_color),
            hoverinfo="text",
            hovertext=[
                f"Index: {i}<br>Hodnota: {values[i]:.3f}"
                for i in node_x_line
            ],
        )
    )

    fig.update_layout(
        title=title,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, title="Index"),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        margin=dict(b=20, l=5, r=5, t=40),
        height=300,
    )

    return fig

def compute_powerlaw_fit(degrees, has_powerlaw=True):
    """
    Provede power-law fit nad stupni grafu.
    Vrací slovník s výsledky nebo informací, proč fit nešel provést.
    """
    result = {
        "success": False,
        "reason": None,
        "degrees_for_fit": None,
        "alpha": None,
        "xmin": None,
        "R": None,
        "p": None,
    }

    if not has_powerlaw:
        result["reason"] = "Balík powerlaw není dostupný."
        return result

    degs_for_fit = np.array([d for d in degrees if d > 0])
    result["degrees_for_fit"] = degs_for_fit

    if len(degs_for_fit) < 10:
        result["reason"] = "Příliš málo hodnot pro smysluplný fit."
        return result

    try:
        import powerlaw

        fit = powerlaw.Fit(
            degs_for_fit,
            discrete=True,
            verbose=False,
        )

        alpha = fit.power_law.alpha
        xmin = fit.power_law.xmin
        R, p = fit.distribution_compare("power_law", "exponential")

        result["success"] = True
        result["alpha"] = alpha
        result["xmin"] = xmin
        result["R"] = R
        result["p"] = p
        return result

    except Exception as e:
        result["reason"] = f"Power-law fit selhal: {e}"
        return result