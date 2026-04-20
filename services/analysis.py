import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

from services.hvg_graph import build_hvg, build_configuration_graph_from_hvg

try:
    import powerlaw
    HAS_POWERLAW = True
except ImportError:
    HAS_POWERLAW = False


# =========================
#  HVG – pomocné funkce
# =========================
@st.cache_data(show_spinner=False)
def build_hvg_cached_edges(data_tuple):
    # Data se převedou na numpy pole kvůli jednotnému zpracování.
    data = np.array(data_tuple, dtype=float)

    # Nejprve vytvoří samotný HVG.
    G = build_hvg(data)

    # Do cache ukládáme pouze hrany a počet vrcholů,
    return list(G.edges()), len(data)


def build_hvg_cached(data):
    # Vstup se převede na tuple, pro cache.
    edges, n = build_hvg_cached_edges(tuple(np.asarray(data, dtype=float)))

    # Z uložených hran se znovu sestaví NetworkX graf.
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    return G

# =========================
#  Entropie / statistika
# =========================

def shannon_entropy(x, bins="auto"):
    """
    odhad Shannonovy entropie z histogramu.
    Používá np.histogram s density=True.
    """
    if len(x) == 0:
        return np.nan

    # Histogram aproximuje rozdělení hodnot v časové řadě.
    hist, _ = np.histogram(x, bins=bins, density=True)

    # Nulové četnosti se vyřadí, aby nevadily v logaritmu.
    hist = hist[hist > 0]

    if len(hist) == 0:
        return np.nan

    return -np.sum(hist * np.log2(hist))


# =========================
#  Small-world + metriky grafu
# =========================

@st.cache_data(show_spinner=False)
def compute_graph_metrics_cached(edges_tuple, n_nodes):
    # Graf se znovu složí z hran a počtu vrcholů,
    # aby bylo možné výpočet metrik efektivně cacheovat.
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from(edges_tuple)

    n_edges = G.number_of_edges()

    # Stupně všech vrcholů tvoří základ pro další metriky.
    degrees = [d for _, d in G.degree()]
    avg_deg = float(np.mean(degrees)) if len(degrees) > 0 else 0.0

    # Clustering (lokální propojenosti sítě).
    try:
        C = nx.average_clustering(G)
    except Exception:
        C = float("nan")

    # Některé metriky mají smysl jen pro souvislý graf.
    is_conn = nx.is_connected(G) if n_nodes > 0 else False

    L = None
    diam = None

    # když je graf souvislý a má alespoň dva vrcholy. -> vypočet diametru a average path lenght
    if is_conn and n_nodes > 1:
        try:
            L = nx.average_shortest_path_length(G)
        except Exception:
            L = None
        try:
            diam = nx.diameter(G)
        except Exception:
            diam = None

    # Assortativita ukazuje, zda se propojují vrcholy podobného stupně.
    try:
        assort = nx.degree_assortativity_coefficient(G)
    except Exception:
        assort = None

    L_rand = None
    C_rand = None

    # Přibližné metriky odpovídajícího náhodného grafu slouží jako reference pro small-world analýzu.
    if n_nodes > 1 and avg_deg > 1:
        try:
            L_rand = np.log(n_nodes) / np.log(avg_deg)
            C_rand = avg_deg / n_nodes
        except Exception:
            L_rand = None
            C_rand = None

    analyzer = SmallWorldAnalyzer(C, L, C_rand, L_rand)
    sigma = analyzer.sigma

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "degrees": degrees,
        "avg_deg": avg_deg,
        "C": C,
        "is_conn": is_conn,
        "L": L,
        "diam": diam,
        "assort": assort,
        "L_rand": L_rand,
        "C_rand": C_rand,
        "sigma": sigma,
    }


def compute_graph_metrics(G):
    # Hrany se převedou do stabilní, seřazené podoby -> konzistejnější cache.
    edges_tuple = tuple(sorted(tuple(sorted(edge)) for edge in G.edges()))
    n_nodes = G.number_of_nodes()

    metrics = compute_graph_metrics_cached(edges_tuple, n_nodes)
    metrics = dict(metrics)

    # K metrikám se doplní i objekt pro slovní interpretaci small-world indexu.
    metrics["analyzer"] = SmallWorldAnalyzer(
        metrics["C"],
        metrics["L"],
        metrics["C_rand"],
        metrics["L_rand"],
    )
    return metrics


def compute_configuration_model_metrics(G, seed=42):
    # Konfigurační graf zachovává stupňovou posloupnost jinak náhodný strukturou
    G_conf = build_configuration_graph_from_hvg(G, seed=seed)

    # Nad tímto null modelem se spočítají stejné metriky jako nad HVG.
    metrics = compute_graph_metrics(G_conf)
    return G_conf, metrics


@st.cache_data(show_spinner=False)
def compute_graph_layout_cached(edges_tuple, n_nodes, layout_type="spring", seed=42):
    # I rozložení vrcholů se rekonstruuje z hran, kvuli uložení v cache
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from(edges_tuple)

    if layout_type == "planar":
        try:
            is_planar, _ = nx.check_planarity(G)
            if is_planar:
                pos = nx.planar_layout(G)
                return {node: (float(x), float(y)) for node, (x, y) in pos.items()}
        except Exception:
            pass

    # Pokud planární rozložení není možné, použije se spring layout.
    pos = nx.spring_layout(G, seed=seed)
    return {node: (float(x), float(y)) for node, (x, y) in pos.items()}


def compute_graph_layout(G, layout_type="spring", seed=42):
    edges_tuple = tuple(sorted(tuple(sorted(edge)) for edge in G.edges()))
    n_nodes = G.number_of_nodes()

    pos_dict = compute_graph_layout_cached(
        edges_tuple,
        n_nodes,
        layout_type=layout_type,
        seed=seed,
    )

    return {node: np.array(coords) for node, coords in pos_dict.items()}


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
        # Pokud některá z potřebných metrik chybí, nejde zpočítat Small-W
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
        - typ: 'success'  -> small-world| 'info'     -> podobné náhodnému grafu (σ ≈ 1)| 'warning'  -> není small-world
        """
        if self.sigma is None or np.isnan(self.sigma):
            return (
                "info",
                "Small-world index σ nelze spolehlivě spočítat "
                "(chybí některá z metrik nebo došlo k numerické chybě)."
            )

        s = self.sigma

        # σ > 1 znamená výraznější small-world charakter než u náhodného grafu.
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

def prepare_network_traces(
    G,
    pos=None,
    node_color="skyblue",
    node_size=10,
    edge_color="#888",
    edge_width=1,
    show_labels=False,
    hover_texts=None,
):
    import networkx as nx
    import plotly.graph_objects as go

    # Ošetření prázdného grafu, aby vizualizace nespadla.
    if G is None or G.number_of_nodes() == 0:
        edge_trace = go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line=dict(width=edge_width, color=edge_color),
            hoverinfo="none",
        )
        node_trace = go.Scatter(
            x=[],
            y=[],
            mode="markers",
            hoverinfo="text",
            marker=dict(size=node_size, color=node_color, line_width=1),
        )
        return edge_trace, node_trace

    # Pokud pozice nejsou dodané, vytvoří se automaticky.
    if pos is None:
        pos = nx.spring_layout(G, seed=42)

    # Pokud v dodaném layoutu některé vrcholy chybí, přepočtení layoutupro graf
    missing_nodes = [node for node in G.nodes() if node not in pos]
    if missing_nodes:
        pos = nx.spring_layout(G, seed=42)

    edge_x = []
    edge_y = []

    # Hrany jsou v Plotly reprezentovány jako jedna dlouhá sekvence bodů
    # oddělená hodnotou None.
    for u, v in G.edges():
        if u not in pos or v not in pos:
            continue

        x0, y0 = pos[u]
        x1, y1 = pos[v]

        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=edge_width, color=edge_color),
        hoverinfo="none",
    )

    node_x = []
    node_y = []
    node_hover = []
    node_labels = []

    nodes_list = list(G.nodes())

    for i, node in enumerate(nodes_list):
        if node not in pos:
            continue

        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        # Pokud jsou dodané vlastní hover texty, použijí se.
        if hover_texts is not None and i < len(hover_texts):
            node_hover.append(hover_texts[i])
        else:
            node_hover.append(f"Vrchol: {node}")

        node_labels.append(str(node))

    node_mode = "markers+text" if show_labels else "markers"

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode=node_mode,
        text=node_labels if show_labels else None,
        textposition="bottom center" if show_labels else None,
        hoverinfo="text",
        hovertext=node_hover,
        marker=dict(
            size=node_size,
            color=node_color,
            line_width=1,
        ),
        textfont=dict(size=10, color="black"),
    )

    return edge_trace, node_trace


@st.cache_data(show_spinner=False)
def compute_degree_distribution_metrics_cached(degrees_tuple):
    degrees = list(degrees_tuple)

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

    # Entropie popisuje rozptýlenost stupňového rozdělení.
    entropy_deg = -np.sum(pk * np.log(pk)) if len(pk) > 0 else 0.0

    # Normalizovaná entropie umožňuje srovnání mezi různými grafy.
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


def compute_degree_distribution_metrics(degrees):
    return compute_degree_distribution_metrics_cached(tuple(degrees))


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

    # Každá hrana ARC diagramu je zobrazena jako půlkružnice nad osou indexů,
    # usnadňuje čtení dlouhodosahových vazeb v HVG.
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

    # Pro fit dávají smysl pouze kladné stupně.
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

        # Fit vrací exponent alpha, spodní hranici xmin a porovnání s exponenciálním rozdělením.
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