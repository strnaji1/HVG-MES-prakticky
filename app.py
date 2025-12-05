import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go

# Zkusíme importovat powerlaw – pokud není, jen nastavíme flag
try:
    import powerlaw
    HAS_POWERLAW = True
except ImportError:
    HAS_POWERLAW = False

# =========================
#  Pomocné funkce – generátory
# =========================

def generate_logistic_map(length, r=3.9, x0=0.2, burn=500):
    """
    Logistická mapa: x_{n+1} = r * x_n * (1 - x_n)
    Vrací posledních `length` hodnot po zahozní burn-in části.
    """
    N = length + burn
    x = np.empty(N)
    x[0] = x0
    for i in range(1, N):
        x[i] = r * x[i-1] * (1 - x[i-1])
    return x[burn:]


def generate_henon_map(length, a=1.4, b=0.3, x0=0.1, y0=0.0, burn=200):
    """
    Henonova mapa:
    x_{n+1} = 1 - a x_n^2 + y_n
    y_{n+1} = b x_n
    Vrací x-sérii po burn-in.
    """
    N = length + burn
    xs = np.empty(N)
    ys = np.empty(N)
    xs[0] = x0
    ys[0] = y0
    for n in range(1, N):
        xs[n] = 1 - a * xs[n-1]**2 + ys[n-1]
        ys[n] = b * xs[n-1]
    return xs[burn:]


def generate_lorenz_x(length, dt=0.01,
                      sigma=10.0, rho=28.0, beta=8/3,
                      x0=1.0, y0=1.0, z0=1.0, burn=1000):
    """
    Lorenzův systém integrovaný jednoduchým Eulerem.
    Vrací x-sérii po burn-in.
    """
    N = length + burn
    xs = np.empty(N)
    ys = np.empty(N)
    zs = np.empty(N)
    xs[0], ys[0], zs[0] = x0, y0, z0

    for i in range(1, N):
        dx = sigma * (ys[i-1] - xs[i-1])
        dy = xs[i-1] * (rho - zs[i-1]) - ys[i-1]
        dz = xs[i-1] * ys[i-1] - beta * zs[i-1]

        xs[i] = xs[i-1] + dx * dt
        ys[i] = ys[i-1] + dy * dt
        zs[i] = zs[i-1] + dz * dt

    return xs[burn:]


def generate_pink_noise(length):
    """
    1/f šum (pink noise) přes frekvenční doménu.
    Vrací normalizovanou sérii délky `length`.
    """
    # nejbližší mocnina 2 >= length kvůli FFT
    N = int(2 ** np.ceil(np.log2(length)))
    freqs = np.fft.rfftfreq(N)
    phases = np.random.uniform(0, 2 * np.pi, len(freqs))

    # amplituda ~ 1/sqrt(f), f=0 nastavíme na 0
    amplitude = np.where(freqs == 0, 0.0, 1.0 / np.sqrt(freqs))
    spectrum = amplitude * (np.cos(phases) + 1j * np.sin(phases))

    signal = np.fft.irfft(spectrum, n=N)
    signal = signal[:length]

    # normalizace
    signal = (signal - signal.mean()) / signal.std()
    return signal


# =========================
#  Funkce pro generování HVG
# =========================

def build_hvg(data):
    G = nx.Graph()
    n = len(data)
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if all(data[k] < data[i] and data[k] < data[j] for k in range(i + 1, j)):
                G.add_edge(i, j)
    return G


# =========================
#  Konfigurační graf (null model)
# =========================

def build_configuration_graph_from_hvg(G, seed=42):
    """
    Vytvoří jednoduchý konfigurační graf (null model)
    se stejnou stupňovou posloupností jako HVG graf G.
    """
    degrees = [d for _, d in G.degree()]
    # multigraf s danou degree sequence
    H_multi = nx.configuration_model(degrees, seed=seed)
    # převod na jednoduchý graf (zahodí paralelní hrany)
    H = nx.Graph(H_multi)
    # odstraníme self-loops
    H.remove_edges_from(nx.selfloop_edges(H))
    return H


# =========================
#  Small-world analyzer třída
# =========================

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


# =========================
#  Inicializace session state
# =========================

for key in ("data", "show_hvg", "show_direct", "show_horiz"):
    if key not in st.session_state:
        st.session_state[key] = None if key == "data" else False

st.set_page_config(page_title="HVG Vizualizátor", layout="wide")

# =========================
#  Hlavička
# =========================

st.title("📊 HVG Vizualizátor")
st.markdown("**Interaktivní vizualizace časových řad a jejich Horizontal Visibility Graphů (HVG)**")

# =========================
#  Sidebar – výběr typu signálu
# =========================

st.sidebar.title("🔧 Nastavení dat")

mode = st.sidebar.radio(
    "Typ vstupu",
    ["Standardní signály", "Chaotické generátory"]
)

typ = None
chaos_typ = None

if mode == "Standardní signály":
    typ = st.sidebar.selectbox(
        "Vyber typ časové řady",
        ["Náhodná uniformní", "Náhodná normální", "Sinusovka",
         "Nahrát CSV", "Ruční vstup"]
    )

    if typ == "Náhodná uniformní":
        length = st.sidebar.slider("Délka řady", 10, 500, 50)
        low = st.sidebar.number_input("Minimální hodnota", value=0.0, step=0.1)
        high = st.sidebar.number_input("Maximální hodnota", value=1.0, step=0.1)
    elif typ == "Náhodná normální":
        length = st.sidebar.slider("Délka řady", 10, 500, 50)
        mu = st.sidebar.number_input("Střední hodnota μ", value=0.0)
        sigma = st.sidebar.number_input("Směrodatná odchylka σ", value=1.0)
    elif typ == "Sinusovka":
        length = st.sidebar.slider("Délka řady", 10, 500, 100)
        amp = st.sidebar.number_input("Amplituda", value=1.0)
        freq = st.sidebar.number_input("Frekvence", value=1.0)
    elif typ == "Nahrát CSV":
        uploaded_file = st.sidebar.file_uploader("Nahraj CSV se sloupcem hodnot", type="csv")
    elif typ == "Ruční vstup":
        raw_text = st.sidebar.text_area("Zadej hodnoty oddělené čárkou", value="10, 5, 3, 7, 6")

else:  # Chaotické generátory
    chaos_typ = st.sidebar.selectbox(
        "Vyber chaotický systém",
        [
            "Logistická mapa",
            "Henonova mapa",
            "Lorenzův systém (x-složka)",
            "1/f šum (pink noise)"
        ]
    )

    if chaos_typ == "Logistická mapa":
        length = st.sidebar.slider("Délka řady", 100, 5000, 1000, step=100)
        r = st.sidebar.slider("Parametr r", 3.5, 4.0, 3.9, step=0.01)
        x0 = st.sidebar.number_input("Počáteční x₀", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
        burn_log = st.sidebar.number_input("Burn-in iterace", 100, 10000, 500, step=100)

    elif chaos_typ == "Henonova mapa":
        length = st.sidebar.slider("Délka řady", 100, 5000, 1000, step=100)
        a = st.sidebar.number_input("Parametr a", value=1.4, step=0.1)
        b = st.sidebar.number_input("Parametr b", value=0.3, step=0.05)
        x0 = st.sidebar.number_input("Počáteční x₀", value=0.1, step=0.05)
        y0 = st.sidebar.number_input("Počáteční y₀", value=0.0, step=0.05)
        burn_henon = st.sidebar.number_input("Burn-in iterace", 100, 10000, 500, step=100)

    elif chaos_typ == "Lorenzův systém (x-složka)":
        length = st.sidebar.slider("Délka řady", 200, 10000, 2000, step=200)
        dt = st.sidebar.number_input("Krok integrace dt", value=0.01, step=0.005, format="%.3f")
        sigma_l = st.sidebar.number_input("σ (sigma)", value=10.0, step=1.0)
        rho_l = st.sidebar.number_input("ρ (rho)", value=28.0, step=1.0)
        beta_l = st.sidebar.number_input("β (beta)", value=8/3, step=0.1)
        burn_lor = st.sidebar.number_input("Burn-in kroků", 500, 20000, 1000, step=500)

    elif chaos_typ == "1/f šum (pink noise)":
        length = st.sidebar.slider("Délka řady", 100, 10000, 2000, step=100)

# tlačítko pro generování
generate = st.sidebar.button("Načíst / generovat řadu")

# =========================
#  Generování dat
# =========================

if generate:
    data = None

    if mode == "Standardní signály":
        if typ == "Náhodná uniformní":
            data = np.random.uniform(low=low, high=high, size=length)
        elif typ == "Náhodná normální":
            data = np.random.normal(loc=mu, scale=sigma, size=length)
        elif typ == "Sinusovka":
            x = np.arange(length)
            data = amp * np.sin(2 * np.pi * freq * x / length)
        elif typ == "Nahrát CSV" and uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            data = df.iloc[:, 0].values
        elif typ == "Ruční vstup":
            try:
                data = np.array([float(v.strip()) for v in raw_text.split(",")])
            except ValueError:
                st.error("Chybný formát ručního vstupu! Zkontroluj čísla.")
                data = None

    else:  # Chaotické generátory
        if chaos_typ == "Logistická mapa":
            data = generate_logistic_map(length, r=r, x0=x0, burn=burn_log)
        elif chaos_typ == "Henonova mapa":
            data = generate_henon_map(length, a=a, b=b, x0=x0, y0=y0, burn=burn_henon)
        elif chaos_typ == "Lorenzův systém (x-složka)":
            data = generate_lorenz_x(length, dt=dt,
                                     sigma=sigma_l, rho=rho_l, beta=beta_l,
                                     burn=burn_lor)
        elif chaos_typ == "1/f šum (pink noise)":
            data = generate_pink_noise(length)

    st.session_state.data = data
    st.session_state.show_hvg = False
    st.session_state.show_direct = False
    st.session_state.show_horiz = False

# =========================
#  Zobrazení časové řady + HVG linky
# =========================

if st.session_state.data is not None:
    arr = st.session_state.data
    st.subheader("📈 Vaše časová řada")

    df_ts = pd.DataFrame({"index": np.arange(len(arr)), "value": arr})
    fig_ts = px.line(
        df_ts, x="index", y="value", markers=True,
        title="Časová řada",
        hover_data={"index": True, "value": ":.3f"}
    )
    fig_ts.update_traces(marker_size=8)

    # Přímé linky
    if st.session_state.show_direct:
        G_tmp = build_hvg(arr)
        shapes = []
        for i, j in G_tmp.edges():
            shapes.append(dict(
                type="line",
                x0=i, y0=arr[i], x1=j, y1=arr[j],
                line=dict(color="gray", width=1)
            ))
        fig_ts.update_layout(shapes=shapes)

    # Vodorovné linky
    if st.session_state.show_horiz:
        G_tmp = build_hvg(arr)
        shapes = []
        for i, j in G_tmp.edges():
            y = min(arr[i], arr[j])
            shapes.append(dict(
                type="line",
                x0=i, y0=y, x1=j, y1=y,
                line=dict(color="gray", width=1)
            ))
        fig_ts.update_layout(shapes=shapes)

    st.plotly_chart(fig_ts, use_container_width=True)

    # Statistiky časové řady
    st.write(
        f"- Délka: **{len(arr)}**, "
        f"Průměr: **{arr.mean():.3f}**, "
        f"Rozptyl: **{arr.var():.3f}**"
    )

    # Tlačítka vedle sebe (toggle)
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🕸️ Vygenerovat HVG"):
            st.session_state.show_hvg = True
    with c2:
        if st.button("🔗 HVG linky (přímé)"):
            st.session_state.show_direct = not st.session_state.show_direct
            if st.session_state.show_direct:
                st.session_state.show_horiz = False
    with c3:
        if st.button("🔗 HVG linky (vodorovné)"):
            st.session_state.show_horiz = not st.session_state.show_horiz
            if st.session_state.show_horiz:
                st.session_state.show_direct = False

# =========================
#  Interaktivní HVG + histogram + power-law + arc diagram
# =========================

if st.session_state.show_hvg and st.session_state.data is not None:
    arr = st.session_state.data
    G = build_hvg(arr)

    st.subheader("🕸️ Interaktivní vizualizace HVG")

    # ====== Analytické statistiky HVG ======
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    degrees = [d for _, d in G.degree()]
    avg_deg = float(np.mean(degrees)) if len(degrees) > 0 else 0.0

    # Clustering
    try:
        C = nx.average_clustering(G)
    except Exception:
        C = float("nan")

    # Souvislost, průměrná délka cesty, průměr
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

    # Teoretické hodnoty pro náhodný graf G(N, p)
    L_rand = None
    C_rand = None

    if n_nodes > 1 and avg_deg > 1:
        try:
            L_rand = np.log(n_nodes) / np.log(avg_deg)
            C_rand = avg_deg / n_nodes
        except Exception:
            L_rand = None
            C_rand = None

    # Small-world analyzer – výpočet σ a interpretace
    analyzer = SmallWorldAnalyzer(C, L, C_rand, L_rand)
    sigma_sw = analyzer.sigma

    col_stats1, col_stats2 = st.columns(2)
    with col_stats1:
        st.markdown("**Základní metriky HVG**")
        st.write(f"- Počet vrcholů: **{n_nodes}**")
        st.write(f"- Počet hran: **{n_edges}**")
        st.write(f"- Průměrný stupeň: **{avg_deg:.3f}**")
        if L is not None:
            st.write(f"- Průměrná délka cesty L: **{L:.3f}**")
        else:
            st.write("- Průměrná délka cesty L: *nelze spočítat (nesouvislý graf)*")
        if diam is not None:
            st.write(f"- Průměr grafu (diameter): **{diam}**")
        else:
            st.write("- Průměr grafu (diameter): *není k dispozici*")

    with col_stats2:
        st.markdown("**Clustering a small-world charakter**")
        st.write(f"- Clustering coefficient C: **{C:.3f}**")
        if assort is not None and not np.isnan(assort):
            st.write(f"- Degree assortativity: **{assort:.3f}**")
        else:
            st.write("- Degree assortativity: *není k dispozici*")

        if L_rand is not None and C_rand is not None and C_rand != 0:
            st.write(
                "- Náhodný graf (pro porovnání):  \n"
                f"  - L_rand ≈ **{L_rand:.3f}**  \n"
                f"  - C_rand ≈ **{C_rand:.5f}**"
            )
        else:
            st.write("- Náhodný graf (L_rand, C_rand): *nelze odhadnout*")

        if sigma_sw is not None and not np.isnan(sigma_sw):
            st.write(
                f"- Small-world index σ "
                f"(σ > 1: small-world, σ ≈ 1: podobné náhodnému grafu, σ < 1: není small-world): "
                f"**{sigma_sw:.2f}**"
            )

            level, msg = analyzer.interpretation(atol=0.05)
            if level == "success":
                st.success(msg)
            elif level == "warning":
                st.warning(msg)
            else:
                st.info(msg)
        else:
            st.write(
                "- Small-world index σ: *nelze spočítat "
                "(chybí některá z metrik L, C, L_rand nebo C_rand nebo je výsledek nespolehlivý)*"
            )

    st.markdown("---")

    # ====== Rozmístění pro vizualizaci HVG ======
    layout_option = st.radio(
        "Rozložení HVG vrcholů",
        ["Síťové (spring layout)", "Planární (pokud možné)"],
        horizontal=True
    )

    if layout_option == "Síťové (spring layout)":
        pos = nx.spring_layout(G, seed=42)
    else:  # "Planární (pokud možné)"
        try:
            is_planar, embedding = nx.check_planarity(G)
            if is_planar:
                pos = nx.planar_layout(G)
            else:
                pos = nx.spring_layout(G, seed=42)
        except Exception:
            pos = nx.spring_layout(G, seed=42)

    # Volba, jestli zobrazit textové popisky vrcholů
    show_labels = st.checkbox("Zobrazit popisky vrcholů (indexy)", value=False)

    # Edges HVG
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode='lines',
        line=dict(width=1, color='#888'), hoverinfo='none'
    )

    # Nodes HVG
    node_x, node_y, node_text = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        neigh = list(G.adj[node])
        node_text.append(f"Index: {node}<br>Stupeň: {len(neigh)}<br>Sousedé: {neigh}")

    if show_labels:
        node_mode = "markers+text"
        node_text_visual = [str(n) for n in G.nodes()]
        text_position = "bottom center"
    else:
        node_mode = "markers"
        node_text_visual = None
        text_position = None

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode=node_mode,
        text=node_text_visual,
        textposition=text_position,
        hoverinfo='text', hovertext=node_text,
        marker=dict(size=10, color='skyblue', line_width=1),
        textfont=dict(size=10, color="black")
    )

    fig_hvg = go.Figure(data=[edge_trace, node_trace])
    fig_hvg.update_layout(
        title="Horizontal Visibility Graph",
        showlegend=False, hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40)
    )
    st.plotly_chart(fig_hvg, use_container_width=True)

    # =========================
    #  Konfigurační graf – pod HVG
    # =========================
    st.markdown("### 🔁 Konfigurační graf (null model)")

    show_conf = st.checkbox(
        "Zobrazit konfigurační graf + metriky (null model se stejnou stupňovou posloupností jako HVG)",
        value=False
    )

    if show_conf:
        G_conf = build_configuration_graph_from_hvg(G, seed=42)

        # --- Metriky konfiguračního grafu ---
        n_nodes_conf = G_conf.number_of_nodes()
        n_edges_conf = G_conf.number_of_edges()
        degrees_conf = [d for _, d in G_conf.degree()]
        avg_deg_conf = float(np.mean(degrees_conf)) if len(degrees_conf) > 0 else 0.0

        try:
            C_conf = nx.average_clustering(G_conf)
        except Exception:
            C_conf = float("nan")

        is_conn_conf = nx.is_connected(G_conf) if n_nodes_conf > 0 else False
        L_conf = None
        diam_conf = None
        if is_conn_conf and n_nodes_conf > 1:
            try:
                L_conf = nx.average_shortest_path_length(G_conf)
            except Exception:
                L_conf = None
            try:
                diam_conf = nx.diameter(G_conf)
            except Exception:
                diam_conf = None

        try:
            assort_conf = nx.degree_assortativity_coefficient(G_conf)
        except Exception:
            assort_conf = None

        # "ER-like" odhad pro konfigurační graf – stejný vzorec
        L_rand_conf = None
        C_rand_conf = None
        if n_nodes_conf > 1 and avg_deg_conf > 1:
            try:
                L_rand_conf = np.log(n_nodes_conf) / np.log(avg_deg_conf)
                C_rand_conf = avg_deg_conf / n_nodes_conf
            except Exception:
                L_rand_conf = None
                C_rand_conf = None

        # Small-world index pro konfigurační graf
        sigma_conf = None
        if (
            C_conf is not None and L_conf is not None and
            L_rand_conf is not None and C_rand_conf not in (None, 0)
        ):
            try:
                sigma_conf = (C_conf / C_rand_conf) / (L_conf / L_rand_conf)
            except Exception:
                sigma_conf = None

        col_conf1, col_conf2 = st.columns(2)
        with col_conf1:
            st.markdown("**Konfigurační graf – základní metriky**")
            st.write(f"- Počet vrcholů: **{n_nodes_conf}**")
            st.write(f"- Počet hran: **{n_edges_conf}**")
            st.write(f"- Průměrný stupeň: **{avg_deg_conf:.3f}**")
            if L_conf is not None:
                st.write(f"- Průměrná délka cesty L_conf: **{L_conf:.3f}**")
            else:
                st.write("- Průměrná délka cesty L_conf: *nelze spočítat (nesouvislý graf)*")
            if diam_conf is not None:
                st.write(f"- Průměr grafu (diameter_conf): **{diam_conf}**")
            else:
                st.write("- Průměr grafu (diameter_conf): *není k dispozici*")

        with col_conf2:
            st.markdown("**Konfigurační graf – clustering, assortativita, σ_conf**")
            st.write(f"- Clustering coefficient C_conf: **{C_conf:.3f}**")
            if assort_conf is not None and not np.isnan(assort_conf):
                st.write(f"- Degree assortativity_conf: **{assort_conf:.3f}**")
            else:
                st.write("- Degree assortativity_conf: *není k dispozici*")

            if L_rand_conf is not None and C_rand_conf is not None and C_rand_conf != 0:
                st.write(
                    "- Náhodný graf pro konfigurační model (odhad):  \n"
                    f"  - L_rand_conf ≈ **{L_rand_conf:.3f}**  \n"
                    f"  - C_rand_conf ≈ **{C_rand_conf:.5f}**"
                )
            else:
                st.write("- L_rand_conf, C_rand_conf: *nelze odhadnout*")

            if sigma_conf is not None and not np.isnan(sigma_conf):
                st.write(
                    f"- Small-world index σ_conf: **{sigma_conf:.2f}** "
                    "(stejná definice jako u HVG)"
                )

        # --- Porovnání HVG vs. konfigurační graf ---
        st.markdown("**📊 Porovnání HVG vs. konfigurační graf (null model)**")

        if not np.isnan(C) and not np.isnan(C_conf):
            st.write(f"- Clustering HVG: **{C:.3f}**, konfigurační graf C_conf: **{C_conf:.3f}**")
            if C > C_conf * 2:
                st.info(
                    "HVG má **výrazně vyšší clustering** než degree-preserving null model – "
                    "to naznačuje silnou nestrukturovanost vůči náhodnému přepojení hran."
                )

        if (L is not None) and (L_conf is not None):
            st.write(f"- Průměrná délka cesty L (HVG): **{L:.3f}**, L_conf: **{L_conf:.3f}**")
            if L >= L_conf:
                st.write(
                    "- HVG má podobné nebo delší cesty než null model, což je konzistentní "
                    "s small-world strukturou (vyšší clustering, cesty pořád krátké)."
                )

        if sigma_sw is not None and sigma_conf is not None:
            st.write(
                f"- Small-world index HVG: **{sigma_sw:.2f}**, "
                f"konfigurační graf σ_conf: **{sigma_conf:.2f}**"
            )
            if sigma_sw > sigma_conf:
                st.success(
                    "σ(HVG) > σ(conf) – skutečný HVG je **víc small-world** než jeho "
                    "degree-preserving null model."
                )

        # --- Vizualizace konfiguračního grafu ---
        st.subheader("🕸️ Konfigurační graf (vizualizace)")
        pos_conf = nx.spring_layout(G_conf, seed=42)
        edge_x_c, edge_y_c = [], []
        for u, v in G_conf.edges():
            x0, y0 = pos_conf[u]
            x1, y1 = pos_conf[v]
            edge_x_c += [x0, x1, None]
            edge_y_c += [y0, y1, None]

        edge_trace_c = go.Scatter(
            x=edge_x_c, y=edge_y_c, mode='lines',
            line=dict(width=1, color='#aaa'), hoverinfo='none'
        )

        node_x_c, node_y_c = [], []
        for node in G_conf.nodes():
            x, y = pos_conf[node]
            node_x_c.append(x)
            node_y_c.append(y)

        node_trace_c = go.Scatter(
            x=node_x_c, y=node_y_c, mode='markers',
            hoverinfo='none',
            marker=dict(size=8, color='lightgreen', line_width=1),
        )

        fig_conf = go.Figure(data=[edge_trace_c, node_trace_c])
        fig_conf.update_layout(
            title="Konfigurační graf se stejnou stupňovou posloupností jako HVG",
            showlegend=False, hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40)
        )
        st.plotly_chart(fig_conf, use_container_width=True)

    st.markdown("---")

    # Histogram stupňů
    degs = degrees
    df_deg = pd.DataFrame({"degree": degs})
    fig_hist = px.histogram(
        df_deg, x="degree", nbins=max(degs) + 1,
        title="Histogram stupňů", labels={"degree": "Stupeň"},
        opacity=0.7
    )
    fig_hist.update_layout(yaxis_title="Počet vrcholů")
    st.plotly_chart(fig_hist, use_container_width=True)

    # Power-law graf P(k) vs k (log–log)
    unique_deg, counts = np.unique(degs, return_counts=True)
    pk = counts / counts.sum()

    df_power = pd.DataFrame({
        "degree": unique_deg,
        "pk": pk
    })

    st.subheader("📉 Power-law (log–log) graf rozdělení stupňů")

    fig_power = px.scatter(
        df_power,
        x="degree",
        y="pk",
        log_x=True,
        log_y=True,
        labels={"degree": "Stupeň k", "pk": "P(k)"},
        title="Log–log graf P(k) vs. k"
    )
    fig_power.update_traces(mode="markers+lines")
    st.plotly_chart(fig_power, use_container_width=True)

    # Volitelný formální power-law test + CCDF graf
    do_pl_test = st.checkbox("🔍 Provést formální power-law test (Clauset–Shalizi–Newman) + CCDF")

    if do_pl_test:
        if not HAS_POWERLAW:
            st.warning(
                "K provedení testu je potřeba balík `powerlaw`. "
                "Přidej ho do `requirements.txt` a nainstaluj pomocí `pip install powerlaw`."
            )
        else:
            # filtrujeme jen stupně >= 1
            degs_for_fit = np.array([d for d in degs if d > 0])

            if len(degs_for_fit) < 10:
                st.info("Graf má příliš málo vrcholů pro smysluplný power-law fit.")
            else:
                try:
                    fit = powerlaw.Fit(degs_for_fit, discrete=True, verbose=False)
                    alpha = fit.power_law.alpha
                    xmin = fit.power_law.xmin

                    # porovnání power-law vs. exponenciální rozdělení
                    R, p = fit.distribution_compare('power_law', 'exponential')

                    st.markdown("**Výsledek power-law analýzy:**")
                    st.write(f"- Odhadnutý exponent \\(\\alpha\\): **{alpha:.3f}**")
                    st.write(f"- Odhadnuté \\(k_\\min\\): **{xmin}**")
                    st.write(f"- Likelihood ratio (power-law vs. exponential): **R = {R:.3f}**")
                    st.write(f"- p-hodnota: **p = {p:.3f}**")

                    if p < 0.1:
                        if R > 0:
                            st.success(
                                "Pro daný HVG jsou data **kompatibilní s power-law** "
                                "(power-law je statisticky preferovaný oproti exponenciálnímu rozdělení)."
                            )
                        else:
                            st.warning(
                                "Power-law model je **horší** než exponenciální (R < 0, p < 0.1). "
                                "Síť pravděpodobně není scale-free."
                            )
                    else:
                        st.info(
                            "Test je **neprůkazný** (p ≥ 0.1). Nelze spolehlivě říct, že rozdělení je power-law, "
                            "ale ani ho jednoznačně vyloučit."
                        )

                    # =========================
                    #  CCDF power-law graf
                    # =========================
                    # Empirická CCDF: P(K >= k)
                    degs_arr = degs_for_fit
                    unique_sorted = np.sort(np.unique(degs_arr))
                    ccdf_vals = np.array([
                        np.sum(degs_arr >= k) / len(degs_arr) for k in unique_sorted
                    ])

                    # používáme jen tail k >= xmin
                    mask = unique_sorted >= xmin
                    if np.sum(mask) >= 2:
                        k_emp = unique_sorted[mask]
                        ccdf_emp = ccdf_vals[mask]

                        # Teoretická power-law CCDF ~ (k/xmin)^{1-α}, znormalizovaná v k_min
                        k_theory = np.linspace(xmin, k_emp.max(), 100)
                        ccdf_theory = (k_theory / xmin) ** (1 - alpha)
                        # přenormování tak, aby se kryla v k_min
                        ccdf_theory *= ccdf_emp[0] / ccdf_theory[0]

                        st.subheader("📈 CCDF power-law graf (log–log)")

                        fig_ccdf = go.Figure()

                        # Empirická CCDF
                        fig_ccdf.add_trace(go.Scatter(
                            x=k_emp,
                            y=ccdf_emp,
                            mode="markers",
                            name="Empirická CCDF",
                        ))

                        # Teoretický power-law fit
                        fig_ccdf.add_trace(go.Scatter(
                            x=k_theory,
                            y=ccdf_theory,
                            mode="lines",
                            name=f"Power-law fit (α={alpha:.2f})",
                        ))

                        fig_ccdf.update_layout(
                            title="CCDF stupňového rozdělení (empirická vs. power-law fit)",
                            xaxis_type="log",
                            yaxis_type="log",
                            xaxis_title="Stupeň k",
                            yaxis_title="P(K ≥ k)",
                            legend=dict(x=0.02, y=0.98),
                            margin=dict(b=40, l=50, r=10, t=50),
                        )

                        st.plotly_chart(fig_ccdf, use_container_width=True)
                        st.caption(
                            "Body představují empirickou komplementární distribuční funkci stupňů pro k ≥ k_min, "
                            "křivka je teoretický power-law fit. "
                            "Pokud se body v tailu (vpravo) přibližně drží křivky, "
                            "je chování rozdělení kompatibilní s power-law."
                        )
                    else:
                        st.info(
                            "Tail rozdělení (k ≥ k_min) je příliš krátký na smysluplný CCDF graf."
                        )

                except Exception as e:
                    st.error(f"Nepodařilo se provést power-law fit: {e}")

    # Arc diagram HVG
    st.subheader("🎨 Arc Diagram HVG")
    n = len(arr)
    node_x_line = np.arange(n)
    node_y_line = np.zeros(n)
    fig_arc = go.Figure()

    for i, j in G.edges():
        r = (j - i) / 2
        mid = i + r
        theta = np.linspace(0, np.pi, 100)
        x_arc = mid + r * np.cos(theta)
        y_arc = r * np.sin(theta)
        fig_arc.add_trace(go.Scatter(
            x=x_arc, y=y_arc, mode='lines',
            line=dict(color='gray', width=1),
            hoverinfo='none'
        ))

    fig_arc.add_trace(go.Scatter(
        x=node_x_line, y=node_y_line, mode='markers',
        marker=dict(size=8, color='skyblue'),
        hoverinfo='text',
        hovertext=[f"Index: {i}<br>Hodnota: {arr[i]:.3f}" for i in node_x_line]
    ))

    fig_arc.update_layout(
        title="Arc Diagram HVG",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, title="Index"),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        margin=dict(b=20, l=5, r=5, t=40),
        height=300
    )
    st.plotly_chart(fig_arc, use_container_width=True)
