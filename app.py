import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import re  # <- kvůli parsování textových vstupů

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
#  Další pomocné funkce
# =========================

def build_hvg(data):
    """
    Vytvoří Horizontal Visibility Graph (HVG) z časové řady `data`.
    Vrcholy jsou indexy časové řady.
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
    H = nx.Graph(H_multi)
    H.remove_edges_from(nx.selfloop_edges(H))
    return H


def shannon_entropy(x, bins="auto"):
    """
    Jednoduchý odhad Shannonovy entropie z histogramu.
    """
    if len(x) == 0:
        return np.nan
    hist, _ = np.histogram(x, bins=bins, density=True)
    hist = hist[hist > 0]
    if len(hist) == 0:
        return np.nan
    return -np.sum(hist * np.log2(hist))


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

for key in ("data", "data2", "show_hvg", "show_direct", "show_horiz", "custom_graph"):
    if key not in st.session_state:
        if key in ("data", "data2"):
            st.session_state[key] = None
        elif key == "custom_graph":
            st.session_state[key] = None
        else:
            st.session_state[key] = False

st.set_page_config(page_title="HVG Vizualizátor", layout="wide")

# =========================
#  Hlavička
# =========================

st.title("📊 HVG Vizualizátor")
st.markdown("**Interaktivní vizualizace časových řad a jejich Horizontal Visibility Graphů (HVG)**")

# =========================
#  Sidebar – volba režimu
# =========================

st.sidebar.title("🔧 Vstup / režim")

analysis_mode = st.sidebar.radio(
    "Co chceš analyzovat?",
    ["Časová řada → HVG", "Vlastní graf (ruční / CSV)", "Porovnat dvě časové řady"]
)

# =====================================================================
#  REŽIM 1: ČASOVÁ ŘADA → HVG
# =====================================================================

if analysis_mode == "Časová řada → HVG":
    st.sidebar.subheader("Nastavení časové řady")

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
            raw_text = st.sidebar.text_area(
                "Zadej hodnoty oddělené čárkou",
                value="10, 5, 3, 7, 6"
            )

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
    #  Interaktivní HVG + další sekce pod ním
    # =========================

    if st.session_state.show_hvg and st.session_state.data is not None:
        arr = st.session_state.data
        G = build_hvg(arr)

        st.subheader("🕸️ Interaktivní vizualizace HVG")

        # ---- Přehledné přepínání sekcí pod HVG ----
        section_options = [
            "📊 Metriky HVG",
            "🔗 Propojení časová řada ↔ HVG",
            "🧮 Lokální analýza úseku časové řady",
            "🧩 Podgraf HVG",
            "📉 Rozdělení stupňů + power-law",
            "🎨 Arc Diagram HVG",
            "🔁 Konfigurační graf (null model)",
            "💾 Export HVG a metrik",
        ]
        selected_sections = st.multiselect(
            "Co chceš pod HVG zobrazit?",
            options=section_options,
            default=[
                "📊 Metriky HVG",
                "📉 Rozdělení stupňů + power-law",
                "🎨 Arc Diagram HVG",
                "💾 Export HVG a metrik",
            ]
        )

        # ====== Analytické statistiky HVG (počítáme vždy, ale zobrazíme jen pokud chceš) ======
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

        # ====== Rozmístění pro vizualizaci HVG (společné pro vše) ======
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

        # Barevné kódování vrcholů (globálně pro HVG vizualizaci)
        color_mode = st.selectbox(
            "Barevné kódování vrcholů HVG",
            ["Jednobarevné", "Podle hodnoty časové řady", "Podle stupně"]
        )

        if color_mode == "Podle hodnoty časové řady":
            node_color_values = [arr[i] for i in G.nodes()]
        elif color_mode == "Podle stupně":
            node_color_values = [G.degree(i) for i in G.nodes()]
        else:
            node_color_values = None

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

        # Nodes HVG – základní trace
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

        marker_kwargs = dict(size=10, line_width=1)
        if node_color_values is None:
            marker_kwargs["color"] = "skyblue"
        else:
            marker_kwargs["color"] = node_color_values
            marker_kwargs["colorscale"] = "Viridis"
            marker_kwargs["showscale"] = True

        node_trace = go.Scatter(
            x=node_x, y=node_y, mode=node_mode,
            text=node_text_visual,
            textposition=text_position,
            hoverinfo='text', hovertext=node_text,
            marker=marker_kwargs,
            textfont=dict(size=10, color="black")
        )

        # ====== případné zvýraznění (už je součást sekce "Propojení") ======
        highlight_trace = None
        neighbors = []
        selected_index = 0

        if "🔗 Propojení časová řada ↔ HVG" in selected_sections and n_nodes > 0:
            st.subheader("🔗 Propojení časové řady a HVG")

            selected_index = st.number_input(
                "Index vrcholu/časového bodu pro zvýraznění",
                min_value=0, max_value=n_nodes - 1, value=0, step=1
            )

            highlight_neighbors = st.checkbox(
                "Zvýraznit také sousedy vybraného vrcholu v časové řadě a HVG",
                value=True
            )

            neighbors = list(G.adj[selected_index])
            st.markdown(
                f"- Vybraný vrchol: **{selected_index}**, "
                f"stupeň: **{G.degree(selected_index)}**, "
                f"sousedé: **{neighbors}**"
            )

            # Zvýraznění vybraného vrcholu + sousedů jako separátní trace
            highlight_x, highlight_y, highlight_text = [], [], []
            highlight_nodes = [selected_index]
            if highlight_neighbors:
                highlight_nodes += neighbors

            for node in highlight_nodes:
                x, y = pos[node]
                highlight_x.append(x)
                highlight_y.append(y)
                highlight_text.append(f"Vybraný / soused: {node}")

            highlight_trace = go.Scatter(
                x=highlight_x, y=highlight_y,
                mode="markers+text",
                text=[str(n) for n in highlight_nodes],
                textposition="top center",
                hoverinfo="text",
                hovertext=highlight_text,
                marker=dict(size=14, color="red", line_width=2),
                textfont=dict(size=12, color="red"),
                showlegend=False
            )

        # ====== finální HVG figure (vždy se vykreslí) ======
        data_traces = [edge_trace, node_trace]
        if highlight_trace is not None:
            data_traces.append(highlight_trace)

        fig_hvg = go.Figure(data=data_traces)
        fig_hvg.update_layout(
            title="Horizontal Visibility Graph",
            showlegend=False, hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40)
        )
        st.plotly_chart(fig_hvg, use_container_width=True)

        # ====== Metriky HVG ======
        if "📊 Metriky HVG" in selected_sections:
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

        # ====== Zvýraznění v časové řadě (jen pokud je sekce propojení aktivní) ======
        if "🔗 Propojení časová řada ↔ HVG" in selected_sections and n_nodes > 0:
            st.subheader("📍 Časová řada se zvýrazněným vrcholem a sousedy")
            df_ts2 = pd.DataFrame({"index": np.arange(len(arr)), "value": arr})
            fig_ts2 = px.line(
                df_ts2, x="index", y="value", markers=True,
                title="Časová řada (highlight)",
                hover_data={"index": True, "value": ":.3f"}
            )
            fig_ts2.update_traces(marker_size=8)

            # vybraný vrchol
            fig_ts2.add_trace(go.Scatter(
                x=[selected_index],
                y=[arr[selected_index]],
                mode="markers",
                marker=dict(size=14, color="red"),
                name="Vybraný bod",
                hovertext=[f"Index: {selected_index}<br>Hodnota: {arr[selected_index]:.3f}"],
                hoverinfo="text"
            ))

            # sousedi
            if len(neighbors) > 0:
                fig_ts2.add_trace(go.Scatter(
                    x=neighbors,
                    y=[arr[i] for i in neighbors],
                    mode="markers",
                    marker=dict(size=12, color="orange"),
                    name="Sousedé",
                    hovertext=[f"Index: {i}<br>Hodnota: {arr[i]:.3f}" for i in neighbors],
                    hoverinfo="text"
                ))

            st.plotly_chart(fig_ts2, use_container_width=True)

        # =========================
        #  Konfigurační graf (null model)
        # =========================
        if "🔁 Konfigurační graf (null model)" in selected_sections:
            st.markdown("### 🔁 Konfigurační graf (null model)")

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

        # =========================
        #  „Kalkulačka“ – lokální analýza úseku časové řady
        # =========================
        if "🧮 Lokální analýza úseku časové řady" in selected_sections:
            st.subheader("🧮 Lokální analýza úseku časové řady")

            if len(arr) >= 2:
                i_start, i_end = st.slider(
                    "Vyber rozsah indexů [i_start, i_end] pro lokální analýzu",
                    min_value=0,
                    max_value=len(arr) - 1,
                    value=(0, min(len(arr) - 1, max(1, len(arr)//5))),
                )
                if i_start > i_end:
                    i_start, i_end = i_end, i_start
            else:
                i_start, i_end = 0, 0

            segment = arr[i_start:i_end + 1]
            st.write(
                f"- Délka úseku: **{len(segment)}**, "
                f"rozsah indexů: **[{i_start}, {i_end}]**"
            )

            if len(segment) > 0:
                ent = shannon_entropy(segment, bins="auto")
                st.write(
                    f"- Průměr (lokální): **{segment.mean():.3f}**  \n"
                    f"- Rozptyl (lokální): **{segment.var():.3f}**  \n"
                    f"- Min: **{segment.min():.3f}**, Max: **{segment.max():.3f}**  \n"
                    f"- Shannonova entropie (odhad): **{ent:.3f}**"
                )

                # Lokální HVG úseku
                if len(segment) >= 2:
                    G_seg = build_hvg(segment)
                    n_seg = G_seg.number_of_nodes()
                    m_seg = G_seg.number_of_edges()
                    degs_seg = [d for _, d in G_seg.degree()]
                    avg_deg_seg = float(np.mean(degs_seg)) if len(degs_seg) > 0 else 0.0

                    try:
                        C_seg = nx.average_clustering(G_seg)
                    except Exception:
                        C_seg = float("nan")

                    is_conn_seg = nx.is_connected(G_seg) if n_seg > 0 else False
                    L_seg = None
                    diam_seg = None
                    if is_conn_seg and n_seg > 1:
                        try:
                            L_seg = nx.average_shortest_path_length(G_seg)
                        except Exception:
                            L_seg = None
                        try:
                            diam_seg = nx.diameter(G_seg)
                        except Exception:
                            diam_seg = None

                    st.markdown("**Lokální HVG pro vybraný úsek**")
                    st.write(f"- Počet vrcholů: **{n_seg}**, počet hran: **{m_seg}**, průměrný stupeň: **{avg_deg_seg:.3f}**")
                    st.write(f"- Clustering (lokální): **{C_seg:.3f}**")
                    if L_seg is not None:
                        st.write(f"- Průměrná délka cesty (lokální): **{L_seg:.3f}**")
                    if diam_seg is not None:
                        st.write(f"- Průměr grafu (lokální): **{diam_seg}**")

        # =========================
        #  Výběr podgrafu z HVG
        # =========================
        if "🧩 Podgraf HVG" in selected_sections:
            st.subheader("🧩 Podgraf HVG podle vybraných vrcholů")

            sub_nodes_text = st.text_input(
                "Seznam vrcholů pro podgraf (indexy oddělené čárkou nebo mezerami)",
                value="0, 1, 2"
            )

            sub_nodes = []
            for token in re.split(r"[,\s;]+", sub_nodes_text):
                token = token.strip()
                if token == "":
                    continue
                try:
                    idx = int(token)
                    if 0 <= idx < n_nodes:
                        sub_nodes.append(idx)
                except ValueError:
                    continue

            sub_nodes = sorted(set(sub_nodes))

            if len(sub_nodes) > 0:
                G_sub = G.subgraph(sub_nodes).copy()
                st.write(f"Podgraf obsahuje **{G_sub.number_of_nodes()}** vrcholů a **{G_sub.number_of_edges()}** hran.")

                degs_sub = [d for _, d in G_sub.degree()]
                avg_deg_sub = float(np.mean(degs_sub)) if len(degs_sub) > 0 else 0.0

                try:
                    C_sub = nx.average_clustering(G_sub)
                except Exception:
                    C_sub = float("nan")

                is_conn_sub = nx.is_connected(G_sub) if G_sub.number_of_nodes() > 0 else False
                L_sub = None
                diam_sub = None
                if is_conn_sub and G_sub.number_of_nodes() > 1:
                    try:
                        L_sub = nx.average_shortest_path_length(G_sub)
                    except Exception:
                        L_sub = None
                    try:
                        diam_sub = nx.diameter(G_sub)
                    except Exception:
                        diam_sub = None

                st.write(f"- Průměrný stupeň v podgrafu: **{avg_deg_sub:.3f}**")
                st.write(f"- Clustering v podgrafu: **{C_sub:.3f}**")
                if L_sub is not None:
                    st.write(f"- Průměrná délka cesty v podgrafu: **{L_sub:.3f}**")
                if diam_sub is not None:
                    st.write(f"- Průměr podgrafu: **{diam_sub}**")

                # vizualizace podgrafu s původním layoutem
                edge_x_sub, edge_y_sub = [], []
                for u, v in G_sub.edges():
                    x0, y0 = pos[u]
                    x1, y1 = pos[v]
                    edge_x_sub += [x0, x1, None]
                    edge_y_sub += [y0, y1, None]

                node_x_sub, node_y_sub = [], []
                for node in G_sub.nodes():
                    x, y = pos[node]
                    node_x_sub.append(x)
                    node_y_sub.append(y)

                edge_trace_sub = go.Scatter(
                    x=edge_x_sub, y=edge_y_sub, mode="lines",
                    line=dict(width=1, color="#888"),
                    hoverinfo="none"
                )
                node_trace_sub = go.Scatter(
                    x=node_x_sub, y=node_y_sub,
                    mode="markers+text",
                    text=[str(n) for n in G_sub.nodes()],
                    textposition="bottom center",
                    marker=dict(size=10, color="lightcoral", line_width=1),
                    hoverinfo="text",
                    hovertext=[f"Vrchol: {n}" for n in G_sub.nodes()]
                )

                fig_sub = go.Figure(data=[edge_trace_sub, node_trace_sub])
                fig_sub.update_layout(
                    title="Podgraf HVG (vybrané vrcholy)",
                    showlegend=False, hovermode="closest",
                    margin=dict(b=20, l=5, r=5, t=40)
                )
                st.plotly_chart(fig_sub, use_container_width=True)

        st.markdown("---")

        # =========================
        #  Histogram + power-law
        # =========================
        if "📉 Rozdělení stupňů + power-law" in selected_sections:
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

        # =========================
        #  Arc diagram HVG
        # =========================
        if "🎨 Arc Diagram HVG" in selected_sections:
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

        # =========================
        #  Export HVG a metrik
        # =========================
        if "💾 Export HVG a metrik" in selected_sections:
            st.subheader("💾 Export HVG a metrik")

            # edge list
            edges_df = pd.DataFrame(list(G.edges()), columns=["source", "target"])
            edges_csv = edges_df.to_csv(index=False).encode("utf-8")

            # adjacency matrix
            adj_df = nx.to_pandas_adjacency(G)
            adj_csv = adj_df.to_csv().encode("utf-8")

            # metriky
            metrics_dict = {
                "n_nodes": n_nodes,
                "n_edges": n_edges,
                "avg_degree": avg_deg,
                "C": C,
                "L": L,
                "diameter": diam,
                "assortativity": assort,
                "L_rand": L_rand,
                "C_rand": C_rand,
                "sigma": sigma_sw,
            }
            metrics_df = pd.DataFrame([metrics_dict])
            metrics_csv = metrics_df.to_csv(index=False).encode("utf-8")

            col_exp1, col_exp2, col_exp3 = st.columns(3)
            with col_exp1:
                st.download_button(
                    "⬇️ Exportovat HVG jako edge list (CSV)",
                    data=edges_csv,
                    file_name="hvg_edgelist.csv",
                    mime="text/csv"
                )
            with col_exp2:
                st.download_button(
                    "⬇️ Exportovat HVG jako adjacency matrix (CSV)",
                    data=adj_csv,
                    file_name="hvg_adjacency.csv",
                    mime="text/csv"
                )
            with col_exp3:
                st.download_button(
                    "⬇️ Exportovat metriky HVG (CSV)",
                    data=metrics_csv,
                    file_name="hvg_metrics.csv",
                    mime="text/csv"
                )

# =====================================================================
#  REŽIM 2: VLASTNÍ GRAF Z NODE/EDGE LISTU NEBO CSV
# =====================================================================

elif analysis_mode == "Vlastní graf (ruční / CSV)":
    st.sidebar.subheader("Vlastní graf – vstup")

    input_mode = st.sidebar.radio(
        "Způsob zadání grafu",
        ["Node list", "Edge list", "Node + Edge list", "CSV (edge list)"]
    )

    custom_graph = None

    if input_mode == "Node list":
        nodes_text = st.sidebar.text_area(
            "Seznam vrcholů (oddělené čárkou, mezerou nebo novým řádkem)",
            value="1, 2, 3, 4"
        )
        if st.sidebar.button("Vytvořit graf z node listu"):
            tokens = [t.strip() for t in re.split(r"[,\s;]+", nodes_text) if t.strip() != ""]
            Gc = nx.Graph()
            Gc.add_nodes_from(tokens)
            custom_graph = Gc

    elif input_mode == "Edge list":
        edges_text = st.sidebar.text_area(
            "Seznam hran – každá hrana na novém řádku ve formátu `u,v` nebo `u v`",
            value="1,2\n2,3\n3,4"
        )
        if st.sidebar.button("Vytvořit graf z edge listu"):
            Gc = nx.Graph()
            for line in edges_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in re.split(r"[,\s;]+", line) if p.strip() != ""]
                if len(parts) >= 2:
                    u, v = parts[0], parts[1]
                    Gc.add_edge(u, v)
            custom_graph = Gc

    elif input_mode == "Node + Edge list":
        nodes_text = st.sidebar.text_area(
            "Seznam vrcholů (oddělené čárkou, mezerou nebo novým řádkem)",
            value="1, 2, 3, 4, 5"
        )
        edges_text = st.sidebar.text_area(
            "Seznam hran – každá hrana na novém řádku ve formátu `u,v` nebo `u v`",
            value="1,2\n2,3\n3,4\n4,5"
        )
        if st.sidebar.button("Vytvořit graf z node + edge listu"):
            tokens = [t.strip() for t in re.split(r"[,\s;]+", nodes_text) if t.strip() != ""]
            Gc = nx.Graph()
            Gc.add_nodes_from(tokens)
            for line in edges_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in re.split(r"[,\s;]+", line) if p.strip() != ""]
                if len(parts) >= 2:
                    u, v = parts[0], parts[1]
                    Gc.add_edge(u, v)
            custom_graph = Gc

    else:  # "CSV (edge list)"
        st.sidebar.write("Očekává se CSV se **dvěma sloupci**: zdroj a cíl hrany (edge list).")
        uploaded_edges = st.sidebar.file_uploader(
            "Nahraj CSV s edge listem", type="csv", key="csv_edges_uploader"
        )
        if uploaded_edges is not None:
            df_edges = pd.read_csv(uploaded_edges)
            if df_edges.shape[1] < 2:
                st.sidebar.error("CSV musí mít alespoň dva sloupce (source, target).")
            else:
                col1 = st.sidebar.selectbox("Sloupec se zdrojem (source)", df_edges.columns, index=0)
                col2 = st.sidebar.selectbox(
                    "Sloupec s cílem (target)",
                    df_edges.columns,
                    index=1 if df_edges.shape[1] > 1 else 0
                )
                if st.sidebar.button("Vytvořit graf z CSV edge listu"):
                    Gc = nx.Graph()
                    for _, row in df_edges.iterrows():
                        u = str(row[col1])
                        v = str(row[col2])
                        Gc.add_edge(u, v)
                    custom_graph = Gc

    # Uložíme, pokud jsme něco vytvořili
    if custom_graph is not None:
        st.session_state.custom_graph = custom_graph

    # Hlavní obsah pro vlastní graf
    st.markdown("## 🧮 Vlastní graf (analýza)")

    if st.session_state.custom_graph is not None:
        Gc = st.session_state.custom_graph

        st.markdown("### 🧷 Metriky a vizualizace vlastního grafu")

        n_nodes_c = Gc.number_of_nodes()
        n_edges_c = Gc.number_of_edges()
        degrees_c = [d for _, d in Gc.degree()]
        avg_deg_c = float(np.mean(degrees_c)) if len(degrees_c) > 0 else 0.0

        # Clustering
        try:
            C_c = nx.average_clustering(Gc)
        except Exception:
            C_c = float("nan")

        # Souvislost, délka cest, průměr
        is_conn_c = nx.is_connected(Gc) if n_nodes_c > 0 else False
        L_c = None
        diam_c = None
        if is_conn_c and n_nodes_c > 1:
            try:
                L_c = nx.average_shortest_path_length(Gc)
            except Exception:
                L_c = None
            try:
                diam_c = nx.diameter(Gc)
            except Exception:
                diam_c = None

        # Náhodný graf pro porovnání
        L_rand_c = None
        C_rand_c = None
        if n_nodes_c > 1 and avg_deg_c > 1:
            try:
                L_rand_c = np.log(n_nodes_c) / np.log(avg_deg_c)
                C_rand_c = avg_deg_c / n_nodes_c
            except Exception:
                L_rand_c = None
                C_rand_c = None

        analyzer_c = SmallWorldAnalyzer(C_c, L_c, C_rand_c, L_rand_c)
        sigma_c = analyzer_c.sigma

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("**Základní metriky vlastního grafu**")
            st.write(f"- Počet vrcholů: **{n_nodes_c}**")
            st.write(f"- Počet hran: **{n_edges_c}**")
            st.write(f"- Průměrný stupeň: **{avg_deg_c:.3f}**")
            if L_c is not None:
                st.write(f"- Průměrná délka cesty L: **{L_c:.3f}**")
            else:
                st.write("- Průměrná délka cesty L: *nelze spočítat (nesouvislý nebo příliš malý graf)*")
            if diam_c is not None:
                st.write(f"- Průměr grafu (diameter): **{diam_c}**")
            else:
                st.write("- Průměr grafu (diameter): *není k dispozici*")

        with col_c2:
            st.markdown("**Clustering a small-world charakter (vlastní graf)**")
            st.write(f"- Clustering coefficient C: **{C_c:.3f}**")
            if L_rand_c is not None and C_rand_c is not None and C_rand_c != 0:
                st.write(
                    "- Náhodný graf (pro porovnání):  \n"
                    f"  - L_rand ≈ **{L_rand_c:.3f}**  \n"
                    f"  - C_rand ≈ **{C_rand_c:.5f}**"
                )
            else:
                st.write("- Náhodný graf (L_rand, C_rand): *nelze odhadnout*")

            if sigma_c is not None and not np.isnan(sigma_c):
                st.write(
                    f"- Small-world index σ "
                    f"(σ > 1: small-world, σ ≈ 1: podobné náhodnému grafu, σ < 1: není small-world): "
                    f"**{sigma_c:.2f}**"
                )
                level_c, msg_c = analyzer_c.interpretation(atol=0.05)
                if level_c == "success":
                    st.success(msg_c)
                elif level_c == "warning":
                    st.warning(msg_c)
                else:
                    st.info(msg_c)
            else:
                st.write(
                    "- Small-world index σ: *nelze spočítat "
                    "(chybí některá z metrik L, C, L_rand nebo C_rand nebo je výsledek nespolehlivý)*"
                )

        # Vizualizace vlastního grafu
        st.subheader("🕸️ Vizuální zobrazení vlastního grafu")

        if n_nodes_c > 0:
            pos_c = nx.spring_layout(Gc, seed=42)
            edge_x_c, edge_y_c = [], []
            for u, v in Gc.edges():
                x0, y0 = pos_c[u]
                x1, y1 = pos_c[v]
                edge_x_c += [x0, x1, None]
                edge_y_c += [y0, y1, None]

            edge_trace_c = go.Scatter(
                x=edge_x_c, y=edge_y_c, mode='lines',
                line=dict(width=1, color='#888'), hoverinfo='none'
            )

            node_x_c, node_y_c, node_text_c = [], [], []
            for node in Gc.nodes():
                x, y = pos_c[node]
                node_x_c.append(x)
                node_y_c.append(y)
                node_text_c.append(f"Vrchol: {node}<br>Stupeň: {Gc.degree(node)}")

            node_trace_c = go.Scatter(
                x=node_x_c, y=node_y_c, mode='markers+text',
                text=[str(n) for n in Gc.nodes()],
                textposition="bottom center",
                hoverinfo='text', hovertext=node_text_c,
                marker=dict(size=10, color='orange', line_width=1),
                textfont=dict(size=10, color="black")
            )

            fig_custom = go.Figure(data=[edge_trace_c, node_trace_c])
            fig_custom.update_layout(
                title="Vlastní graf (node/edge list nebo CSV)",
                showlegend=False, hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40)
            )
            st.plotly_chart(fig_custom, use_container_width=True)
        else:
            st.info("Graf neobsahuje žádné vrcholy – zadej alespoň jeden vrchol nebo hranu.")
    else:
        st.info("👈 Nejprve zadej vlastní graf v levém panelu (node/edge list nebo CSV).")

# =====================================================================
#  REŽIM 3: POROVNÁNÍ DVOU ČASOVÝCH ŘAD / HVG
# =====================================================================

else:  # "Porovnat dvě časové řady"
    st.markdown("## ⚖️ Porovnání dvou časových řad a jejich HVG")

    if st.session_state.data is None:
        st.info("Nejdřív vygeneruj časovou řadu v režimu **„Časová řada → HVG“**. "
                "Tahle série pak bude použita jako *Série 1* pro porovnání.")
    else:
        # =============================
        # Série 1 = už vygenerovaná časová řada
        # =============================
        data1 = st.session_state.data
        st.markdown("### Série 1 – aktuálně vygenerovaná časová řada")

        st.write(
            f"- Délka: **{len(data1)}**, "
            f"Průměr: **{data1.mean():.3f}**, "
            f"Rozptyl: **{data1.var():.3f}**"
        )

        # HVG pro sérii 1
        G1 = build_hvg(data1)
        n1 = G1.number_of_nodes()
        m1 = G1.number_of_edges()
        degs1 = [d for _, d in G1.degree()]
        avg_deg1 = float(np.mean(degs1)) if len(degs1) > 0 else 0.0
        try:
            C1 = nx.average_clustering(G1)
        except Exception:
            C1 = float("nan")
        is_conn1 = nx.is_connected(G1) if n1 > 0 else False
        L1 = None
        diam1 = None
        if is_conn1 and n1 > 1:
            try:
                L1 = nx.average_shortest_path_length(G1)
            except Exception:
                L1 = None
            try:
                diam1 = nx.diameter(G1)
            except Exception:
                diam1 = None
        L_rand1 = np.log(n1) / np.log(avg_deg1) if n1 > 1 and avg_deg1 > 1 else None
        C_rand1 = avg_deg1 / n1 if n1 > 0 else None
        try:
            assort1 = nx.degree_assortativity_coefficient(G1)
        except Exception:
            assort1 = None
        analyzer1 = SmallWorldAnalyzer(C1, L1, C_rand1, L_rand1)
        sigma1 = analyzer1.sigma

        # =============================
        # Sidebar – nastavení série 2
        # =============================
        st.sidebar.subheader("Série 2 – nastavení")

        src2 = st.sidebar.selectbox(
            "Zdroj série 2",
            ["Nahrát CSV", "Ruční vstup", "Náhodná normální", "Sinusovka", "Chaotický generátor"],
            index=0
        )

        data2_candidate = None

        if src2 == "Nahrát CSV":
            file2 = st.sidebar.file_uploader("CSV pro sérii 2", type="csv", key="csv_cmp_2")
            if file2 is not None:
                df2 = pd.read_csv(file2)
                data2_candidate = df2.iloc[:, 0].values

        elif src2 == "Ruční vstup":
            txt2 = st.sidebar.text_area("Hodnoty série 2 (čárka)", "2, 4, 6, 8, 10")
            try:
                data2_candidate = np.array([float(v.strip()) for v in txt2.split(",")])
            except ValueError:
                st.sidebar.error("Chybný formát série 2.")

        elif src2 == "Náhodná normální":
            length2 = st.sidebar.slider("Délka série 2", 10, 1000, 100, key="len_cmp2")
            mu2 = st.sidebar.number_input("μ (série 2)", value=0.0, key="mu_cmp2")
            sigma2 = st.sidebar.number_input("σ (série 2)", value=1.0, key="sigma_cmp2")
            data2_candidate = np.random.normal(mu2, sigma2, size=length2)

        elif src2 == "Sinusovka":
            length2 = st.sidebar.slider("Délka série 2", 10, 1000, 200, key="len_sin2")
            amp2 = st.sidebar.number_input("Amplituda 2", value=1.0, key="amp_sin2")
            freq2 = st.sidebar.number_input("Frekvence 2", value=1.0, key="frq_sin2")
            x2 = np.arange(length2)
            data2_candidate = amp2 * np.sin(2 * np.pi * freq2 * x2 / length2)

        else:  # Chaotický generátor – série 2
            chaos2 = st.sidebar.selectbox(
                "Typ chaotického generátoru (série 2)",
                ["Logistická mapa", "Henonova mapa", "Lorenzův systém (x-složka)", "1/f šum (pink noise)"],
                key="chaos_type_2"
            )

            if chaos2 == "Logistická mapa":
                length2 = st.sidebar.slider("Délka série 2", 100, 5000, 1000, step=100, key="len_log_2")
                r2 = st.sidebar.slider("Parametr r (série 2)", 3.5, 4.0, 3.9, step=0.01, key="r_log_2")
                x02 = st.sidebar.number_input("Počáteční x₀ (série 2)", min_value=0.0, max_value=1.0, value=0.2, step=0.01, key="x0_log_2")
                burn2 = st.sidebar.number_input("Burn-in iterace (série 2)", 100, 10000, 500, step=100, key="burn_log_2")
                data2_candidate = generate_logistic_map(length2, r=r2, x0=x02, burn=burn2)

            elif chaos2 == "Henonova mapa":
                length2 = st.sidebar.slider("Délka série 2", 100, 5000, 1000, step=100, key="len_hen_2")
                a2 = st.sidebar.number_input("Parametr a (série 2)", value=1.4, step=0.1, key="a_hen_2")
                b2 = st.sidebar.number_input("Parametr b (série 2)", value=0.3, step=0.05, key="b_hen_2")
                x02 = st.sidebar.number_input("Počáteční x₀ (série 2)", value=0.1, step=0.05, key="x0_hen_2")
                y02 = st.sidebar.number_input("Počáteční y₀ (série 2)", value=0.0, step=0.05, key="y0_hen_2")
                burn2 = st.sidebar.number_input("Burn-in iterace (série 2)", 100, 10000, 500, step=100, key="burn_hen_2")
                data2_candidate = generate_henon_map(length2, a=a2, b=b2, x0=x02, y0=y02, burn=burn2)

            elif chaos2 == "Lorenzův systém (x-složka)":
                length2 = st.sidebar.slider("Délka série 2", 200, 10000, 2000, step=200, key="len_lor_2")
                dt2 = st.sidebar.number_input("Krok integrace dt (série 2)", value=0.01, step=0.005, format="%.3f", key="dt_lor_2")
                sigma_l2 = st.sidebar.number_input("σ (série 2)", value=10.0, step=1.0, key="sigma_lor_2")
                rho_l2 = st.sidebar.number_input("ρ (série 2)", value=28.0, step=1.0, key="rho_lor_2")
                beta_l2 = st.sidebar.number_input("β (série 2)", value=8/3, step=0.1, key="beta_lor_2")
                burn2 = st.sidebar.number_input("Burn-in kroků (série 2)", 500, 20000, 1000, step=500, key="burn_lor_2")
                data2_candidate = generate_lorenz_x(length2, dt=dt2,
                                                    sigma=sigma_l2, rho=rho_l2, beta=beta_l2,
                                                    burn=burn2)

            else:  # 1/f šum
                length2 = st.sidebar.slider("Délka série 2", 100, 10000, 2000, step=100, key="len_pink_2")
                data2_candidate = generate_pink_noise(length2)

        generate2 = st.sidebar.button("Načíst / generovat sérii 2")

        if generate2:
            if data2_candidate is None:
                st.sidebar.error("Série 2 zatím není připravená – zkontroluj nastavení / CSV.")
            else:
                st.session_state.data2 = data2_candidate

        data2 = st.session_state.data2

        if data2 is None:
            st.info("👈 V levém panelu nastav parametry **Série 2** a klikni na "
                    "**„Načíst / generovat sérii 2“**.")
        else:
            # =============================
            # Série 2 – výpočet HVG a metrik
            # =============================
            st.markdown("### Série 2 – nastavená v levém panelu")

            st.write(
                f"- Délka: **{len(data2)}**, "
                f"Průměr: **{data2.mean():.3f}**, "
                f"Rozptyl: **{data2.var():.3f}**"
            )

            G2 = build_hvg(data2)
            n2 = G2.number_of_nodes()
            m2 = G2.number_of_edges()
            degs2 = [d for _, d in G2.degree()]
            avg_deg2 = float(np.mean(degs2)) if len(degs2) > 0 else 0.0
            try:
                C2 = nx.average_clustering(G2)
            except Exception:
                C2 = float("nan")
            is_conn2 = nx.is_connected(G2) if n2 > 0 else False
            L2 = None
            diam2 = None
            if is_conn2 and n2 > 1:
                try:
                    L2 = nx.average_shortest_path_length(G2)
                except Exception:
                    L2 = None
                try:
                    diam2 = nx.diameter(G2)
                except Exception:
                    diam2 = None
            L_rand2 = np.log(n2) / np.log(avg_deg2) if n2 > 1 and avg_deg2 > 1 else None
            C_rand2 = avg_deg2 / n2 if n2 > 0 else None
            try:
                assort2 = nx.degree_assortativity_coefficient(G2)
            except Exception:
                assort2 = None
            analyzer2 = SmallWorldAnalyzer(C2, L2, C_rand2, L_rand2)
            sigma2 = analyzer2.sigma

            # =============================
            # Společný výběr sekcí pro obě HVG
            # =============================
            section_options_cmp = [
                "📊 Metriky HVG",
                "📉 Rozdělení stupňů",
                "🎨 Arc Diagram HVG",
                "💾 Export HVG a metrik",
            ]
            selected_sections_cmp = st.multiselect(
                "Co chceš pod porovnáním zobrazit pro **obě** HVG?",
                options=section_options_cmp,
                default=section_options_cmp  # všechno defaultně
            )

            # =============================
            # Časové řady vedle sebe
            # =============================
            st.markdown("### 📈 Časové řady vedle sebe")

            col_ts1, col_ts2 = st.columns(2)
            with col_ts1:
                df1 = pd.DataFrame({"index": np.arange(len(data1)), "value": data1})
                fig1 = px.line(df1, x="index", y="value", markers=True, title="Série 1")
                fig1.update_traces(marker_size=6)
                st.plotly_chart(fig1, use_container_width=True)
            with col_ts2:
                df2 = pd.DataFrame({"index": np.arange(len(data2)), "value": data2})
                fig2 = px.line(df2, x="index", y="value", markers=True, title="Série 2")
                fig2.update_traces(marker_size=6)
                st.plotly_chart(fig2, use_container_width=True)

            # =============================
            # HVG vizualizace vedle sebe
            # =============================
            st.markdown("### 🕸️ HVG grafy vedle sebe")

            col_g1, col_g2 = st.columns(2)
            with col_g1:
                pos1 = nx.spring_layout(G1, seed=42)
                edge_x1, edge_y1 = [], []
                for u, v in G1.edges():
                    x0, y0 = pos1[u]
                    x1_, y1_ = pos1[v]
                    edge_x1 += [x0, x1_, None]
                    edge_y1 += [y0, y1_, None]
                edge_trace1 = go.Scatter(
                    x=edge_x1, y=edge_y1, mode="lines",
                    line=dict(width=1, color="#888"),
                    hoverinfo="none"
                )
                node_x1, node_y1 = [], []
                for node in G1.nodes():
                    x, y = pos1[node]
                    node_x1.append(x)
                    node_y1.append(y)
                node_trace1 = go.Scatter(
                    x=node_x1, y=node_y1, mode="markers",
                    marker=dict(size=10, color="skyblue"),
                    hoverinfo="none"
                )
                fig_g1 = go.Figure(data=[edge_trace1, node_trace1])
                fig_g1.update_layout(
                    title="HVG – série 1",
                    showlegend=False, hovermode="closest",
                    margin=dict(b=20, l=5, r=5, t=40)
                )
                st.plotly_chart(fig_g1, use_container_width=True)

            with col_g2:
                pos2 = nx.spring_layout(G2, seed=42)
                edge_x2, edge_y2 = [], []
                for u, v in G2.edges():
                    x0, y0 = pos2[u]
                    x2_, y2_ = pos2[v]
                    edge_x2 += [x0, x2_, None]
                    edge_y2 += [y0, y2_, None]
                edge_trace2 = go.Scatter(
                    x=edge_x2, y=edge_y2, mode="lines",
                    line=dict(width=1, color="#888"),
                    hoverinfo="none"
                )
                node_x2, node_y2 = [], []
                for node in G2.nodes():
                    x, y = pos2[node]
                    node_x2.append(x)
                    node_y2.append(y)
                node_trace2 = go.Scatter(
                    x=node_x2, y=node_y2, mode="markers",
                    marker=dict(size=10, color="lightgreen"),
                    hoverinfo="none"
                )
                fig_g2 = go.Figure(data=[edge_trace2, node_trace2])
                fig_g2.update_layout(
                    title="HVG – série 2",
                    showlegend=False, hovermode="closest",
                    margin=dict(b=20, l=5, r=5, t=40)
                )
                st.plotly_chart(fig_g2, use_container_width=True)

            # =============================
            # 📊 Metriky HVG
            # =============================
            if "📊 Metriky HVG" in selected_sections_cmp:
                st.markdown("### 📊 Porovnání metrik HVG")

                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.markdown("**Série 1 – metriky HVG**")
                    st.write(f"- Počet vrcholů: **{n1}**")
                    st.write(f"- Počet hran: **{m1}**")
                    st.write(f"- Průměrný stupeň: **{avg_deg1:.3f}**")
                    if L1 is not None:
                        st.write(f"- Průměrná délka cesty L: **{L1:.3f}**")
                    else:
                        st.write("- Průměrná délka cesty L: *nelze spočítat (nesouvislý graf)*")
                    if diam1 is not None:
                        st.write(f"- Průměr grafu (diameter): **{diam1}**")
                    else:
                        st.write("- Průměr grafu (diameter): *není k dispozici*")
                    st.write(f"- Clustering coefficient C: **{C1:.3f}**")
                    if assort1 is not None and not np.isnan(assort1):
                        st.write(f"- Degree assortativity: **{assort1:.3f}**")
                    else:
                        st.write("- Degree assortativity: *není k dispozici*")
                    if L_rand1 is not None and C_rand1 is not None and C_rand1 != 0:
                        st.write(
                            "- Náhodný graf:  \n"
                            f"  - L_rand ≈ **{L_rand1:.3f}**  \n"
                            f"  - C_rand ≈ **{C_rand1:.5f}**"
                        )
                    if sigma1 is not None and not np.isnan(sigma1):
                        st.write(f"- Small-world index σ: **{sigma1:.2f}**")
                        level1, msg1 = analyzer1.interpretation(atol=0.05)
                        if level1 == "success":
                            st.success(msg1)
                        elif level1 == "warning":
                            st.warning(msg1)
                        else:
                            st.info(msg1)

                with col_m2:
                    st.markdown("**Série 2 – metriky HVG**")
                    st.write(f"- Počet vrcholů: **{n2}**")
                    st.write(f"- Počet hran: **{m2}**")
                    st.write(f"- Průměrný stupeň: **{avg_deg2:.3f}**")
                    if L2 is not None:
                        st.write(f"- Průměrná délka cesty L: **{L2:.3f}**")
                    else:
                        st.write("- Průměrná délka cesty L: *nelze spočítat (nesouvislý graf)*")
                    if diam2 is not None:
                        st.write(f"- Průměr grafu (diameter): **{diam2}**")
                    else:
                        st.write("- Průměr grafu (diameter): *není k dispozici*")
                    st.write(f"- Clustering coefficient C: **{C2:.3f}**")
                    if assort2 is not None and not np.isnan(assort2):
                        st.write(f"- Degree assortativity: **{assort2:.3f}**")
                    else:
                        st.write("- Degree assortativity: *není k dispozici*")
                    if L_rand2 is not None and C_rand2 is not None and C_rand2 != 0:
                        st.write(
                            "- Náhodný graf:  \n"
                            f"  - L_rand ≈ **{L_rand2:.3f}**  \n"
                            f"  - C_rand ≈ **{C_rand2:.5f}**"
                        )
                    if sigma2 is not None and not np.isnan(sigma2):
                        st.write(f"- Small-world index σ: **{sigma2:.2f}**")
                        level2, msg2 = analyzer2.interpretation(atol=0.05)
                        if level2 == "success":
                            st.success(msg2)
                        elif level2 == "warning":
                            st.warning(msg2)
                        else:
                            st.info(msg2)

            # =============================
            # 📉 Porovnání stupňového rozdělení
            # =============================
            if "📉 Rozdělení stupňů" in selected_sections_cmp:
                st.markdown("### 📉 Porovnání stupňového rozdělení")

                df_deg_cmp = pd.DataFrame({
                    "degree": degs1 + degs2,
                    "serie": (["Série 1"] * len(degs1)) + (["Série 2"] * len(degs2))
                })

                max_deg = max(
                    max(degs1) if len(degs1) > 0 else 1,
                    max(degs2) if len(degs2) > 0 else 1
                )

                fig_deg_cmp = px.histogram(
                    df_deg_cmp,
                    x="degree",
                    color="serie",
                    barmode="overlay",
                    opacity=0.6,
                    nbins=max_deg + 1,
                    title="Histogram stupňů – série 1 vs. série 2",
                    labels={"degree": "Stupeň", "count": "Počet vrcholů"}
                )
                fig_deg_cmp.update_layout(yaxis_title="Počet vrcholů")
                st.plotly_chart(fig_deg_cmp, use_container_width=True)

            # =============================
            # 🎨 Arc Diagram pro obě HVG
            # =============================
            if "🎨 Arc Diagram HVG" in selected_sections_cmp:
                st.markdown("### 🎨 Arc Diagramy HVG – porovnání")

                col_arc1, col_arc2 = st.columns(2)

                with col_arc1:
                    n = len(data1)
                    node_x_line = np.arange(n)
                    node_y_line = np.zeros(n)
                    fig_arc1 = go.Figure()

                    for i, j in G1.edges():
                        r = (j - i) / 2
                        mid = i + r
                        theta = np.linspace(0, np.pi, 100)
                        x_arc = mid + r * np.cos(theta)
                        y_arc = r * np.sin(theta)
                        fig_arc1.add_trace(go.Scatter(
                            x=x_arc, y=y_arc, mode='lines',
                            line=dict(color='gray', width=1),
                            hoverinfo='none'
                        ))

                    fig_arc1.add_trace(go.Scatter(
                        x=node_x_line, y=node_y_line, mode='markers',
                        marker=dict(size=8, color='skyblue'),
                        hoverinfo='text',
                        hovertext=[f"Index: {i}<br>Hodnota: {data1[i]:.3f}" for i in node_x_line]
                    ))

                    fig_arc1.update_layout(
                        title="Arc Diagram HVG – série 1",
                        showlegend=False,
                        xaxis=dict(showgrid=False, zeroline=False, title="Index"),
                        yaxis=dict(showgrid=False, zeroline=False, visible=False),
                        margin=dict(b=20, l=5, r=5, t=40),
                        height=300
                    )
                    st.plotly_chart(fig_arc1, use_container_width=True)

                with col_arc2:
                    n = len(data2)
                    node_x_line = np.arange(n)
                    node_y_line = np.zeros(n)
                    fig_arc2 = go.Figure()

                    for i, j in G2.edges():
                        r = (j - i) / 2
                        mid = i + r
                        theta = np.linspace(0, np.pi, 100)
                        x_arc = mid + r * np.cos(theta)
                        y_arc = r * np.sin(theta)
                        fig_arc2.add_trace(go.Scatter(
                            x=x_arc, y=y_arc, mode='lines',
                            line=dict(color='gray', width=1),
                            hoverinfo='none'
                        ))

                    fig_arc2.add_trace(go.Scatter(
                        x=node_x_line, y=node_y_line, mode='markers',
                        marker=dict(size=8, color='lightgreen'),
                        hoverinfo='text',
                        hovertext=[f"Index: {i}<br>Hodnota: {data2[i]:.3f}" for i in node_x_line]
                    ))

                    fig_arc2.update_layout(
                        title="Arc Diagram HVG – série 2",
                        showlegend=False,
                        xaxis=dict(showgrid=False, zeroline=False, title="Index"),
                        yaxis=dict(showgrid=False, zeroline=False, visible=False),
                        margin=dict(b=20, l=5, r=5, t=40),
                        height=300
                    )
                    st.plotly_chart(fig_arc2, use_container_width=True)

            # =============================
            # 💾 Export HVG a metrik pro obě série
            # =============================
            if "💾 Export HVG a metrik" in selected_sections_cmp:
                st.markdown("### 💾 Export HVG a metrik pro obě série")

                # Série 1
                edges_df1 = pd.DataFrame(list(G1.edges()), columns=["source", "target"])
                edges_csv1 = edges_df1.to_csv(index=False).encode("utf-8")
                adj_df1 = nx.to_pandas_adjacency(G1)
                adj_csv1 = adj_df1.to_csv().encode("utf-8")
                metrics_dict1 = {
                    "n_nodes": n1,
                    "n_edges": m1,
                    "avg_degree": avg_deg1,
                    "C": C1,
                    "L": L1,
                    "diameter": diam1,
                    "assortativity": assort1,
                    "L_rand": L_rand1,
                    "C_rand": C_rand1,
                    "sigma": sigma1,
                }
                metrics_df1 = pd.DataFrame([metrics_dict1])
                metrics_csv1 = metrics_df1.to_csv(index=False).encode("utf-8")

                # Série 2
                edges_df2 = pd.DataFrame(list(G2.edges()), columns=["source", "target"])
                edges_csv2 = edges_df2.to_csv(index=False).encode("utf-8")
                adj_df2 = nx.to_pandas_adjacency(G2)
                adj_csv2 = adj_df2.to_csv().encode("utf-8")
                metrics_dict2 = {
                    "n_nodes": n2,
                    "n_edges": m2,
                    "avg_degree": avg_deg2,
                    "C": C2,
                    "L": L2,
                    "diameter": diam2,
                    "assortativity": assort2,
                    "L_rand": L_rand2,
                    "C_rand": C_rand2,
                    "sigma": sigma2,
                }
                metrics_df2 = pd.DataFrame([metrics_dict2])
                metrics_csv2 = metrics_df2.to_csv(index=False).encode("utf-8")

                col_exp1, col_exp2 = st.columns(2)

                with col_exp1:
                    st.markdown("**Série 1 – exporty**")
                    st.download_button(
                        "⬇️ HVG (edge list, CSV) – série 1",
                        data=edges_csv1,
                        file_name="hvg_series1_edgelist.csv",
                        mime="text/csv"
                    )
                    st.download_button(
                        "⬇️ HVG (adjacency matrix, CSV) – série 1",
                        data=adj_csv1,
                        file_name="hvg_series1_adjacency.csv",
                        mime="text/csv"
                    )
                    st.download_button(
                        "⬇️ Metriky HVG – série 1",
                        data=metrics_csv1,
                        file_name="hvg_series1_metrics.csv",
                        mime="text/csv"
                    )

                with col_exp2:
                    st.markdown("**Série 2 – exporty**")
                    st.download_button(
                        "⬇️ HVG (edge list, CSV) – série 2",
                        data=edges_csv2,
                        file_name="hvg_series2_edgelist.csv",
                        mime="text/csv"
                    )
                    st.download_button(
                        "⬇️ HVG (adjacency matrix, CSV) – série 2",
                        data=adj_csv2,
                        file_name="hvg_series2_adjacency.csv",
                        mime="text/csv"
                    )
                    st.download_button(
                        "⬇️ Metriky HVG – série 2",
                        data=metrics_csv2,
                        file_name="hvg_series2_metrics.csv",
                        mime="text/csv"
                    )
