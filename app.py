import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go

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

    # Statistiky
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
    pos = nx.spring_layout(G, seed=42)

    # Edges
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

    # Nodes
    node_x, node_y, node_text = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        neigh = list(G.adj[node])
        node_text.append(f"Index: {node}<br>Stupeň: {len(neigh)}<br>Sousedé: {neigh}")

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        text=[str(n) for n in G.nodes()], textposition="bottom center",
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

    # Histogram stupňů
    degs = [d for _, d in G.degree()]
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

    # Arc diagram HVG
    st.subheader("🎨 Arc diagram HVG")
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
