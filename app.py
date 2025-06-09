import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go

# --- Funkce pro generování HVG ---
def build_hvg(data):
    G = nx.Graph()
    n = len(data)
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if all(data[k] < data[i] and data[k] < data[j] for k in range(i + 1, j)):
                G.add_edge(i, j)
    return G

# --- Inicializace session state ---
for key in ("data", "show_hvg", "show_direct", "show_horiz"):
    if key not in st.session_state:
        st.session_state[key] = None if key == "data" else False

st.set_page_config(page_title="HVG Vizualizátor", layout="wide")

# --- Hlavička ---
st.title("📊 HVG Vizualizátor")
st.markdown("**Interaktivní vizualizace časových řad a jejich HVG**")

# --- Sidebar: vstup dat ---
st.sidebar.title("🔧 Nastavení dat")
typ = st.sidebar.selectbox("Vyber typ časové řady", [
    "Náhodná uniformní", "Náhodná normální", "Sinusovka",
    "Nahrát CSV", "Ruční vstup"
])

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

generate = st.sidebar.button("Načíst / generovat řadu")

# --- Generování dat ---
if generate:
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
    else:
        data = None

    st.session_state.data = data
    st.session_state.show_hvg = False
    st.session_state.show_direct = False
    st.session_state.show_horiz = False

# --- Zobrazení časové řady + linky HVG ---
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
        G = build_hvg(arr)
        shapes = []
        for i, j in G.edges():
            shapes.append(dict(
                type="line",
                x0=i, y0=arr[i], x1=j, y1=arr[j],
                line=dict(color="gray", width=1)
            ))
        fig_ts.update_layout(shapes=shapes)

    # Vodorovné linky
    if st.session_state.show_horiz:
        G = build_hvg(arr)
        shapes = []
        for i, j in G.edges():
            y = min(arr[i], arr[j])
            shapes.append(dict(
                type="line",
                x0=i, y0=y, x1=j, y1=y,
                line=dict(color="gray", width=1)
            ))
        fig_ts.update_layout(shapes=shapes)

    st.plotly_chart(fig_ts, use_container_width=True)

    # Statistiky
    st.write(f"- Délka: **{len(arr)}**, Průměr: **{arr.mean():.3f}**, Rozptyl: **{arr.var():.3f}**")

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

# --- Interaktivní HVG + histogram ---
if st.session_state.show_hvg and st.session_state.data is not None:
    arr = st.session_state.data
    G = build_hvg(arr)

    st.subheader("🕸️ Interaktivní vizualizace HVG")
    pos = nx.spring_layout(G, seed=42)

    # Edges
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines',
                            line=dict(width=1, color='#888'), hoverinfo='none')

    # Nodes
    node_x, node_y, node_text = [], [], []
    for node in G.nodes():
        x, y = pos[node]; node_x.append(x); node_y.append(y)
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
    fig_hist = px.histogram(df_deg, x="degree", nbins=max(degs) + 1,
                            title="Histogram stupňů", labels={"degree": "Stupeň"},
                            opacity=0.7)
    fig_hist.update_layout(yaxis_title="Počet vrcholů")
    st.plotly_chart(fig_hist, use_container_width=True)

    # --- Uhlazený Arc diagram HVG ---
    st.subheader("🎨 Arc diagram HVG")
    arr = st.session_state.data
    n = len(arr)
    node_x = np.arange(n)
    node_y = np.zeros(n)
    fig_arc = go.Figure()

    # Parametrické oblouky
    G = build_hvg(arr)
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

    # Vrcholy na ose
    fig_arc.add_trace(go.Scatter(
        x=node_x, y=node_y, mode='markers',
        marker=dict(size=8, color='skyblue'),
        hoverinfo='text',
        hovertext=[f"Index: {i}<br>Hodnota: {arr[i]:.3f}" for i in node_x]
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
