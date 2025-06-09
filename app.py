import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import io

st.set_page_config(page_title="HVG Vizualizátor", layout="wide")
st.title("📊 HVG Vizualizátor.")
st.title("Vizualizace hierarchických časových řad pomocí HVG")
st.sidebar.title("Nastavení")

with st.sidebar:
    st.header("🔧 Vstup dat")

    typ = st.selectbox(
        "Vyber typ časové řady",
        [
            "Náhodná uniformní",
            "Náhodná normální",
            "Sinusovka",
            "Nahrát CSV",
            "Ruční vstup"
        ]
    )

    # Parametry pro jednotlivé typy
    if typ == "Náhodná uniformní":
        length = st.slider("Délka řady", 10, 500, 50)
        low, high = st.number_input("Rozmezí hodnot (min, max)", value=(0.0, 1.0))
    elif typ == "Náhodná normální":
        length = st.slider("Délka řady", 10, 500, 50)
        mu = st.number_input("Střední hodnota μ", value=0.0)
        sigma = st.number_input("Směrodatná odchylka σ", value=1.0)
    elif typ == "Sinusovka":
        length = st.slider("Délka řady", 10, 500, 100)
        amp = st.number_input("Amplituda", value=1.0)
        freq = st.number_input("Frekvence", value=1.0)
    elif typ == "Nahrát CSV":
        uploaded_file = st.file_uploader("Nahraj CSV se sloupcem hodnot", type="csv")
    elif typ == "Ruční vstup":
        raw_text = st.text_area(
            "Zadej hodnoty oddělené čárkou",
            value="10, 5, 3, 7, 6"
        )

    # Tlačítko pro vygenerování/načtení řady
    generate = st.button("Načíst / generovat řadu")

# --- GENEROVÁNÍ / NAČTENÍ ŘADY A VYKRESLENÍ --- 
data = None

if generate:
    # 1) Náhodná uniformní
    if typ == "Náhodná uniformní":
        data = np.random.uniform(low=low, high=high, size=length)
    # 2) Náhodná normální
    elif typ == "Náhodná normální":
        data = np.random.normal(loc=mu, scale=sigma, size=length)
    # 3) Sinusovka
    elif typ == "Sinusovka":
        x = np.arange(length)
        data = amp * np.sin(2 * np.pi * freq * x / length)
    # 4) CSV
    elif typ == "Nahrát CSV" and uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # vezmeme první sloupec
        data = df.iloc[:, 0].values
    # 5) Ruční vstup
    elif typ == "Ruční vstup":
        try:
            data = np.array([float(v.strip()) for v in raw_text.split(",")])
        except:
            st.error("Chybný formát ručního vstupu! Zkontroluj čárky a čísla.")
    
    # Pokud data existují, vykresli je
    if data is not None:
        st.subheader("📈 Vaše časová řada")
        # Základní statistiky
        st.write(f"- Délka řady: **{len(data)}**")
        st.write(f"- Průměr: **{data.mean():.3f}**")
        st.write(f"- Rozptyl: **{data.var():.3f}**")
        
        # Plot
        fig, ax = plt.subplots()
        ax.plot(data, marker="o", linestyle="-")
        ax.set_xlabel("Index")
        ax.set_ylabel("Hodnota")
        ax.set_title("Časová řada")
        st.pyplot(fig)
    else:
        st.warning("Data nebyla vygenerována ani načtena.")