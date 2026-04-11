

import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import re 
import datetime as dt
from io import BytesIO

st.set_page_config(page_title="HVG Vizualizátor", layout="wide")
st.markdown(
    """
    <style>
    div[data-baseweb="popover"] {
        left: 80px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# naše služby / třídy
from services.generators import (
    generate_logistic_map,
    generate_henon_map,
    generate_lorenz_x,
    generate_pink_noise,
)

from services.analysis import (
    compute_degree_distribution_metrics,
    shannon_entropy,
    HAS_POWERLAW,
    compute_configuration_model_metrics,
    compute_graph_metrics,
    compute_graph_layout,
    prepare_network_traces,
    classify_entropy_level,
    create_degree_histogram_figure,
    create_degree_pdf_figure,
    create_degree_cdf_figure,
    create_arc_diagram_figure,
    compute_powerlaw_fit,
    build_hvg_cached,
)

@st.cache_data(show_spinner=False)
def read_csv_cached(file_bytes, has_header=True):
    buffer = BytesIO(file_bytes)

    if has_header:
        df = pd.read_csv(buffer, sep=None, engine="python")
    else:
        df = pd.read_csv(buffer, sep=None, engine="python", header=None)
        df.columns = [f"sloupec_{i}" for i in range(df.shape[1])]

    return df

def load_csv_series(
    uploaded_file=None,
    df_input=None,
    selected_column=None,
    normalize=False,
    start_index=0,
    end_index=None,
    has_header=True,
    datetime_column=None,
    selection_mode="index",   # "index" nebo "date"
    start_date=None,
    end_date=None,
    aggregation_freq=None,    # např. "5min", "1h", "1D"
    aggregation_method="mean",
):
    if uploaded_file is None and df_input is None:
        return None, None, None, "Nebyl nahrán žádný soubor."

    try:
        if df_input is not None:
            df = df_input.copy()
        else:
            uploaded_file.seek(0)
            file_bytes = uploaded_file.getvalue()
            df = read_csv_cached(file_bytes, has_header=has_header).copy()

        if df.empty:
            return None, None, None, "CSV soubor je prázdný."

        # Jen preview tabulky
        if selected_column is None:
            return df, None, None, None

        if selected_column not in df.columns:
            return df, None, None, f"Sloupec '{selected_column}' nebyl nalezen."

        work_df = df.copy()

        # převod hodnotového sloupce na čísla
        work_df[selected_column] = pd.to_numeric(work_df[selected_column], errors="coerce")
        work_df = work_df.dropna(subset=[selected_column])

        if work_df.empty:
            return df, None, None, "Vybraný sloupec neobsahuje žádné číselné hodnoty."

        # =========================
        # Výběr podle data/času
        # =========================
        if datetime_column is not None and datetime_column != "Žádný":
            if datetime_column not in work_df.columns:
                return df, None, None, f"Sloupec '{datetime_column}' nebyl nalezen."

            work_df[datetime_column] = pd.to_datetime(work_df[datetime_column], errors="coerce")
            work_df = work_df.dropna(subset=[datetime_column])

            if work_df.empty:
                return df, None, None, "Ve sloupci s datem/časem nejsou platná data."

            work_df = work_df.sort_values(datetime_column)

            if selection_mode == "date":
                if start_date is None or end_date is None:
                    return df, None, None, "Není zadaný rozsah data/času."

                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)

                if start_dt > end_dt:
                    start_dt, end_dt = end_dt, start_dt

                work_df = work_df[
                    (work_df[datetime_column] >= start_dt) &
                    (work_df[datetime_column] <= end_dt)
                ]

                if work_df.empty:
                    return df, None, None, "Ve zvoleném datovém rozsahu nejsou žádná data."

            elif selection_mode == "index":
                if start_index < 0:
                    start_index = 0

                if end_index is None:
                    end_index = len(work_df) - 1

                if end_index >= len(work_df):
                    end_index = len(work_df) - 1

                if start_index > end_index:
                    return df, None, None, "Počáteční index je větší než koncový index."

                work_df = work_df.iloc[start_index:end_index + 1]

                if work_df.empty:
                    return df, None, None, "Ve zvoleném rozsahu indexů nejsou žádná data."

            # agregace podle času
            if aggregation_freq is not None and aggregation_freq != "bez agregace":
                work_df = work_df.set_index(datetime_column)

                if aggregation_method == "mean":
                    agg_df = work_df[[selected_column]].resample(aggregation_freq).mean()
                elif aggregation_method == "median":
                    agg_df = work_df[[selected_column]].resample(aggregation_freq).median()
                elif aggregation_method == "min":
                    agg_df = work_df[[selected_column]].resample(aggregation_freq).min()
                elif aggregation_method == "max":
                    agg_df = work_df[[selected_column]].resample(aggregation_freq).max()
                elif aggregation_method == "sum":
                    agg_df = work_df[[selected_column]].resample(aggregation_freq).sum()
                elif aggregation_method == "last":
                    agg_df = work_df[[selected_column]].resample(aggregation_freq).last()
                else:
                    return df, None, None, f"Neznámá agregační metoda: {aggregation_method}"

                agg_df = agg_df.dropna().reset_index()
                work_df = agg_df

                if work_df.empty:
                    return df, None, None, "Po agregaci nezůstala žádná data."

            data = work_df[selected_column].values.astype(float)
            original_data = data.copy()

            if normalize:
                std = float(np.std(data))
                if std > 0:
                    data = (data - np.mean(data)) / std

            meta = {
                "n_points": len(data),
                "datetime_used": True,
                "datetime_column": datetime_column,
                "min_time": work_df[datetime_column].min() if datetime_column in work_df.columns else None,
                "max_time": work_df[datetime_column].max() if datetime_column in work_df.columns else None,
                "selection_mode": selection_mode,
                "aggregation_freq": aggregation_freq,
                "aggregation_method": aggregation_method,
                "normalized": normalize,
                "original_mean": float(np.mean(original_data)),
                "original_var": float(np.var(original_data)),
                "processed_mean": float(np.mean(data)),
                "processed_var": float(np.var(data)),
            }

            return df, data, meta, None

        
        # =========================
        # Bez datového sloupce
        # =========================
        else:
            if start_index < 0:
                start_index = 0

            if end_index is None:
                end_index = len(work_df) - 1

            if end_index >= len(work_df):
                end_index = len(work_df) - 1

            if start_index > end_index:
                return df, None, None, "Počáteční index je větší než koncový index."

            work_df = work_df.iloc[start_index:end_index + 1]

            if work_df.empty:
                return df, None, None, "Ve zvoleném rozsahu nejsou žádná data."

            data = work_df[selected_column].values.astype(float)
            original_data = data.copy()

            if normalize:
                std = float(np.std(data))
                if std > 0:
                    data = (data - np.mean(data)) / std

            meta = {
                "n_points": len(data),
                "datetime_used": False,
                "datetime_column": None,
                "min_time": None,
                "max_time": None,
                "selection_mode": "index",
                "aggregation_freq": None,
                "aggregation_method": None,
                "normalized": normalize,
                "original_mean": float(np.mean(original_data)),
                "original_var": float(np.var(original_data)),
                "processed_mean": float(np.mean(data)),
                "processed_var": float(np.var(data)),
            }

            return df, data, meta, None

    except Exception as e:
        return None, None, None, f"Chyba při načítání CSV: {e}"

def sync_main_from_slider():
    start, end = st.session_state.csv_main_range
    st.session_state.csv_main_start_manual = start
    st.session_state.csv_main_end_manual = end


def sync_main_from_manual():
    start = st.session_state.csv_main_start_manual
    end = st.session_state.csv_main_end_manual

    if start > end:
        start, end = end, start

    st.session_state.csv_main_start_manual = start
    st.session_state.csv_main_end_manual = end
    st.session_state.csv_main_range = (start, end)


def sync_cmp_from_slider():
    start, end = st.session_state.csv2_range
    st.session_state.csv2_start_manual = start
    st.session_state.csv2_end_manual = end


def sync_cmp_from_manual():
    start = st.session_state.csv2_start_manual
    end = st.session_state.csv2_end_manual

    if start > end:
        start, end = end, start

    st.session_state.csv2_start_manual = start
    st.session_state.csv2_end_manual = end
    st.session_state.csv2_range = (start, end)
    
def infer_series_type(series_name):
    if not series_name:
        return None

    s = str(series_name).lower()

    if "light" in s:
        return "light"
    if "plug" in s:
        return "plug"
    if "ac" in s:
        return "ac"

    return None

def normalize_graph_node(value):
    return str(value).strip()

def generate_hvg_summary_text(
    n_nodes,
    n_edges,
    avg_deg,
    C,
    L,
    sigma_sw,
    assort,
    is_normalized=False,
    aggregation_freq=None,
    series_name=None,
):
    technical_parts = []
    interpretation_parts = []
    verdict_parts = []

    # Technické shrnutí
    technical_parts.append(
        f"HVG obsahuje {n_nodes} vrcholů a {n_edges} hran, přičemž průměrný stupeň vrcholu je {avg_deg:.3f}."
    )

    if C is not None and not np.isnan(C):
        if C >= 0.4:
            technical_parts.append(
                "Graf vykazuje vyšší lokální propojenost, což naznačuje výraznější lokální strukturu v časové řadě."
            )
        elif C >= 0.2:
            technical_parts.append(
                "Graf vykazuje střední lokální propojenost, takže časová řada obsahuje určitou vnitřní strukturu."
            )
        else:
            technical_parts.append(
                "Graf má nízkou lokální propojenost, takže časová řada působí méně strukturovaně."
            )

    if L is not None:
        technical_parts.append(
            f"Průměrná délka cesty je {L:.3f}, což popisuje průměrnou vzdálenost mezi vrcholy v síti."
        )

    if sigma_sw is not None and not np.isnan(sigma_sw):
        if sigma_sw > 1.1:
            technical_parts.append(
                f"Hodnota small-world indexu σ = {sigma_sw:.2f} ukazuje na small-world charakter sítě."
            )
        elif sigma_sw >= 0.9:
            technical_parts.append(
                f"Hodnota small-world indexu σ = {sigma_sw:.2f} je blízká náhodnému grafu."
            )
        else:
            technical_parts.append(
                f"Hodnota small-world indexu σ = {sigma_sw:.2f} nenaznačuje výrazný small-world charakter."
            )

    if assort is not None and not np.isnan(assort):
        if assort > 0.1:
            technical_parts.append(
                "Kladná assortativita naznačuje, že se častěji propojují vrcholy podobného stupně."
            )
        elif assort < -0.1:
            technical_parts.append(
                "Záporná assortativita naznačuje, že se častěji propojují vrcholy odlišného stupně."
            )

    # Interpretace časové řady
    if C is not None and not np.isnan(C) and sigma_sw is not None and not np.isnan(sigma_sw):
        if C >= 0.3 and sigma_sw > 1:
            interpretation_parts.append(
                "Časová řada nepůsobí jako čistě náhodná, ale vykazuje vnitřní organizaci a opakující se strukturální vzory."
            )
        elif C < 0.2 and sigma_sw < 1:
            interpretation_parts.append(
                "Časová řada působí méně strukturovaně a může být více proměnlivá nebo blízká náhodnému chování."
            )
        else:
            interpretation_parts.append(
                "Časová řada kombinuje jak strukturální prvky, tak proměnlivost, bez jednoznačně dominantního charakteru."
            )

    if is_normalized:
        interpretation_parts.append(
            "Analýza byla provedena nad normalizovanou časovou řadou, takže se hodnotí především tvar a struktura vývoje, nikoli absolutní velikost hodnot."
        )

    if aggregation_freq not in (None, "bez agregace"):
        interpretation_parts.append(
            f"Data byla před konstrukcí HVG agregována krokem {aggregation_freq}, takže výsledná síť zachycuje strukturu řady na této časové škále."
        )

    series_type = infer_series_type(series_name)

    if series_type == "light":
        interpretation_parts.append(
            "U osvětlení lze očekávat pravidelnější režimy související s denním provozem budovy nebo využíváním prostoru."
        )
    elif series_type == "plug":
        interpretation_parts.append(
            "U zásuvkových okruhů bývá chování často proměnlivější, protože závisí na skutečném používání zařízení uživateli."
        )
    elif series_type == "ac":
        interpretation_parts.append(
            "U klimatizačních jednotek mohou být patrné provozní cykly, reakce na okolní podmínky a blokové změny výkonu."
        )

    # Závěrečný verdikt
    if sigma_sw is not None and not np.isnan(sigma_sw) and C is not None and not np.isnan(C):
        if sigma_sw > 1.1 and C >= 0.3:
            verdict_parts.append(
                "Výsledný HVG ukazuje na výraznější strukturální organizaci časové řady. "
                "Tento výsledek naznačuje přítomnost vnitřních vzorů, ale sám o sobě nerozlišuje mezi periodickým a chaotickým chováním."
            )
        elif sigma_sw < 1 and C < 0.2:
            verdict_parts.append(
                "Výsledný HVG odpovídá spíše méně strukturované a variabilnější časové řadě, která může být bližší náhodnému chování."
            )
        else:
            verdict_parts.append(
                "Výsledný HVG ukazuje středně výraznou strukturu časové řady bez jednoznačně pravidelného ani zcela náhodného charakteru."
            )

    technical_text = " ".join(technical_parts)
    interpretation_text = " ".join(interpretation_parts)
    verdict_text = " ".join(verdict_parts)

    return technical_text, interpretation_text, verdict_text   

def classify_series_from_hvg(
    avg_deg,
    C,
    L,
    sigma_sw,
    assort,
    entropy_deg_norm,
    powerlaw_p=None,
    powerlaw_R=None,
    C_rand=None,
    L_rand=None,
    sigma_conf=None,
):
    scores = {
        "Spíše pravidelná / periodická": 0.0,
        "Spíše komplexní deterministická / chaotická": 0.0,
        "Spíše stochastická / náhodná": 0.0,
    }

    reason_parts = []
    structure_parts = []
    evidence_parts = []

    warning_text = (
        "Tato klasifikace je orientační a vychází ze síťové reprezentace časové řady pomocí HVG. "
        "Nejde o formální důkaz chaosu ani stochasticity."
    )

    # =========================
    # 1) Lokální propojenost C
    # =========================
    if C is not None and not np.isnan(C):
        if C < 0.18:
            structure_parts.append(
                "Síť má nízkou lokální propojenost."
            )
            scores["Spíše stochastická / náhodná"] += 2.5
        elif C < 0.32:
            structure_parts.append(
                "Síť má střední lokální propojenost."
            )
            scores["Spíše komplexní deterministická / chaotická"] += 1.5
            scores["Spíše stochastická / náhodná"] += 0.5
        else:
            structure_parts.append(
                "Síť má vyšší lokální propojenost."
            )
            scores["Spíše pravidelná / periodická"] += 2.0
            scores["Spíše komplexní deterministická / chaotická"] += 1.0

    # =========================
    # 2) Small-world index
    # =========================
    if sigma_sw is not None and not np.isnan(sigma_sw):
        if sigma_sw < 0.95:
            structure_parts.append(
                "Síť nevykazuje výrazný small-world charakter."
            )
            scores["Spíše stochastická / náhodná"] += 1.5
        elif sigma_sw < 1.15:
            structure_parts.append(
                "Síť je small-world charakterem blízká náhodnému grafu."
            )
            scores["Spíše komplexní deterministická / chaotická"] += 1.0
            scores["Spíše stochastická / náhodná"] += 1.0
        else:
            structure_parts.append(
                "Síť vykazuje výraznější small-world charakter."
            )
            scores["Spíše komplexní deterministická / chaotická"] += 2.0
            scores["Spíše pravidelná / periodická"] += 1.0

    # =========================
    # 3) Variabilita stupňového rozdělení
    # =========================
    if entropy_deg_norm is not None and not np.isnan(entropy_deg_norm):
        if entropy_deg_norm < 0.30:
            structure_parts.append(
                "Variabilita stupňového rozdělení je nízká."
            )
            scores["Spíše pravidelná / periodická"] += 2.5
        elif entropy_deg_norm < 0.55:
            structure_parts.append(
                "Variabilita stupňového rozdělení je střední."
            )
            scores["Spíše komplexní deterministická / chaotická"] += 2.0
        else:
            structure_parts.append(
                "Variabilita stupňového rozdělení je vyšší."
            )
            scores["Spíše stochastická / náhodná"] += 2.0
            scores["Spíše komplexní deterministická / chaotická"] += 0.5

    # =========================
    # 4) Assortativita
    # =========================
    if assort is not None and not np.isnan(assort):
        if assort > 0.10:
            structure_parts.append(
                "Kladná assortativita ukazuje tendenci propojování vrcholů podobného stupně."
            )
            scores["Spíše pravidelná / periodická"] += 1.0
            scores["Spíše komplexní deterministická / chaotická"] += 0.5
        elif assort < -0.10:
            structure_parts.append(
                "Záporná assortativita ukazuje častější propojení vrcholů odlišného stupně."
            )
            scores["Spíše stochastická / náhodná"] += 1.0

    # =========================
    # 5) Průměrný stupeň
    # =========================
    if avg_deg is not None and not np.isnan(avg_deg):
        evidence_parts.append(f"Průměrný stupeň sítě je {avg_deg:.3f}.")

        if avg_deg < 2.4:
            scores["Spíše stochastická / náhodná"] += 0.5
        elif avg_deg > 3.2:
            scores["Spíše pravidelná / periodická"] += 0.5
            scores["Spíše komplexní deterministická / chaotická"] += 0.5

    # =========================
    # 6) Průměrná délka cesty
    # =========================
    if L is not None and not np.isnan(L):
        reason_parts.append(
            f"Průměrná délka cesty L = {L:.3f} popisuje dosažitelnost vzdálenějších částí sítě."
        )

    # =========================
    # 7) Srovnání s náhodným grafem
    # =========================
    if C_rand is not None and L_rand is not None:
        reason_parts.append(
            "Do interpretace je zahrnuto i srovnání s odpovídajícím náhodným grafem."
        )

        if C is not None and C_rand not in (None, 0) and not np.isnan(C):
            ratio_c = C / C_rand if C_rand != 0 else None

            if ratio_c is not None:
                if ratio_c > 2.5:
                    evidence_parts.append(
                        "Clustering HVG je výrazně vyšší než u odpovídajícího náhodného grafu."
                    )
                    scores["Spíše pravidelná / periodická"] += 1.0
                    scores["Spíše komplexní deterministická / chaotická"] += 1.0
                elif ratio_c < 1.3:
                    evidence_parts.append(
                        "Clustering HVG je jen málo odlišný od náhodného grafu."
                    )
                    scores["Spíše stochastická / náhodná"] += 1.0

    # =========================
    # 8) Srovnání s konfiguračním null modelem
    # =========================
    if sigma_conf is not None and not np.isnan(sigma_conf) and sigma_sw is not None and not np.isnan(sigma_sw):
        if sigma_sw > sigma_conf + 0.15:
            evidence_parts.append(
                "Skutečný HVG je strukturálně výraznější než konfigurační null model."
            )
            scores["Spíše komplexní deterministická / chaotická"] += 1.0
            scores["Spíše pravidelná / periodická"] += 1.0
        elif sigma_sw < sigma_conf - 0.15:
            evidence_parts.append(
                "Konfigurační null model vykazuje srovnatelný nebo vyšší small-world charakter než skutečný HVG."
            )
            scores["Spíše stochastická / náhodná"] += 1.0

    # =========================
    # 9) Power-law test
    # =========================
    if powerlaw_p is not None and powerlaw_R is not None:
        if powerlaw_p < 0.1 and powerlaw_R > 0:
            evidence_parts.append(
                "Tail stupňového rozdělení je kompatibilní s power-law."
            )
            scores["Spíše komplexní deterministická / chaotická"] += 1.0
        elif powerlaw_p < 0.1 and powerlaw_R < 0:
            evidence_parts.append(
                "Exponenciální model je vhodnější než power-law."
            )
            scores["Spíše pravidelná / periodická"] += 0.5
            scores["Spíše stochastická / náhodná"] += 0.5
        else:
            evidence_parts.append(
                "Power-law test je neprůkazný."
            )

    # =========================
    # 10) Rozhodnutí
    # =========================
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_label, best_score = sorted_scores[0]
    second_label, second_score = sorted_scores[1]
    third_label, third_score = sorted_scores[2]

    gap = best_score - second_score
    total_score = sum(scores.values())
    relative_gap = gap / total_score if total_score > 0 else 0.0

    if best_score < 3.0:
        label = "Smíšená / neurčitá"
        confidence = "nízká"
        reason_parts.insert(
            0,
            "Dostupné metriky neposkytují dostatečně silný signál pro jednoznačnou interpretaci."
        )
    elif gap < 1.0:
        label = "Smíšená / neurčitá"
        confidence = "nízká až střední"
        reason_parts.insert(
            0,
            "Jednotlivé metriky podporují více konkurenčních interpretací a výsledek proto není jednoznačný."
        )
    elif relative_gap < 0.12:
        label = "Smíšená / neurčitá"
        confidence = "střední"
        reason_parts.insert(
            0,
            "Rozdíl mezi nejsilnější a druhou nejsilnější interpretací je vzhledem k celkovému skóre příliš malý."
        )
    else:
        label = best_label

        if gap >= 3.0 and best_score >= 5.0 and relative_gap >= 0.25:
            confidence = "vyšší"
        elif gap >= 2.0 and relative_gap >= 0.18:
            confidence = "střední"
        else:
            confidence = "nižší"

        if label == "Spíše pravidelná / periodická":
            reason_parts.insert(
                0,
                "Klasifikace směřuje k pravidelnému nebo periodickému charakteru, protože síť působí organizovaněji a její stupňové rozdělení je méně variabilní."
            )
        elif label == "Spíše komplexní deterministická / chaotická":
            reason_parts.insert(
                0,
                "Klasifikace směřuje ke komplexní deterministické nebo chaotické dynamice, protože síť kombinuje strukturální organizaci s vyšší komplexitou."
            )
        elif label == "Spíše stochastická / náhodná":
            reason_parts.insert(
                0,
                "Klasifikace směřuje ke stochastickému nebo náhodnému charakteru, protože síť je méně organizovaná a vykazuje vyšší míru nahodilosti."
            )

    # =========================
    # 11) Dominantní a alternativní interpretace
    # =========================
    alternative_label = second_label
    alternative_gap_text = (
        f"Hlavní interpretace získala skóre {best_score:.1f}, "
        f"druhá nejsilnější interpretace {second_score:.1f}."
    )
    score_sum = sum(scores.values())

    if score_sum > 0:
        normalized_scores = {
            k: round((v / score_sum) * 100, 1) for k, v in scores.items()
        }
    else:
        normalized_scores = {k: 0.0 for k in scores}
        
    if total_score > 0:
        dominance_ratio = best_score / total_score
    else:
        dominance_ratio = 0.0
    
    ranking_text = (
        f"Nejsilněji podporovaná interpretace je „{best_label}“ "
        f"se skóre {best_score:.1f}. "
        f"Druhá v pořadí je „{second_label}“ se skóre {second_score:.1f}. "
        f"Nejslabší podporu má „{third_label}“ se skóre {third_score:.1f}."
    )
    if dominance_ratio >= 0.60:
        dominance_text = "Dominantní interpretace je poměrně výrazná."
    elif dominance_ratio >= 0.45:
        dominance_text = "Dominantní interpretace je středně výrazná."
    else:
        dominance_text = "Dominantní interpretace není příliš výrazná."
    if gap >= 3:
        gap_text = "Rozdíl mezi nejsilnější a druhou nejsilnější interpretací je výrazný."
    elif gap >= 1.5:
        gap_text = "Rozdíl mezi nejsilnější a druhou nejsilnější interpretací je střední."
    else:
        gap_text = "Rozdíl mezi nejsilnější a druhou nejsilnější interpretací je malý."    

    if label == "Smíšená / neurčitá":
        stability_text = "Výsledek je nejednoznačný a jednotlivé interpretace jsou si relativně blízké."
    elif confidence == "vyšší":
        stability_text = "Klasifikace působí stabilně a dominantní interpretace je zřetelně oddělená od ostatních."
    elif confidence == "střední":
        stability_text = "Klasifikace je použitelná, ale část metrik stále připouští i alternativní interpretaci."
    else:
        stability_text = "Klasifikace je spíše hraniční a je vhodné ji chápat opatrně."
    
    if label == "Smíšená / neurčitá":
        mixed_text = (
            f"Nejsilnější interpretace je **{best_label}**, "
            f"ale velmi blízko zůstává i **{second_label}**."
        )
    else:
        mixed_text = (
            f"Vedle dominantní interpretace **{label}** zůstává jako druhá nejsilnější možnost "
            f"**{alternative_label}**."
        )
    return {
        "label": label,
        "confidence": confidence,
        "reason_text": " ".join(reason_parts + evidence_parts),
        "structure_text": " ".join(structure_parts),
        "warning_text": warning_text,
        "scores": scores,
        "normalized_scores": normalized_scores,
        "dominance_ratio": dominance_ratio,
         "dominance_text": dominance_text,
         "mixed_text": mixed_text,
        "stability_text": stability_text,
        "alternative_label": alternative_label,
        "score_gap_text": alternative_gap_text,
        "gap_text": gap_text,
        "ranking_text": ranking_text,
        "best_score": best_score,
        "second_score": second_score,
        "total_score": total_score,
    }

def get_classification_status_text(classification):
    label = classification["label"]
    confidence = classification["confidence"]
    dominance_ratio = classification.get("dominance_ratio", 0.0)

    if label == "Smíšená / neurčitá":
        if dominance_ratio >= 0.45:
            return "Výsledek je smíšený, ale jedna interpretace mírně převažuje."
        return "Výsledek je nejednoznačný a interpretace zůstává otevřená."

    if confidence == "vyšší":
        return "Výsledek působí poměrně přesvědčivě a dominantní interpretace je výrazná."
    elif confidence == "střední":
        return "Výsledek je použitelný, ale stále je vhodné zachovat interpretační opatrnost."
    else:
        return "Výsledek naznačuje určitý směr, ale důkaz není příliš silný."


# =========================
#  Inicializace session state
# =========================

for key in (
    "data",
    "data2",
    "meta",
    "meta2",
    "show_hvg",
    "show_horiz",
    "show_cmp_horiz1",
    "show_cmp_horiz2",
    "custom_graph",
    "series_name",
    "series_normalized",
    "series_aggregation",
    "series_name2",
    "series_normalized2",
    "series_aggregation2",
):
    if key not in st.session_state:
        if key in (
            "data",
            "data2",
            "meta",
            "meta2",
            "series_name",
            "series_aggregation",
            "series_name2",
            "series_aggregation2",
        ):
            st.session_state[key] = None
        elif key in ("series_normalized", "series_normalized2"):
            st.session_state[key] = False
        elif key == "custom_graph":
            st.session_state[key] = None
        else:
            st.session_state[key] = False



# =========================
#  Hlavička
# =========================

st.title("HVG Vizualizátor")
st.markdown(
    "**Interaktivní vizualizace časových řad a jejich Horizontal Visibility Graphů (HVG)**"
)

# =========================
#  Sidebar – volba režimu
# =========================

st.sidebar.title("🔧 Vstup / režim")

analysis_mode = st.sidebar.radio(
    "Co chceš analyzovat?",
    ["Časová řada → HVG", "Porovnat dvě časové řady", "Vlastní HVG graf (ruční / CSV)"],
)

# =====================================================================
#  REŽIM 1: ČASOVÁ ŘADA → HVG
# =====================================================================

if analysis_mode == "Časová řada → HVG":
    st.sidebar.subheader("Nastavení časové řady")

    mode = st.sidebar.radio(
        "Typ vstupu",
        ["Standardní signály", "Chaotické generátory", "Nahrát CSV"],
    )

    typ = None
    chaos_typ = None

    # bezpečné výchozí hodnoty pro CSV větev
    uploaded_file = None
    df_preview = None
    csv_column = None
    normalize_csv = False
    csv_start_index = 0
    csv_end_index = 0
    csv_has_header = True
    csv_datetime_column = "Žádný"
    selection_mode_main = "index"
    csv_start_time = None
    csv_end_time = None
    csv_start_datetime = None
    csv_end_datetime = None
    aggregation_freq_main = "bez agregace"
    aggregation_method_main = "mean"

    # =========================
    # Standardní signály
    # =========================
    if mode == "Standardní signály":
        typ = st.sidebar.selectbox(
            "Vyber typ časové řady",
            [
                "Náhodná uniformní",
                "Náhodná normální",
                "Sinusovka",
                "Ruční vstup",
            ],
        )

        if typ == "Náhodná uniformní":
            length = st.sidebar.slider("Délka řady", 10, 5000, 50)
            low = st.sidebar.number_input("Minimální hodnota", value=0.0, step=0.1)
            high = st.sidebar.number_input("Maximální hodnota", value=1.0, step=0.1)

        elif typ == "Náhodná normální":
            length = st.sidebar.slider("Délka řady", 10, 5000, 50)
            mu = st.sidebar.number_input("Střední hodnota μ", value=0.0)
            sigma = st.sidebar.number_input("Směrodatná odchylka σ", value=1.0)

        elif typ == "Sinusovka":
            length = st.sidebar.slider("Délka řady", 10, 5000, 100)
            amp = st.sidebar.number_input("Amplituda", value=1.0)
            freq = st.sidebar.number_input("Frekvence", value=1.0)

        elif typ == "Ruční vstup":
            raw_text = st.sidebar.text_area(
                "Zadej hodnoty oddělené čárkou",
                value="10, 5, 3, 7, 6",
            )

    # =========================
    # Chaotické generátory
    # =========================
    elif mode == "Chaotické generátory":
        chaos_typ = st.sidebar.selectbox(
            "Vyber chaotický systém",
            [
                "Logistická mapa",
                "Henonova mapa",
                "Lorenzův systém (x-složka)",
                "1/f šum (pink noise)",
            ],
        )

        if chaos_typ == "Logistická mapa":
            length = st.sidebar.slider("Délka řady", 100, 5000, 1000, step=100)
            r = st.sidebar.slider("Parametr r", 3.5, 4.0, 3.9, step=0.01)
            x0 = st.sidebar.number_input(
                "Počáteční x₀",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.01,
            )
            burn_log = st.sidebar.number_input(
                "Burn-in iterace", 100, 10000, 500, step=100
            )

        elif chaos_typ == "Henonova mapa":
            length = st.sidebar.slider("Délka řady", 100, 5000, 1000, step=100)
            a = st.sidebar.number_input("Parametr a", value=1.4, step=0.1)
            b = st.sidebar.number_input("Parametr b", value=0.3, step=0.05)
            x0 = st.sidebar.number_input("Počáteční x₀", value=0.1, step=0.05)
            y0 = st.sidebar.number_input("Počáteční y₀", value=0.0, step=0.05)
            burn_henon = st.sidebar.number_input(
                "Burn-in iterace", 100, 10000, 500, step=100
            )

        elif chaos_typ == "Lorenzův systém (x-složka)":
            length = st.sidebar.slider("Délka řady", 200, 10000, 2000, step=200)
            dt = st.sidebar.number_input(
                "Krok integrace dt", value=0.01, step=0.005, format="%.3f"
            )
            sigma_l = st.sidebar.number_input("σ (sigma)", value=10.0, step=1.0)
            rho_l = st.sidebar.number_input("ρ (rho)", value=28.0, step=1.0)
            beta_l = st.sidebar.number_input("β (beta)", value=8 / 3, step=0.1)
            burn_lor = st.sidebar.number_input(
                "Burn-in kroků", 500, 20000, 1000, step=500
            )

        elif chaos_typ == "1/f šum (pink noise)":
            length = st.sidebar.slider("Délka řady", 100, 10000, 2000, step=100)

    # =========================
    # Nahrát CSV
    # =========================
    elif mode == "Nahrát CSV":
        uploaded_file = st.sidebar.file_uploader(
            "Nahraj CSV soubor",
            type="csv",
            key="csv_main",
        )

        if uploaded_file is not None:
            csv_has_header = st.sidebar.checkbox(
                "CSV má hlavičku",
                value=True,
                key="csv_main_header",
            )

            try:
                df_preview = read_csv_cached(
                    uploaded_file.getvalue(),
                    has_header=csv_has_header,
                )
                err = None
            except Exception as e:
                df_preview = None
                err = f"Chyba při načítání CSV: {e}"

            if err:
                st.sidebar.error(err)
            else:
                st.sidebar.caption("Náhled (prvních 5 řádků):")
                st.sidebar.dataframe(df_preview.head(), use_container_width=True)

                csv_column = st.sidebar.selectbox(
                    "Vyber sloupec s hodnotami časové řady",
                    options=df_preview.columns.tolist(),
                    key="csv_main_col",
                )

                datetime_options = ["Žádný"] + df_preview.columns.tolist()
                csv_datetime_column = st.sidebar.selectbox(
                    "Sloupec s datem/časem (volitelné)",
                    options=datetime_options,
                    key="csv_main_datetime_col",
                )

                if csv_datetime_column != "Žádný":
                    selection_mode_main = st.sidebar.radio(
                        "Jak chceš vybírat rozsah?",
                        ["Podle indexu", "Podle data"],
                        key="csv_main_selection_mode",
                    )

                normalize_csv = st.sidebar.checkbox(
                    "Normalizovat (z-score)",
                    value=False,
                    key="csv_main_norm",
                )

                if normalize_csv:
                    st.sidebar.caption(
                        "Data jsou převedena na bezrozměrnou škálu (z-score). "
                        "Každá hodnota říká, o kolik směrodatných odchylek se liší od průměru."
                    )

                if csv_datetime_column != "Žádný":
                    st.sidebar.markdown("**Agregace časové řady**")

                    aggregation_freq_main = st.sidebar.selectbox(
                        "Agregační krok",
                        options=[
                            "bez agregace",
                            "1min",
                            "5min",
                            "10min",
                            "15min",
                            "30min",
                            "1h",
                            "1D",
                        ],
                        key="csv_main_agg_freq",
                    )

                    if aggregation_freq_main != "bez agregace":
                        aggregation_method_main = st.sidebar.selectbox(
                            "Agregační metoda",
                            options=["mean", "median", "min", "max", "sum", "last"],
                            key="csv_main_agg_method",
                        )
                
                if csv_datetime_column != "Žádný" and selection_mode_main == "Podle data":
                    dt_series = pd.to_datetime(
                        df_preview[csv_datetime_column],
                        errors="coerce",
                    ).dropna()
                    dt_series = pd.to_datetime(
                        df_preview[csv_datetime_column],
                        errors="coerce",
                    ).dropna()

                    if len(dt_series) > 0:
                        min_dt = dt_series.min()
                        max_dt = dt_series.max()

                        st.sidebar.markdown("**Výběr časového rozsahu**")

                        col_dt_1, col_dt_2 = st.sidebar.columns(2)

                        with col_dt_1:
                            csv_start_date = st.date_input(
                                "Datum od",
                                value=min_dt.date(),
                                min_value=min_dt.date(),
                                max_value=max_dt.date(),
                                key="csv_main_date_start",
                            )
                            start_hour = st.selectbox(
                                "Hodina od",
                                options=list(range(24)),
                                index=min_dt.hour,
                                key="csv_main_start_hour",
                            )
                            start_minute = st.selectbox(
                                "Minuta od",
                                options=list(range(60)),
                                index=min_dt.minute,
                                key="csv_main_start_minute",
                            )

                        with col_dt_2:
                            csv_end_date = st.date_input(
                                "Datum do",
                                value=max_dt.date(),
                                min_value=min_dt.date(),
                                max_value=max_dt.date(),
                                key="csv_main_date_end",
                            )
                            end_hour = st.selectbox(
                                "Hodina do",
                                options=list(range(24)),
                                index=max_dt.hour,
                                key="csv_main_end_hour",
                            )
                            end_minute = st.selectbox(
                                "Minuta do",
                                options=list(range(60)),
                                index=max_dt.minute,
                                key="csv_main_end_minute",
                            )

                        csv_start_datetime = dt.datetime.combine(
                            csv_start_date,
                            dt.time(start_hour, start_minute),
                        )
                        csv_end_datetime = dt.datetime.combine(
                            csv_end_date,
                            dt.time(end_hour, end_minute),
                        )

                        _, _, preview_meta, preview_err = load_csv_series(
                            uploaded_file,
                            df_input=df_preview,
                            selected_column=csv_column,
                            normalize=False,
                            start_index=0,
                            end_index=None,
                            has_header=csv_has_header,
                            datetime_column=csv_datetime_column,
                            selection_mode="date",
                            start_date=csv_start_datetime,
                            end_date=csv_end_datetime,
                            aggregation_freq=aggregation_freq_main,
                            aggregation_method=aggregation_method_main,
                        )

                        if preview_err:
                            st.sidebar.warning(preview_err)
                        elif preview_meta is not None:
                            st.sidebar.caption(
                                f"Po načtení vznikne přibližně {preview_meta['n_points']} bodů časové řady."
                            )
                    else:
                        st.sidebar.warning("Ve vybraném datetime sloupci nejsou platná data.")
                    

                else:
                    max_possible_index = max(0, len(df_preview) - 1)
                    default_end_main = min(999, max_possible_index)

                    if "csv_main_range" not in st.session_state:
                        st.session_state.csv_main_range = (0, default_end_main)

                    if "csv_main_start_manual" not in st.session_state:
                        st.session_state.csv_main_start_manual = st.session_state.csv_main_range[0]

                    if "csv_main_end_manual" not in st.session_state:
                        st.session_state.csv_main_end_manual = st.session_state.csv_main_range[1]

                    start_tmp, end_tmp = st.session_state.csv_main_range
                    start_tmp = min(max(0, start_tmp), max_possible_index)
                    end_tmp = min(max(0, end_tmp), max_possible_index)

                    if start_tmp > end_tmp:
                        start_tmp, end_tmp = end_tmp, start_tmp

                    st.session_state.csv_main_range = (start_tmp, end_tmp)
                    st.session_state.csv_main_start_manual = start_tmp
                    st.session_state.csv_main_end_manual = end_tmp

                    csv_start_index, csv_end_index = st.sidebar.slider(
                        "Vyber rozsah řádků",
                        min_value=0,
                        max_value=max_possible_index,
                        step=1,
                        key="csv_main_range",
                        on_change=sync_main_from_slider,
                    )

                    col_range_1, col_range_2 = st.sidebar.columns(2)

                    with col_range_1:
                        st.number_input(
                            "Od",
                            min_value=0,
                            max_value=max_possible_index,
                            step=1,
                            key="csv_main_start_manual",
                            on_change=sync_main_from_manual,
                        )

                    with col_range_2:
                        st.number_input(
                            "Do",
                            min_value=0,
                            max_value=max_possible_index,
                            step=1,
                            key="csv_main_end_manual",
                            on_change=sync_main_from_manual,
                        )

                    csv_start_index = st.session_state.csv_main_start_manual
                    csv_end_index = st.session_state.csv_main_end_manual

                    _, _, preview_meta, preview_err = load_csv_series(
                        uploaded_file,
                        df_input=df_preview,
                        selected_column=csv_column,
                        normalize=False,
                        start_index=csv_start_index,
                        end_index=csv_end_index,
                        has_header=csv_has_header,
                        datetime_column=None if csv_datetime_column == "Žádný" else csv_datetime_column,
                        selection_mode="index",
                        start_date=None,
                        end_date=None,
                        aggregation_freq=aggregation_freq_main,
                        aggregation_method=aggregation_method_main,
                    )

                    if preview_err:
                        st.sidebar.warning(preview_err)
                    elif preview_meta is not None:
                        st.sidebar.caption(
                            f"Po načtení vznikne přibližně {preview_meta['n_points']} bodů časové řady."
                        )

    # =========================
    # Tlačítko pro generování
    # =========================
    generate = st.sidebar.button("Načíst / generovat řadu")

    # =========================
    # Generování dat
    # =========================
    if generate:
        data = None
        meta = None

        if mode == "Standardní signály":
            if typ == "Náhodná uniformní":
                data = np.random.uniform(low=low, high=high, size=length)

            elif typ == "Náhodná normální":
                data = np.random.normal(loc=mu, scale=sigma, size=length)

            elif typ == "Sinusovka":
                x = np.arange(length)
                data = amp * np.sin(2 * np.pi * freq * x / length)

            elif typ == "Ruční vstup":
                try:
                    data = np.array([float(v.strip()) for v in raw_text.split(",")])
                except ValueError:
                    st.error("Chybný formát ručního vstupu! Zkontroluj čísla.")
                    data = None

        elif mode == "Nahrát CSV":
            if uploaded_file is None:
                st.error("Nejprve nahraj CSV soubor.")
                data = None

            elif csv_column is None:
                st.error("Vyber sloupec s časovou řadou.")
                data = None

            else:
                _, data, meta, err = load_csv_series(
                    uploaded_file,
                    df_input=df_preview,
                    selected_column=csv_column,
                    normalize=normalize_csv,
                    start_index=csv_start_index,
                    end_index=csv_end_index,
                    has_header=csv_has_header,
                    datetime_column=None if csv_datetime_column == "Žádný" else csv_datetime_column,
                    selection_mode=(
                        "date"
                        if (csv_datetime_column != "Žádný" and selection_mode_main == "Podle data")
                        else "index"
                    ),
                    start_date=(
                        csv_start_datetime
                        if (csv_datetime_column != "Žádný" and selection_mode_main == "Podle data")
                        else None
                    ),
                    end_date=(
                        csv_end_datetime
                        if (csv_datetime_column != "Žádný" and selection_mode_main == "Podle data")
                        else None
                    ),
                    aggregation_freq=aggregation_freq_main,
                    aggregation_method=aggregation_method_main,
                )

                if err:
                    st.error(err)
                    data = None

        elif mode == "Chaotické generátory":
            if chaos_typ == "Logistická mapa":
                data = generate_logistic_map(length, r=r, x0=x0, burn=burn_log)

            elif chaos_typ == "Henonova mapa":
                data = generate_henon_map(
                    length,
                    a=a,
                    b=b,
                    x0=x0,
                    y0=y0,
                    burn=burn_henon,
                )

            elif chaos_typ == "Lorenzův systém (x-složka)":
                data = generate_lorenz_x(
                    length,
                    dt=dt,
                    sigma=sigma_l,
                    rho=rho_l,
                    beta=beta_l,
                    burn=burn_lor,
                )

            elif chaos_typ == "1/f šum (pink noise)":
                data = generate_pink_noise(length)

        if data is not None:
            st.session_state.data = data
            st.session_state.meta = meta

            if mode == "Nahrát CSV":
                st.session_state.series_name = csv_column
                st.session_state.series_normalized = normalize_csv
                st.session_state.series_aggregation = aggregation_freq_main
            else:
                st.session_state.series_name = typ if typ is not None else chaos_typ
                st.session_state.series_normalized = False
                st.session_state.series_aggregation = None

            st.success(f"Načteno {len(data)} hodnot.")

            if mode == "Nahrát CSV":
                if meta is not None and meta["datetime_used"]:
                    if meta["selection_mode"] == "date":
                        st.caption(
                            f"Datový rozsah: {csv_start_datetime.strftime('%d.%m.%Y %H:%M:%S')} → "
                            f"{csv_end_datetime.strftime('%d.%m.%Y %H:%M:%S')}"
                        )
                    else:
                        st.caption(f"Indexy: {csv_start_index} → {csv_end_index}")

                    if meta["aggregation_freq"] not in (None, "bez agregace"):
                        st.caption(
                            f"Agregace: {meta['aggregation_freq']} | metoda: {meta['aggregation_method']}"
                        )

                    if meta["min_time"] is not None and meta["max_time"] is not None:
                        st.caption(
                            f"Výsledná časová řada pokrývá: {meta['min_time']} → {meta['max_time']}"
                        )
                else:
                    st.caption(f"Indexy: {csv_start_index} → {csv_end_index}")

            st.session_state.show_hvg = False
            st.session_state.show_horiz = False

    # =========================
    # Zobrazení časové řady + HVG linky
    # =========================
    if st.session_state.data is not None:
        arr = st.session_state.data
        meta = st.session_state.meta
        st.subheader("Vaše časová řada")

        df_ts = pd.DataFrame({"index": np.arange(len(arr)), "value": arr})
        fig_ts = px.line(
            df_ts,
            x="index",
            y="value",
            markers=True,
            title="Časová řada",
            hover_data={"index": True, "value": ":.3f"},
        )
        fig_ts.update_traces(marker_size=8)

        if st.session_state.show_horiz:
            G_tmp = build_hvg_cached(arr)
            shapes = []
            for i, j in G_tmp.edges():
                y = min(arr[i], arr[j])
                shapes.append(
                    dict(
                        type="line",
                        x0=i,
                        y0=y,
                        x1=j,
                        y1=y,
                        line=dict(color="gray", width=1),
                    )
                )
            fig_ts.update_layout(shapes=shapes)

        st.plotly_chart(fig_ts, use_container_width=True)

        st.write(f"- Délka: **{len(arr)}**")

        if meta is not None and meta.get("normalized", False):
            st.write(
                f"- Původní průměr: **{meta['original_mean']:.3f}**, "
                f"Původní rozptyl: **{meta['original_var']:.3f}**"
            )
            st.write(
                f"- Průměr po normalizaci: **{meta['processed_mean']:.3f}**, "
                f"Rozptyl po normalizaci: **{meta['processed_var']:.3f}**"
            )
        else:
            st.write(
                f"- Průměr: **{arr.mean():.3f}**, "
                f"Rozptyl: **{arr.var():.3f}**"
            )

        c1, c2 = st.columns(2)

        with c1:
            if st.button("Vygenerovat HVG"):
                st.session_state.show_hvg = True

        with c2:
            if st.button("HVG linky (vodorovné)"):
                st.session_state.show_horiz = not st.session_state.show_horiz

    # =========================
    # Interaktivní HVG + další sekce pod ním
    # =========================
    if st.session_state.show_hvg and st.session_state.data is not None:
        arr = st.session_state.data
        G = build_hvg_cached(arr)
        powerlaw_p_result = None
        powerlaw_R_result = None

        st.subheader("Interaktivní vizualizace HVG")

        section_options = [
            "Metriky HVG",
            "Propojení časová řada ↔ HVG",
            "Lokální analýza úseku časové řady",
            "Podgraf HVG",
            "Rozdělení stupňů + power-law",
            "Arc Diagram HVG",
            "Konfigurační graf (null model)",
            "Shrnutí analýzy",
            "Export HVG a metrik",
        ]
        selected_sections = st.multiselect(
            "Co chceš pod HVG zobrazit?",
            options=section_options,
            default=[
                "Metriky HVG",
                "Shrnutí analýzy",
                "Rozdělení stupňů + power-law",
                "Arc Diagram HVG",
                "Export HVG a metrik",
            ],
        )

        # ====== Analytické statistiky HVG ======
        metrics_main = compute_graph_metrics(G)

        n_nodes = metrics_main["n_nodes"]
        n_edges = metrics_main["n_edges"]
        degrees = metrics_main["degrees"]
        avg_deg = metrics_main["avg_deg"]
        C = metrics_main["C"]
        L = metrics_main["L"]
        diam = metrics_main["diam"]
        assort = metrics_main["assort"]
        L_rand = metrics_main["L_rand"]
        C_rand = metrics_main["C_rand"]
        sigma_sw = metrics_main["sigma"]
        analyzer = metrics_main["analyzer"]

        degree_metrics_main = compute_degree_distribution_metrics(degrees)
        unique_deg_all = degree_metrics_main["unique_deg"]
        counts_all = degree_metrics_main["counts"]
        pk_all = degree_metrics_main["pk"]
        entropy_deg_global = degree_metrics_main["entropy_deg"]
        entropy_deg_norm_global = degree_metrics_main["entropy_deg_norm"]

        alpha_powerlaw = None
        xmin_powerlaw = None
        powerlaw_p_result = None
        powerlaw_R_result = None

        # ====== Konfigurační graf ======
        G_conf, metrics_conf = compute_configuration_model_metrics(G, seed=42)

        n_nodes_conf = metrics_conf["n_nodes"]
        n_edges_conf = metrics_conf["n_edges"]
        degrees_conf = metrics_conf["degrees"]
        avg_deg_conf = metrics_conf["avg_deg"]
        C_conf = metrics_conf["C"]
        L_conf = metrics_conf["L"]
        diam_conf = metrics_conf["diam"]
        assort_conf = metrics_conf["assort"]
        L_rand_conf = metrics_conf["L_rand"]
        C_rand_conf = metrics_conf["C_rand"]
        sigma_conf = metrics_conf["sigma"]

        # ====== Rozmístění HVG ======
        layout_option = st.radio(
            "Rozložení HVG vrcholů",
            ["Síťové (spring layout)", "Planární (pokud možné)"],
            horizontal=True,
        )

        if layout_option == "Síťové (spring layout)":
            pos = compute_graph_layout(G, layout_type="spring", seed=42)
        else:
            pos = compute_graph_layout(G, layout_type="planar", seed=42)

        color_mode = st.selectbox(
            "Barevné kódování vrcholů HVG",
            ["Jednobarevné", "Podle hodnoty časové řady", "Podle stupně"],
        )

        if color_mode == "Podle hodnoty časové řady":
            node_color_values = [arr[i] for i in G.nodes()]
        elif color_mode == "Podle stupně":
            node_color_values = [G.degree(i) for i in G.nodes()]
        else:
            node_color_values = None

        show_labels = st.checkbox("Zobrazit popisky vrcholů (indexy)", value=False)

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
            line=dict(width=1, color="#888"),
            hoverinfo="none",
        )

        node_x, node_y, node_text = [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            neigh = list(G.adj[node])
            node_text.append(
                f"Index: {node}<br>Stupeň: {len(neigh)}<br>Sousedé: {neigh}"
            )

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
            x=node_x,
            y=node_y,
            mode=node_mode,
            text=node_text_visual,
            textposition=text_position,
            hoverinfo="text",
            hovertext=node_text,
            marker=marker_kwargs,
            textfont=dict(size=10, color="black"),
        )

        highlight_trace = None
        neighbors = []
        selected_index = 0

        if "Propojení časová řada ↔ HVG" in selected_sections and n_nodes > 0:
            st.subheader("Propojení časové řady a HVG")

            selected_index = st.number_input(
                "Index vrcholu/časového bodu pro zvýraznění",
                min_value=0,
                max_value=n_nodes - 1,
                value=0,
                step=1,
            )

            highlight_neighbors = st.checkbox(
                "Zvýraznit také sousedy vybraného vrcholu v časové řadě a HVG",
                value=True,
            )

            neighbors = list(G.adj[selected_index])
            st.markdown(
                f"- Vybraný vrchol: **{selected_index}**, "
                f"stupeň: **{G.degree(selected_index)}**, "
                f"sousedé: **{neighbors}**"
            )

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
                x=highlight_x,
                y=highlight_y,
                mode="markers+text",
                text=[str(n) for n in highlight_nodes],
                textposition="top center",
                hoverinfo="text",
                hovertext=highlight_text,
                marker=dict(size=14, color="red", line_width=2),
                textfont=dict(size=12, color="red"),
                showlegend=False,
            )

        data_traces = [edge_trace, node_trace]
        if highlight_trace is not None:
            data_traces.append(highlight_trace)

        fig_hvg = go.Figure(data=data_traces)
        fig_hvg.update_layout(
            title="Horizontal Visibility Graph",
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
        )
        st.plotly_chart(fig_hvg, use_container_width=True)

        # ====== Metriky HVG ======
        if "Metriky HVG" in selected_sections:
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
                    st.write(f"- Diametr grafu: **{diam}**")
                else:
                    st.write("- Diametr grafu: *není k dispozici*")

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

        # ====== Zvýraznění v časové řadě ======
        if "Propojení časová řada ↔ HVG" in selected_sections and n_nodes > 0:
            st.subheader("Časová řada se zvýrazněným vrcholem a sousedy")

            df_ts2 = pd.DataFrame({"index": np.arange(len(arr)), "value": arr})
            fig_ts2 = px.line(
                df_ts2,
                x="index",
                y="value",
                markers=True,
                title="Časová řada (highlight)",
                hover_data={"index": True, "value": ":.3f"},
            )
            fig_ts2.update_traces(marker_size=8)

            fig_ts2.add_trace(
                go.Scatter(
                    x=[selected_index],
                    y=[arr[selected_index]],
                    mode="markers",
                    marker=dict(size=14, color="red"),
                    name="Vybraný bod",
                    hovertext=[
                        f"Index: {selected_index}<br>Hodnota: {arr[selected_index]:.3f}"
                    ],
                    hoverinfo="text",
                )
            )

            if len(neighbors) > 0:
                fig_ts2.add_trace(
                    go.Scatter(
                        x=neighbors,
                        y=[arr[i] for i in neighbors],
                        mode="markers",
                        marker=dict(size=12, color="orange"),
                        name="Sousedé",
                        hovertext=[
                            f"Index: {i}<br>Hodnota: {arr[i]:.3f}" for i in neighbors
                        ],
                        hoverinfo="text",
                    )
                )

            st.plotly_chart(fig_ts2, use_container_width=True)

        # ====== Konfigurační graf ======
        if "Konfigurační graf (null model)" in selected_sections:
            st.markdown("### Konfigurační graf (null model)")

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

            st.markdown("**Porovnání HVG vs. konfigurační graf (null model)**")

            if not np.isnan(C) and not np.isnan(C_conf):
                st.write(
                    f"- Clustering HVG: **{C:.3f}**, konfigurační graf C_conf: **{C_conf:.3f}**"
                )
                if C > C_conf * 2:
                    st.info(
                        "HVG má **výrazně vyšší clustering** než degree-preserving null model – "
                        "to naznačuje silnou nestrukturovanost vůči náhodnému přepojení hran."
                    )

            if (L is not None) and (L_conf is not None):
                st.write(
                    f"- Průměrná délka cesty L (HVG): **{L:.3f}**, L_conf: **{L_conf:.3f}**"
                )
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

            st.subheader("Konfigurační graf (vizualizace)")
            pos_conf = compute_graph_layout(G_conf, layout_type="spring", seed=42)

            edge_trace_c, node_trace_c = prepare_network_traces(
                G_conf,
                pos_conf,
                node_color="lightgreen",
                node_size=8,
                edge_color="#aaa",
                edge_width=1,
                show_labels=False,
                hover_texts=[f"Vrchol: {n}" for n in G_conf.nodes()],
            )

            fig_conf = go.Figure(data=[edge_trace_c, node_trace_c])
            fig_conf.update_layout(
                title="Konfigurační graf se stejnou stupňovou posloupností jako HVG",
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
            )
            st.plotly_chart(fig_conf, use_container_width=True)

        # ====== Lokální analýza ======
        if "Lokální analýza úseku časové řady" in selected_sections:
            st.subheader("Lokální analýza úseku časové řady")

            if len(arr) >= 2:
                i_start, i_end = st.slider(
                    "Vyber rozsah indexů [i_start, i_end] pro lokální analýzu",
                    min_value=0,
                    max_value=len(arr) - 1,
                    value=(0, min(len(arr) - 1, max(1, len(arr) // 5))),
                )
                if i_start > i_end:
                    i_start, i_end = i_end, i_start
            else:
                i_start, i_end = 0, 0

            segment = arr[i_start : i_end + 1]
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

                if len(segment) >= 2:
                    G_seg = build_hvg_cached(segment)
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
                    st.write(
                        f"- Počet vrcholů: **{n_seg}**, počet hran: **{m_seg}**, průměrný stupeň: **{avg_deg_seg:.3f}**"
                    )
                    st.write(f"- Clustering (lokální): **{C_seg:.3f}**")
                    if L_seg is not None:
                        st.write(f"- Průměrná délka cesty (lokální): **{L_seg:.3f}**")
                    if diam_seg is not None:
                        st.write(f"- Průměr grafu (lokální): **{diam_seg}**")

        # ====== Podgraf HVG ======
        if "Podgraf HVG" in selected_sections:
            st.subheader("Podgraf HVG podle vybraných vrcholů")

            sub_nodes_text = st.text_input(
                "Seznam vrcholů pro podgraf (indexy oddělené čárkou nebo mezerami)",
                value="0, 1, 2",
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
                st.write(
                    f"Podgraf obsahuje **{G_sub.number_of_nodes()}** vrcholů a **{G_sub.number_of_edges()}** hran."
                )

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

                edge_trace_sub, node_trace_sub = prepare_network_traces(
                    G_sub,
                    pos,
                    node_color="lightcoral",
                    node_size=10,
                    edge_color="#888",
                    edge_width=1,
                    show_labels=True,
                    hover_texts=[f"Vrchol: {n}" for n in G_sub.nodes()],
                )

                fig_sub = go.Figure(data=[edge_trace_sub, node_trace_sub])
                fig_sub.update_layout(
                    title="Podgraf HVG (vybrané vrcholy)",
                    showlegend=False,
                    hovermode="closest",
                    margin=dict(b=20, l=5, r=5, t=40),
                )
                st.plotly_chart(fig_sub, use_container_width=True)

        st.markdown("---")

        # ====== Rozdělení stupňů + power-law ======
        if "Rozdělení stupňů + power-law" in selected_sections:
            degs = degrees
            unique_deg = unique_deg_all
            counts = counts_all
            pk = pk_all
            entropy_deg = entropy_deg_global
            entropy_deg_norm = entropy_deg_norm_global

            entropy_level, entropy_text = classify_entropy_level(entropy_deg_norm)

            st.subheader("Základní metriky stupňového rozdělení")

            col_deg_1, col_deg_2, col_deg_3, col_deg_4, col_deg_5 = st.columns(5)

            with col_deg_1:
                st.metric("Průměrný stupeň", f"{np.mean(degs):.3f}")
            with col_deg_2:
                st.metric("Medián stupně", f"{np.median(degs):.3f}")
            with col_deg_3:
                st.metric("Maximální stupeň", f"{np.max(degs)}")
            with col_deg_4:
                st.metric("Shannonova entropie", f"{entropy_deg:.3f}")
            with col_deg_5:
                st.metric("Norm. entropie", f"{entropy_deg_norm:.3f}")
                st.caption(entropy_level)

            st.subheader("Stručná interpretace rozdělení stupňů")

            interp_parts = []
            interp_parts.append(
                f"Normalizovaná entropie stupňového rozdělení je **{entropy_deg_norm:.3f}**, "
                f"což odpovídá kategorii **{entropy_level}**. {entropy_text}"
            )

            degree_range = np.max(degs) - np.min(degs)
            if degree_range <= 2:
                interp_parts.append(
                    "Rozsah stupňů je poměrně malý, takže většina vrcholů má podobnou konektivitu."
                )
            elif degree_range <= 5:
                interp_parts.append(
                    "Rozsah stupňů je střední, což naznačuje kombinaci běžných i výrazněji propojených vrcholů."
                )
            else:
                interp_parts.append(
                    "Rozsah stupňů je poměrně široký, takže v síti existují jak slabě propojené, tak výrazně propojené vrcholy."
                )

            peak_degree = unique_deg[np.argmax(pk)]
            interp_parts.append(
                f"PDF dosahuje maxima při stupni **k = {peak_degree}**, takže právě tento stupeň je v síti nejčastější."
            )

            median_degree = np.median(degs)
            interp_parts.append(
                f"CDF ukazuje, jak rychle se kumuluje podíl vrcholů do nižších stupňů; medián stupně je **{median_degree:.3f}**."
            )

            st.info(" ".join(interp_parts))

            fig_hist = create_degree_histogram_figure(
                degs,
                title="Histogram stupňů",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            st.subheader("PDF stupňového rozdělení")
            fig_pdf = create_degree_pdf_figure(
                unique_deg,
                pk,
                title="PDF stupňového rozdělení P(k)",
            )
            st.plotly_chart(fig_pdf, use_container_width=True)

            st.caption(
                "PDF (Probability Distribution Function) ukazuje pravděpodobnost, "
                "že náhodně vybraný vrchol v HVG má právě stupeň k."
            )

            st.subheader("CDF stupňového rozdělení")
            cdf_vals = degree_metrics_main["cdf"]
            fig_cdf = create_degree_cdf_figure(
                unique_deg,
                cdf_vals,
                title="CDF stupňového rozdělení F(k)",
            )
            st.plotly_chart(fig_cdf, use_container_width=True)

            st.caption(
                "CDF (Cumulative Distribution Function) ukazuje pravděpodobnost, "
                "že náhodně vybraný vrchol v HVG má stupeň menší nebo roven k."
            )

            do_powerlaw_global = st.checkbox(
                "🔍 Provést formální power-law test (Clauset–Shalizi–Newman) + CCDF",
                key="powerlaw_main_global",
            )

            if do_powerlaw_global:
                if not HAS_POWERLAW:
                    st.warning(
                        "K provedení testu je potřeba balík `powerlaw`. "
                        "Přidej ho do `requirements.txt` a nainstaluj pomocí `pip install powerlaw`."
                    )
                else:
                    powerlaw_result = compute_powerlaw_fit(degs, has_powerlaw=HAS_POWERLAW)

                    if not powerlaw_result["success"]:
                        if powerlaw_result["reason"] == "Příliš málo hodnot pro smysluplný fit.":
                            st.info("Graf má příliš málo vrcholů pro smysluplný power-law fit.")
                        else:
                            st.info(
                                f"Power-law test se nepodařilo spolehlivě vyhodnotit: {powerlaw_result['reason']}"
                            )
                    else:
                        alpha_powerlaw = powerlaw_result["alpha"]
                        xmin_powerlaw = powerlaw_result["xmin"]
                        powerlaw_R_result = powerlaw_result["R"]
                        powerlaw_p_result = powerlaw_result["p"]

                        alpha = alpha_powerlaw
                        xmin = xmin_powerlaw
                        R = powerlaw_R_result
                        p = powerlaw_p_result

                        st.markdown("**Výsledek power-law analýzy:**")
                        st.write(f"- Odhadnutý exponent α: **{alpha:.3f}**")
                        st.write(f"- Odhadnuté k_min: **{xmin}**")
                        st.write(f"- Likelihood ratio: **R = {R:.3f}**")
                        st.write(f"- p-hodnota: **p = {p:.3f}**")

                        if p < 0.1:
                            if R > 0:
                                st.success(
                                    "Rozdělení je kompatibilní s power-law (lepší než exponenciální)."
                                )
                            else:
                                st.warning(
                                    "Power-law model je horší než exponenciální rozdělení."
                                )
                        else:
                            st.info("Test je neprůkazný – nelze rozhodnout.")

                        degs_for_fit = powerlaw_result["degrees_for_fit"]
                        unique_sorted = np.sort(np.unique(degs_for_fit))
                        ccdf_vals = np.array(
                            [
                                np.sum(degs_for_fit >= k) / len(degs_for_fit)
                                for k in unique_sorted
                            ]
                        )

                        mask = unique_sorted >= xmin
                        if np.sum(mask) >= 2:
                            k_emp = unique_sorted[mask]
                            ccdf_emp = ccdf_vals[mask]

                            k_theory = np.linspace(xmin, k_emp.max(), 100)
                            ccdf_theory = (k_theory / xmin) ** (1 - alpha)
                            ccdf_theory *= ccdf_emp[0] / ccdf_theory[0]

                            fig_ccdf = go.Figure()
                            fig_ccdf.add_trace(
                                go.Scatter(
                                    x=k_emp,
                                    y=ccdf_emp,
                                    mode="markers",
                                    name="Empirická CCDF",
                                )
                            )
                            fig_ccdf.add_trace(
                                go.Scatter(
                                    x=k_theory,
                                    y=ccdf_theory,
                                    mode="lines",
                                    name=f"Power-law fit (α={alpha:.2f})",
                                )
                            )
                            fig_ccdf.update_layout(
                                title="CCDF (log–log)",
                                xaxis_type="log",
                                yaxis_type="log",
                                xaxis_title="Stupeň k",
                                yaxis_title="P(K ≥ k)",
                            )
                            st.plotly_chart(fig_ccdf, use_container_width=True)
                        else:
                            st.info("Tail je příliš krátký pro CCDF graf.")

        # ====== Shrnutí analýzy ======
        if "Shrnutí analýzy" in selected_sections:
            st.subheader("Shrnutí analýzy")

            experiment_name = st.text_input(
                "Název experimentu",
                value="Analýza jedné časové řady",
                key="main_experiment_name",
            )

            tech, interp, verdict = generate_hvg_summary_text(
                n_nodes=n_nodes,
                n_edges=n_edges,
                avg_deg=avg_deg,
                C=C,
                L=L,
                sigma_sw=sigma_sw,
                assort=assort,
                is_normalized=st.session_state.get("series_normalized", False),
                aggregation_freq=st.session_state.get("series_aggregation", None),
                series_name=st.session_state.get("series_name", "Časová řada"),
            )

            classification = classify_series_from_hvg(
                avg_deg=avg_deg,
                C=C,
                L=L,
                sigma_sw=sigma_sw,
                assort=assort,
                entropy_deg_norm=entropy_deg_norm_global,
                powerlaw_p=powerlaw_p_result,
                powerlaw_R=powerlaw_R_result,
                C_rand=C_rand,
                L_rand=L_rand,
                sigma_conf=sigma_conf,
            )

            validation_messages = []

            if len(arr) < 10:
                validation_messages.append(
                    "Časová řada je velmi krátká (méně než 10 bodů), takže výsledky mohou být silně nestabilní."
                )
            elif len(arr) < 30:
                validation_messages.append(
                    "Časová řada je poměrně krátká (méně než 30 bodů), takže interpretace může být méně spolehlivá."
                )

            if n_nodes < 10:
                validation_messages.append(
                    "HVG má velmi málo vrcholů, takže některé síťové metriky a klasifikační závěry mohou být méně robustní."
                )

            if validation_messages:
                st.markdown("**Upozornění k interpretaci**")
                for msg in validation_messages:
                    st.warning(msg)

            st.markdown("**Technické shrnutí**")
            st.info(tech)

            st.markdown("**Interpretace řady**")
            st.write(interp)

            st.markdown("**Orientační klasifikace**")
            classification_text = (
                f"**{classification['label']}** "
                f"(jistota: **{classification['confidence']}**)"
            )

            if classification["confidence"] == "vyšší":
                st.success(classification_text)
            elif classification["confidence"] == "střední":
                st.warning(classification_text)
            else:
                st.info(classification_text)

            st.caption(get_classification_status_text(classification))

            st.markdown("**Zdůvodnění**")
            st.write(classification["reason_text"])

            st.markdown("**Stabilita a charakter výsledku**")
            st.info(
                f"{classification['structure_text']} "
                f"Alternativní interpretace: {classification['alternative_label']}. "
                f"{classification['score_gap_text']} "
                f"{classification['gap_text']} "
                f"{classification['dominance_text']}"
            )

            st.markdown("**Závěr**")
            st.warning(verdict)

            st.markdown("**Skóre jednotlivých interpretací**")
            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "Pravidelná / periodická",
                    f"{classification['scores']['Spíše pravidelná / periodická']:.1f}",
                    delta=f"{classification['normalized_scores']['Spíše pravidelná / periodická']:.1f} %",
                )

            with c2:
                st.metric(
                    "Komplexní / chaotická",
                    f"{classification['scores']['Spíše komplexní deterministická / chaotická']:.1f}",
                    delta=f"{classification['normalized_scores']['Spíše komplexní deterministická / chaotická']:.1f} %",
                )

            with c3:
                st.metric(
                    "Stochastická / náhodná",
                    f"{classification['scores']['Spíše stochastická / náhodná']:.1f}",
                    delta=f"{classification['normalized_scores']['Spíše stochastická / náhodná']:.1f} %",
                )

            st.caption("Hlavní číslo = bodové skóre. Procento = relativní podíl interpretace.")

            st.markdown("**Relativní podpora interpretací**")

            mapping = [
                ("Pravidelná / periodická", "Spíše pravidelná / periodická"),
                ("Komplexní / chaotická", "Spíše komplexní deterministická / chaotická"),
                ("Stochastická / náhodná", "Spíše stochastická / náhodná"),
            ]

            for label, key in mapping:
                st.write(label)
                st.progress(classification["normalized_scores"][key] / 100)
                st.caption(f"{classification['normalized_scores'][key]:.1f} %")

            st.caption(classification["stability_text"])
            st.caption(classification["mixed_text"])
            st.caption(classification["warning_text"])

            st.markdown("### Export souhrnné klasifikace")

            summary_export_df = pd.DataFrame([
                {
                    "experiment_name": experiment_name,
                    "n_points": len(arr),
                    "n_nodes": n_nodes,
                    "n_edges": n_edges,
                    "avg_degree": avg_deg,
                    "clustering": C,
                    "avg_path_length": L,
                    "sigma": sigma_sw,
                    "entropy_deg_norm": entropy_deg_norm_global,
                    "label": classification["label"],
                    "confidence": classification["confidence"],
                    "alternative_label": classification["alternative_label"],
                    "dominance_ratio": classification["dominance_ratio"],
                    "best_score": classification["best_score"],
                    "second_score": classification["second_score"],
                    "verdict": verdict,
                }
            ])

            summary_export_csv = summary_export_df.to_csv(index=False).encode("utf-8-sig")

            st.download_button(
                "Exportovat souhrnnou klasifikaci (CSV)",
                data=summary_export_csv,
                file_name="single_series_summary_classification.csv",
                mime="text/csv",
            )

        st.markdown("---")

        # ====== Arc diagram HVG ======
        if "Arc Diagram HVG" in selected_sections:
            st.subheader("Arc Diagram HVG")
            fig_arc = create_arc_diagram_figure(
                G,
                arr,
                title="Arc Diagram HVG",
                node_color="skyblue",
            )
            st.plotly_chart(fig_arc, use_container_width=True)

        # ====== Export HVG a metrik ======
        if "Export HVG a metrik" in selected_sections:
            st.subheader("Export HVG a metrik")

            edges_df = pd.DataFrame(list(G.edges()), columns=["source", "target"])
            edges_csv = edges_df.to_csv(index=False).encode("utf-8-sig")

            adj_df = nx.to_pandas_adjacency(G)
            adj_csv = adj_df.to_csv().encode("utf-8-sig")

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
                "entropy_deg_global": entropy_deg_global,
                "entropy_deg_norm_global": entropy_deg_norm_global,
                "sigma_conf": sigma_conf,
                "powerlaw_p": powerlaw_p_result,
                "powerlaw_R": powerlaw_R_result,
            }
            metrics_df = pd.DataFrame([metrics_dict])
            metrics_csv = metrics_df.to_csv(index=False).encode("utf-8-sig")

            col_exp1, col_exp2, col_exp3 = st.columns(3)

            with col_exp1:
                st.download_button(
                    "Exportovat HVG jako edge list (CSV)",
                    data=edges_csv,
                    file_name="hvg_edgelist.csv",
                    mime="text/csv",
                )

            with col_exp2:
                st.download_button(
                    "Exportovat HVG jako adjacency matrix (CSV)",
                    data=adj_csv,
                    file_name="hvg_adjacency.csv",
                    mime="text/csv",
                )

            with col_exp3:
                st.download_button(
                    "Exportovat metriky HVG (CSV)",
                    data=metrics_csv,
                    file_name="hvg_metrics.csv",
                    mime="text/csv",
                )
# =====================================================================
#  REŽIM 2: VLASTNÍ GRAF Z NODE/EDGE LISTU NEBO CSV
# =====================================================================

elif analysis_mode == "Vlastní HVG graf (ruční / CSV)":
    st.sidebar.subheader("Vlastní HVG graf – vstup")

    input_mode = st.sidebar.radio(
        "Způsob zadání grafu",
        [
            "Node list",
            "Edge list",
            "Node + Edge list",
            "Edge list (CSV)",
            "Node + edge list (CSV)",
        ],
    )

    custom_graph = None

    # =========================
    # Node list
    # =========================
    if input_mode == "Node list":
        nodes_text = st.sidebar.text_area(
            "Seznam vrcholů (oddělené čárkou, mezerou nebo novým řádkem)",
            value="1, 2, 3, 4",
        )

        if st.sidebar.button("Vytvořit graf z node listu"):
            tokens = [
                normalize_graph_node(t)
                for t in re.split(r"[,\s;]+", nodes_text)
                if str(t).strip() != ""
            ]
            Gc = nx.Graph()
            Gc.add_nodes_from(tokens)
            custom_graph = Gc

    # =========================
    # Edge list
    # =========================
    elif input_mode == "Edge list":
        edges_text = st.sidebar.text_area(
            "Seznam hran – každá hrana na novém řádku ve formátu `u,v` nebo `u v`",
            value="1,2\n2,3\n3,4",
        )

        if st.sidebar.button("Vytvořit graf z edge listu"):
            Gc = nx.Graph()
            for line in edges_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = [normalize_graph_node(p) for p in re.split(r"[,\s;]+", line) if str(p).strip() != ""]
                if len(parts) >= 2:
                    u, v = parts[0], parts[1]
                    Gc.add_edge(u, v)
            custom_graph = Gc

    # =========================
    # Node + Edge list
    # =========================
    elif input_mode == "Node + Edge list":
        nodes_text = st.sidebar.text_area(
            "Seznam vrcholů (oddělené čárkou, mezerou nebo novým řádkem)",
            value="1, 2, 3, 4, 5",
        )
        edges_text = st.sidebar.text_area(
            "Seznam hran – každá hrana na novém řádku ve formátu `u,v` nebo `u v`",
            value="1,2\n2,3\n3,4\n4,5",
        )

        if st.sidebar.button("Vytvořit graf z node + edge listu"):
            tokens = [
                normalize_graph_node(t)
                for t in re.split(r"[,\s;]+", nodes_text)
                if str(t).strip() != ""
            ]
            Gc = nx.Graph()
            Gc.add_nodes_from(tokens)

            for line in edges_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = [normalize_graph_node(p) for p in re.split(r"[,\s;]+", line) if str(p).strip() != ""]
                if len(parts) >= 2:
                    u, v = parts[0], parts[1]
                    Gc.add_edge(u, v)

            custom_graph = Gc

    # =========================
    # Edge list CSV
    # =========================
    elif input_mode == "Edge list (CSV)":
        st.sidebar.write(
            "Očekává se CSV se **dvěma sloupci**: zdroj a cíl hrany (edge list)."
        )

        uploaded_edges = st.sidebar.file_uploader(
            "Nahraj CSV s edge listem",
            type="csv",
            key="csv_edges_uploader",
        )

        if uploaded_edges is not None:
            try:
                df_edges = pd.read_csv(uploaded_edges)

                if df_edges.shape[1] < 2:
                    st.sidebar.error("CSV musí mít alespoň dva sloupce (source, target).")
                else:
                    col1 = st.sidebar.selectbox(
                        "Sloupec se zdrojem (source)",
                        df_edges.columns,
                        index=0,
                    )
                    col2 = st.sidebar.selectbox(
                        "Sloupec s cílem (target)",
                        df_edges.columns,
                        index=1 if df_edges.shape[1] > 1 else 0,
                    )

                    if st.sidebar.button("Vytvořit graf z CSV edge listu"):
                        Gc = nx.Graph()
                        for _, row in df_edges.iterrows():
                            u = normalize_graph_node(row[col1])
                            v = normalize_graph_node(row[col2])
                            Gc.add_edge(u, v)

                        custom_graph = Gc

            except Exception as e:
                st.sidebar.error(f"Chyba při načítání CSV: {e}")

    # =========================
    # Node + edge list CSV
    # =========================
    elif input_mode == "Node + edge list (CSV)":
        st.sidebar.markdown("### 📂 Načtení grafu z CSV")

        nodes_file = st.sidebar.file_uploader(
            "CSV – seznam vrcholů (nodes)",
            type="csv",
            key="nodes_csv",
        )
        edges_file = st.sidebar.file_uploader(
            "CSV – seznam hran (edges)",
            type="csv",
            key="edges_csv",
        )

        if nodes_file is not None and edges_file is not None:
            try:
                nodes_df = pd.read_csv(nodes_file)
                edges_df = pd.read_csv(edges_file)

                st.sidebar.caption("Náhled nodes:")
                st.sidebar.dataframe(nodes_df.head(), use_container_width=True)

                st.sidebar.caption("Náhled edges:")
                st.sidebar.dataframe(edges_df.head(), use_container_width=True)

                G_custom = nx.Graph()

                if nodes_df.shape[1] == 1:
                    node_col = nodes_df.columns[0]
                else:
                    node_col = st.sidebar.selectbox(
                        "Sloupec s ID vrcholu",
                        nodes_df.columns.tolist(),
                        key="nodes_col_select",
                    )

                for n in nodes_df[node_col]:
                    G_custom.add_node(normalize_graph_node(n))

                if edges_df.shape[1] >= 2:
                    col1, col2 = st.sidebar.columns(2)

                    with col1:
                        source_col = col1.selectbox(
                            "Zdroj (source)",
                            edges_df.columns.tolist(),
                            key="edge_source_col",
                        )

                    with col2:
                        target_col = col2.selectbox(
                            "Cíl (target)",
                            edges_df.columns.tolist(),
                            key="edge_target_col",
                        )

                    for _, row in edges_df.iterrows():
                        G_custom.add_edge(
                            normalize_graph_node(row[source_col]),
                            normalize_graph_node(row[target_col]),
                        )

                st.session_state.custom_graph = G_custom

                st.sidebar.success(
                    f"Načten graf: {G_custom.number_of_nodes()} vrcholů, {G_custom.number_of_edges()} hran"
                )

            except Exception as e:
                st.sidebar.error(f"Chyba při načítání CSV: {e}")

    # =========================
    # Uložení grafu do session
    # =========================
    if custom_graph is not None:
        st.session_state.custom_graph = custom_graph

    # =========================
    # Hlavní obsah
    # =========================
    st.markdown("## Vlastní HVG graf (analýza)")
    selected_sections_custom = []

    if st.session_state.custom_graph is not None:
        Gc = st.session_state.custom_graph

        st.markdown("### Metriky a vizualizace vlastního grafu")

        metrics_custom = compute_graph_metrics(Gc)

        n_nodes_c = metrics_custom["n_nodes"]
        n_edges_c = metrics_custom["n_edges"]
        degrees_c = metrics_custom["degrees"]
        avg_deg_c = metrics_custom["avg_deg"]
        C_c = metrics_custom["C"]
        L_c = metrics_custom["L"]
        diam_c = metrics_custom["diam"]
        assort_c = metrics_custom["assort"]
        L_rand_c = metrics_custom["L_rand"]
        C_rand_c = metrics_custom["C_rand"]
        sigma_c = metrics_custom["sigma"]
        analyzer_c = metrics_custom["analyzer"]

        degree_metrics_custom = compute_degree_distribution_metrics(degrees_c)
        unique_deg_c = degree_metrics_custom["unique_deg"]
        counts_c = degree_metrics_custom["counts"]
        pk_c = degree_metrics_custom["pk"]
        entropy_deg_c = degree_metrics_custom["entropy_deg"]
        entropy_deg_norm_c = degree_metrics_custom["entropy_deg_norm"]

        # =========================
        # Konfigurační graf
        # =========================
        Gc_conf, conf_metrics_c = compute_configuration_model_metrics(Gc, seed=42)

        n_nodes_conf_c = conf_metrics_c["n_nodes"]
        n_edges_conf_c = conf_metrics_c["n_edges"]
        degrees_conf_c = conf_metrics_c["degrees"]
        avg_deg_conf_c = conf_metrics_c["avg_deg"]
        C_conf_c = conf_metrics_c["C"]
        L_conf_c = conf_metrics_c["L"]
        diam_conf_c = conf_metrics_c["diam"]
        assort_conf_c = conf_metrics_c["assort"]
        L_rand_conf_c = conf_metrics_c["L_rand"]
        C_rand_conf_c = conf_metrics_c["C_rand"]
        sigma_conf_c = conf_metrics_c["sigma"]

        # =========================
        # Výběr sekcí
        # =========================
        section_options_custom = [
            "Metriky HVG",
            "Podgraf HVG",
            "Rozdělení stupňů + power-law",
            "Konfigurační graf (null model)",
            "Shrnutí analýzy",
            "Export HVG a metrik",
        ]

        selected_sections_custom = st.multiselect(
            "Co chceš u vlastního HVG grafu zobrazit?",
            options=section_options_custom,
            default=[
                "Metriky HVG",
                "Rozdělení stupňů + power-law",
                "Shrnutí analýzy",
                "Export HVG a metrik",
            ],
            key="custom_hvg_sections",
        )

        # =========================
        # Metriky HVG
        # =========================
        if "Metriky HVG" in selected_sections_custom:
            col_c1, col_c2 = st.columns(2)

            with col_c1:
                st.markdown("**Základní metriky vlastního HVG grafu**")
                st.write(f"- Počet vrcholů: **{n_nodes_c}**")
                st.write(f"- Počet hran: **{n_edges_c}**")
                st.write(f"- Průměrný stupeň: **{avg_deg_c:.3f}**")
                st.write(f"- Shannonova entropie stupňů: **{entropy_deg_c:.3f}**")
                st.write(f"- Normalizovaná entropie stupňů: **{entropy_deg_norm_c:.3f}**")

                if L_c is not None:
                    st.write(f"- Průměrná délka cesty L: **{L_c:.3f}**")
                else:
                    st.write("- Průměrná délka cesty L: *nelze spočítat (nesouvislý nebo příliš malý graf)*")

                if diam_c is not None:
                    st.write(f"- Diametr grafu: **{diam_c}**")
                else:
                    st.write("- Diametr grafu: *není k dispozici*")

            with col_c2:
                st.markdown("**Clustering a small-world charakter (vlastní HVG graf)**")
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

        # =========================
        # Vizualizace vlastního grafu
        # =========================
        st.subheader("Vizuální zobrazení vlastního grafu")

        if n_nodes_c > 0:
            pos_c = compute_graph_layout(Gc, layout_type="spring", seed=42)

            edge_trace_c, node_trace_c = prepare_network_traces(
                Gc,
                pos_c,
                node_color="orange",
                node_size=10,
                edge_color="#888",
                edge_width=1,
                show_labels=True,
                hover_texts=[f"Vrchol: {node}<br>Stupeň: {Gc.degree(node)}" for node in Gc.nodes()],
            )

            fig_custom = go.Figure(data=[edge_trace_c, node_trace_c])
            fig_custom.update_layout(
                title="Vlastní HVG graf (node/edge list nebo CSV)",
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
            )
            st.plotly_chart(fig_custom, use_container_width=True)
        else:
            st.info("Graf neobsahuje žádné vrcholy – zadej alespoň jeden vrchol nebo hranu.")

        # =========================
        # Podgraf HVG
        # =========================
        if "Podgraf HVG" in selected_sections_custom:
            st.subheader("Podgraf HVG podle vybraných vrcholů")

            sub_nodes_text_c = st.text_input(
                "Seznam vrcholů pro podgraf (oddělené čárkou, mezerou nebo středníkem)",
                value="1, 2, 3",
                key="custom_subgraph_nodes",
            )

            tokens_c = [
                normalize_graph_node(t)
                for t in re.split(r"[,\s;]+", sub_nodes_text_c)
                if str(t).strip() != ""
            ]
            available_nodes_c = {normalize_graph_node(n): n for n in Gc.nodes()}
            valid_nodes_c = [available_nodes_c[t] for t in tokens_c if t in available_nodes_c]
            valid_nodes_c = list(dict.fromkeys(valid_nodes_c))

            if len(valid_nodes_c) == 0:
                st.info("Nebyl zadán žádný platný vrchol existující v grafu.")
            else:
                Gc_sub = Gc.subgraph(valid_nodes_c).copy()

                st.write(
                    f"Podgraf obsahuje **{Gc_sub.number_of_nodes()}** vrcholů a **{Gc_sub.number_of_edges()}** hran."
                )

                degs_sub_c = [d for _, d in Gc_sub.degree()]
                avg_deg_sub_c = float(np.mean(degs_sub_c)) if len(degs_sub_c) > 0 else 0.0

                try:
                    C_sub_c = nx.average_clustering(Gc_sub)
                except Exception:
                    C_sub_c = float("nan")

                is_conn_sub_c = nx.is_connected(Gc_sub) if Gc_sub.number_of_nodes() > 0 else False
                L_sub_c = None
                diam_sub_c = None

                if is_conn_sub_c and Gc_sub.number_of_nodes() > 1:
                    try:
                        L_sub_c = nx.average_shortest_path_length(Gc_sub)
                    except Exception:
                        L_sub_c = None
                    try:
                        diam_sub_c = nx.diameter(Gc_sub)
                    except Exception:
                        diam_sub_c = None

                st.write(f"- Průměrný stupeň v podgrafu: **{avg_deg_sub_c:.3f}**")
                st.write(f"- Clustering v podgrafu: **{C_sub_c:.3f}**")
                if L_sub_c is not None:
                    st.write(f"- Průměrná délka cesty v podgrafu: **{L_sub_c:.3f}**")
                if diam_sub_c is not None:
                    st.write(f"- Průměr podgrafu: **{diam_sub_c}**")

                pos_sub_c = compute_graph_layout(Gc_sub, layout_type="spring", seed=42)

                edge_trace_sub_c, node_trace_sub_c = prepare_network_traces(
                    Gc_sub,
                    pos_sub_c,
                    node_color="lightcoral",
                    node_size=10,
                    edge_color="#888",
                    edge_width=1,
                    show_labels=True,
                    hover_texts=[f"Vrchol: {n}" for n in Gc_sub.nodes()],
                )

                fig_sub_c = go.Figure(data=[edge_trace_sub_c, node_trace_sub_c])
                fig_sub_c.update_layout(
                    title="Podgraf vlastního HVG grafu",
                    showlegend=False,
                    hovermode="closest",
                    margin=dict(b=20, l=5, r=5, t=40),
                )
                st.plotly_chart(fig_sub_c, use_container_width=True)

        # =========================
        # Rozdělení stupňů + power-law
        # =========================
        if "Rozdělení stupňů + power-law" in selected_sections_custom:
            st.subheader("Rozdělení stupňů vlastního HVG grafu")

            degs_c = degrees_c
            unique_deg_c = degree_metrics_custom["unique_deg"]
            pk_c = degree_metrics_custom["pk"]
            entropy_deg_c = degree_metrics_custom["entropy_deg"]
            entropy_deg_norm_c = degree_metrics_custom["entropy_deg_norm"]

            entropy_level_c, entropy_text_c = classify_entropy_level(entropy_deg_norm_c)

            col_deg_c1, col_deg_c2, col_deg_c3, col_deg_c4, col_deg_c5 = st.columns(5)
            with col_deg_c1:
                st.metric("Průměrný stupeň", f"{np.mean(degs_c):.3f}")
            with col_deg_c2:
                st.metric("Medián stupně", f"{np.median(degs_c):.3f}")
            with col_deg_c3:
                st.metric("Maximální stupeň", f"{np.max(degs_c)}")
            with col_deg_c4:
                st.metric("Shannonova entropie", f"{entropy_deg_c:.3f}")
            with col_deg_c5:
                st.metric("Norm. entropie", f"{entropy_deg_norm_c:.3f}")
                st.caption(entropy_level_c)

            st.info(
                f"Normalizovaná entropie stupňového rozdělení je **{entropy_deg_norm_c:.3f}**, "
                f"což odpovídá kategorii **{entropy_level_c}**. {entropy_text_c}"
            )

            fig_hist_c = create_degree_histogram_figure(
                degs_c,
                title="Histogram stupňů",
            )
            st.plotly_chart(fig_hist_c, use_container_width=True)

            fig_pdf_c = create_degree_pdf_figure(
                unique_deg_c,
                pk_c,
                title="PDF stupňového rozdělení P(k)",
            )
            st.plotly_chart(fig_pdf_c, use_container_width=True)

            st.subheader("CDF stupňového rozdělení")
            cdf_vals_c = np.cumsum(pk_c)
            fig_cdf_c = create_degree_cdf_figure(
                unique_deg_c,
                cdf_vals_c,
                title="CDF stupňového rozdělení F(k)",
            )
            st.plotly_chart(fig_cdf_c, use_container_width=True)

            do_powerlaw_custom = st.checkbox(
                "🔍 Provést formální power-law test (Clauset–Shalizi–Newman) + CCDF",
                key="powerlaw_custom_global",
            )

            powerlaw_p_result_c = None
            powerlaw_R_result_c = None
            alpha_powerlaw_c = None
            xmin_powerlaw_c = None

            if do_powerlaw_custom:
                if not HAS_POWERLAW:
                    st.warning(
                        "K provedení testu je potřeba balík `powerlaw`. "
                        "Přidej ho do `requirements.txt` a nainstaluj pomocí `pip install powerlaw`."
                    )
                else:
                    result_c = compute_powerlaw_fit(degs_c, has_powerlaw=HAS_POWERLAW)

                    if not result_c["success"]:
                        if result_c["reason"] == "Příliš málo hodnot pro smysluplný fit.":
                            st.info("Graf má příliš málo vrcholů pro smysluplný power-law fit.")
                        else:
                            st.info(
                                f"Power-law test se nepodařilo spolehlivě vyhodnotit: {result_c['reason']}"
                            )
                    else:
                        alpha_powerlaw_c = result_c["alpha"]
                        xmin_powerlaw_c = result_c["xmin"]
                        powerlaw_R_result_c = result_c["R"]
                        powerlaw_p_result_c = result_c["p"]

                        st.markdown("**Výsledek power-law analýzy:**")
                        st.write(f"- Odhadnutý exponent α: **{alpha_powerlaw_c:.3f}**")
                        st.write(f"- Odhadnuté k_min: **{xmin_powerlaw_c}**")
                        st.write(f"- Likelihood ratio R: **{powerlaw_R_result_c:.3f}**")
                        st.write(f"- p-hodnota: **{powerlaw_p_result_c:.3f}**")

                        if powerlaw_p_result_c < 0.1:
                            if powerlaw_R_result_c > 0:
                                st.success(
                                    "Graf je kompatibilní s power-law a power-law je preferovaný oproti exponenciálnímu rozdělení."
                                )
                            else:
                                st.warning("Power-law model je horší než exponenciální rozdělení.")
                        else:
                            st.info("Test je neprůkazný. Nelze spolehlivě rozhodnout.")

                        degs_for_fit_c = result_c["degrees_for_fit"]
                        unique_sorted_c = np.sort(np.unique(degs_for_fit_c))
                        ccdf_vals_c = np.array(
                            [
                                np.sum(degs_for_fit_c >= k) / len(degs_for_fit_c)
                                for k in unique_sorted_c
                            ]
                        )

                        mask_c = unique_sorted_c >= xmin_powerlaw_c
                        if np.sum(mask_c) >= 2:
                            k_emp_c = unique_sorted_c[mask_c]
                            ccdf_emp_c = ccdf_vals_c[mask_c]

                            k_theory_c = np.linspace(xmin_powerlaw_c, k_emp_c.max(), 100)
                            ccdf_theory_c = (k_theory_c / xmin_powerlaw_c) ** (1 - alpha_powerlaw_c)
                            ccdf_theory_c *= ccdf_emp_c[0] / ccdf_theory_c[0]

                            fig_ccdf_c = go.Figure()
                            fig_ccdf_c.add_trace(
                                go.Scatter(
                                    x=k_emp_c,
                                    y=ccdf_emp_c,
                                    mode="markers",
                                    name="Empirická CCDF",
                                )
                            )
                            fig_ccdf_c.add_trace(
                                go.Scatter(
                                    x=k_theory_c,
                                    y=ccdf_theory_c,
                                    mode="lines",
                                    name=f"Power-law fit (α={alpha_powerlaw_c:.2f})",
                                )
                            )
                            fig_ccdf_c.update_layout(
                                title="CCDF stupňového rozdělení (empirická vs. power-law fit)",
                                xaxis_type="log",
                                yaxis_type="log",
                                xaxis_title="Stupeň k",
                                yaxis_title="P(K ≥ k)",
                                legend=dict(x=0.02, y=0.98),
                                margin=dict(b=40, l=50, r=10, t=50),
                            )
                            st.plotly_chart(fig_ccdf_c, use_container_width=True)
                        else:
                            st.info("Tail rozdělení je příliš krátký na smysluplný CCDF graf.")

        # =========================
        # Konfigurační graf
        # =========================
        if "Konfigurační graf (null model)" in selected_sections_custom:
            st.subheader("Konfigurační graf (null model)")

            col_conf_c1, col_conf_c2 = st.columns(2)

            with col_conf_c1:
                st.markdown("**Konfigurační graf – základní metriky**")
                st.write(f"- Počet vrcholů: **{n_nodes_conf_c}**")
                st.write(f"- Počet hran: **{n_edges_conf_c}**")
                st.write(f"- Průměrný stupeň: **{avg_deg_conf_c:.3f}**")
                if L_conf_c is not None:
                    st.write(f"- Průměrná délka cesty L_conf: **{L_conf_c:.3f}**")
                else:
                    st.write("- Průměrná délka cesty L_conf: *nelze spočítat (nesouvislý graf)*")
                if diam_conf_c is not None:
                    st.write(f"- Průměr grafu (diameter_conf): **{diam_conf_c}**")
                else:
                    st.write("- Průměr grafu (diameter_conf): *není k dispozici*")

            with col_conf_c2:
                st.markdown("**Konfigurační graf – clustering, assortativita, σ_conf**")
                st.write(f"- Clustering coefficient C_conf: **{C_conf_c:.3f}**")
                if assort_conf_c is not None and not np.isnan(assort_conf_c):
                    st.write(f"- Degree assortativity_conf: **{assort_conf_c:.3f}**")
                else:
                    st.write("- Degree assortativity_conf: *není k dispozici*")

                if L_rand_conf_c is not None and C_rand_conf_c is not None and C_rand_conf_c != 0:
                    st.write(
                        "- Náhodný graf pro konfigurační model (odhad):  \n"
                        f"  - L_rand_conf ≈ **{L_rand_conf_c:.3f}**  \n"
                        f"  - C_rand_conf ≈ **{C_rand_conf_c:.5f}**"
                    )
                else:
                    st.write("- L_rand_conf, C_rand_conf: *nelze odhadnout*")

                if sigma_conf_c is not None and not np.isnan(sigma_conf_c):
                    st.write(f"- Small-world index σ_conf: **{sigma_conf_c:.2f}**")

            st.markdown("**Porovnání vlastního HVG grafu vs. konfigurační graf**")

            if not np.isnan(C_c) and not np.isnan(C_conf_c):
                st.write(
                    f"- Clustering HVG: **{C_c:.3f}**, konfigurační graf C_conf: **{C_conf_c:.3f}**"
                )

            if (L_c is not None) and (L_conf_c is not None):
                st.write(
                    f"- Průměrná délka cesty L (HVG): **{L_c:.3f}**, L_conf: **{L_conf_c:.3f}**"
                )

            if sigma_c is not None and sigma_conf_c is not None:
                st.write(
                    f"- Small-world index HVG: **{sigma_c:.2f}**, konfigurační graf σ_conf: **{sigma_conf_c:.2f}**"
                )

            st.subheader("Konfigurační graf (vizualizace)")
            pos_conf_c = compute_graph_layout(Gc_conf, layout_type="spring", seed=42)

            edge_trace_conf_c, node_trace_conf_c = prepare_network_traces(
                Gc_conf,
                pos_conf_c,
                node_color="lightgreen",
                node_size=8,
                edge_color="#aaa",
                edge_width=1,
                show_labels=False,
                hover_texts=[f"Vrchol: {n}" for n in Gc_conf.nodes()],
            )

            fig_conf_c = go.Figure(data=[edge_trace_conf_c, node_trace_conf_c])
            fig_conf_c.update_layout(
                title="Konfigurační graf se stejnou stupňovou posloupností jako vlastní HVG graf",
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
            )
            st.plotly_chart(fig_conf_c, use_container_width=True)

        # =========================
        # Shrnutí analýzy
        # =========================
        if "Shrnutí analýzy" in selected_sections_custom:
            st.subheader("Shrnutí analýzy")

            col_sum_c1, col_sum_c2, col_sum_c3 = st.columns(3)

            with col_sum_c1:
                st.metric("Počet vrcholů", n_nodes_c)
                st.metric("Počet hran", n_edges_c)

            with col_sum_c2:
                st.metric("Průměrný stupeň", f"{avg_deg_c:.3f}")
                st.metric("Clustering", f"{C_c:.3f}" if not np.isnan(C_c) else "N/A")

            with col_sum_c3:
                st.metric("Průměrná délka cesty", f"{L_c:.3f}" if L_c is not None else "N/A")
                st.metric(
                    "Small-world index σ",
                    f"{sigma_c:.2f}" if sigma_c is not None and not np.isnan(sigma_c) else "N/A"
                )

            technical_parts_c = []
            interpretation_parts_c = []
            verdict_parts_c = []

            technical_parts_c.append(
                f"Graf obsahuje {n_nodes_c} vrcholů a {n_edges_c} hran, přičemž průměrný stupeň vrcholu je {avg_deg_c:.3f}."
            )

            if C_c is not None and not np.isnan(C_c):
                if C_c >= 0.4:
                    technical_parts_c.append("Graf vykazuje vyšší lokální propojenost.")
                elif C_c >= 0.2:
                    technical_parts_c.append("Graf vykazuje střední lokální propojenost.")
                else:
                    technical_parts_c.append("Graf má nízkou lokální propojenost.")

            if L_c is not None:
                technical_parts_c.append(f"Průměrná délka cesty je {L_c:.3f}.")

            if sigma_c is not None and not np.isnan(sigma_c):
                if sigma_c > 1.1:
                    interpretation_parts_c.append("Graf vykazuje výraznější small-world charakter.")
                elif sigma_c >= 0.9:
                    interpretation_parts_c.append("Graf je svým small-world charakterem blízký náhodnému grafu.")
                else:
                    interpretation_parts_c.append("Graf nevykazuje výrazný small-world charakter.")

            if entropy_deg_norm_c < 0.35:
                interpretation_parts_c.append(
                    "Stupňové rozdělení je spíše koncentrované a graf působí strukturálně pravidelněji."
                )
            elif entropy_deg_norm_c < 0.65:
                interpretation_parts_c.append(
                    "Stupňové rozdělení je středně variabilní a graf kombinuje pravidelnost i heterogenitu."
                )
            else:
                interpretation_parts_c.append(
                    "Stupňové rozdělení je výrazně variabilní a graf působí heterogenněji."
                )

            if sigma_c is not None and not np.isnan(sigma_c) and C_c is not None and not np.isnan(C_c):
                if sigma_c > 1.1 and C_c >= 0.3:
                    verdict_parts_c.append(
                        "Celkově graf vykazuje organizovanější a strukturálně výraznější topologii."
                    )
                elif sigma_c < 1 and C_c < 0.2:
                    verdict_parts_c.append(
                        "Celkově graf působí méně strukturovaně a je bližší náhodnému charakteru."
                    )
                else:
                    verdict_parts_c.append(
                        "Celkově graf vykazuje středně výraznou strukturu bez jednoznačně extrémního charakteru."
                    )
            else:
                verdict_parts_c.append(
                    "Pro tento graf nelze na základě dostupných metrik jednoznačně posoudit celkový charakter struktury."
                )

            st.markdown("**Technické shrnutí**")
            st.info(" ".join(technical_parts_c))

            st.markdown("**Interpretace grafu**")
            if interpretation_parts_c:
                st.write(" ".join(interpretation_parts_c))
            else:
                st.write("Interpretaci grafu zatím nelze jednoznačně formulovat z dostupných metrik.")

            st.markdown("**Závěrečný verdikt**")
            if verdict_parts_c:
                st.success(" ".join(verdict_parts_c))
            else:
                st.info("Pro tento graf zatím nelze vytvořit jednoznačný závěrečný verdikt.")

        # =========================
        # Export HVG a metrik
        # =========================
        if "Export HVG a metrik" in selected_sections_custom:
            st.subheader("Export HVG a metrik")

            edges_df_c = pd.DataFrame(list(Gc.edges()), columns=["source", "target"])
            edges_csv_c = edges_df_c.to_csv(index=False).encode("utf-8-sig")

            adj_df_c = nx.to_pandas_adjacency(Gc)
            adj_csv_c = adj_df_c.to_csv().encode("utf-8-sig")

            metrics_dict_c = {
                "n_nodes": n_nodes_c,
                "n_edges": n_edges_c,
                "avg_degree": avg_deg_c,
                "C": C_c,
                "L": L_c,
                "diameter": diam_c,
                "L_rand": L_rand_c,
                "C_rand": C_rand_c,
                "sigma": sigma_c,
                "entropy_deg": entropy_deg_c,
                "entropy_deg_norm": entropy_deg_norm_c,
                "sigma_conf": sigma_conf_c,
            }

            metrics_df_c = pd.DataFrame([metrics_dict_c])
            metrics_csv_c = metrics_df_c.to_csv(index=False).encode("utf-8-sig")

            col_exp_c1, col_exp_c2, col_exp_c3 = st.columns(3)

            with col_exp_c1:
                st.download_button(
                    "Exportovat HVG jako edge list (CSV)",
                    data=edges_csv_c,
                    file_name="custom_hvg_edgelist.csv",
                    mime="text/csv",
                )

            with col_exp_c2:
                st.download_button(
                    "Exportovat HVG jako adjacency matrix (CSV)",
                    data=adj_csv_c,
                    file_name="custom_hvg_adjacency.csv",
                    mime="text/csv",
                )

            with col_exp_c3:
                st.download_button(
                    "Exportovat metriky HVG (CSV)",
                    data=metrics_csv_c,
                    file_name="custom_hvg_metrics.csv",
                    mime="text/csv",
                )

    else:
        st.info("Nejprve zadej vlastní HVG graf v levém panelu (node/edge list nebo CSV).")
# =====================================================================
#  REŽIM 3: POROVNÁNÍ DVOU ČASOVÝCH ŘAD / HVG
# =====================================================================

elif analysis_mode == "Porovnat dvě časové řady":
    st.markdown("## Porovnání dvou časových řad a jejich HVG")

    if st.session_state.data is None:
        st.info(
            "Nejdřív vygeneruj časovou řadu v režimu **„Časová řada → HVG“**. "
            "Tahle série pak bude použita jako *Série 1* pro porovnání."
        )
    else:
        # =============================
        # Série 1 = už vygenerovaná časová řada
        # =============================
        data1 = st.session_state.data

        G1 = build_hvg_cached(data1)
        metrics1 = compute_graph_metrics(G1)

        n1 = metrics1["n_nodes"]
        m1 = metrics1["n_edges"]
        degs1 = metrics1["degrees"]
        avg_deg1 = metrics1["avg_deg"]
        C1 = metrics1["C"]
        L1 = metrics1["L"]
        diam1 = metrics1["diam"]
        assort1 = metrics1["assort"]
        L_rand1 = metrics1["L_rand"]
        C_rand1 = metrics1["C_rand"]
        sigma1 = metrics1["sigma"]
        analyzer1 = metrics1["analyzer"]

        degree_metrics_1 = compute_degree_distribution_metrics(degs1)
        unique_deg1_main = degree_metrics_1["unique_deg"]
        counts1_main = degree_metrics_1["counts"]
        pk1_main = degree_metrics_1["pk"]
        entropy_deg1 = degree_metrics_1["entropy_deg"]
        entropy_deg_norm1 = degree_metrics_1["entropy_deg_norm"]

        powerlaw_p_result_1 = None
        powerlaw_R_result_1 = None

        G1_conf = None
        conf_metrics1 = None
        n1c = None
        m1c = None
        degs1c = None
        avg_deg1c = None
        C1c = None
        L1c = None
        diam1c = None
        assort1c = None
        L_rand1c = None
        C_rand1c = None
        sigma1c = None

        # =============================
        # Bezpečné výchozí hodnoty pro sérii 2
        # =============================
        mode2 = None
        typ2 = None
        chaos_typ2 = None
        data2_candidate = None
        meta2 = None

        file2 = None
        df2_preview = None
        selected_column2 = None
        normalize_csv2 = False
        aggregation_freq_cmp = "bez agregace"
        aggregation_method_cmp = "mean"
        csv2_start_index = 0
        csv2_end_index = 0
        csv2_start_date = None
        csv2_end_date = None
        csv2_start_datetime = None
        csv2_end_datetime = None
        csv2_datetime_column = "Žádný"
        selection_mode_cmp = "index"
        csv2_has_header = True

        # =============================
        # Sidebar – nastavení série 2
        # =============================
        st.sidebar.subheader("Nastavení časové řady – Série 2")

        mode2 = st.sidebar.radio(
            "Typ vstupu",
            ["Standardní signály", "Chaotické generátory", "Nahrát CSV"],
            key="mode_series_2",
        )

        # -----------------------------
        # Standardní signály – Série 2
        # -----------------------------
        if mode2 == "Standardní signály":
            typ2 = st.sidebar.selectbox(
                "Vyber typ časové řady",
                [
                    "Náhodná uniformní",
                    "Náhodná normální",
                    "Sinusovka",
                    "Ruční vstup",
                ],
                key="typ_series_2",
            )

            if typ2 == "Náhodná uniformní":
                length2 = st.sidebar.slider("Délka řady", 10, 5000, 50, key="len_uni_2")
                low2 = st.sidebar.number_input(
                    "Minimální hodnota",
                    value=0.0,
                    step=0.1,
                    key="low_uni_2",
                )
                high2 = st.sidebar.number_input(
                    "Maximální hodnota",
                    value=1.0,
                    step=0.1,
                    key="high_uni_2",
                )

            elif typ2 == "Náhodná normální":
                length2 = st.sidebar.slider("Délka řady", 10, 5000, 50, key="len_norm_2")
                mu2 = st.sidebar.number_input(
                    "Střední hodnota μ",
                    value=0.0,
                    key="mu_norm_2",
                )
                sigma_input_2 = st.sidebar.number_input(
                    "Směrodatná odchylka σ",
                    value=1.0,
                    key="sigma_norm_2",
                )

            elif typ2 == "Sinusovka":
                length2 = st.sidebar.slider("Délka řady", 10, 5000, 100, key="len_sin_2")
                amp2 = st.sidebar.number_input("Amplituda", value=1.0, key="amp_sin_2")
                freq2 = st.sidebar.number_input("Frekvence", value=1.0, key="freq_sin_2")

            elif typ2 == "Ruční vstup":
                txt2 = st.sidebar.text_area(
                    "Zadej hodnoty oddělené čárkou",
                    value="2, 4, 6, 8, 10",
                    key="manual_series_2",
                )

        # -----------------------------
        # Chaotické generátory – Série 2
        # -----------------------------
        elif mode2 == "Chaotické generátory":
            chaos_typ2 = st.sidebar.selectbox(
                "Vyber chaotický systém",
                [
                    "Logistická mapa",
                    "Henonova mapa",
                    "Lorenzův systém (x-složka)",
                    "1/f šum (pink noise)",
                ],
                key="chaos_type_2",
            )

            if chaos_typ2 == "Logistická mapa":
                length2 = st.sidebar.slider(
                    "Délka řady",
                    100,
                    5000,
                    1000,
                    step=100,
                    key="len_log_2",
                )
                r2 = st.sidebar.slider(
                    "Parametr r",
                    3.5,
                    4.0,
                    3.9,
                    step=0.01,
                    key="r_log_2",
                )
                x02 = st.sidebar.number_input(
                    "Počáteční x₀",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.2,
                    step=0.01,
                    key="x0_log_2",
                )
                burn2 = st.sidebar.number_input(
                    "Burn-in iterace",
                    100,
                    10000,
                    500,
                    step=100,
                    key="burn_log_2",
                )

            elif chaos_typ2 == "Henonova mapa":
                length2 = st.sidebar.slider(
                    "Délka řady",
                    100,
                    5000,
                    1000,
                    step=100,
                    key="len_hen_2",
                )
                a2 = st.sidebar.number_input("Parametr a", value=1.4, step=0.1, key="a_hen_2")
                b2 = st.sidebar.number_input("Parametr b", value=0.3, step=0.05, key="b_hen_2")
                x02 = st.sidebar.number_input("Počáteční x₀", value=0.1, step=0.05, key="x0_hen_2")
                y02 = st.sidebar.number_input("Počáteční y₀", value=0.0, step=0.05, key="y0_hen_2")
                burn2 = st.sidebar.number_input(
                    "Burn-in iterace",
                    100,
                    10000,
                    500,
                    step=100,
                    key="burn_hen_2",
                )

            elif chaos_typ2 == "Lorenzův systém (x-složka)":
                length2 = st.sidebar.slider(
                    "Délka řady",
                    200,
                    10000,
                    2000,
                    step=200,
                    key="len_lor_2",
                )
                dt2 = st.sidebar.number_input(
                    "Krok integrace dt",
                    value=0.01,
                    step=0.005,
                    format="%.3f",
                    key="dt_lor_2",
                )
                sigma_l2 = st.sidebar.number_input("σ (sigma)", value=10.0, step=1.0, key="sigma_lor_2")
                rho_l2 = st.sidebar.number_input("ρ (rho)", value=28.0, step=1.0, key="rho_lor_2")
                beta_l2 = st.sidebar.number_input("β (beta)", value=8 / 3, step=0.1, key="beta_lor_2")
                burn2 = st.sidebar.number_input(
                    "Burn-in kroků",
                    500,
                    20000,
                    1000,
                    step=500,
                    key="burn_lor_2",
                )

            elif chaos_typ2 == "1/f šum (pink noise)":
                length2 = st.sidebar.slider(
                    "Délka řady",
                    100,
                    10000,
                    2000,
                    step=100,
                    key="len_pink_2",
                )

        # -----------------------------
        # Nahrát CSV – Série 2
        # -----------------------------
        elif mode2 == "Nahrát CSV":
            file2 = st.sidebar.file_uploader(
                "Nahraj CSV soubor",
                type="csv",
                key="csv_cmp_2",
            )

            if file2 is not None:
                csv2_has_header = st.sidebar.checkbox(
                    "CSV má hlavičku",
                    value=True,
                    key="csv2_header",
                )

                try:
                    df2_preview = read_csv_cached(
                        file2.getvalue(),
                        has_header=csv2_has_header,
                    )
                    err = None
                except Exception as e:
                    df2_preview = None
                    err = f"Chyba při načítání CSV: {e}"

                if err:
                    st.sidebar.error(err)
                else:
                    st.sidebar.caption("Náhled (prvních 5 řádků):")
                    st.sidebar.dataframe(df2_preview.head(), use_container_width=True)

                    selected_column2 = st.sidebar.selectbox(
                        "Vyber sloupec s hodnotami časové řady",
                        df2_preview.columns.tolist(),
                        key="csv2_col",
                    )

                    datetime_options2 = ["Žádný"] + df2_preview.columns.tolist()

                    csv2_datetime_column = st.sidebar.selectbox(
                        "Sloupec s datem/časem (volitelné)",
                        options=datetime_options2,
                        key="csv2_datetime_col",
                    )

                    if csv2_datetime_column != "Žádný":
                        selection_mode_cmp = st.sidebar.radio(
                            "Jak chceš vybírat rozsah?",
                            ["Podle indexu", "Podle data"],
                            key="csv2_selection_mode",
                        )

                    normalize_csv2 = st.sidebar.checkbox(
                        "Normalizovat (z-score)",
                        value=False,
                        key="csv2_norm",
                    )

                    if normalize_csv2:
                        st.sidebar.caption(
                            "Data jsou převedena na bezrozměrnou škálu (z-score). "
                            "Každá hodnota říká, o kolik směrodatných odchylek se liší od průměru."
                        )

                    if csv2_datetime_column != "Žádný":
                        st.sidebar.markdown("**Agregace časové řady**")

                        aggregation_freq_cmp = st.sidebar.selectbox(
                            "Agregační krok",
                            options=[
                                "bez agregace",
                                "1min",
                                "5min",
                                "10min",
                                "15min",
                                "30min",
                                "1h",
                                "1D",
                            ],
                            key="csv2_agg_freq",
                        )

                        if aggregation_freq_cmp != "bez agregace":
                            aggregation_method_cmp = st.sidebar.selectbox(
                                "Agregační metoda",
                                options=["mean", "median", "min", "max", "sum", "last"],
                                key="csv2_agg_method",
                            )

                    st.sidebar.markdown("**Výběr rozsahu dat z CSV**")

                    preview_meta2 = None
                    preview_err2 = None

                    if csv2_datetime_column != "Žádný" and selection_mode_cmp == "Podle data":
                        dt_series2 = pd.to_datetime(
                            df2_preview[csv2_datetime_column],
                            errors="coerce",
                        ).dropna()

                        if len(dt_series2) > 0:
                            min_dt2 = dt_series2.min()
                            max_dt2 = dt_series2.max()

                            st.sidebar.markdown("**Výběr časového rozsahu**")

                            col_dt2_1, col_dt2_2 = st.sidebar.columns(2)

                            with col_dt2_1:
                                csv2_start_date = st.date_input(
                                    "Datum od",
                                    value=min_dt2.date(),
                                    min_value=min_dt2.date(),
                                    max_value=max_dt2.date(),
                                    key="csv2_date_start",
                                )
                                start_hour_2 = st.selectbox(
                                    "Hodina od",
                                    options=list(range(24)),
                                    index=min_dt2.hour,
                                    key="csv2_start_hour",
                                )
                                start_minute_2 = st.selectbox(
                                    "Minuta od",
                                    options=list(range(60)),
                                    index=min_dt2.minute,
                                    key="csv2_start_minute",
                                )

                            with col_dt2_2:
                                csv2_end_date = st.date_input(
                                    "Datum do",
                                    value=max_dt2.date(),
                                    min_value=min_dt2.date(),
                                    max_value=max_dt2.date(),
                                    key="csv2_date_end",
                                )
                                end_hour_2 = st.selectbox(
                                    "Hodina do",
                                    options=list(range(24)),
                                    index=max_dt2.hour,
                                    key="csv2_end_hour",
                                )
                                end_minute_2 = st.selectbox(
                                    "Minuta do",
                                    options=list(range(60)),
                                    index=max_dt2.minute,
                                    key="csv2_end_minute",
                                )

                            csv2_start_datetime = dt.datetime.combine(
                                csv2_start_date,
                                dt.time(start_hour_2, start_minute_2),
                            )
                            csv2_end_datetime = dt.datetime.combine(
                                csv2_end_date,
                                dt.time(end_hour_2, end_minute_2),
                            )

                            _, _, preview_meta2, preview_err2 = load_csv_series(
                                file2,
                                df_input=df2_preview,
                                selected_column=selected_column2,
                                normalize=False,
                                start_index=0,
                                end_index=None,
                                has_header=csv2_has_header,
                                datetime_column=csv2_datetime_column,
                                selection_mode="date",
                                start_date=csv2_start_datetime,
                                end_date=csv2_end_datetime,
                                aggregation_freq=aggregation_freq_cmp,
                                aggregation_method=aggregation_method_cmp,
                            )

                            if preview_err2:
                                st.sidebar.warning(preview_err2)
                            elif preview_meta2 is not None:
                                st.sidebar.caption(
                                    f"Po načtení vznikne přibližně {preview_meta2['n_points']} bodů časové řady."
                                )
                        else:
                            st.sidebar.warning("Ve vybraném datetime sloupci nejsou platná data.")

                    else:
                        max_possible_index2 = max(0, len(df2_preview) - 1)
                        default_end_cmp = min(999, max_possible_index2)

                        if "csv2_range" not in st.session_state:
                            st.session_state.csv2_range = (0, default_end_cmp)

                        if "csv2_start_manual" not in st.session_state:
                            st.session_state.csv2_start_manual = st.session_state.csv2_range[0]

                        if "csv2_end_manual" not in st.session_state:
                            st.session_state.csv2_end_manual = st.session_state.csv2_range[1]

                        start_tmp2, end_tmp2 = st.session_state.csv2_range
                        start_tmp2 = min(max(0, start_tmp2), max_possible_index2)
                        end_tmp2 = min(max(0, end_tmp2), max_possible_index2)

                        if start_tmp2 > end_tmp2:
                            start_tmp2, end_tmp2 = end_tmp2, start_tmp2

                        st.session_state.csv2_range = (start_tmp2, end_tmp2)
                        st.session_state.csv2_start_manual = start_tmp2
                        st.session_state.csv2_end_manual = end_tmp2

                        csv2_start_index, csv2_end_index = st.sidebar.slider(
                            "Vyber rozsah řádků",
                            min_value=0,
                            max_value=max_possible_index2,
                            step=1,
                            key="csv2_range",
                            on_change=sync_cmp_from_slider,
                        )

                        col_range2_1, col_range2_2 = st.sidebar.columns(2)

                        with col_range2_1:
                            st.number_input(
                                "Od",
                                min_value=0,
                                max_value=max_possible_index2,
                                step=1,
                                key="csv2_start_manual",
                                on_change=sync_cmp_from_manual,
                            )

                        with col_range2_2:
                            st.number_input(
                                "Do",
                                min_value=0,
                                max_value=max_possible_index2,
                                step=1,
                                key="csv2_end_manual",
                                on_change=sync_cmp_from_manual,
                            )

                        csv2_start_index = st.session_state.csv2_start_manual
                        csv2_end_index = st.session_state.csv2_end_manual

                        _, _, preview_meta2, preview_err2 = load_csv_series(
                            file2,
                            df_input=df2_preview,
                            selected_column=selected_column2,
                            normalize=False,
                            start_index=csv2_start_index,
                            end_index=csv2_end_index,
                            has_header=csv2_has_header,
                            datetime_column=None if csv2_datetime_column == "Žádný" else csv2_datetime_column,
                            selection_mode="index",
                            start_date=None,
                            end_date=None,
                            aggregation_freq=aggregation_freq_cmp,
                            aggregation_method=aggregation_method_cmp,
                        )

                        if preview_err2:
                            st.sidebar.warning(preview_err2)
                        elif preview_meta2 is not None:
                            st.sidebar.caption(
                                f"Po načtení vznikne přibližně {preview_meta2['n_points']} bodů časové řady."
                            )

        generate2 = st.sidebar.button("Načíst / generovat sérii 2")

        if generate2:
            if mode2 == "Standardní signály":
                if typ2 == "Náhodná uniformní":
                    data2_candidate = np.random.uniform(low=low2, high=high2, size=length2)

                elif typ2 == "Náhodná normální":
                    data2_candidate = np.random.normal(
                        loc=mu2,
                        scale=sigma_input_2,
                        size=length2,
                    )

                elif typ2 == "Sinusovka":
                    x2 = np.arange(length2)
                    data2_candidate = amp2 * np.sin(2 * np.pi * freq2 * x2 / length2)

                elif typ2 == "Ruční vstup":
                    try:
                        data2_candidate = np.array([float(v.strip()) for v in txt2.split(",")])
                    except ValueError:
                        st.sidebar.error("Chybný formát série 2.")
                        data2_candidate = None

            elif mode2 == "Chaotické generátory":
                if chaos_typ2 == "Logistická mapa":
                    data2_candidate = generate_logistic_map(length2, r=r2, x0=x02, burn=burn2)

                elif chaos_typ2 == "Henonova mapa":
                    data2_candidate = generate_henon_map(
                        length2,
                        a=a2,
                        b=b2,
                        x0=x02,
                        y0=y02,
                        burn=burn2,
                    )

                elif chaos_typ2 == "Lorenzův systém (x-složka)":
                    data2_candidate = generate_lorenz_x(
                        length2,
                        dt=dt2,
                        sigma=sigma_l2,
                        rho=rho_l2,
                        beta=beta_l2,
                        burn=burn2,
                    )

                elif chaos_typ2 == "1/f šum (pink noise)":
                    data2_candidate = generate_pink_noise(length2)

            elif mode2 == "Nahrát CSV":
                if file2 is None:
                    st.sidebar.error("Nejprve nahraj CSV soubor.")
                    data2_candidate = None

                elif selected_column2 is None:
                    st.sidebar.error("Vyber sloupec s časovou řadou.")
                    data2_candidate = None

                else:
                    _, data2_candidate, meta2, err2 = load_csv_series(
                        file2,
                        df_input=df2_preview,
                        selected_column=selected_column2,
                        normalize=normalize_csv2,
                        start_index=csv2_start_index,
                        end_index=csv2_end_index,
                        has_header=csv2_has_header,
                        datetime_column=None if csv2_datetime_column == "Žádný" else csv2_datetime_column,
                        selection_mode=(
                            "date"
                            if (csv2_datetime_column != "Žádný" and selection_mode_cmp == "Podle data")
                            else "index"
                        ),
                        start_date=(
                            csv2_start_datetime
                            if (csv2_datetime_column != "Žádný" and selection_mode_cmp == "Podle data")
                            else None
                        ),
                        end_date=(
                            csv2_end_datetime
                            if (csv2_datetime_column != "Žádný" and selection_mode_cmp == "Podle data")
                            else None
                        ),
                        aggregation_freq=aggregation_freq_cmp,
                        aggregation_method=aggregation_method_cmp,
                    )

                    if err2:
                        st.sidebar.error(err2)
                        data2_candidate = None
                        meta2 = None

            if data2_candidate is None:
                st.sidebar.error("Série 2 zatím není připravená – zkontroluj nastavení.")
            else:
                st.session_state.data2 = data2_candidate.copy()
                st.session_state.meta2 = meta2
                st.session_state.show_cmp_horiz1 = False
                st.session_state.show_cmp_horiz2 = False

                if mode2 == "Nahrát CSV":
                    st.session_state.series_name2 = selected_column2
                    st.session_state.series_normalized2 = normalize_csv2
                    st.session_state.series_aggregation2 = aggregation_freq_cmp
                else:
                    st.session_state.series_name2 = typ2 if typ2 is not None else chaos_typ2
                    st.session_state.series_normalized2 = False
                    st.session_state.series_aggregation2 = None

                st.sidebar.success(f"Načteno {len(data2_candidate)} hodnot pro sérii 2.")

                if mode2 == "Nahrát CSV":
                    if meta2 is not None and meta2["datetime_used"]:
                        if meta2["selection_mode"] == "date":
                            st.sidebar.caption(
                                f"Datový rozsah série 2: {csv2_start_datetime.strftime('%d.%m.%Y %H:%M:%S')} → "
                                f"{csv2_end_datetime.strftime('%d.%m.%Y %H:%M:%S')}"
                            )
                        else:
                            st.sidebar.caption(f"Indexy série 2: {csv2_start_index} → {csv2_end_index}")

                        if meta2["aggregation_freq"] not in (None, "bez agregace"):
                            st.sidebar.caption(
                                f"Agregace série 2: {meta2['aggregation_freq']} | metoda: {meta2['aggregation_method']}"
                            )

                    else:
                        st.sidebar.caption(f"Indexy série 2: {csv2_start_index} → {csv2_end_index}")

        data2 = st.session_state.data2
        meta2_saved = st.session_state.meta2

        if data2 is None:
            st.info(
                "V levém panelu nastav parametry **Série 2** a klikni na "
                "**„Načíst / generovat sérii 2“**."
            )
        else:
            G2 = build_hvg_cached(data2)
            metrics2 = compute_graph_metrics(G2)

            n2 = metrics2["n_nodes"]
            m2 = metrics2["n_edges"]
            degs2 = metrics2["degrees"]
            avg_deg2 = metrics2["avg_deg"]
            C2 = metrics2["C"]
            L2 = metrics2["L"]
            diam2 = metrics2["diam"]
            assort2 = metrics2["assort"]
            L_rand2 = metrics2["L_rand"]
            C_rand2 = metrics2["C_rand"]
            sigma2 = metrics2["sigma"]
            analyzer2 = metrics2["analyzer"]

            degree_metrics_2 = compute_degree_distribution_metrics(degs2)
            unique_deg2_main = degree_metrics_2["unique_deg"]
            counts2_main = degree_metrics_2["counts"]
            pk2_main = degree_metrics_2["pk"]
            entropy_deg2 = degree_metrics_2["entropy_deg"]
            entropy_deg_norm2 = degree_metrics_2["entropy_deg_norm"]

            powerlaw_p_result_2 = None
            powerlaw_R_result_2 = None

            G2_conf = None
            conf_metrics2 = None
            n2c = None
            m2c = None
            degs2c = None
            avg_deg2c = None
            C2c = None
            L2c = None
            diam2c = None
            assort2c = None
            L_rand2c = None
            C_rand2c = None
            sigma2c = None

            # =============================
            # Časové řady série 1 a 2
            # =============================
            col_series1, col_series2 = st.columns(2)

            with col_series1:
                st.markdown("### Série 1 – aktuálně vygenerovaná časová řada")
                st.write(
                    f"- Délka: **{len(data1)}**, "
                    f"Průměr: **{data1.mean():.3f}**, "
                    f"Rozptyl: **{data1.var():.3f}**"
                )

                df1 = pd.DataFrame({"index": np.arange(len(data1)), "value": data1})
                fig1 = px.line(df1, x="index", y="value", markers=True, title="Série 1")
                fig1.update_traces(marker_size=6)

                if st.session_state.show_cmp_horiz1:
                    shapes1 = []
                    for i, j in G1.edges():
                        y = min(data1[i], data1[j])
                        shapes1.append(
                            dict(
                                type="line",
                                x0=i,
                                y0=y,
                                x1=j,
                                y1=y,
                                line=dict(color="gray", width=1),
                            )
                        )
                    fig1.update_layout(shapes=shapes1)

                st.plotly_chart(fig1, use_container_width=True)

            with col_series2:
                st.markdown("### Série 2 – nastavená v levém panelu")

                if meta2_saved is not None and meta2_saved.get("normalized", False):
                    st.write(
                        f"- Délka: **{len(data2)}**, "
                        f"Původní průměr: **{meta2_saved['original_mean']:.3f}**, "
                        f"Původní rozptyl: **{meta2_saved['original_var']:.3f}**"
                    )
                    st.write(
                        f"- Průměr po normalizaci: **{meta2_saved['processed_mean']:.3f}**, "
                        f"Rozptyl po normalizaci: **{meta2_saved['processed_var']:.3f}**"
                    )
                else:
                    st.write(
                        f"- Délka: **{len(data2)}**, "
                        f"Průměr: **{data2.mean():.3f}**, "
                        f"Rozptyl: **{data2.var():.3f}**"
                    )

                df2 = pd.DataFrame({"index": np.arange(len(data2)), "value": data2})
                fig2 = px.line(df2, x="index", y="value", markers=True, title="Série 2")
                fig2.update_traces(marker_size=6)

                if st.session_state.show_cmp_horiz2:
                    shapes2 = []
                    for i, j in G2.edges():
                        y = min(data2[i], data2[j])
                        shapes2.append(
                            dict(
                                type="line",
                                x0=i,
                                y0=y,
                                x1=j,
                                y1=y,
                                line=dict(color="gray", width=1),
                            )
                        )
                    fig2.update_layout(shapes=shapes2)

                st.plotly_chart(fig2, use_container_width=True)

            btn_col1, btn_col2 = st.columns(2)

            with btn_col1:
                if st.button("HVG linky (vodorovné) – Série 1", key="btn_cmp_horiz1"):
                    st.session_state.show_cmp_horiz1 = not st.session_state.show_cmp_horiz1
                    st.rerun()

            with btn_col2:
                if st.button("HVG linky (vodorovné) – Série 2", key="btn_cmp_horiz2"):
                    st.session_state.show_cmp_horiz2 = not st.session_state.show_cmp_horiz2
                    st.rerun()

            # =============================
            # Společný výběr sekcí
            # =============================
            section_options_cmp = [
                "Metriky HVG",
                "Propojení časová řada ↔ HVG",
                "Lokální analýza úseku časové řady",
                "Podgraf HVG",
                "Rozdělení stupňů + power-law",
                "Arc Diagram HVG",
                "Konfigurační graf (null model)",
                "Shrnutí analýzy",
                "Export HVG a metrik",
            ]
            selected_sections_cmp = st.multiselect(
                "Co chceš pod porovnáním zobrazit pro **obě** HVG?",
                options=section_options_cmp,
                default=section_options_cmp,
            )

            # =============================
            # HVG grafy vedle sebe
            # =============================
            st.markdown("### HVG grafy vedle sebe")

            col_g1, col_g2 = st.columns(2)

            with col_g1:
                pos1 = compute_graph_layout(G1, layout_type="spring", seed=42)
                edge_trace1, node_trace1 = prepare_network_traces(
                    G1,
                    pos1,
                    node_color="skyblue",
                    node_size=10,
                    edge_color="#888",
                    edge_width=1,
                    show_labels=False,
                    hover_texts=[f"Vrchol: {n}" for n in G1.nodes()],
                )
                fig_g1 = go.Figure(data=[edge_trace1, node_trace1])
                fig_g1.update_layout(
                    title="HVG – série 1",
                    showlegend=False,
                    hovermode="closest",
                    margin=dict(b=20, l=5, r=5, t=40),
                )
                st.plotly_chart(fig_g1, use_container_width=True)

            with col_g2:
                pos2 = compute_graph_layout(G2, layout_type="spring", seed=42)
                edge_trace2, node_trace2 = prepare_network_traces(
                    G2,
                    pos2,
                    node_color="lightgreen",
                    node_size=10,
                    edge_color="#888",
                    edge_width=1,
                    show_labels=False,
                    hover_texts=[f"Vrchol: {n}" for n in G2.nodes()],
                )
                fig_g2 = go.Figure(data=[edge_trace2, node_trace2])
                fig_g2.update_layout(
                    title="HVG – série 2",
                    showlegend=False,
                    hovermode="closest",
                    margin=dict(b=20, l=5, r=5, t=40),
                )
                st.plotly_chart(fig_g2, use_container_width=True)

            # =============================
            # Metriky HVG
            # =============================
            if "Metriky HVG" in selected_sections_cmp:
                st.markdown("### Porovnání metrik HVG")

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
                        st.write(f"- Diametr grafu: **{diam1}**")
                    else:
                        st.write("- Diametr grafu: *není k dispozici*")
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
                        st.write(f"- Diametr grafu: **{diam2}**")
                    else:
                        st.write("- Diametr grafu: *není k dispozici*")
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
            # Propojení časová řada ↔ HVG
            # =============================
            if "Propojení časová řada ↔ HVG" in selected_sections_cmp:
                st.markdown("### Propojení časové řady a HVG (Série 1 & 2)")

                tab1, tab2 = st.tabs(["Série 1", "Série 2"])

                with tab1:
                    if n1 > 0:
                        idx1 = st.number_input(
                            "Index pro zvýraznění (Série 1)",
                            min_value=0,
                            max_value=n1 - 1,
                            value=0,
                            step=1,
                            key="cmp_idx1",
                        )
                        neighbors1 = list(G1.adj[idx1])
                        st.markdown(
                            f"- Vybraný vrchol: **{idx1}**, "
                            f"stupeň: **{G1.degree(idx1)}**, "
                            f"sousedé: **{neighbors1}**"
                        )

                        df_ts1 = pd.DataFrame({"index": np.arange(len(data1)), "value": data1})
                        fig_ts1 = px.line(
                            df_ts1,
                            x="index",
                            y="value",
                            markers=True,
                            title="Série 1 – časová řada (highlight)",
                        )
                        fig_ts1.update_traces(marker_size=6)
                        fig_ts1.add_trace(
                            go.Scatter(
                                x=[idx1],
                                y=[data1[idx1]],
                                mode="markers",
                                marker=dict(size=14, color="red"),
                                name="Vybraný bod",
                            )
                        )
                        if len(neighbors1) > 0:
                            fig_ts1.add_trace(
                                go.Scatter(
                                    x=neighbors1,
                                    y=[data1[i] for i in neighbors1],
                                    mode="markers",
                                    marker=dict(size=10, color="orange"),
                                    name="Sousedé",
                                )
                            )
                        st.plotly_chart(fig_ts1, use_container_width=True)

                        pos1_h = compute_graph_layout(G1, layout_type="spring", seed=42)

                        edge_x1h, edge_y1h = [], []
                        for u, v in G1.edges():
                            x0, y0 = pos1_h[u]
                            x1_, y1_ = pos1_h[v]
                            edge_x1h += [x0, x1_, None]
                            edge_y1h += [y0, y1_, None]

                        edge_trace1h = go.Scatter(
                            x=edge_x1h,
                            y=edge_y1h,
                            mode="lines",
                            line=dict(width=1, color="#aaa"),
                            hoverinfo="none",
                        )

                        node_x1h, node_y1h = [], []
                        for node in G1.nodes():
                            x, y = pos1_h[node]
                            node_x1h.append(x)
                            node_y1h.append(y)

                        node_trace1h = go.Scatter(
                            x=node_x1h,
                            y=node_y1h,
                            mode="markers",
                            marker=dict(size=10, color="skyblue"),
                            hoverinfo="none",
                        )

                        hl_nodes1 = [idx1] + neighbors1
                        hl_x1, hl_y1 = [], []
                        for node in hl_nodes1:
                            x, y = pos1_h[node]
                            hl_x1.append(x)
                            hl_y1.append(y)

                        highlight1 = go.Scatter(
                            x=hl_x1,
                            y=hl_y1,
                            mode="markers+text",
                            text=[str(i) for i in hl_nodes1],
                            textposition="top center",
                            marker=dict(size=14, color="red"),
                            hoverinfo="text",
                            hovertext=[f"Vrchol: {i}" for i in hl_nodes1],
                        )

                        fig_h1 = go.Figure(data=[edge_trace1h, node_trace1h, highlight1])
                        fig_h1.update_layout(
                            title="HVG – Série 1 (highlight)",
                            showlegend=False,
                            hovermode="closest",
                            margin=dict(b=20, l=5, r=5, t=40),
                        )
                        st.plotly_chart(fig_h1, use_container_width=True)

                with tab2:
                    if n2 > 0:
                        idx2 = st.number_input(
                            "Index pro zvýraznění (Série 2)",
                            min_value=0,
                            max_value=n2 - 1,
                            value=0,
                            step=1,
                            key="cmp_idx2",
                        )
                        neighbors2 = list(G2.adj[idx2])
                        st.markdown(
                            f"- Vybraný vrchol: **{idx2}**, "
                            f"stupeň: **{G2.degree(idx2)}**, "
                            f"sousedé: **{neighbors2}**"
                        )

                        df_ts2 = pd.DataFrame({"index": np.arange(len(data2)), "value": data2})
                        fig_ts2 = px.line(
                            df_ts2,
                            x="index",
                            y="value",
                            markers=True,
                            title="Série 2 – časová řada (highlight)",
                        )
                        fig_ts2.update_traces(marker_size=6)
                        fig_ts2.add_trace(
                            go.Scatter(
                                x=[idx2],
                                y=[data2[idx2]],
                                mode="markers",
                                marker=dict(size=14, color="red"),
                                name="Vybraný bod",
                            )
                        )
                        if len(neighbors2) > 0:
                            fig_ts2.add_trace(
                                go.Scatter(
                                    x=neighbors2,
                                    y=[data2[i] for i in neighbors2],
                                    mode="markers",
                                    marker=dict(size=10, color="orange"),
                                    name="Sousedé",
                                )
                            )
                        st.plotly_chart(fig_ts2, use_container_width=True)

                        pos2_h = compute_graph_layout(G2, layout_type="spring", seed=42)

                        edge_x2h, edge_y2h = [], []
                        for u, v in G2.edges():
                            x0, y0 = pos2_h[u]
                            x2_, y2_ = pos2_h[v]
                            edge_x2h += [x0, x2_, None]
                            edge_y2h += [y0, y2_, None]

                        edge_trace2h = go.Scatter(
                            x=edge_x2h,
                            y=edge_y2h,
                            mode="lines",
                            line=dict(width=1, color="#aaa"),
                            hoverinfo="none",
                        )

                        node_x2h, node_y2h = [], []
                        for node in G2.nodes():
                            x, y = pos2_h[node]
                            node_x2h.append(x)
                            node_y2h.append(y)

                        node_trace2h = go.Scatter(
                            x=node_x2h,
                            y=node_y2h,
                            mode="markers",
                            marker=dict(size=10, color="lightgreen"),
                            hoverinfo="none",
                        )

                        hl_nodes2 = [idx2] + neighbors2
                        hl_x2, hl_y2 = [], []
                        for node in hl_nodes2:
                            x, y = pos2_h[node]
                            hl_x2.append(x)
                            hl_y2.append(y)

                        highlight2 = go.Scatter(
                            x=hl_x2,
                            y=hl_y2,
                            mode="markers+text",
                            text=[str(i) for i in hl_nodes2],
                            textposition="top center",
                            marker=dict(size=14, color="red"),
                            hoverinfo="text",
                            hovertext=[f"Vrchol: {i}" for i in hl_nodes2],
                        )

                        fig_h2 = go.Figure(data=[edge_trace2h, node_trace2h, highlight2])
                        fig_h2.update_layout(
                            title="HVG – Série 2 (highlight)",
                            showlegend=False,
                            hovermode="closest",
                            margin=dict(b=20, l=5, r=5, t=40),
                        )
                        st.plotly_chart(fig_h2, use_container_width=True)

            # =============================
            # Lokální analýza
            # =============================
            if "Lokální analýza úseku časové řady" in selected_sections_cmp:
                st.markdown("### Lokální analýza úseku časové řady (Série 1 & 2)")

                col_loc1, col_loc2 = st.columns(2)

                with col_loc1:
                    st.markdown("**Série 1 – lokální úsek**")
                    if len(data1) >= 2:
                        i1_start, i1_end = st.slider(
                            "Rozsah indexů (Série 1)",
                            min_value=0,
                            max_value=len(data1) - 1,
                            value=(0, min(len(data1) - 1, max(1, len(data1) // 5))),
                            key="loc_range_1",
                        )
                        if i1_start > i1_end:
                            i1_start, i1_end = i1_end, i1_start

                        seg1 = data1[i1_start : i1_end + 1]
                        st.write(
                            f"- Délka úseku: **{len(seg1)}**, "
                            f"rozsah indexů: **[{i1_start}, {i1_end}]**"
                        )

                        if len(seg1) > 0:
                            ent1 = shannon_entropy(seg1, bins="auto")
                            st.write(
                                f"- Průměr (lokální): **{seg1.mean():.3f}**  \n"
                                f"- Rozptyl (lokální): **{seg1.var():.3f}**  \n"
                                f"- Min: **{seg1.min():.3f}**, Max: **{seg1.max():.3f}**  \n"
                                f"- Shannonova entropie: **{ent1:.3f}**"
                            )

                            if len(seg1) >= 2:
                                G1_seg = build_hvg_cached(seg1)
                                n1s = G1_seg.number_of_nodes()
                                m1s = G1_seg.number_of_edges()
                                degs1s = [d for _, d in G1_seg.degree()]
                                avg_deg1s = float(np.mean(degs1s)) if len(degs1s) > 0 else 0.0

                                try:
                                    C1s = nx.average_clustering(G1_seg)
                                except Exception:
                                    C1s = float("nan")

                                is_conn1s = nx.is_connected(G1_seg) if n1s > 0 else False
                                L1s, diam1s = None, None

                                if is_conn1s and n1s > 1:
                                    try:
                                        L1s = nx.average_shortest_path_length(G1_seg)
                                    except Exception:
                                        L1s = None
                                    try:
                                        diam1s = nx.diameter(G1_seg)
                                    except Exception:
                                        diam1s = None

                                st.markdown("**Lokální HVG – Série 1**")
                                st.write(
                                    f"- Počet vrcholů: **{n1s}**, počet hran: **{m1s}**, průměrný stupeň: **{avg_deg1s:.3f}**"
                                )
                                st.write(f"- Clustering: **{C1s:.3f}**")
                                if L1s is not None:
                                    st.write(f"- Průměrná délka cesty: **{L1s:.3f}**")
                                if diam1s is not None:
                                    st.write(f"- Průměr grafu: **{diam1s}**")

                with col_loc2:
                    st.markdown("**Série 2 – lokální úsek**")
                    if len(data2) >= 2:
                        i2_start, i2_end = st.slider(
                            "Rozsah indexů (Série 2)",
                            min_value=0,
                            max_value=len(data2) - 1,
                            value=(0, min(len(data2) - 1, max(1, len(data2) // 5))),
                            key="loc_range_2",
                        )
                        if i2_start > i2_end:
                            i2_start, i2_end = i2_end, i2_start

                        seg2 = data2[i2_start : i2_end + 1]
                        st.write(
                            f"- Délka úseku: **{len(seg2)}**, "
                            f"rozsah indexů: **[{i2_start}, {i2_end}]**"
                        )

                        if len(seg2) > 0:
                            ent2 = shannon_entropy(seg2, bins="auto")
                            st.write(
                                f"- Průměr (lokální): **{seg2.mean():.3f}**  \n"
                                f"- Rozptyl (lokální): **{seg2.var():.3f}**  \n"
                                f"- Min: **{seg2.min():.3f}**, Max: **{seg2.max():.3f}**  \n"
                                f"- Shannonova entropie: **{ent2:.3f}**"
                            )

                            if len(seg2) >= 2:
                                G2_seg = build_hvg_cached(seg2)
                                n2s = G2_seg.number_of_nodes()
                                m2s = G2_seg.number_of_edges()
                                degs2s = [d for _, d in G2_seg.degree()]
                                avg_deg2s = float(np.mean(degs2s)) if len(degs2s) > 0 else 0.0

                                try:
                                    C2s = nx.average_clustering(G2_seg)
                                except Exception:
                                    C2s = float("nan")

                                is_conn2s = nx.is_connected(G2_seg) if n2s > 0 else False
                                L2s, diam2s = None, None

                                if is_conn2s and n2s > 1:
                                    try:
                                        L2s = nx.average_shortest_path_length(G2_seg)
                                    except Exception:
                                        L2s = None
                                    try:
                                        diam2s = nx.diameter(G2_seg)
                                    except Exception:
                                        diam2s = None

                                st.markdown("**Lokální HVG – Série 2**")
                                st.write(
                                    f"- Počet vrcholů: **{n2s}**, počet hran: **{m2s}**, průměrný stupeň: **{avg_deg2s:.3f}**"
                                )
                                st.write(f"- Clustering: **{C2s:.3f}**")
                                if L2s is not None:
                                    st.write(f"- Průměrná délka cesty: **{L2s:.3f}**")
                                if diam2s is not None:
                                    st.write(f"- Průměr grafu: **{diam2s}**")

            # =============================
            # Podgraf HVG
            # =============================
            if "Podgraf HVG" in selected_sections_cmp:
                st.markdown("### Podgraf HVG pro obě série")

                sub_nodes_text = st.text_input(
                    "Seznam vrcholů pro podgraf (indexy oddělené čárkou nebo mezerami) – použijí se na obě HVG",
                    value="0, 1, 2",
                    key="sub_nodes_cmp",
                )

                sub_nodes = []
                for token in re.split(r"[,\s;]+", sub_nodes_text):
                    token = token.strip()
                    if token == "":
                        continue
                    try:
                        idx = int(token)
                        if idx >= 0:
                            sub_nodes.append(idx)
                    except ValueError:
                        continue
                sub_nodes = sorted(set(sub_nodes))

                col_sub1, col_sub2 = st.columns(2)

                with col_sub1:
                    st.markdown("**Podgraf – Série 1**")
                    valid1 = [i for i in sub_nodes if i < n1]
                    if len(valid1) == 0:
                        st.info("Žádný zadaný index nepadá do rozsahu vrcholů Série 1.")
                    else:
                        G1_sub = G1.subgraph(valid1).copy()
                        st.write(
                            f"- Vrcholy: **{G1_sub.number_of_nodes()}**, hrany: **{G1_sub.number_of_edges()}**"
                        )

                        degs1_sub = [d for _, d in G1_sub.degree()]
                        avg_deg1_sub = float(np.mean(degs1_sub)) if len(degs1_sub) > 0 else 0.0

                        try:
                            C1_sub = nx.average_clustering(G1_sub)
                        except Exception:
                            C1_sub = float("nan")

                        is_conn1_sub = nx.is_connected(G1_sub) if G1_sub.number_of_nodes() > 0 else False
                        L1_sub, diam1_sub = None, None

                        if is_conn1_sub and G1_sub.number_of_nodes() > 1:
                            try:
                                L1_sub = nx.average_shortest_path_length(G1_sub)
                            except Exception:
                                L1_sub = None
                            try:
                                diam1_sub = nx.diameter(G1_sub)
                            except Exception:
                                diam1_sub = None

                        st.write(f"- Průměrný stupeň: **{avg_deg1_sub:.3f}**")
                        st.write(f"- Clustering: **{C1_sub:.3f}**")
                        if L1_sub is not None:
                            st.write(f"- Průměrná délka cesty: **{L1_sub:.3f}**")
                        if diam1_sub is not None:
                            st.write(f"- Průměr grafu: **{diam1_sub}**")

                        edge_trace1_sub, node_trace1_sub = prepare_network_traces(
                            G1_sub,
                            pos1,
                            node_color="lightcoral",
                            node_size=10,
                            edge_color="#888",
                            edge_width=1,
                            show_labels=True,
                            hover_texts=[f"Vrchol: {n}" for n in G1_sub.nodes()],
                        )
                        fig1_sub = go.Figure(data=[edge_trace1_sub, node_trace1_sub])
                        fig1_sub.update_layout(
                            title="Podgraf HVG – Série 1",
                            showlegend=False,
                            hovermode="closest",
                            margin=dict(b=20, l=5, r=5, t=40),
                        )
                        st.plotly_chart(fig1_sub, use_container_width=True)

                with col_sub2:
                    st.markdown("**Podgraf – Série 2**")
                    valid2 = [i for i in sub_nodes if i < n2]
                    if len(valid2) == 0:
                        st.info("Žádný zadaný index nepadá do rozsahu vrcholů Série 2.")
                    else:
                        G2_sub = G2.subgraph(valid2).copy()
                        st.write(
                            f"- Vrcholy: **{G2_sub.number_of_nodes()}**, hrany: **{G2_sub.number_of_edges()}**"
                        )

                        degs2_sub = [d for _, d in G2_sub.degree()]
                        avg_deg2_sub = float(np.mean(degs2_sub)) if len(degs2_sub) > 0 else 0.0

                        try:
                            C2_sub = nx.average_clustering(G2_sub)
                        except Exception:
                            C2_sub = float("nan")

                        is_conn2_sub = nx.is_connected(G2_sub) if G2_sub.number_of_nodes() > 0 else False
                        L2_sub, diam2_sub = None, None

                        if is_conn2_sub and G2_sub.number_of_nodes() > 1:
                            try:
                                L2_sub = nx.average_shortest_path_length(G2_sub)
                            except Exception:
                                L2_sub = None
                            try:
                                diam2_sub = nx.diameter(G2_sub)
                            except Exception:
                                diam2_sub = None

                        st.write(f"- Průměrný stupeň: **{avg_deg2_sub:.3f}**")
                        st.write(f"- Clustering: **{C2_sub:.3f}**")
                        if L2_sub is not None:
                            st.write(f"- Průměrná délka cesty: **{L2_sub:.3f}**")
                        if diam2_sub is not None:
                            st.write(f"- Průměr grafu: **{diam2_sub}**")

                        edge_trace2_sub, node_trace2_sub = prepare_network_traces(
                            G2_sub,
                            pos2,
                            node_color="lightcoral",
                            node_size=10,
                            edge_color="#888",
                            edge_width=1,
                            show_labels=True,
                            hover_texts=[f"Vrchol: {n}" for n in G2_sub.nodes()],
                        )
                        fig2_sub = go.Figure(data=[edge_trace2_sub, node_trace2_sub])
                        fig2_sub.update_layout(
                            title="Podgraf HVG – Série 2",
                            showlegend=False,
                            hovermode="closest",
                            margin=dict(b=20, l=5, r=5, t=40),
                        )
                        st.plotly_chart(fig2_sub, use_container_width=True)

            # =============================
            # Konfigurační graf (null model)
            # =============================
            if "Konfigurační graf (null model)" in selected_sections_cmp:
                st.markdown("### Konfigurační graf (null model) pro obě série")

                if conf_metrics1 is None:
                    G1_conf, conf_metrics1 = compute_configuration_model_metrics(G1, seed=42)
                    n1c = conf_metrics1["n_nodes"]
                    m1c = conf_metrics1["n_edges"]
                    degs1c = conf_metrics1["degrees"]
                    avg_deg1c = conf_metrics1["avg_deg"]
                    C1c = conf_metrics1["C"]
                    L1c = conf_metrics1["L"]
                    diam1c = conf_metrics1["diam"]
                    assort1c = conf_metrics1["assort"]
                    L_rand1c = conf_metrics1["L_rand"]
                    C_rand1c = conf_metrics1["C_rand"]
                    sigma1c = conf_metrics1["sigma"]

                if conf_metrics2 is None:
                    G2_conf, conf_metrics2 = compute_configuration_model_metrics(G2, seed=42)
                    n2c = conf_metrics2["n_nodes"]
                    m2c = conf_metrics2["n_edges"]
                    degs2c = conf_metrics2["degrees"]
                    avg_deg2c = conf_metrics2["avg_deg"]
                    C2c = conf_metrics2["C"]
                    L2c = conf_metrics2["L"]
                    diam2c = conf_metrics2["diam"]
                    assort2c = conf_metrics2["assort"]
                    L_rand2c = conf_metrics2["L_rand"]
                    C_rand2c = conf_metrics2["C_rand"]
                    sigma2c = conf_metrics2["sigma"]

                col_conf1, col_conf2 = st.columns(2)

                with col_conf1:
                    st.markdown("**Konfigurační graf – Série 1**")
                    st.write(f"- Počet vrcholů: **{n1c}**")
                    st.write(f"- Počet hran: **{m1c}**")
                    st.write(f"- Průměrný stupeň: **{avg_deg1c:.3f}**")
                    if L1c is not None:
                        st.write(f"- Průměrná délka cesty L_conf: **{L1c:.3f}**")
                    else:
                        st.write("- Průměrná délka cesty L_conf: *nelze spočítat (nesouvislý graf)*")
                    if diam1c is not None:
                        st.write(f"- Průměr grafu (diameter_conf): **{diam1c}**")
                    else:
                        st.write("- Průměr grafu (diameter_conf): *není k dispozici*")
                    st.write(f"- Clustering C_conf: **{C1c:.3f}**")
                    if L_rand1c is not None and C_rand1c is not None and C_rand1c != 0:
                        st.write(
                            "- Náhodný graf (odhad):  \n"
                            f"  - L_rand_conf ≈ **{L_rand1c:.3f}**  \n"
                            f"  - C_rand_conf ≈ **{C_rand1c:.5f}**"
                        )
                    if sigma1c is not None and not np.isnan(sigma1c):
                        st.write(f"- Small-world index σ_conf: **{sigma1c:.2f}**")

                    pos1c = compute_graph_layout(G1_conf, layout_type="spring", seed=42)
                    edge_trace1c, node_trace1c = prepare_network_traces(
                        G1_conf,
                        pos1c,
                        node_color="lightgreen",
                        node_size=8,
                        edge_color="#aaa",
                        edge_width=1,
                        show_labels=False,
                        hover_texts=[f"Vrchol: {n}" for n in G1_conf.nodes()],
                    )
                    fig_conf1 = go.Figure(data=[edge_trace1c, node_trace1c])
                    fig_conf1.update_layout(
                        title="Konfigurační graf – Série 1",
                        showlegend=False,
                        hovermode="closest",
                        margin=dict(b=20, l=5, r=5, t=40),
                    )
                    st.plotly_chart(fig_conf1, use_container_width=True)

                with col_conf2:
                    st.markdown("**Konfigurační graf – Série 2**")
                    st.write(f"- Počet vrcholů: **{n2c}**")
                    st.write(f"- Počet hran: **{m2c}**")
                    st.write(f"- Průměrný stupeň: **{avg_deg2c:.3f}**")
                    if L2c is not None:
                        st.write(f"- Průměrná délka cesty L_conf: **{L2c:.3f}**")
                    else:
                        st.write("- Průměrná délka cesty L_conf: *nelze spočítat (nesouvislý graf)*")
                    if diam2c is not None:
                        st.write(f"- Průměr grafu (diameter_conf): **{diam2c}**")
                    else:
                        st.write("- Průměr grafu (diameter_conf): *není k dispozici*")
                    st.write(f"- Clustering C_conf: **{C2c:.3f}**")
                    if L_rand2c is not None and C_rand2c is not None and C_rand2c != 0:
                        st.write(
                            "- Náhodný graf (odhad):  \n"
                            f"  - L_rand_conf ≈ **{L_rand2c:.3f}**  \n"
                            f"  - C_rand_conf ≈ **{C_rand2c:.5f}**"
                        )
                    if sigma2c is not None and not np.isnan(sigma2c):
                        st.write(f"- Small-world index σ_conf: **{sigma2c:.2f}**")

                    pos2c = compute_graph_layout(G2_conf, layout_type="spring", seed=42)
                    edge_trace2c, node_trace2c = prepare_network_traces(
                        G2_conf,
                        pos2c,
                        node_color="lightgreen",
                        node_size=8,
                        edge_color="#aaa",
                        edge_width=1,
                        show_labels=False,
                        hover_texts=[f"Vrchol: {n}" for n in G2_conf.nodes()],
                    )
                    fig_conf2 = go.Figure(data=[edge_trace2c, node_trace2c])
                    fig_conf2.update_layout(
                        title="Konfigurační graf – Série 2",
                        showlegend=False,
                        hovermode="closest",
                        margin=dict(b=20, l=5, r=5, t=40),
                    )
                    st.plotly_chart(fig_conf2, use_container_width=True)

            # =============================
            # Rozdělení stupňů + power-law
            # =============================
            if "Rozdělení stupňů + power-law" in selected_sections_cmp:
                st.markdown("### Porovnání stupňového rozdělení")

                unique_deg1 = degree_metrics_1["unique_deg"]
                counts1 = degree_metrics_1["counts"]
                pk1 = degree_metrics_1["pk"]
                entropy_deg1 = degree_metrics_1["entropy_deg"]
                entropy_deg_norm1 = degree_metrics_1["entropy_deg_norm"]

                unique_deg2 = degree_metrics_2["unique_deg"]
                counts2 = degree_metrics_2["counts"]
                pk2 = degree_metrics_2["pk"]
                entropy_deg2 = degree_metrics_2["entropy_deg"]
                entropy_deg_norm2 = degree_metrics_2["entropy_deg_norm"]

                entropy_level1, entropy_text1 = classify_entropy_level(entropy_deg_norm1)
                entropy_level2, entropy_text2 = classify_entropy_level(entropy_deg_norm2)

                col_deg_s1, col_deg_s2 = st.columns(2)

                with col_deg_s1:
                    st.markdown("**Série 1 – základní metriky stupňového rozdělení**")
                    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
                    with metric_col1:
                        st.metric("Průměrný stupeň", f"{np.mean(degs1):.3f}")
                    with metric_col2:
                        st.metric("Medián", f"{np.median(degs1):.3f}")
                    with metric_col3:
                        st.metric("Maximum", f"{np.max(degs1)}")
                    with metric_col4:
                        st.metric("Entropie", f"{entropy_deg1:.3f}")
                    with metric_col5:
                        st.metric("Norm. entropie", f"{entropy_deg_norm1:.3f}", delta=entropy_level1)

                with col_deg_s2:
                    st.markdown("**Série 2 – základní metriky stupňového rozdělení**")
                    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
                    with metric_col1:
                        st.metric("Průměrný stupeň", f"{np.mean(degs2):.3f}")
                    with metric_col2:
                        st.metric("Medián", f"{np.median(degs2):.3f}")
                    with metric_col3:
                        st.metric("Maximum", f"{np.max(degs2)}")
                    with metric_col4:
                        st.metric("Entropie", f"{entropy_deg2:.3f}")
                    with metric_col5:
                        st.metric("Norm. entropie", f"{entropy_deg_norm2:.3f}", delta=entropy_level2)

                st.markdown("#### Stručná interpretace porovnání")

                comparison_deg_parts = []

                if entropy_deg_norm1 > entropy_deg_norm2:
                    comparison_deg_parts.append(
                        "Série 1 má vyšší normalizovanou entropii stupňového rozdělení než Série 2, takže její HVG vykazuje větší rozmanitost stupňů."
                    )
                elif entropy_deg_norm2 > entropy_deg_norm1:
                    comparison_deg_parts.append(
                        "Série 2 má vyšší normalizovanou entropii stupňového rozdělení než Série 1, takže její HVG vykazuje větší rozmanitost stupňů."
                    )
                else:
                    comparison_deg_parts.append(
                        "Obě série mají velmi podobnou normalizovanou entropii stupňového rozdělení."
                    )

                peak_degree1 = unique_deg1[np.argmax(pk1)]
                peak_degree2 = unique_deg2[np.argmax(pk2)]

                comparison_deg_parts.append(
                    f"Nejčastější stupeň je u Série 1 roven **{peak_degree1}** a u Série 2 roven **{peak_degree2}**."
                )

                range1 = np.max(degs1) - np.min(degs1)
                range2 = np.max(degs2) - np.min(degs2)

                if range1 > range2:
                    comparison_deg_parts.append(
                        "Série 1 má širší rozsah stupňů, což naznačuje větší rozdíly mezi slabě a silně propojenými vrcholy."
                    )
                elif range2 > range1:
                    comparison_deg_parts.append(
                        "Série 2 má širší rozsah stupňů, což naznačuje větší rozdíly mezi slabě a silně propojenými vrcholy."
                    )
                else:
                    comparison_deg_parts.append("Obě série mají podobný rozsah stupňů.")

                if np.median(degs1) > np.median(degs2):
                    comparison_deg_parts.append(
                        "Medián stupně je vyšší u Série 1, takže typický vrchol bývá o něco více propojený."
                    )
                elif np.median(degs2) > np.median(degs1):
                    comparison_deg_parts.append(
                        "Medián stupně je vyšší u Série 2, takže typický vrchol bývá o něco více propojený."
                    )
                else:
                    comparison_deg_parts.append("Typický vrchol má v obou sériích podobný stupeň.")

                st.info(" ".join(comparison_deg_parts))

                df_deg_cmp = pd.DataFrame(
                    {
                        "degree": degs1 + degs2,
                        "serie": (["Série 1"] * len(degs1)) + (["Série 2"] * len(degs2)),
                    }
                )

                max_deg = max(
                    max(degs1) if len(degs1) > 0 else 1,
                    max(degs2) if len(degs2) > 0 else 1,
                )

                fig_deg_cmp = px.histogram(
                    df_deg_cmp,
                    x="degree",
                    color="serie",
                    barmode="overlay",
                    opacity=0.6,
                    nbins=max_deg + 1,
                    title="Histogram stupňů – série 1 vs. série 2",
                    labels={"degree": "Stupeň", "count": "Počet vrcholů"},
                )
                fig_deg_cmp.update_layout(yaxis_title="Počet vrcholů")
                st.plotly_chart(fig_deg_cmp, use_container_width=True)

                st.markdown("#### PDF stupňového rozdělení")

                col_pdf1, col_pdf2 = st.columns(2)

                with col_pdf1:
                    fig_pdf1 = create_degree_pdf_figure(
                        unique_deg1,
                        pk1,
                        title="Série 1 – PDF P(k)",
                    )
                    st.plotly_chart(fig_pdf1, use_container_width=True)

                with col_pdf2:
                    fig_pdf2 = create_degree_pdf_figure(
                        unique_deg2,
                        pk2,
                        title="Série 2 – PDF P(k)",
                    )
                    st.plotly_chart(fig_pdf2, use_container_width=True)

                st.caption(
                    "PDF ukazuje pravděpodobnost, že náhodně vybraný vrchol má právě stupeň k."
                )

                st.markdown("#### CDF stupňového rozdělení")

                cdf1 = np.cumsum(pk1)
                cdf2 = np.cumsum(pk2)

                col_cdf1, col_cdf2 = st.columns(2)

                with col_cdf1:
                    fig_cdf1 = create_degree_cdf_figure(
                        unique_deg1,
                        cdf1,
                        title="Série 1 – CDF F(k)",
                    )
                    st.plotly_chart(fig_cdf1, use_container_width=True)

                with col_cdf2:
                    fig_cdf2 = create_degree_cdf_figure(
                        unique_deg2,
                        cdf2,
                        title="Série 2 – CDF F(k)",
                    )
                    st.plotly_chart(fig_cdf2, use_container_width=True)

                st.caption(
                    "CDF ukazuje, jak rychle se kumuluje podíl vrcholů do nižších stupňů."
                )

                st.markdown("#### Formální power-law test + CCDF")

                do_powerlaw_cmp = st.checkbox(
                    "🔍 Provést formální power-law test pro obě série (Clauset–Shalizi–Newman) + CCDF",
                    key="cmp_powerlaw_global",
                )

                alpha1 = None
                alpha2 = None
                xmin1 = None
                xmin2 = None
                result_text_1 = None
                result_text_2 = None

                if do_powerlaw_cmp:
                    if not HAS_POWERLAW:
                        st.warning(
                            "K provedení testu je potřeba balík `powerlaw`. "
                            "Přidej ho do `requirements.txt` a nainstaluj pomocí `pip install powerlaw`."
                        )
                    else:
                        result1 = compute_powerlaw_fit(degs1, has_powerlaw=HAS_POWERLAW)
                        result2 = compute_powerlaw_fit(degs2, has_powerlaw=HAS_POWERLAW)

                        col_pl1, col_pl2 = st.columns(2)

                        with col_pl1:
                            st.markdown("**Série 1 – power-law analýza**")

                            if not result1["success"]:
                                if result1["reason"] == "Příliš málo hodnot pro smysluplný fit.":
                                    st.info("Série 1 má příliš málo vrcholů pro smysluplný power-law fit.")
                                else:
                                    st.info(
                                        f"Power-law test pro Sérii 1 se nepodařilo spolehlivě vyhodnotit: {result1['reason']}"
                                    )
                            else:
                                alpha1 = result1["alpha"]
                                xmin1 = result1["xmin"]
                                powerlaw_R_result_1 = result1["R"]
                                powerlaw_p_result_1 = result1["p"]

                                st.write(f"- Odhadnutý exponent α: **{alpha1:.3f}**")
                                st.write(f"- Odhadnuté k_min: **{xmin1}**")
                                st.write(f"- Likelihood ratio R: **{powerlaw_R_result_1:.3f}**")
                                st.write(f"- p-hodnota: **{powerlaw_p_result_1:.3f}**")

                                if powerlaw_p_result_1 < 0.1:
                                    if powerlaw_R_result_1 > 0:
                                        result_text_1 = "kompatibilní s power-law"
                                        st.success(
                                            "Rozdělení je kompatibilní s power-law a power-law je preferovaný oproti exponenciálnímu rozdělení."
                                        )
                                    else:
                                        result_text_1 = "spíše neodpovídá power-law"
                                        st.warning("Power-law model je horší než exponenciální rozdělení.")
                                else:
                                    result_text_1 = "neprůkazné"
                                    st.info("Test je neprůkazný. Nelze spolehlivě rozhodnout.")

                                degs1_for_fit = result1["degrees_for_fit"]
                                unique_sorted1 = np.sort(np.unique(degs1_for_fit))
                                ccdf_vals1 = np.array(
                                    [np.sum(degs1_for_fit >= k) / len(degs1_for_fit) for k in unique_sorted1]
                                )

                                mask1 = unique_sorted1 >= xmin1
                                if np.sum(mask1) >= 2:
                                    k_emp1 = unique_sorted1[mask1]
                                    ccdf_emp1 = ccdf_vals1[mask1]

                                    k_theory1 = np.linspace(xmin1, k_emp1.max(), 100)
                                    ccdf_theory1 = (k_theory1 / xmin1) ** (1 - alpha1)
                                    ccdf_theory1 *= ccdf_emp1[0] / ccdf_theory1[0]

                                    fig_ccdf1 = go.Figure()
                                    fig_ccdf1.add_trace(
                                        go.Scatter(
                                            x=k_emp1,
                                            y=ccdf_emp1,
                                            mode="markers",
                                            name="Empirická CCDF",
                                        )
                                    )
                                    fig_ccdf1.add_trace(
                                        go.Scatter(
                                            x=k_theory1,
                                            y=ccdf_theory1,
                                            mode="lines",
                                            name=f"Power-law fit (α={alpha1:.2f})",
                                        )
                                    )
                                    fig_ccdf1.update_layout(
                                        title="Série 1 – CCDF (log–log)",
                                        xaxis_type="log",
                                        yaxis_type="log",
                                        xaxis_title="Stupeň k",
                                        yaxis_title="P(K ≥ k)",
                                        legend=dict(x=0.02, y=0.98),
                                        margin=dict(b=40, l=50, r=10, t=50),
                                    )
                                    st.plotly_chart(fig_ccdf1, use_container_width=True)
                                else:
                                    st.info("Tail rozdělení Série 1 je příliš krátký na smysluplný CCDF graf.")

                        with col_pl2:
                            st.markdown("**Série 2 – power-law analýza**")

                            if not result2["success"]:
                                if result2["reason"] == "Příliš málo hodnot pro smysluplný fit.":
                                    st.info("Série 2 má příliš málo vrcholů pro smysluplný power-law fit.")
                                else:
                                    st.info(
                                        f"Power-law test pro Sérii 2 se nepodařilo spolehlivě vyhodnotit: {result2['reason']}"
                                    )
                            else:
                                alpha2 = result2["alpha"]
                                xmin2 = result2["xmin"]
                                powerlaw_R_result_2 = result2["R"]
                                powerlaw_p_result_2 = result2["p"]

                                st.write(f"- Odhadnutý exponent α: **{alpha2:.3f}**")
                                st.write(f"- Odhadnuté k_min: **{xmin2}**")
                                st.write(f"- Likelihood ratio R: **{powerlaw_R_result_2:.3f}**")
                                st.write(f"- p-hodnota: **{powerlaw_p_result_2:.3f}**")

                                if powerlaw_p_result_2 < 0.1:
                                    if powerlaw_R_result_2 > 0:
                                        result_text_2 = "kompatibilní s power-law"
                                        st.success(
                                            "Rozdělení je kompatibilní s power-law a power-law je preferovaný oproti exponenciálnímu rozdělení."
                                        )
                                    else:
                                        result_text_2 = "spíše neodpovídá power-law"
                                        st.warning("Power-law model je horší než exponenciální rozdělení.")
                                else:
                                    result_text_2 = "neprůkazné"
                                    st.info("Test je neprůkazný. Nelze spolehlivě rozhodnout.")

                                degs2_for_fit = result2["degrees_for_fit"]
                                unique_sorted2 = np.sort(np.unique(degs2_for_fit))
                                ccdf_vals2 = np.array(
                                    [np.sum(degs2_for_fit >= k) / len(degs2_for_fit) for k in unique_sorted2]
                                )

                                mask2 = unique_sorted2 >= xmin2
                                if np.sum(mask2) >= 2:
                                    k_emp2 = unique_sorted2[mask2]
                                    ccdf_emp2 = ccdf_vals2[mask2]

                                    k_theory2 = np.linspace(xmin2, k_emp2.max(), 100)
                                    ccdf_theory2 = (k_theory2 / xmin2) ** (1 - alpha2)
                                    ccdf_theory2 *= ccdf_emp2[0] / ccdf_theory2[0]

                                    fig_ccdf2 = go.Figure()
                                    fig_ccdf2.add_trace(
                                        go.Scatter(
                                            x=k_emp2,
                                            y=ccdf_emp2,
                                            mode="markers",
                                            name="Empirická CCDF",
                                        )
                                    )
                                    fig_ccdf2.add_trace(
                                        go.Scatter(
                                            x=k_theory2,
                                            y=ccdf_theory2,
                                            mode="lines",
                                            name=f"Power-law fit (α={alpha2:.2f})",
                                        )
                                    )
                                    fig_ccdf2.update_layout(
                                        title="Série 2 – CCDF (log–log)",
                                        xaxis_type="log",
                                        yaxis_type="log",
                                        xaxis_title="Stupeň k",
                                        yaxis_title="P(K ≥ k)",
                                        legend=dict(x=0.02, y=0.98),
                                        margin=dict(b=40, l=50, r=10, t=50),
                                    )
                                    st.plotly_chart(fig_ccdf2, use_container_width=True)
                                else:
                                    st.info("Tail rozdělení Série 2 je příliš krátký na smysluplný CCDF graf.")

                        st.markdown("#### Porovnání power-law výsledků")

                        compare_powerlaw_parts = []

                        if result_text_1 is not None and result_text_2 is not None:
                            compare_powerlaw_parts.append(
                                f"Série 1 je vyhodnocena jako **{result_text_1}**, zatímco Série 2 jako **{result_text_2}**."
                            )

                        if alpha1 is not None and alpha2 is not None:
                            if alpha1 > alpha2:
                                compare_powerlaw_parts.append(
                                    "Série 1 má vyšší odhad exponentu α, takže tail jejího rozdělení klesá strměji."
                                )
                            elif alpha2 > alpha1:
                                compare_powerlaw_parts.append(
                                    "Série 2 má vyšší odhad exponentu α, takže tail jejího rozdělení klesá strměji."
                                )
                            else:
                                compare_powerlaw_parts.append(
                                    "Obě série mají velmi podobný odhad exponentu α."
                                )

                        if not compare_powerlaw_parts:
                            compare_powerlaw_parts.append(
                                "Power-law výsledky zatím nelze mezi sériemi smysluplně porovnat."
                            )

                        st.warning(" ".join(compare_powerlaw_parts))

            # =============================
            # Shrnutí analýzy pro obě série
            # =============================
            if "Shrnutí analýzy" in selected_sections_cmp:
                st.markdown("### Shrnutí analýzy")

                if conf_metrics1 is None:
                    G1_conf, conf_metrics1 = compute_configuration_model_metrics(G1, seed=42)
                    n1c = conf_metrics1["n_nodes"]
                    m1c = conf_metrics1["n_edges"]
                    degs1c = conf_metrics1["degrees"]
                    avg_deg1c = conf_metrics1["avg_deg"]
                    C1c = conf_metrics1["C"]
                    L1c = conf_metrics1["L"]
                    diam1c = conf_metrics1["diam"]
                    assort1c = conf_metrics1["assort"]
                    L_rand1c = conf_metrics1["L_rand"]
                    C_rand1c = conf_metrics1["C_rand"]
                    sigma1c = conf_metrics1["sigma"]

                if conf_metrics2 is None:
                    G2_conf, conf_metrics2 = compute_configuration_model_metrics(G2, seed=42)
                    n2c = conf_metrics2["n_nodes"]
                    m2c = conf_metrics2["n_edges"]
                    degs2c = conf_metrics2["degrees"]
                    avg_deg2c = conf_metrics2["avg_deg"]
                    C2c = conf_metrics2["C"]
                    L2c = conf_metrics2["L"]
                    diam2c = conf_metrics2["diam"]
                    assort2c = conf_metrics2["assort"]
                    L_rand2c = conf_metrics2["L_rand"]
                    C_rand2c = conf_metrics2["C_rand"]
                    sigma2c = conf_metrics2["sigma"]

                experiment_name_cmp = st.text_input(
                    "Název experimentu / porovnání",
                    value="Porovnání série 1 a série 2",
                    key="cmp_experiment_name",
                )

                tech1, interp1, verdict1 = generate_hvg_summary_text(
                    n_nodes=n1,
                    n_edges=m1,
                    avg_deg=avg_deg1,
                    C=C1,
                    L=L1,
                    sigma_sw=sigma1,
                    assort=assort1,
                    is_normalized=st.session_state.get("series_normalized", False),
                    aggregation_freq=st.session_state.get("series_aggregation", None),
                    series_name=st.session_state.get("series_name", "Série 1"),
                )

                tech2, interp2, verdict2 = generate_hvg_summary_text(
                    n_nodes=n2,
                    n_edges=m2,
                    avg_deg=avg_deg2,
                    C=C2,
                    L=L2,
                    sigma_sw=sigma2,
                    assort=assort2,
                    is_normalized=st.session_state.get("series_normalized2", False),
                    aggregation_freq=st.session_state.get("series_aggregation2", None),
                    series_name=st.session_state.get("series_name2", "Série 2"),
                )

                classification1 = classify_series_from_hvg(
                    avg_deg=avg_deg1,
                    C=C1,
                    L=L1,
                    sigma_sw=sigma1,
                    assort=assort1,
                    entropy_deg_norm=entropy_deg_norm1,
                    powerlaw_p=powerlaw_p_result_1,
                    powerlaw_R=powerlaw_R_result_1,
                    C_rand=C_rand1,
                    L_rand=L_rand1,
                    sigma_conf=sigma1c,
                )

                classification2 = classify_series_from_hvg(
                    avg_deg=avg_deg2,
                    C=C2,
                    L=L2,
                    sigma_sw=sigma2,
                    assort=assort2,
                    entropy_deg_norm=entropy_deg_norm2,
                    powerlaw_p=powerlaw_p_result_2,
                    powerlaw_R=powerlaw_R_result_2,
                    C_rand=C_rand2,
                    L_rand=L_rand2,
                    sigma_conf=sigma2c,
                )

                def render_series_summary(title, tech, interp, verdict, classification, n_points, n_nodes_local):
                    st.markdown(f"## {title}")

                    validation_messages = []

                    if n_points < 10:
                        validation_messages.append(
                            "Časová řada je velmi krátká (méně než 10 bodů), takže výsledky mohou být silně nestabilní."
                        )
                    elif n_points < 30:
                        validation_messages.append(
                            "Časová řada je poměrně krátká (méně než 30 bodů), takže interpretace může být méně spolehlivá."
                        )

                    if n_nodes_local < 10:
                        validation_messages.append(
                            "HVG má velmi málo vrcholů, takže některé síťové metriky a klasifikační závěry mohou být méně robustní."
                        )

                    if validation_messages:
                        st.markdown("**Upozornění k interpretaci**")
                        for msg in validation_messages:
                            st.warning(msg)

                    st.markdown("**Technické shrnutí**")
                    st.info(tech)

                    st.markdown("**Interpretace řady**")
                    st.write(interp)

                    st.markdown("**Orientační klasifikace**")
                    classification_text = (
                        f"**{classification['label']}** "
                        f"(jistota: **{classification['confidence']}**)"
                    )

                    if classification["confidence"] == "vyšší":
                        st.success(classification_text)
                    elif classification["confidence"] == "střední":
                        st.warning(classification_text)
                    else:
                        st.info(classification_text)

                    st.caption(get_classification_status_text(classification))

                    st.markdown("**Zdůvodnění**")
                    st.write(classification["reason_text"])

                    st.markdown("**Stabilita a charakter výsledku**")
                    st.info(
                        f"{classification['structure_text']} "
                        f"Alternativní interpretace: {classification['alternative_label']}. "
                        f"{classification['score_gap_text']} "
                        f"{classification['gap_text']} "
                        f"{classification['dominance_text']}"
                    )

                    st.markdown("**Závěr**")
                    st.warning(verdict)

                    st.markdown("**Skóre jednotlivých interpretací**")
                    c1, c2, c3 = st.columns(3)

                    with c1:
                        st.metric(
                            "Pravidelná / periodická",
                            f"{classification['scores']['Spíše pravidelná / periodická']:.1f}",
                            delta=f"{classification['normalized_scores']['Spíše pravidelná / periodická']:.1f} %",
                        )

                    with c2:
                        st.metric(
                            "Komplexní / chaotická",
                            f"{classification['scores']['Spíše komplexní deterministická / chaotická']:.1f}",
                            delta=f"{classification['normalized_scores']['Spíše komplexní deterministická / chaotická']:.1f} %",
                        )

                    with c3:
                        st.metric(
                            "Stochastická / náhodná",
                            f"{classification['scores']['Spíše stochastická / náhodná']:.1f}",
                            delta=f"{classification['normalized_scores']['Spíše stochastická / náhodná']:.1f} %",
                        )

                    st.caption("Hlavní číslo = bodové skóre. Procento = relativní podíl interpretace.")

                    st.markdown("**Relativní podpora interpretací**")

                    mapping = [
                        ("Pravidelná / periodická", "Spíše pravidelná / periodická"),
                        ("Komplexní / chaotická", "Spíše komplexní deterministická / chaotická"),
                        ("Stochastická / náhodná", "Spíše stochastická / náhodná"),
                    ]

                    for label, key in mapping:
                        st.write(label)
                        st.progress(classification["normalized_scores"][key] / 100)
                        st.caption(f"{classification['normalized_scores'][key]:.1f} %")

                    st.caption(classification["stability_text"])
                    st.caption(classification["mixed_text"])
                    st.caption(classification["warning_text"])

                col_s1, col_s2 = st.columns(2)

                with col_s1:
                    render_series_summary(
                        "Série 1",
                        tech1,
                        interp1,
                        verdict1,
                        classification1,
                        n_points=len(data1),
                        n_nodes_local=n1,
                    )

                with col_s2:
                    render_series_summary(
                        "Série 2",
                        tech2,
                        interp2,
                        verdict2,
                        classification2,
                        n_points=len(data2),
                        n_nodes_local=n2,
                    )

                st.markdown("---")
                st.markdown("## Porovnání sérií")

                topology_parts = []

                if not np.isnan(C1) and not np.isnan(C2):
                    if C1 > C2:
                        topology_parts.append("Série 1 vykazuje vyšší lokální propojenost než Série 2.")
                    elif C2 > C1:
                        topology_parts.append("Série 2 vykazuje vyšší lokální propojenost než Série 1.")
                    else:
                        topology_parts.append("Obě série mají podobnou lokální propojenost.")

                if sigma1 is not None and sigma2 is not None and not np.isnan(sigma1) and not np.isnan(sigma2):
                    if sigma1 > sigma2:
                        topology_parts.append("Z hlediska small-world indexu je Série 1 strukturálně výraznější.")
                    elif sigma2 > sigma1:
                        topology_parts.append("Z hlediska small-world indexu je Série 2 strukturálně výraznější.")
                    else:
                        topology_parts.append("Obě série mají podobný small-world charakter.")

                if avg_deg1 > avg_deg2:
                    topology_parts.append("HVG Série 1 je v průměru propojenější než HVG Série 2.")
                elif avg_deg2 > avg_deg1:
                    topology_parts.append("HVG Série 2 je v průměru propojenější než HVG Série 1.")
                else:
                    topology_parts.append("Obě HVG mají podobnou průměrnou propojenost.")

                if entropy_deg_norm1 > entropy_deg_norm2:
                    topology_parts.append("Série 1 má vyšší variabilitu stupňového rozdělení než Série 2.")
                elif entropy_deg_norm2 > entropy_deg_norm1:
                    topology_parts.append("Série 2 má vyšší variabilitu stupňového rozdělení než Série 1.")
                else:
                    topology_parts.append("Obě série mají podobnou variabilitu stupňového rozdělení.")

                st.markdown("**Porovnání topologie HVG**")
                st.info(" ".join(topology_parts))

                st.markdown("**Porovnání výsledné interpretace**")

                same_label = classification1["label"] == classification2["label"]
                conf1 = classification1["confidence"]
                conf2 = classification2["confidence"]
                dom1 = classification1["dominance_ratio"]
                dom2 = classification2["dominance_ratio"]
                best_score_1 = classification1["best_score"]
                best_score_2 = classification2["best_score"]
                gap1 = classification1["best_score"] - classification1["second_score"]
                gap2 = classification2["best_score"] - classification2["second_score"]

                if same_label:
                    st.success(
                        f"Obě časové řady vykazují podobný orientační charakter: "
                        f"**{classification1['label']}**."
                    )
                else:
                    st.warning(
                        f"Série 1 je orientačně klasifikována jako **{classification1['label']}**, "
                        f"zatímco Série 2 jako **{classification2['label']}**."
                    )

                interpretation_compare_parts = []

                if dom1 > dom2 + 0.08:
                    interpretation_compare_parts.append(
                        "Klasifikace Série 1 působí přesvědčivěji, protože dominantní interpretace je zde výraznější."
                    )
                elif dom2 > dom1 + 0.08:
                    interpretation_compare_parts.append(
                        "Klasifikace Série 2 působí přesvědčivěji, protože dominantní interpretace je zde výraznější."
                    )
                else:
                    interpretation_compare_parts.append(
                        "Obě série mají podobně výraznou dominantní interpretaci."
                    )

                if gap1 > gap2 + 1.0:
                    interpretation_compare_parts.append(
                        "Rozdíl mezi nejsilnější a druhou nejsilnější interpretací je větší u Série 1, takže její závěr je relativně stabilnější."
                    )
                elif gap2 > gap1 + 1.0:
                    interpretation_compare_parts.append(
                        "Rozdíl mezi nejsilnější a druhou nejsilnější interpretací je větší u Série 2, takže její závěr je relativně stabilnější."
                    )
                else:
                    interpretation_compare_parts.append(
                        "Obě série mají podobnou míru vnitřní jednoznačnosti klasifikace."
                    )

                if conf1 != conf2:
                    interpretation_compare_parts.append(
                        f"Jistota klasifikace je u Série 1: **{conf1}**, zatímco u Série 2: **{conf2}**."
                    )
                else:
                    interpretation_compare_parts.append(
                        f"Obě série mají stejnou slovní úroveň jistoty klasifikace: **{conf1}**."
                    )

                if best_score_1 > best_score_2 + 1.0:
                    interpretation_compare_parts.append(
                        "Dominantní interpretace Série 1 má vyšší absolutní podporu metrik než dominantní interpretace Série 2."
                    )
                elif best_score_2 > best_score_1 + 1.0:
                    interpretation_compare_parts.append(
                        "Dominantní interpretace Série 2 má vyšší absolutní podporu metrik než dominantní interpretace Série 1."
                    )
                else:
                    interpretation_compare_parts.append(
                        "Obě dominantní interpretace mají podobně silnou absolutní podporu metrik."
                    )

                st.info(" ".join(interpretation_compare_parts))
                st.caption(
                    "Porovnání vychází z topologie HVG, zejména z lokální propojenosti, "
                    "small-world charakteru, variability stupňového rozdělení a relativní dominance výsledné interpretace."
                )

                st.markdown("### Grafické porovnání procentuální podpory interpretací")

                df_class_cmp = pd.DataFrame(
                    {
                        "Interpretace": [
                            "Pravidelná / periodická",
                            "Komplexní / chaotická",
                            "Stochastická / náhodná",
                            "Pravidelná / periodická",
                            "Komplexní / chaotická",
                            "Stochastická / náhodná",
                        ],
                        "Podpora (%)": [
                            classification1["normalized_scores"]["Spíše pravidelná / periodická"],
                            classification1["normalized_scores"]["Spíše komplexní deterministická / chaotická"],
                            classification1["normalized_scores"]["Spíše stochastická / náhodná"],
                            classification2["normalized_scores"]["Spíše pravidelná / periodická"],
                            classification2["normalized_scores"]["Spíše komplexní deterministická / chaotická"],
                            classification2["normalized_scores"]["Spíše stochastická / náhodná"],
                        ],
                        "Série": [
                            "Série 1",
                            "Série 1",
                            "Série 1",
                            "Série 2",
                            "Série 2",
                            "Série 2",
                        ],
                    }
                )

                fig_class_cmp = px.bar(
                    df_class_cmp,
                    x="Interpretace",
                    y="Podpora (%)",
                    color="Série",
                    barmode="group",
                    title="Porovnání podpory jednotlivých interpretací",
                    labels={
                        "Interpretace": "Typ interpretace",
                        "Podpora (%)": "Podpora (%)",
                    },
                )

                fig_class_cmp.update_layout(yaxis_range=[0, 100])
                st.plotly_chart(fig_class_cmp, use_container_width=True)

                st.markdown("### Grafické porovnání surového skóre")

                scores_compare_df = pd.DataFrame(
                    {
                        "Interpretace": list(classification1["scores"].keys()),
                        "Série 1": list(classification1["scores"].values()),
                        "Série 2": list(classification2["scores"].values()),
                    }
                )

                scores_compare_long = scores_compare_df.melt(
                    id_vars="Interpretace",
                    var_name="Série",
                    value_name="Skóre",
                )

                fig_scores_cmp = px.bar(
                    scores_compare_long,
                    x="Interpretace",
                    y="Skóre",
                    color="Série",
                    barmode="group",
                    title="Porovnání skóre klasifikace obou sérií",
                    text="Skóre",
                )

                fig_scores_cmp.update_traces(textposition="outside")
                fig_scores_cmp.update_layout(
                    xaxis_title="Typ interpretace",
                    yaxis_title="Skóre podpory",
                )

                st.plotly_chart(fig_scores_cmp, use_container_width=True)

                st.caption(
                    "Procentuální graf ukazuje relativní rozdělení podpory mezi interpretace. "
                    "Graf se surovým skóre ukazuje absolutní sílu podpory jednotlivých směrů."
                )

                st.markdown("### Stručný závěr porovnání")

                final_compare_parts = [
                    f"U Série 1 dominuje interpretace **{classification1['label']}** se silou **{classification1['best_score']:.1f} bodu**.",
                    f"U Série 2 dominuje interpretace **{classification2['label']}** se silou **{classification2['best_score']:.1f} bodu**.",
                ]

                if classification1["label"] == classification2["label"]:
                    final_compare_parts.append(
                        "Obě série směřují ke stejnému orientačnímu typu dynamiky."
                    )
                else:
                    final_compare_parts.append(
                        "Série se z hlediska HVG klasifikace liší, takže jejich strukturální charakter není stejný."
                    )

                if classification1["dominance_ratio"] > classification2["dominance_ratio"]:
                    final_compare_parts.append(
                        "U Série 1 je dominantní interpretace výraznější než u Série 2."
                    )
                elif classification2["dominance_ratio"] > classification1["dominance_ratio"]:
                    final_compare_parts.append(
                        "U Série 2 je dominantní interpretace výraznější než u Série 1."
                    )
                else:
                    final_compare_parts.append(
                        "Dominance hlavní interpretace je u obou sérií podobná."
                    )

                st.success(" ".join(final_compare_parts))

                if experiment_name_cmp.strip():
                    st.caption(f"Název porovnání: {experiment_name_cmp}")

                same_label_cmp = classification1["label"] == classification2["label"]

                if not np.isnan(C1) and not np.isnan(C2):
                    if C1 > C2:
                        topology_winner = "Série 1"
                    elif C2 > C1:
                        topology_winner = "Série 2"
                    else:
                        topology_winner = "obě série podobně"
                else:
                    topology_winner = "nelze jednoznačně určit"

                if classification1["dominance_ratio"] > classification2["dominance_ratio"]:
                    dominance_winner = "Série 1"
                elif classification2["dominance_ratio"] > classification1["dominance_ratio"]:
                    dominance_winner = "Série 2"
                else:
                    dominance_winner = "obě série podobně"

                report_col1, report_col2, report_col3 = st.columns(3)

                with report_col1:
                    st.markdown("**Hlavní interpretace – Série 1**")
                    st.success(classification1["label"])
                    st.caption(f"Jistota: {classification1['confidence']}")

                with report_col2:
                    st.markdown("**Hlavní interpretace – Série 2**")
                    st.success(classification2["label"])
                    st.caption(f"Jistota: {classification2['confidence']}")

                with report_col3:
                    st.metric(
                        "Shoda interpretace",
                        "ANO" if same_label_cmp else "NE",
                    )
                    st.metric(
                        "Výraznější dominance",
                        dominance_winner,
                    )

                report_text_parts = []

                if same_label_cmp:
                    report_text_parts.append(
                        "Obě série směřují ke stejnému hlavnímu typu interpretace."
                    )
                else:
                    report_text_parts.append(
                        "Každá série směřuje k odlišné hlavní interpretaci."
                    )

                report_text_parts.append(
                    f"Z hlediska lokální struktury a topologie působí výrazněji: **{topology_winner}**."
                )

                report_text_parts.append(
                    f"Z hlediska jednoznačnosti klasifikace působí přesvědčivěji: **{dominance_winner}**."
                )

                st.info(" ".join(report_text_parts))

                st.markdown("### Export souhrnné klasifikace")

                summary_export_df = pd.DataFrame(
                    [
                        {
                            "experiment_name": experiment_name_cmp,
                            "series": "Série 1",
                            "n_points": len(data1),
                            "n_nodes": n1,
                            "n_edges": m1,
                            "avg_degree": avg_deg1,
                            "clustering": C1,
                            "avg_path_length": L1,
                            "sigma": sigma1,
                            "entropy_deg_norm": entropy_deg_norm1,
                            "label": classification1["label"],
                            "confidence": classification1["confidence"],
                            "alternative_label": classification1["alternative_label"],
                            "dominance_ratio": classification1["dominance_ratio"],
                            "best_score": classification1["best_score"],
                            "second_score": classification1["second_score"],
                            "verdict": verdict1,
                        },
                        {
                            "experiment_name": experiment_name_cmp,
                            "series": "Série 2",
                            "n_points": len(data2),
                            "n_nodes": n2,
                            "n_edges": m2,
                            "avg_degree": avg_deg2,
                            "clustering": C2,
                            "avg_path_length": L2,
                            "sigma": sigma2,
                            "entropy_deg_norm": entropy_deg_norm2,
                            "label": classification2["label"],
                            "confidence": classification2["confidence"],
                            "alternative_label": classification2["alternative_label"],
                            "dominance_ratio": classification2["dominance_ratio"],
                            "best_score": classification2["best_score"],
                            "second_score": classification2["second_score"],
                            "verdict": verdict2,
                        },
                    ]
                )

                summary_export_csv = summary_export_df.to_csv(index=False).encode("utf-8-sig")

                st.download_button(
                    "Exportovat souhrnnou klasifikaci porovnání (CSV)",
                    data=summary_export_csv,
                    file_name="comparison_summary_classification.csv",
                    mime="text/csv",
                )

            # =============================
            # Arc Diagram HVG – obě série
            # =============================
            if "Arc Diagram HVG" in selected_sections_cmp:
                st.markdown("### Arc Diagramy HVG – porovnání")

                col_arc1, col_arc2 = st.columns(2)

                with col_arc1:
                    fig_arc1 = create_arc_diagram_figure(
                        G1,
                        data1,
                        title="Arc Diagram HVG – série 1",
                        node_color="skyblue",
                    )
                    st.plotly_chart(fig_arc1, use_container_width=True)

                with col_arc2:
                    fig_arc2 = create_arc_diagram_figure(
                        G2,
                        data2,
                        title="Arc Diagram HVG – série 2",
                        node_color="lightgreen",
                    )
                    st.plotly_chart(fig_arc2, use_container_width=True)

            # =============================
            # Export HVG a metrik pro obě série
            # =============================
            if "Export HVG a metrik" in selected_sections_cmp:
                st.markdown("### Export HVG a metrik pro obě série")

                edges_df1 = pd.DataFrame(list(G1.edges()), columns=["source", "target"])
                edges_csv1 = edges_df1.to_csv(index=False).encode("utf-8-sig")
                adj_df1 = nx.to_pandas_adjacency(G1)
                adj_csv1 = adj_df1.to_csv().encode("utf-8-sig")
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
                    "entropy_deg_norm": entropy_deg_norm1,
                    "sigma_conf": sigma1c,
                    "powerlaw_p": powerlaw_p_result_1,
                    "powerlaw_R": powerlaw_R_result_1,
                }
                metrics_df1 = pd.DataFrame([metrics_dict1])
                metrics_csv1 = metrics_df1.to_csv(index=False).encode("utf-8-sig")

                edges_df2 = pd.DataFrame(list(G2.edges()), columns=["source", "target"])
                edges_csv2 = edges_df2.to_csv(index=False).encode("utf-8-sig")
                adj_df2 = nx.to_pandas_adjacency(G2)
                adj_csv2 = adj_df2.to_csv().encode("utf-8-sig")
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
                    "entropy_deg_norm": entropy_deg_norm2,
                    "sigma_conf": sigma2c,
                    "powerlaw_p": powerlaw_p_result_2,
                    "powerlaw_R": powerlaw_R_result_2,
                }
                metrics_df2 = pd.DataFrame([metrics_dict2])
                metrics_csv2 = metrics_df2.to_csv(index=False).encode("utf-8-sig")

                col_exp1, col_exp2 = st.columns(2)

                with col_exp1:
                    st.markdown("**Série 1 – exporty**")
                    st.download_button(
                        "⬇️ HVG (edge list, CSV) – série 1",
                        data=edges_csv1,
                        file_name="hvg_series1_edgelist.csv",
                        mime="text/csv",
                    )
                    st.download_button(
                        "⬇️ HVG (adjacency matrix, CSV) – série 1",
                        data=adj_csv1,
                        file_name="hvg_series1_adjacency.csv",
                        mime="text/csv",
                    )
                    st.download_button(
                        "⬇️ Metriky HVG – série 1",
                        data=metrics_csv1,
                        file_name="hvg_series1_metrics.csv",
                        mime="text/csv",
                    )

                with col_exp2:
                    st.markdown("**Série 2 – exporty**")
                    st.download_button(
                        "⬇️ HVG (edge list, CSV) – série 2",
                        data=edges_csv2,
                        file_name="hvg_series2_edgelist.csv",
                        mime="text/csv",
                    )
                    st.download_button(
                        "⬇️ HVG (adjacency matrix, CSV) – série 2",
                        data=adj_csv2,
                        file_name="hvg_series2_adjacency.csv",
                        mime="text/csv",
                    )
                    st.download_button(
                        "⬇️ Metriky HVG – série 2",
                        data=metrics_csv2,
                        file_name="hvg_series2_metrics.csv",
                        mime="text/csv",
                    )