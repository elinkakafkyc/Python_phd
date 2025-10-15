

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="TRC.py",
                   page_icon="🛠️",
                    layout="wide")
st.title(
"Demo aplikace")
st.subheader("Ing. Eliška Kafková")
st.caption("DISCLAIMER:"    \   
"Aktuálně je aplikace ve formě rozpracovanosti a nelze ji brát jako funkční.")

# ========================
# Reálná data (Plain & TRC, Hollow/Filled, 8 a 18 mm)
# ========================
CUSTOM_DATA_BEND = {
    "Plain (bez výztuže)": {"Hollow": [(8, 11.42), (18, 12.91)], "Filled": [(8, 17.72), (18, 20.93)]},
    "TRC (uhlíková mříž)": {"Hollow": [(8, 23.69), (18, 43.09)], "Filled": [(8, 23.82), (18, 47.87)]},
}
CUSTOM_DATA_COMP = {
    "Plain (bez výztuže)": {"Hollow": [(8, 47.93), (18, 64.63)], "Filled": [(8, 24.62), (18, 53.95)]},
    "TRC (uhlíková mříž)": {"Hollow": [(8, 59.45), (18, 74.49)], "Filled": [(8, 35.41), (18, 56.87)]},
}

# ========================
# Dutý NAC panel (ŽB reference, fck ~125 MPa krychlová)
# ========================
DEFAULT_BEND = {
    "Železobeton (NAC s výztuží)": {
        "Hollow": [(8, 15.0), (18, 18.0)],
        "Filled": [(8, 15.0), (18, 18.0)],
    }
}
DEFAULT_COMP = {
    "Železobeton (NAC s výztuží)": {
        "Hollow": [(8, 45.0), (18, 55.0)],
        "Filled": [(8, 45.0), (18, 55.0)],
    }
}

SYSTEMS_LEFT = ["Plain (bez výztuže)", "TRC (uhlíková mříž)"]
SYSTEM_RC = "Železobeton (NAC s výztuží)"

# ========================
# Sidebar
# ========================
with st.sidebar:
    st.header("Nastavení")
    system = st.selectbox("Systém vlevo", SYSTEMS_LEFT, index=1)
    thickness = st.slider("Tloušťka [mm]", min_value=8, max_value=30, value=13, step=1)
    filled = st.toggle("Vyplněno RAC jádrem", value=True)

# ========================
# Fit funkce 
# ========================
def auto_polyfit(x, y):
    n = len(x)
    deg = 2 if n >= 3 else 1
    coef = np.polyfit(x, y, deg=deg)
    p = np.poly1d(coef)
    return coef, p(x), p, deg

def predict_from_pairs(pairs, t_val):
    x = np.array([p[0] for p in pairs], dtype=float)
    y = np.array([p[1] for p in pairs], dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    _, _, model_fn, _ = auto_polyfit(x, y)
    return float(model_fn(t_val))

def get_pairs(data_dict, default_dict, sys_name, cfg):
    pairs = data_dict.get(sys_name, {}).get(cfg, None)
    if pairs and len(pairs) >= 2:
        return pairs
    return default_dict.get(sys_name, {}).get(cfg, None)

def predict_from_custom(data_dict, default_dict, sys_name, cfg, t_val):
    pairs = get_pairs(data_dict, default_dict, sys_name, cfg)
    if pairs and len(pairs) >= 2:
        return predict_from_pairs(pairs, t_val)
    return float("nan")

# ========================
# Výpočet predikcí
# ========================
cfg = "Filled" if filled else "Hollow"

# Vlevo – Plain & TRC
bend_vals_left = [predict_from_custom(CUSTOM_DATA_BEND, {}, s, cfg, thickness) for s in SYSTEMS_LEFT]
comp_vals_left = [predict_from_custom(CUSTOM_DATA_COMP, {}, s, cfg, thickness) for s in SYSTEMS_LEFT]

# Vpravo – referenční NAC panel
bend_val_rc = predict_from_custom({}, DEFAULT_BEND, SYSTEM_RC, cfg, thickness)
comp_val_rc = predict_from_custom({}, DEFAULT_COMP, SYSTEM_RC, cfg, thickness)

# ========================
# Dva sloupce layout
# ========================
col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.subheader("HPC skořepina + RAC jádro")
    idx_sel = SYSTEMS_LEFT.index(system)
    st.metric(f"{system}: Ohyb", f"{bend_vals_left[idx_sel]:.2f} MPa" if not np.isnan(bend_vals_left[idx_sel]) else "—")
    st.metric(f"{system}: Tlak", f"{comp_vals_left[idx_sel]:.2f} MPa" if not np.isnan(comp_vals_left[idx_sel]) else "—")
    st.caption(f"Nastavení: {cfg} • Tloušťka = {thickness} mm")

    # Bar chart (Plain/TRC)
    labels = SYSTEMS_LEFT
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.bar(x - width/2, bend_vals_left, width, label="Ohyb", color="#f4a2c3",alpha=0.85)
    ax.bar(x + width/2, comp_vals_left, width, label="Tlak", color="#c89fca",alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.set_ylabel("Pevnost [MPa]")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    st.pyplot(fig, clear_figure=True)

with col_right:
    st.subheader("Referenční dutý ŽB")
    st.metric("ŽB: Ohyb", f"{bend_val_rc:.2f} MPa" if not np.isnan(bend_val_rc) else "—")
    st.metric("ŽB: Tlak", f"{comp_val_rc:.2f} MPa" if not np.isnan(comp_val_rc) else "—")
    st.caption(f"Nastavení: {cfg} • Tloušťka = {thickness} mm")

    # Bar chart (ŽB)
    labels_rc = ["ŽB"]
    x = np.arange(len(labels_rc))
    width = 0.35
    fig2, ax2 = plt.subplots(figsize=(6, 4), dpi=150)

# růžová + fialová paleta
    ax2.bar(x - width/2, [bend_val_rc], width, label="Ohyb", color="#f4a2c3", alpha=0.9)
    ax2.bar(x + width/2, [comp_val_rc], width, label="Tlak", color="#c89fca", alpha=0.9)
    

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_rc)
    ax2.set_ylabel("Pevnost [MPa]")
    ax2.legend()
    ax2.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    st.pyplot(fig2, clear_figure=True)


    
    df_rc = pd.DataFrame({
        "Tloušťka [mm]": [8, 18],
        "Ohyb [MPa]": [DEFAULT_BEND[SYSTEM_RC]["Hollow"][0][1], DEFAULT_BEND[SYSTEM_RC]["Hollow"][1][1]],
        "Tlak [MPa]": [DEFAULT_COMP[SYSTEM_RC]["Hollow"][0][1], DEFAULT_COMP[SYSTEM_RC]["Hollow"][1][1]],
    })
    st.subheader("Tabulka referenčních hodnot (ŽB dutý panel)")
    st.dataframe(df_rc, use_container_width=True)
    st.caption("Hodnoty odpovídají dutému NAC panelu z materiálu ~125 MPa (krychlová pevnost).")
