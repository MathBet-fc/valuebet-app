import streamlit as st
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Mathbet fc",
                   page_icon="⚽",
                   layout="wide")

# --- AUTOMAZIONE ELO (CACHING) ---
@st.cache_data(ttl=3600) 
def get_clubelo_database():
    try:
        date_str = date.today().strftime("%Y-%m-%d")
        url = f"http://api.clubelo.com/{date_str}"
        df = pd.read_csv(url)
        elo_dict = dict(zip(df.Club, df.Elo))
        return elo_dict
    except:
        return {}

ELO_DB = get_clubelo_database()

# --- INIZIALIZZAZIONE SESSION STATE ---
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
    st.session_state.xg_h = 0
    st.session_state.xg_a = 0
    st.session_state.prob_1 = 0
    st.session_state.prob_X = 0
    st.session_state.prob_2 = 0
    st.session_state.score_matrix = None
    st.session_state.all_scores = []
    st.session_state.hist_uo_h = {}
    st.session_state.hist_uo_a = {}

# --- DATABASE CAMPIONATI ---
LEAGUES = {
    "🌐 Generico (Media)": { "avg": 1.35, "ha": 0.30, "w_elo_base": 0.40, "rho": -0.13 }, 
    "🇮🇹 Serie A":          { "avg": 1.30, "ha": 0.20, "w_elo_base": 0.50, "rho": -0.14 },
    "🇮🇹 Serie B":          { "avg": 1.15, "ha": 0.25, "w_elo_base": 0.30, "rho": -0.18 },
    "🇬🇧 Premier League":   { "avg": 1.55, "ha": 0.30, "w_elo_base": 0.55, "rho": -0.12 },
    "🇩🇪 Bundesliga":       { "avg": 1.65, "ha": 0.35, "w_elo_base": 0.45, "rho": -0.10 },
    "🇪🇸 La Liga":          { "avg": 1.25, "ha": 0.25, "w_elo_base": 0.55, "rho": -0.14 },
    "🇫🇷 Ligue 1":          { "avg": 1.30, "ha": 0.24, "w_elo_base": 0.45, "rho": -0.15 },
}

# Parametri Globali
SCALING_FACTOR = 400.0  
KELLY_FRACTION = 0.25
WEIGHT_HIST_UO = 0.40 

# --- FUNZIONI MATEMATICHE ---
def dixon_coles_probability(h_goals, a_goals, mu_h, mu_a, rho):
    prob = (math.exp(-mu_h) * (mu_h**h_goals) / math.factorial(h_goals)) * \
           (math.exp(-mu_a) * (mu_a**a_goals) / math.factorial(a_goals))
    correction = 1.0
    if h_goals == 0 and a_goals == 0: correction = 1.0 - (mu_h * mu_a * rho)
    elif h_goals == 0 and a_goals == 1: correction = 1.0 + (mu_h * rho)
    elif h_goals == 1 and a_goals == 0: correction = 1.0 + (mu_a * rho)
    elif h_goals == 1 and a_goals == 1: correction = 1.0 - rho
    return max(0.0, prob * correction)

def calculate_kelly(prob_true, odds_book):
    if odds_book <= 1.01 or prob_true <= 0: return 0.0
    b = odds_book - 1
    p = prob_true
    q = 1 - p
    kelly = ((b * p) - q) / b
    return max(0.0, (kelly * KELLY_FRACTION) * 100)

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Configurazione")
    league_name = st.selectbox("Campionato", list(LEAGUES.keys()))
    L_DATA = LEAGUES[league_name]
    matchday = st.slider("Giornata Attuale", 1, 38, 10)
    
    if matchday <= 8: w_elo = max(L_DATA["w_elo_base"], 0.75)
    elif matchday <= 19: w_elo = L_DATA["w_elo_base"] + 0.10
    else: w_elo = L_DATA["w_elo_base"]
    w_stats = 1.0 - w_elo
    
    CURRENT_RHO = L_DATA.get("rho", -0.13)
    st.caption(f"🎯 Elo Weight: {w_elo:.0%}")

st.title("Mathbet fc ⚽")

# --- LINK UTILI ---
with st.expander("🔗 Link Utili", expanded=False):
    lc1, lc2, lc3 = st.columns(3)
    lc1.link_button("ClubElo", "http://clubelo.com")
    lc2.link_button("FootyStats", "https://footystats.org/it/")
    lc3.link_button("FBref", "https://fbref.com")

st.markdown("---")

# --- INPUT DATI ---
col_h, col_a = st.columns(2)
h_uo_input = {}; a_uo_input = {}

with col_h:
    st.subheader("🏠 Squadra Casa")
    h_name = st.text_input("Nome Casa", "Inter", key="h_name_input")
    auto_elo_h = float(ELO_DB.get(h_name, 1600.0))
    h_elo = st.number_input("ClubElo Rating", 1000.0, 2500.0, value=auto_elo_h, step=1.0, key="helo")
    h_str = st.slider("Disponibilità Titolari %", 50, 100, 100, key="hstr")
    
    st.write("🚑 **Assenze**")
    c_h1, c_h2 = st.columns(2)
    h_miss_att = c_h1.checkbox("Manca Top Scorer", key="hma")
    h_miss_def = c_h2.checkbox("Manca Difesa Top", key="hmd")
    
    with st.expander("📊 Stats & Gol"):
        h_gf_s = st.number_input("GF Media Stagione", 0.0, 5.0, 1.4, key="h1")
        h_gs_s = st.number_input("GS Media Stagione", 0.0, 5.0, 1.0, key="h2")
        h_gf_h = st.number_input("GF Media in Casa", 0.0, 5.0, 1.6, key="h3")
        h_gs_h = st.number_input("GS Media in Casa", 0.0, 5.0, 0.8, key="h4")
        h_gf_l5 = st.number_input("GOL FATTI (Ultime 5)", 0, 25, 7, key="h5")
        h_gs_l5 = st.number_input("GOL SUBITI (Ultime 5)", 0, 25, 5, key="h6")
    
    with st.expander("📈 Storico Under/Over %"):
        for l in [0.5, 1.5, 2.5, 3.5, 4.5]:
            h_uo_input[l] = st.slider(f"% Over {l} Casa", 0, 100, 50, key=f"ho{l}")

with col_a:
    st.subheader("✈️ Squadra Ospite")
    a_name = st.text_input("Nome Ospite", "Milan", key="a_name_input")
    auto_elo_a = float(ELO_DB.get(a_name, 1550.0))
    a_elo = st.number_input("ClubElo Rating", 1000.0, 2500.0, value=auto_elo_a, step=1.0, key="aelo")
    a_str = st.slider("Disponibilità Titolari %", 50, 100, 100, key="astr")
    
    st.write("🚑 **Assenze**")
    c_a1, c_a2 = st.columns(2)
    a_miss_att = c_a1.checkbox("Manca Top Scorer", key="ama")
    a_miss_def = c_a2.checkbox("Manca Difesa Top", key="amd")
    
    with st.expander("📊 Stats & Gol"):
        a_gf_s = st.number_input("GF Media Stagione", 0.0, 5.0, 1.2, key="a1")
        a_gs_s = st.number_input("GS Media Stagione", 0.0, 5.0, 1.3, key="a2")
        a_gf_a = st.number_input("GF Media Fuori", 0.0, 5.0, 1.0, key="a3")
        a_gs_a = st.number_input("GS Media Fuori", 0.0, 5.0, 1.5, key="a4")
        a_gf_l5 = st.number_input("GOL FATTI (Ultime 5)", 0, 25, 5, key="a5")
        a_gs_l5 = st.number_input("GOL SUBITI (Ultime 5)", 0, 25, 6, key="a6")

    with st.expander("📈 Storico Under/Over %"):
        for l in [0.5, 1.5, 2.5, 3.5, 4.5]:
            a_uo_input[l] = st.slider(f"% Over {l} Ospite", 0, 100, 50, key=f"ao{l}")

st.markdown("---")
st.subheader("💰 Quote Bookmaker")
qc1, qc2, qc3 = st.columns(3)
b1 = qc1.number_input("Quota 1", 1.01, 100.0, 2.20); bX = qc2.number_input("Quota X", 1.01, 100.0, 3.10); b2 = qc3.number_input("Quota 2", 1.01, 100.0, 3.40)

# --- CALCOLO ---
if st.button("🚀 ANALIZZA PARTITA", type="primary", use_container_width=True):
    h_form_att, h_form_def = h_gf_l5 / 5.0, h_gs_l5 / 5.0
    a_form_att, a_form_def = a_gf_l5 / 5.0, a_gs_l5 / 5.0
    h_att = (h_gf_s * 0.4) + (h_gf_h * 0.35) + (h_form_att * 0.25)
    h_def = (h_gs_s * 0.4) + (h_gs_h * 0.35) + (h_form_def * 0.25)
    a_att = (a_gf_s * 0.4) + (a_gf_a * 0.35) + (a_form_att * 0.25)
    a_def = (a_gs_s * 0.4) + (a_gs_a * 0.35) + (a_form_def * 0.25)
    xg_s_h = (h_att * a_def) / L_DATA["avg"]; xg_s_a = (a_att * h_def) / L_DATA["avg"]
    diff_h = (h_elo + (L_DATA["ha"] * 400)) - a_elo
    exp_h = 1 / (1 + 10 ** (-diff_h / 400.0))
    xg_e_h = L_DATA["avg"] * (exp_h / 0.5) ** 0.85; xg_e_a = L_DATA["avg"] * ((1-exp_h) / 0.5) ** 0.85
    f_xg_h = (xg_e_h * w_elo) + (xg_s_h * w_stats); f_xg_a = (xg_e_a * w_elo) + (xg_s_a * w_stats)
    f_xg_h *= (h_str/100.0); f_xg_a *= (a_str/100.0)
    if h_miss_att: f_xg_h *= 0.85; 
    if h_miss_def: f_xg_a *= 1.20; 
    if a_miss_att: f_xg_a *= 0.85; 
    if a_miss_def: f_xg_h *= 1.20; 

    p1, pX, p2, pGG = 0, 0, 0, 0
    matrix = np.zeros((10, 10)); scores = []
    for h in range(10):
        for a in range(10):
            p = dixon_coles_probability(h, a, f_xg_h, f_xg_a, CURRENT_RHO)
            matrix[h, a] = p
            if h > a: p1 += p
            elif h == a: pX += p
            else: p2 += p
            if h > 0 and a > 0: pGG += p
            if h < 6 and a < 6: scores.append({"Risultato": f"{h}-{a}", "Prob": p})
    
    total = p1 + pX + p2; p1 /= total; pX /= total; p2 /= total; pGG /= total; matrix /= total
    
    n_sims = 10000; sim_res = []
    for _ in range(n_sims):
        xh = max(0.1, np.random.normal(f_xg_h, 0.18 * f_xg_h))
        xa = max(0.1, np.random.normal(f_xg_a, 0.18 * f_xg_a))
        gh, ga = np.random.poisson(xh), np.random.poisson(xa)
        if gh > ga: sim_res.append(1);
        elif gh == ga: sim_res.append(0);
        else: sim_res.append(2)
    s1, sX, s2 = sim_res.count(1)/n_sims, sim_res.count(0)/n_sims, sim_res.count(2)/n_sims
    stability = max(0, 100 - ((abs(p1 - s1) + abs(pX - sX) + abs(p2 - s2)) / 3 * 400))

    # --- DISPLAY ---
    st.header(f"📊 {h_name} vs {a_name} (xG {f_xg_h:.2f}-{f_xg_a:.2f})")
    s_col1, s_col2 = st.columns([1, 3])
    s_col1.metric("Stabilità", f"{stability:.1f}%")
    status = "ALTA" if stability > 85 else "MEDIA" if stability > 70 else "BASSA"
    s_col2.info(f"Stabilità Predittiva: **{status}**. (Sim 1: {s1:.1%}|X: {sX:.1%}|2: {s2:.1%})")

    st.subheader("🏆 Esito Finale & Valore")
    k1, kX, k2 = calculate_kelly(p1, b1), calculate_kelly(pX, bX), calculate_kelly(p2, b2)
    st.table(pd.DataFrame({
        "Esito": ["1", "X", "2"], "Prob %": [f"{p1:.1%}", f"{pX:.1%}", f"{p2:.1%}"],
        "Fair": [f"{1/p1:.2f}", f"{1/pX:.2f}", f"{1/p2:.2f}"], "Book": [b1, bX, b2],
        "Value": [f"{((b1*p1)-1):.1%}", f"{((bX*pX)-1):.1%}", f"{((b2*p2)-1):.1%}"], "Stake": [f"{k1:.1f}%", f"{kX:.1f}%", f"{k2:.1f}%"]
    }))

    # --- RISULTATI & HEATMAP (MODIFICATO v4.8) ---
    c_res, c_heat = st.columns([1, 2])
    scores.sort(key=lambda x: x["Prob"], reverse=True)
    c_res.subheader("🎯 Top 5 Risultati")
    
    # Nuova logica formattazione richiesta
    top_5_formatted = []
    for s in scores[:5]:
        top_5_formatted.append({
            "Risultato": s["Risultato"],
            "Prob %": f"{s['Prob']:.1%}",
            "Fair": f"{1/s['Prob']:.2f}"
        })
    c_res.table(pd.DataFrame(top_5_formatted))
    
    with c_heat:
        st.subheader("🔥 Distribuzione Gol")
        fig, ax = plt.subplots(figsize=(6, 3)); sns.heatmap(matrix[:5, :5], annot=True, fmt=".0%", cmap="Blues", cbar=False); plt.xlabel(a_name); plt.ylabel(h_name); st.pyplot(fig)

    # UNDER/OVER
    st.subheader("📉 Under / Over")
    uo_data = []
    for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
        p_over_m = np.sum(matrix[np.indices((10,10))[0] + np.indices((10,10))[1] > line])
        p_over_h = (h_uo_input[line] + a_uo_input[line]) / 200.0
        final_o = (p_over_m * 0.6) + (p_over_h * 0.4); final_u = 1 - final_o
        uo_data.append({"Linea": line, "Under %": f"{final_u:.1%}", "Fair U": f"{1/final_u:.2f}", "Over %": f"{final_o:.1%}", "Fair O": f"{1/final_o:.2f}"})
    st.table(pd.DataFrame(uo_data))

    # GG & MULTIGOL & HANDICAP
    cg, ch = st.columns(2)
    with cg:
        st.subheader("⚽ Gol / No Gol"); st.table(pd.DataFrame([{"Esito": "GG", "Prob": f"{pGG:.1%}", "Fair": f"{1/pGG:.2f}"}, {"Esito": "NG", "Prob": f"{(1-pGG):.1%}", "Fair": f"{1/(1-pGG):.2f}"}]))
        st.subheader("🔢 Multigol"); mg_data = []
        for r in [(1,2), (1,3), (2,3), (2,4), (3,5)]:
            pmg = np.sum(matrix[(np.indices((10,10))[0] + np.indices((10,10))[1] >= r[0]) & (np.indices((10,10))[0] + np.indices((10,10))[1] <= r[1])])
            mg_data.append({"Range": f"{r[0]}-{r[1]}", "Prob": f"{pmg:.1%}", "Fair": f"{1/pmg:.2f}"})
        st.table(pd.DataFrame(mg_data))
    with ch:
        st.subheader("🏁 Handicap")
        eh1_1 = np.sum(matrix[np.indices((10,10))[0] - 1 > np.indices((10,10))[1]])
        eh1_X = np.sum(matrix[np.indices((10,10))[0] - 1 == np.indices((10,10))[1]])
        ehP1_1 = np.sum(matrix[np.indices((10,10))[0] + 1 > np.indices((10,10))[1]])
        st.write("**Europeo**"); st.table(pd.DataFrame([{"Tipo": "H-1", "1": f"{eh1_1:.1%}", "Fair 1": f"{1/eh1_1:.2f}", "X": f"{eh1_X:.1%}", "Fair X": f"{1/eh1_X:.2f}"}, {"Tipo": "H+1", "1X": f"{ehP1_1:.1%}", "Fair": f"{1/ehP1_1:.2f}"}]))
        st.write("**Asiatico**"); st.table(pd.DataFrame([{"Linea": "DNB (0.0)", "Prob": f"{(p1/(p1+p2)):.1%}", "Fair": f"{(1/(p1/(p1+p2))):.2f}"}, {"Linea": "H-0.5", "Prob": f"{p1:.1%}", "Fair": f"{1/p1:.2f}"}]))

# --- TOOLS EXTRA ---
st.markdown("---")
with st.expander("🛠️ Strumenti Extra"):
    et1, et2 = st.columns(2)
    q_rev = et1.number_input("Reverse Quota", 1.01, 100.0, 2.0); et1.write(f"Prob: {1/q_rev:.1%}")
    k_mp = et2.number_input("Tua Stima %", 1.0, 100.0, 50.0)/100; k_mq = et2.number_input("Quota", 1.01, 100.0, 2.0)
    if (k_mp * k_mq) - 1 > 0: et2.success(f"Value! Stake: {(((k_mq-1)*k_mp)-(1-k_mp))/(k_mq-1)*0.25:.1%}")
