import streamlit as st
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Mathbet fc", page_icon="⚽", layout="wide")

# --- AUTOMAZIONE ELO (CACHING) ---
@st.cache_data(ttl=3600) 
def get_clubelo_database():
    try:
        date_str = date.today().strftime("%Y-%m-%d")
        url = f"http://api.clubelo.com/{date_str}"
        df = pd.read_csv(url)
        return dict(zip(df.Club, df.Elo))
    except: return {}

ELO_DB = get_clubelo_database()

# --- INIZIALIZZAZIONE SESSION STATE ---
if 'history' not in st.session_state:
    st.session_state.history = []
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False

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

# --- FUNZIONI CORE ---
def dixon_coles_probability(h_goals, a_goals, mu_h, mu_a, rho):
    prob = (math.exp(-mu_h) * (mu_h**h_goals) / math.factorial(h_goals)) * \
           (math.exp(-mu_a) * (mu_a**a_goals) / math.factorial(a_goals))
    if h_goals == 0 and a_goals == 0: prob *= (1.0 - (mu_h * mu_a * rho))
    elif h_goals == 0 and a_goals == 1: prob *= (1.0 + (mu_h * rho))
    elif h_goals == 1 and a_goals == 0: prob *= (1.0 + (mu_a * rho))
    elif h_goals == 1 and a_goals == 1: prob *= (1.0 - rho)
    return max(0.0, prob)

def calculate_kelly(prob_true, odds_book):
    if odds_book <= 1.01 or prob_true <= 0: return 0.0
    kelly = (((odds_book - 1) * prob_true) - (1 - prob_true)) / (odds_book - 1)
    return max(0.0, kelly * 0.25 * 100)

def calculate_player_probability(metric_per90, expected_mins, team_match_xg, team_avg_xg):
    base_lambda = (metric_per90 / 90.0) * expected_mins
    match_factor = team_match_xg / team_avg_xg if team_avg_xg > 0 else 1.0
    final_lambda = base_lambda * match_factor
    return 1 - math.exp(-final_lambda), final_lambda

# --- SIDEBAR & SETUP ---
with st.sidebar:
    st.title("⚙️ Configurazione")
    league_name = st.selectbox("Campionato", list(LEAGUES.keys()))
    L_DATA = LEAGUES[league_name]
    matchday = st.slider("Giornata", 1, 38, 10)
    w_elo = (L_DATA["w_elo_base"] + 0.10) if 8 < matchday <= 19 else (max(L_DATA["w_elo_base"], 0.75) if matchday <= 8 else L_DATA["w_elo_base"])
    CURRENT_RHO = L_DATA.get("rho", -0.13)

st.title("Mathbet fc ⚽")

with st.expander("🔗 Link Utili (Scraper Dati)", expanded=False):
    lc1, lc2, lc3 = st.columns(3)
    lc1.link_button("ClubElo", "http://clubelo.com")
    lc2.link_button("FootyStats", "https://footystats.org/it/")
    lc3.link_button("FBref", "https://fbref.com")

st.markdown("---")

# --- INPUT SQUADRE ---
col_h, col_a = st.columns(2)
h_uo_input, a_uo_input = {}, {}

with col_h:
    st.subheader("🏠 Squadra Casa")
    h_name = st.text_input("Nome Casa", "Inter", key="h_n")
    auto_elo_h = float(ELO_DB.get(h_name, 1600.0))
    if h_name not in ELO_DB and h_name != "":
        matches = [k for k in ELO_DB.keys() if h_name.lower() in k.lower()]
        if matches: st.info(f"Forse intendevi: {', '.join(matches[:3])}?")
    h_elo = st.number_input("Rating Casa", 1000.0, 2500.0, value=auto_elo_h, step=1.0, key="helo")
    h_str = st.slider("Titolari Casa %", 50, 100, 100, key="hs")
    c_h1, c_h2 = st.columns(2)
    h_m_a, h_m_d = c_h1.checkbox("Manca Bomber (C)", key="ha"), c_h2.checkbox("Manca Difesa (C)", key="hd")
    with st.expander("📊 Stats Gol"):
        h_gf_s, h_gs_s = st.number_input("GF Stag.", 0.0, 5.0, 1.4, key="hgf"), st.number_input("GS Stag.", 0.0, 5.0, 1.0, key="hgs")
        h_gf_h, h_gs_h = st.number_input("GF Casa", 0.0, 5.0, 1.6, key="hgfh"), st.number_input("GS Casa", 0.0, 5.0, 0.8, key="hgsh")
        h_gf_l5, h_gs_l5 = st.number_input("GF L5", 0, 25, 7, key="hgfl5"), st.number_input("GS L5", 0, 25, 5, key="hgsl5")
    with st.expander("📈 Over %"):
        for l in [0.5, 1.5, 2.5, 3.5, 4.5]: h_uo_input[l] = st.slider(f"O{l} Casa", 0, 100, 50, key=f"ho{l}")

with col_a:
    st.subheader("✈️ Squadra Ospite")
    a_name = st.text_input("Nome Ospite", "Milan", key="a_n")
    auto_elo_a = float(ELO_DB.get(a_name, 1550.0))
    if a_name not in ELO_DB and a_name != "":
        matches = [k for k in ELO_DB.keys() if a_name.lower() in k.lower()]
        if matches: st.info(f"Forse intendevi: {', '.join(matches[:3])}?")
    a_elo = st.number_input("Rating Ospite", 1000.0, 2500.0, value=auto_elo_a, step=1.0, key="aelo")
    a_str = st.slider("Titolari Ospite %", 50, 100, 100, key="as")
    c_a1, c_a2 = st.columns(2)
    a_m_a, a_m_d = c_a1.checkbox("Manca Bomber (O)", key="aa"), c_a2.checkbox("Manca Difesa (O)", key="ad")
    with st.expander("📊 Stats Gol "):
        a_gf_s, a_gs_s = st.number_input("GF Stag. ", 0.0, 5.0, 1.2, key="agf"), st.number_input("GS Stag. ", 0.0, 5.0, 1.3, key="ags")
        a_gf_a, a_gs_a = st.number_input("GF Fuori", 0.0, 5.0, 1.0, key="agfa"), st.number_input("GS Fuori", 0.0, 5.0, 1.5, key="agsa")
        a_gf_l5, a_gs_l5 = st.number_input("GF L5 ", 0, 25, 5, key="agfl5"), st.number_input("GS L5 ", 0, 25, 6, key="agsl5")
    with st.expander("📈 Over % "):
        for l in [0.5, 1.5, 2.5, 3.5, 4.5]: a_uo_input[l] = st.slider(f"O{l} Ospite", 0, 100, 50, key=f"ao{l}")

st.subheader("💰 Quote Bookmaker")
qc1, qc2, qc3 = st.columns(3)
b1, bX, b2 = qc1.number_input("Q1", 1.01, 100.0, 2.20), qc2.number_input("QX", 1.01, 100.0, 3.10), qc3.number_input("Q2", 1.01, 100.0, 3.40)

# --- ANALISI ---
if st.button("🚀 ANALIZZA PARTITA", type="primary", use_container_width=True):
    h_att = (h_gf_s * 0.4) + (h_gf_h * 0.35) + (h_gf_l5/5.0 * 0.25)
    h_def = (h_gs_s * 0.4) + (h_gs_h * 0.35) + (h_gs_l5/5.0 * 0.25)
    a_att = (a_gf_s * 0.4) + (a_gf_a * 0.35) + (a_gf_l5/5.0 * 0.25)
    a_def = (a_gs_s * 0.4) + (a_gs_a * 0.35) + (a_gs_l5/5.0 * 0.25)
    xg_s_h, xg_s_a = (h_att * a_def)/L_DATA["avg"], (a_att * h_def)/L_DATA["avg"]
    exp_h = 1 / (1 + 10 ** (-((h_elo + L_DATA["ha"]*400) - a_elo)/400.0))
    xg_e_h, xg_e_a = L_DATA["avg"]*(exp_h/0.5)**0.85, L_DATA["avg"]*((1-exp_h)/0.5)**0.85
    f_xh = ((xg_e_h * w_elo) + (xg_s_h * (1-w_elo))) * (h_str/100.0)
    f_xa = ((xg_e_a * w_elo) + (xg_s_a * (1-w_elo))) * (a_str/100.0)
    if h_m_a: f_xh *= 0.85
    if h_m_d: f_xa *= 1.20
    if a_m_a: f_xa *= 0.85
    if a_m_d: f_xh *= 1.20

    p1, pX, p2, pGG = 0, 0, 0, 0
    matrix = np.zeros((10,10)); scores = []
    for h in range(10):
        for a in range(10):
            p = dixon_coles_probability(h, a, f_xh, f_xa, CURRENT_RHO)
            matrix[h,a] = p
            if h > a: p1 += p
            elif h == a: pX += p
            else: p2 += p
            if h>0 and a>0: pGG += p
            if h<6 and a<6: scores.append({"Risultato": f"{h}-{a}", "Prob": p})
    
    matrix /= np.sum(matrix); p1, pX, p2, pGG = p1/np.sum(matrix), pX/np.sum(matrix), p2/np.sum(matrix), pGG/np.sum(matrix)
    
    sim = []
    for _ in range(5000):
        gh, ga = np.random.poisson(max(0.1, np.random.normal(f_xh, 0.18*f_xh))), np.random.poisson(max(0.1, np.random.normal(f_xa, 0.18*f_xa)))
        sim.append(1 if gh>ga else (0 if gh==ga else 2))
    s1, sX, s2 = sim.count(1)/5000, sim.count(0)/5000, sim.count(2)/5000
    stability = max(0, 100 - ((abs(p1-s1)+abs(pX-sX)+abs(p2-s2))/3*400))

    st.session_state.analyzed, st.session_state.f_xh, st.session_state.f_xa = True, f_xh, f_xa
    st.session_state.h_gf_s, st.session_state.a_gf_s = h_gf_s, a_gf_s
    st.session_state.h_name, st.session_state.a_name = h_name, a_name

    st.header(f"📊 {h_name} - {a_name} ({f_xh:.2f} - {f_xa:.2f})")
    st.metric("Stabilità Modello", f"{stability:.1f}%")
    
    st.subheader("🏆 Probabilità 1X2")
    st.table(pd.DataFrame({"Esito":["1","X","2"], "Prob %":[f"{p1:.1%}",f"{pX:.1%}",f"{p2:.1%}"], "Fair":[f"{1/p1:.2f}",f"{1/pX:.2f}",f"{1/p2:.2f}"], "Value":[f"{(b1*p1-1):.1%}",f"{(bX*pX-1):.1%}",f"{(b2*p2-1):.1%}"], "Stake":[f"{calculate_kelly(p1,b1):.1f}%",f"{calculate_kelly(pX,bX):.1f}%",f"{calculate_kelly(p2,b2):.1f}%"]}))

    if st.button("💾 SALVA IN STORICO"):
        st.session_state.history.append({"Data": date.today().strftime("%d/%m"), "Match": f"{h_name}-{a_name}", "P1": p1, "PX": pX, "P2": p2, "Stabilità": stability, "Risultato": "In attesa"})
        st.toast("Salvato!")

    c1, c2 = st.columns([1,2])
    scores.sort(key=lambda x: x["Prob"], reverse=True)
    c1.subheader("🎯 Top 5 Score")
    c1.table(pd.DataFrame([{"Score": s["Risultato"], "%": f"{s['Prob']:.1%}", "Fair": f"{1/s['Prob']:.2f}"} for s in scores[:5]]))
    with c2: 
        fig, ax = plt.subplots(figsize=(5,3)); sns.heatmap(matrix[:5,:5], annot=True, fmt=".0%", cmap="Blues", cbar=False); st.pyplot(fig)

    st.subheader("📉 Under / Over")
    uo_d = []
    for l in [0.5, 1.5, 2.5, 3.5, 4.5]:
        fo = (np.sum(matrix[np.indices((10,10))[0] + np.indices((10,10))[1] > l]) * 0.6) + ((h_uo_input[l] + a_uo_input[l])/200.0 * 0.4)
        uo_d.append({"Linea": l, "Under %": f"{(1-fo):.1%}", "Fair U": f"{1/(1-fo):.2f}", "Over %": f"{fo:.1%}", "Fair O": f"{1/fo:.2f}"})
    st.table(pd.DataFrame(uo_d))

    # --- REINTEGRATI: GG/NG & MULTIGOL & HANDICAP ---
    cg, ch = st.columns(2)
    with cg:
        st.subheader("⚽ Mercati Gol")
        st.table(pd.DataFrame([{"Esito": "GG", "Prob": f"{pGG:.1%}", "Fair": f"{1/pGG:.2f}"}, {"Esito": "NG", "Prob": f"{(1-pGG):.1%}", "Fair": f"{1/(1-pGG):.2f}"}]))
        
        st.subheader("🔢 Multigol")
        mg_data = []
        for r in [(1,2), (1,3), (2,3), (2,4), (3,5)]:
            pmg = np.sum(matrix[(np.indices((10,10))[0] + np.indices((10,10))[1] >= r[0]) & (np.indices((10,10))[0] + np.indices((10,10))[1] <= r[1])])
            mg_data.append({"Multigol": f"{r[0]}-{r[1]}", "Prob": f"{pmg:.1%}", "Fair": f"{1/pmg:.2f}"})
        st.table(pd.DataFrame(mg_data))
    
    with ch:
        st.subheader("🏁 Handicap")
        eh1_1 = np.sum(matrix[np.indices((10,10))[0] - 1 > np.indices((10,10))[1]])
        eh1_X = np.sum(matrix[np.indices((10,10))[0] - 1 == np.indices((10,10))[1]])
        ehP1_1 = np.sum(matrix[np.indices((10,10))[0] + 1 > np.indices((10,10))[1]])
        st.write("**Handicap Europeo**")
        st.table(pd.DataFrame([{"Tipo": "H-1", "1": f"{eh1_1:.1%}", "Fair": f"{1/eh1_1:.2f}", "X": f"{eh1_X:.1%}", "F.X": f"{1/eh1_X:.2f}"}, {"Tipo": "H+1", "1X": f"{ehP1_1:.1%}", "Fair": f"{1/ehP1_1:.2f}"}]))
        st.write("**Asian Handicap**")
        st.table(pd.DataFrame([{"Linea": "DNB (0.0)", "Prob": f"{(p1/(p1+p2)):.1%}" if (p1+p2)>0 else "0%", "Fair": f"{(1/(p1/(p1+p2))):.2f}" if (p1+p2)>0 else "0.00"}, {"Linea": "H-0.5 (Win)", "Prob": f"{p1:.1%}", "Fair": f"{1/p1:.2f}"}]))

# --- PLAYER PROP ---
if st.session_state.get('analyzed'):
    st.markdown("---")
    st.header("👤 Marcatore / Assist")
    with st.expander("Calcolatore Giocatore Avanzato", expanded=True):
        pcol1, pcol2 = st.columns(2)
        with pcol1:
            p_n = st.text_input("Giocatore", "Lautaro")
            p_t = st.radio("Squadra", [st.session_state.h_name, st.session_state.a_name], horizontal=True)
            p_m = st.number_input("Minuti attesi", 1, 100, 80)
        with pcol2:
            p_v = st.number_input("xG/90 o xA/90", 0.01, 2.0, 0.45)
            p_b = st.number_input("Quota Bookmaker ", 1.01, 100.0, 2.50)
        
        ctx_xg = st.session_state.f_xh if p_t == st.session_state.h_name else st.session_state.f_xa
        ctx_avg = st.session_state.h_gf_s if p_t == st.session_state.h_name else st.session_state.a_gf_s
        
        prob_p, _ = calculate_player_probability(p_v, p_m, ctx_xg, ctx_avg)
        fair_p = 1/prob_p if prob_p > 0 else 0
        edge_p = ((p_b*prob_p)-1)*100
        
        pr1, pr2, pr3 = st.columns(3)
        pr1.metric("Probabilità", f"{prob_p:.1%}")
        pr2.metric("Fair Odd", f"{fair_p:.2f}")
        pr3.metric("Valore %", f"{edge_p:+.1f}%")

# --- BACKTESTING ---
st.markdown("---")
st.header("📈 Backtesting Performance")
if st.session_state.history:
    df_h = pd.DataFrame(st.session_state.history)
    ed_df = st.data_editor(df_h, column_config={"Risultato": st.column_config.SelectboxColumn("Esito Reale", options=["1", "X", "2", "In attesa"])})
    val = ed_df[ed_df["Risultato"] != "In attesa"]
    if not val.empty:
        br = []
        for _, r in val.iterrows():
            o = [1 if r["Risultato"]=="1" else 0, 1 if r["Risultato"]=="X" else 0, 1 if r["Risultato"]=="2" else 0]
            br.append((r["P1"]-o[0])**2 + (r["PX"]-o[1])**2 + (r["P2"]-o[2])**2)
        st.metric("Brier Score Medio", f"{np.mean(br):.3f}")
    if st.button("🗑️ Reset Storico"): st.session_state.history = []; st.rerun()

# --- EXTRA ---
st.markdown("---")
with st.expander("🛠️ Strumenti Manuali"):
    col1, col2 = st.columns(2)
    q_in = col1.number_input("Reverse Quota", 1.01, 100.0, 2.0); col1.write(f"Prob: {1/q_in:.1%}")
    mp = col2.number_input("Tua Stima %", 1.0, 100.0, 50.0)/100; mq = col2.number_input("Quota", 1.01, 100.0, 2.0)
    if (mp*mq)-1 > 0: col2.success(f"Value! Stake: {(((mq-1)*mp)-(1-mp))/(mq-1)*25:.1f}%")
 
