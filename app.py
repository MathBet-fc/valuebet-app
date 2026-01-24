import streamlit as st
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Mathbet fc Pro - Ultimate Edition", page_icon="⚽", layout="wide")

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
    if team_avg_xg <= 0: team_avg_xg = 0.01
    match_factor = team_match_xg / team_avg_xg
    final_lambda = base_lambda * match_factor
    return 1 - math.exp(-final_lambda), final_lambda

# --- SIDEBAR & SETUP ---
with st.sidebar:
    st.title("⚙️ Configurazione")
    league_name = st.selectbox("Campionato", list(LEAGUES.keys()))
    L_DATA = LEAGUES[league_name]
    matchday = st.slider("Giornata", 1, 38, 10)
    
    # Peso Elo Automatico
    w_elo = (L_DATA["w_elo_base"] + 0.10) if 8 < matchday <= 19 else (max(L_DATA["w_elo_base"], 0.75) if matchday <= 8 else L_DATA["w_elo_base"])
    
    st.markdown("---")
    st.subheader("🏟️ Contesto Partita")
    m_type = st.radio("Tipo di Incontro", ["Standard", "Derby (Stesso Stadio)", "Campo Neutro"])
    is_big_match = st.checkbox("🔥 Big Match (Scontro Diretto)", help="Riduce l'aspettativa di gol per tensione tattica.")
    
    st.markdown("---")
    # Toggle xG
    use_xg_mode = st.toggle("📊 Usa Modalità xG (Expected Goals)", value=False, help="Attivalo se hai i dati xG. Nota: Le ultime 5 restano sempre Gol Reali.")
    
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

# Definizioni etichette dinamiche (Gol vs xG) per STAGIONE e CASA/FUORI
lbl_gf_s = "xG Fatti Totali" if use_xg_mode else "GF Stag."
lbl_gs_s = "xG Subiti Totali" if use_xg_mode else "GS Stag."
lbl_gf_ha = "xG Fatti Casa" if use_xg_mode else "GF Casa"
lbl_gs_ha = "xG Subiti Casa" if use_xg_mode else "GS Casa"

lbl_gf_s_a = "xG Fatti Totali" if use_xg_mode else "GF Stag."
lbl_gs_s_a = "xG Subiti Totali" if use_xg_mode else "GS Stag."
lbl_gf_ha_a = "xG Fatti Fuori" if use_xg_mode else "GF Fuori"
lbl_gs_ha_a = "xG Subiti Fuori" if use_xg_mode else "GS Fuori"

# Options per Strength of Schedule
sos_options = ["Media (Standard)", "Difficili (Top Team)", "Facili (Bassa Classifica)"]

with col_h:
    st.subheader("🏠 Squadra Casa")
    h_name = st.text_input("Nome Casa", "Inter", key="h_n")
    
    auto_elo_h = float(ELO_DB.get(h_name, 1600.0))
    if h_name not in ELO_DB and h_name != "":
        matches = [k for k in ELO_DB.keys() if h_name.lower() in k.lower()]
        if matches: st.info(f"💡 Suggerimento: {', '.join(matches[:3])}")
    
    h_elo = st.number_input("Rating Casa", 1000.0, 2500.0, value=auto_elo_h, key="helo")
    h_str = st.slider("Titolari Casa %", 50, 100, 100, key="hs")
    
    # Slider Riposo
    h_rest = st.slider("Giorni Riposo (vs Ultima Partita)", 2, 10, 7, key="h_rest", help="Meno di 4 giorni applica penalità per stanchezza.")
    
    c_h1, c_h2 = st.columns(2)
    h_m_a, h_m_d = c_h1.checkbox("Manca Bomber (C)", key="h_ma"), c_h2.checkbox("Manca Difesa (C)", key="h_md")
    
    with st.expander("📊 Stats Gol / xG", expanded=True):
        # Dati Stagionali (Cambiano in xG se attivo)
        h_gf_s = st.number_input(lbl_gf_s, 0.0, 5.0, 1.45, step=0.01, key="h_gf_s")
        h_gs_s = st.number_input(lbl_gs_s, 0.0, 5.0, 1.05, step=0.01, key="h_gs_s")
        h_gf_h = st.number_input(lbl_gf_ha, 0.0, 5.0, 1.65, step=0.01, key="h_gf_h")
        h_gs_h = st.number_input(lbl_gs_ha, 0.0, 5.0, 0.85, step=0.01, key="h_gs_h")
        
        st.markdown("---")
        # SoS (Strength of Schedule)
        h_sos = st.selectbox("Livello Avversari L5", sos_options, index=0, key="h_sos", help="Chi hanno affrontato nelle ultime 5?")
        # --- MODIFICA: Qui l'etichetta è SEMPRE "GF Reali", mai xG ---
        h_gf_l5 = st.number_input("GF Reali Ultime 5 (Forma)", 0.0, 25.0, 7.0, step=0.5, key="h_gf_l5")
        h_gs_l5 = st.number_input("GS Reali Ultime 5 (Forma)", 0.0, 25.0, 5.0, step=0.5, key="h_gs_l5")

    with st.expander("📈 Over %"):
        for l in [0.5, 1.5, 2.5, 3.5, 4.5]: h_uo_input[l] = st.slider(f"O{l} Casa", 0, 100, 50, key=f"ho{l}")

with col_a:
    st.subheader("✈️ Squadra Ospite")
    a_name = st.text_input("Nome Ospite", "Milan", key="a_n")
    
    auto_elo_a = float(ELO_DB.get(a_name, 1550.0))
    if a_name not in ELO_DB and a_name != "":
        matches = [k for k in ELO_DB.keys() if a_name.lower() in k.lower()]
        if matches: st.info(f"💡 Suggerimento: {', '.join(matches[:3])}")

    a_elo = st.number_input("Rating Ospite", 1000.0, 2500.0, value=auto_elo_a, key="aelo")
    a_str = st.slider("Titolari Ospite %", 50, 100, 100, key="as")

    # Slider Riposo Ospite
    a_rest = st.slider("Giorni Riposo (vs Ultima Partita)", 2, 10, 7, key="a_rest", help="Meno di 4 giorni applica penalità per stanchezza.")

    c_a1, c_a2 = st.columns(2)
    a_m_a, a_m_d = c_a1.checkbox("Manca Bomber (O)", key="a_ma"), c_a2.checkbox("Manca Difesa (O)", key="a_md")
    
    with st.expander("📊 Stats Gol / xG ", expanded=True):
        # Dati Stagionali (Cambiano in xG se attivo)
        a_gf_s = st.number_input(lbl_gf_s_a, 0.0, 5.0, 1.25, step=0.01, key="a_gf_s")
        a_gs_s = st.number_input(lbl_gs_s_a, 0.0, 5.0, 1.35, step=0.01, key="a_gs_s")
        a_gf_a = st.number_input(lbl_gf_ha_a, 0.0, 5.0, 1.10, step=0.01, key="a_gf_a")
        a_gs_a = st.number_input(lbl_gs_ha_a, 0.0, 5.0, 1.55, step=0.01, key="a_gs_a")
        
        st.markdown("---")
        # SoS Ospite
        a_sos = st.selectbox("Livello Avversari L5 ", sos_options, index=0, key="a_sos", help="Chi hanno affrontato nelle ultime 5?")
        # --- MODIFICA: Qui l'etichetta è SEMPRE "GF Reali", mai xG ---
        a_gf_l5 = st.number_input("GF Reali Ultime 5 (Forma) ", 0.0, 25.0, 5.0, step=0.5, key="a_gf_l5")
        a_gs_l5 = st.number_input("GS Reali Ultime 5 (Forma) ", 0.0, 25.0, 6.0, step=0.5, key="a_gs_l5")

    with st.expander("📈 Over % "):
        for l in [0.5, 1.5, 2.5, 3.5, 4.5]: a_uo_input[l] = st.slider(f"O{l} Ospite", 0, 100, 50, key=f"ao{l}")

st.subheader("💰 Quote Bookmaker")
qc1, qc2, qc3 = st.columns(3)
b1 = qc1.number_input("Q1", 1.01, 100.0, 2.20, key="b1")
bX = qc2.number_input("QX", 1.01, 100.0, 3.10, key="bX")
b2 = qc3.number_input("Q2", 1.01, 100.0, 3.40, key="b2")

# --- ANALISI ---
if st.button("🚀 ANALIZZA PARTITA", type="primary", use_container_width=True):
    
    # 1. Gestione Home Advantage (HA)
    ha_val = L_DATA["ha"]
    if m_type == "Campo Neutro": ha_val = 0.0
    elif m_type == "Derby (Stesso Stadio)": ha_val *= 0.5
    
    # 2. Applicazione Strength of Schedule (SoS) alle statistiche L5
    h_gf_l5_c, h_gs_l5_c = h_gf_l5, h_gs_l5
    a_gf_l5_c, a_gs_l5_c = a_gf_l5, a_gs_l5

    if h_sos == "Difficili (Top Team)":
        h_gf_l5_c *= 1.25 # Segnare ai forti vale di più
        h_gs_l5_c *= 0.85 # Subire dai forti è perdonabile
    elif h_sos == "Facili (Bassa Classifica)":
        h_gf_l5_c *= 0.85 
        h_gs_l5_c *= 1.20 
    
    if a_sos == "Difficili (Top Team)":
        a_gf_l5_c *= 1.25
        a_gs_l5_c *= 0.85
    elif a_sos == "Facili (Bassa Classifica)":
        a_gf_l5_c *= 0.85
        a_gs_l5_c *= 1.20

    # 3. Calcolo Attacco/Difesa (Logica Avanzata xG vs Standard)
    if use_xg_mode:
        # PESI xG: Fiducia alta nei dati stagionali e Casa/Fuori (85%), bassa nella forma (15%)
        # Nota: La forma usa i GOL REALI (come richiesto), ma pesano meno nel mix totale
        w_seas, w_ha, w_l5 = 0.50, 0.35, 0.15
    else:
        # PESI STANDARD: La forma recente conta di più (25%)
        w_seas, w_ha, w_l5 = 0.40, 0.35, 0.25

    if m_type == "Campo Neutro":
        w_tot_neutro = w_seas + w_l5 
        h_att_val = (h_gf_s * (w_seas/w_tot_neutro) + h_gf_l5_c/5.0 * (w_l5/w_tot_neutro))
        h_def_val = (h_gs_s * (w_seas/w_tot_neutro) + h_gs_l5_c/5.0 * (w_l5/w_tot_neutro))
        a_att_val = (a_gf_s * (w_seas/w_tot_neutro) + a_gf_l5_c/5.0 * (w_l5/w_tot_neutro))
        a_def_val = (a_gs_s * (w_seas/w_tot_neutro) + a_gs_l5_c/5.0 * (w_l5/w_tot_neutro))
    else:
        h_att_val = (h_gf_s * w_seas + h_gf_h * w_ha + h_gf_l5_c/5.0 * w_l5)
        h_def_val = (h_gs_s * w_seas + h_gs_h * w_ha + h_gs_l5_c/5.0 * w_l5)
        a_att_val = (a_gf_s * w_seas + a_gf_a * w_ha + a_gf_l5_c/5.0 * w_l5)
        a_def_val = (a_gs_s * w_seas + a_gs_a * w_ha + a_gs_l5_c/5.0 * w_l5)
    
    xg_s_h, xg_s_a = (h_att_val * a_def_val)/L_DATA["avg"], (a_att_val * h_def_val)/L_DATA["avg"]
    
    # 4. Calcolo Elo
    exp_h = 1 / (1 + 10 ** (-((h_elo + ha_val*400) - a_elo)/400.0))
    xg_e_h, xg_e_a = L_DATA["avg"]*(exp_h/0.5)**0.85, L_DATA["avg"]*((1-exp_h)/0.5)**0.85
    
    # 5. Fusione Finale e Applicazione Fattori Extra
    f_xh = ((xg_e_h * w_elo) + (xg_s_h * (1-w_elo))) * (h_str/100.0)
    f_xa = ((xg_e_a * w_elo) + (xg_s_a * (1-w_elo))) * (a_str/100.0)
    
    # APPLICAZIONE FATTORE STANCHEZZA
    fatigue_malus = 0.05 
    if h_rest <= 3: 
        f_xh *= (1 - fatigue_malus) 
        f_xa *= (1 + fatigue_malus) 
    if a_rest <= 3: 
        f_xa *= (1 - fatigue_malus)
        f_xh *= (1 + fatigue_malus)
    
    if is_big_match:
        f_xh *= 0.90
        f_xa *= 0.90
    
    if h_m_a: f_xh *= 0.85
    if h_m_d: f_xa *= 1.20
    if a_m_a: f_xa *= 0.85
    if a_m_d: f_xh *= 1.20

    # 6. Dixon-Coles Matrix
    p1, pX, p2, pGG = 0, 0, 0, 0
    matrix = np.zeros((10,10)); scores = []
    for h_g in range(10):
        for a_g in range(10):
            p = dixon_coles_probability(h_g, a_g, f_xh, f_xa, CURRENT_RHO)
            matrix[h_g,a_g] = p
            if h_g > a_g: p1 += p
            elif h_g == a_g: pX += p
            else: p2 += p
            if h_g>0 and a_g>0: pGG += p
            if h_g<6 and a_g<6: scores.append({"Risultato": f"{h_g}-{a_g}", "Prob": p})
    
    total_prob = np.sum(matrix)
    if total_prob > 0:
        matrix /= total_prob
        p1, pX, p2, pGG = p1/total_prob, pX/total_prob, p2/total_prob, pGG/total_prob
    
    # Simulazione Stabilità
    sim = []
    for _ in range(5000):
        gh = np.random.poisson(max(0.1, np.random.normal(f_xh, 0.15*f_xh)))
        ga = np.random.poisson(max(0.1, np.random.normal(f_xa, 0.15*f_xa)))
        sim.append(1 if gh>ga else (0 if gh==ga else 2))
    s1, sX, s2 = sim.count(1)/5000, sim.count(0)/5000, sim.count(2)/5000
    stability = max(0, 100 - ((abs(p1-s1)+abs(pX-sX)+abs(p2-s2))/3*400))

    # --- SALVATAGGIO SESSIONE ---
    st.session_state.analyzed = True
    st.session_state.f_xh = f_xh
    st.session_state.f_xa = f_xa
    st.session_state.home_name_display = h_name
    st.session_state.away_name_display = a_name
    st.session_state.p1, st.session_state.pX, st.session_state.p2 = p1, pX, p2
    st.session_state.stability = stability

    # --- OUTPUT GRAFICO ---
    st.header(f"📊 {h_name} - {a_name} ({f_xh:.2f} - {f_xa:.2f})")
    st.metric("Stabilità Modello", f"{stability:.1f}%")
    
    st.subheader("🏆 Probabilità 1X2")
    st.table(pd.DataFrame({
        "Esito":["1","X","2"], 
        "Prob %":[f"{p1:.1%}",f"{pX:.1%}",f"{p2:.1%}"], 
        "Fair Odd":[f"{1/p1:.2f}",f"{1/pX:.2f}",f"{1/p2:.2f}"], 
        "Value %":[f"{(b1*p1-1):.1%}",f"{(bX*pX-1):.1%}",f"{(b2*p2-1):.1%}"], 
        "Stake (Kelly)":[f"{calculate_kelly(p1,b1):.1f}%",f"{calculate_kelly(pX,bX):.1f}%",f"{calculate_kelly(p2,b2):.1f}%"]
    }))

    c1, c2 = st.columns([1,2])
    scores.sort(key=lambda x: x["Prob"], reverse=True)
    with c1:
        st.subheader("🎯 Top 5 Score")
        st.table(pd.DataFrame([{"Score": s["Risultato"], "%": f"{s['Prob']:.1%}", "Fair": f"{1/s['Prob']:.2f}"} for s in scores[:5]]))
    with c2: 
        fig, ax = plt.subplots(figsize=(5,3))
        sns.heatmap(matrix[:5,:5], annot=True, fmt=".0%", cmap="Greens", cbar=False)
        plt.xlabel("Ospite"); plt.ylabel("Casa")
        st.pyplot(fig)

    st.subheader("📉 Under / Over")
    uo_data = []
    for l in [0.5, 1.5, 2.5, 3.5, 4.5]:
        p_over = (np.sum(matrix[np.indices((10,10))[0] + np.indices((10,10))[1] > l]) * 0.65) + ((h_uo_input[l] + a_uo_input[l])/200.0 * 0.35)
        uo_data.append({"Linea": l, "Under %": f"{(1-p_over):.1%}", "Fair U": f"{1/(1-p_over):.2f}", "Over %": f"{p_over:.1%}", "Fair O": f"{1/p_over:.2f}"})
    st.table(pd.DataFrame(uo_data))

    cm1, cm2 = st.columns(2)
    with cm1:
        st.subheader("⚽ Mercati Gol")
        st.table(pd.DataFrame([{"Esito": "GG", "Prob": f"{pGG:.1%}", "Fair": f"{1/pGG:.2f}"}, {"Esito": "NG", "Prob": f"{(1-pGG):.1%}", "Fair": f"{1/(1-pGG):.2f}"}]))
        st.subheader("🔢 Multigol")
        mg_res = []
        for r in [(1,2), (1,3), (2,3), (2,4), (3,5)]:
            pm = np.sum(matrix[(np.indices((10,10))[0] + np.indices((10,10))[1] >= r[0]) & (np.indices((10,10))[0] + np.indices((10,10))[1] <= r[1])])
            mg_res.append({"Range": f"{r[0]}-{r[1]}", "Prob": f"{pm:.1%}", "Fair": f"{1/pm:.2f}"})
        st.table(pd.DataFrame(mg_res))
    with cm2:
        st.subheader("🏁 Handicap & Asian")
        h1_1 = np.sum(matrix[np.indices((10,10))[0] - 1 > np.indices((10,10))[1]])
        st.write(f"**Handicap Europeo (-1):** Prob 1: {h1_1:.1%} | Fair: {1/h1_1:.2f}")
        dnb_p = (p1/(p1+p2)) if (p1+p2)>0 else 0
        st.write(f"**Asian DNB (0.0):** Prob 1: {dnb_p:.1%} | Fair: {1/dnb_p:.2f}")

# --- SALVATAGGIO STORICO ---
if st.session_state.get('analyzed'):
    if st.button("💾 SALVA IN STORICO"):
        st.session_state.history.append({
            "Data": date.today().strftime("%d/%m"), 
            "Match": f"{st.session_state.home_name_display}-{st.session_state.away_name_display}", 
            "P1": st.session_state.p1, 
            "PX": st.session_state.pX, 
            "P2": st.session_state.p2, 
            "Stabilità": st.session_state.stability, 
            "Risultato": "In attesa"
        })
        st.toast("Salvato con successo!")

# --- PLAYER PROP ---
if st.session_state.get('analyzed'):
    st.markdown("---")
    st.header("👤 Marcatore / Assist")
    with st.expander("Calcolatore Giocatore Avanzato", expanded=True):
        pcol1, pcol2 = st.columns(2)
        n_h = st.session_state.h_n
        n_a = st.session_state.a_n
        
        p_t = pcol1.radio("Squadra", [n_h, n_a], horizontal=True)
        p_v = pcol2.number_input("xG/90 o xA/90", 0.01, 2.0, 0.40)
        p_m = pcol1.number_input("Minuti attesi", 1, 100, 80)
        p_b = pcol2.number_input("Quota Bookie", 1.01, 100.0, 2.50)
        
        ctx_xg = st.session_state.f_xh if p_t == n_h else st.session_state.f_xa
        # Legge il valore medio dalla widget corretta
        ctx_avg = st.session_state.h_gf_s if p_t == n_h else st.session_state.a_gf_s
        
        prob_p, _ = calculate_player_probability(p_v, p_m, ctx_xg, ctx_avg)
        
        r1, r2, r3 = st.columns(3)
        r1.metric("Probabilità", f"{prob_p:.1%}")
        r2.metric("Fair Odd", f"{1/prob_p:.2f}")
        r3.metric("Valore %", f"{((p_b*prob_p)-1)*100:+.1f}%")

# --- BACKTESTING ---
st.markdown("---")
st.header("📈 Backtesting Performance")
if st.session_state.history:
    df_h = pd.DataFrame(st.session_state.history)
    ed_df = st.data_editor(df_h, column_config={"Risultato": st.column_config.SelectboxColumn("Esito Reale", options=["1", "X", "2", "In attesa"])})
    val = ed_df[ed_df["Risultato"] != "In attesa"]
    if not val.empty:
        brier = []
        for _, r in val.iterrows():
            o = [1 if r["Risultato"]=="1" else 0, 1 if r["Risultato"]=="X" else 0, 1 if r["Risultato"]=="2" else 0]
            brier.append((r["P1"]-o[0])**2 + (r["PX"]-o[1])**2 + (r["P2"]-o[2])**2)
        st.metric("Brier Score Medio", f"{np.mean(brier):.3f}", help="0=Perfetto, 2=Pessimo. Un buon modello sta sotto 0.60.")
    if st.button("🗑️ Reset Storico"): st.session_state.history = []; st.rerun()

# --- EXTRA: STRUMENTI MANUALI ---
st.markdown("---")
with st.expander("🛠️ Strumenti Manuali Rapidi"):
    col1, col2 = st.columns(2)
    with col1:
        q_in = st.number_input("Reverse Quota", 1.01, 100.0, 2.0, key="rev_q")
        st.write(f"Probabilità: **{1/q_in:.1%}**")
    with col2:
        mp = st.number_input("Tua Stima %", 1.0, 100.0, 50.0, key="my_p") / 100
        mq = st.number_input("Quota", 1.01, 100.0, 2.0, key="my_q")
        if (mp * mq) - 1 > 0:
            val_kelly = (((mq - 1) * mp) - (1 - mp)) / (mq - 1) * 25
            st.success(f"Value! Stake: **{val_kelly:.1f}%**")
        else:
            st.error("Nessun valore.")
