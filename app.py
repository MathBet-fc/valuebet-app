import streamlit as st
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Mathbet fc - ML Edition", page_icon="🧠", layout="wide")

# --- PARAMETRI CAMPIONATI (IL CUORE DEL MACHINE LEARNING) ---
# Questi sono i valori che il tuo "Allenatore" calcolerà.
# Aggiornali qui quando fai girare lo script di training.
LEAGUES = {
    # FORMATO: "Nome": { "avg": MediaGol, "ha": VantaggioCasa, "rho": FattorePareggio }
    "🌐 Generico (Default)": { "avg": 1.35, "ha": 0.25, "rho": -0.13 }, 
    "🇮🇹 Serie A":          { "avg": 1.30, "ha": 0.22, "rho": -0.11 },
    "🇬🇧 Premier League":   { "avg": 1.55, "ha": 0.32, "rho": -0.10 },
    "🇪🇸 La Liga":          { "avg": 1.25, "ha": 0.28, "rho": -0.14 },
    "🇩🇪 Bundesliga":       { "avg": 1.65, "ha": 0.35, "rho": -0.09 },
}

# --- FUNZIONI MATEMATICHE CORE (Dixon-Coles) ---
def dixon_coles_probability(h_goals, a_goals, mu_h, mu_a, rho):
    # Formula base di Poisson
    prob = (math.exp(-mu_h) * (mu_h**h_goals) / math.factorial(h_goals)) * \
           (math.exp(-mu_a) * (mu_a**a_goals) / math.factorial(a_goals))
    
    # Correzione Dixon-Coles per dipendenza Under/Pareggi (Rho)
    if h_goals == 0 and a_goals == 0: prob *= (1.0 - (mu_h * mu_a * rho))
    elif h_goals == 0 and a_goals == 1: prob *= (1.0 + (mu_h * rho))
    elif h_goals == 1 and a_goals == 0: prob *= (1.0 + (mu_a * rho))
    elif h_goals == 1 and a_goals == 1: prob *= (1.0 - rho)
    
    return max(0.0, prob)

def calculate_kelly(prob_true, odds_book):
    if odds_book <= 1.01 or prob_true <= 0: return 0.0
    kelly = (((odds_book - 1) * prob_true) - (1 - prob_true)) / (odds_book - 1)
    return max(0.0, kelly * 0.25 * 100) # Kelly frazionato (1/4) per sicurezza

def calculate_player_probability(metric_per90, expected_mins, team_match_xg, team_avg_xg):
    base_lambda = (metric_per90 / 90.0) * expected_mins
    if team_avg_xg <= 0: team_avg_xg = 0.01
    match_factor = team_match_xg / team_avg_xg
    final_lambda = base_lambda * match_factor
    return 1 - math.exp(-final_lambda)

# --- INIZIALIZZAZIONE SESSION STATE ---
if 'history' not in st.session_state: st.session_state.history = []
if 'analyzed' not in st.session_state: st.session_state.analyzed = False

# --- SIDEBAR (CONFIGURAZIONE) ---
with st.sidebar:
    st.title("🧠 Configurazione ML")
    
    # Selezione Campionato (Carica i parametri ottimizzati)
    league_name = st.selectbox("Campionato (Parametri)", list(LEAGUES.keys()))
    L_DATA = LEAGUES[league_name]
    
    st.info(f"📊 Parametri attivi:\n- Home Adv: {L_DATA['ha']}\n- Rho (Pari): {L_DATA['rho']}")
    
    st.markdown("---")
    matchday = st.slider("Giornata (Peso Stagione)", 1, 38, 10)
    # Peso dinamico: più avanti nella stagione, più contano i dati stagionali rispetto all'Elo/Base
    w_seas = min(0.90, 0.30 + (matchday * 0.02)) 
    
    st.markdown("---")
    m_type = st.radio("Contesto", ["Standard", "Derby", "Campo Neutro"])
    is_big_match = st.checkbox("🔥 Big Match (Tattica chiusa)")

st.title("Mathbet fc - Machine Learning Core 🧠")

# --- INPUT PRINCIPALE (MANUALE) ---
col_h, col_a = st.columns(2)
h_uo_input, a_uo_input = {}, {}

# CASA
with col_h:
    st.subheader("🏠 Squadra Casa")
    h_name = st.text_input("Nome Casa", "Inter")
    # Rating Elo (Opzionale, usato per bilanciare)
    h_elo = st.number_input("Rating Elo Casa", 1000.0, 2500.0, 1600.0, step=10.0)
    
    with st.expander("📊 Dati Offensivi/Difensivi", expanded=True):
        st.caption("Inserisci Media Gol Fatti/Subiti (o xG)")
        h_att = st.number_input("Media Gol Fatti Casa", 0.0, 5.0, 1.85, 0.01)
        h_def = st.number_input("Media Gol Subiti Casa", 0.0, 5.0, 0.95, 0.01)
        
        st.caption("Fattore Forma (Ultime 5)")
        h_form_att = st.number_input("Gol Fatti L5 (Totale)", 0.0, 25.0, 9.0, 0.5)
        h_form_def = st.number_input("Gol Subiti L5 (Totale)", 0.0, 25.0, 4.0, 0.5)

    with st.expander("Trend Over % (Opzionale)"):
        for l in [0.5, 1.5, 2.5, 3.5, 4.5]: h_uo_input[l] = st.slider(f"O{l} H", 0, 100, 50, key=f"ho{l}")

# OSPITE
with col_a:
    st.subheader("✈️ Squadra Ospite")
    a_name = st.text_input("Nome Ospite", "Juventus")
    a_elo = st.number_input("Rating Elo Ospite", 1000.0, 2500.0, 1550.0, step=10.0)

    with st.expander("📊 Dati Offensivi/Difensivi", expanded=True):
        st.caption("Inserisci Media Gol Fatti/Subiti (o xG)")
        a_att = st.number_input("Media Gol Fatti Ospite", 0.0, 5.0, 1.45, 0.01)
        a_def = st.number_input("Media Gol Subiti Ospite", 0.0, 5.0, 0.85, 0.01)
        
        st.caption("Fattore Forma (Ultime 5)")
        a_form_att = st.number_input("Gol Fatti L5 (Totale)", 0.0, 25.0, 7.0, 0.5)
        a_form_def = st.number_input("Gol Subiti L5 (Totale)", 0.0, 25.0, 3.0, 0.5)

    with st.expander("Trend Over % (Opzionale)"):
        for l in [0.5, 1.5, 2.5, 3.5, 4.5]: a_uo_input[l] = st.slider(f"O{l} A", 0, 100, 50, key=f"ao{l}")

st.subheader("💰 Quote Bookmaker (Per verifica valore)")
qc1, qc2, qc3 = st.columns(3)
b1 = qc1.number_input("Quota 1", 1.01, 100.0, 2.10)
bX = qc2.number_input("Quota X", 1.01, 100.0, 3.20)
b2 = qc3.number_input("Quota 2", 1.01, 100.0, 3.60)

# --- MOTORE DI CALCOLO ---
if st.button("🚀 ANALIZZA CON ML", type="primary", use_container_width=True):
    
    # 1. Recupero Parametri ML dal dizionario
    home_adv_goals = L_DATA["ha"]
    rho_val = L_DATA["rho"]
    avg_goals_league = L_DATA["avg"]
    
    if m_type == "Campo Neutro": home_adv_goals = 0.0
    elif m_type == "Derby": home_adv_goals *= 0.5

    # 2. Calcolo Forza Squadre (Mix Stagione + Forma + Elo)
    # Forza Attacco/Difesa pesata (Stagione vs Forma)
    # Forma L5 è un totale, dividiamo per 5 per avere media
    h_a_val = (h_att * w_seas) + ((h_form_att/5.0) * (1-w_seas))
    h_d_val = (h_def * w_seas) + ((h_form_def/5.0) * (1-w_seas))
    a_a_val = (a_att * w_seas) + ((a_form_att/5.0) * (1-w_seas))
    a_d_val = (a_def * w_seas) + ((a_form_def/5.0) * (1-w_seas))
    
    # 3. Calcolo Expected Goals (xG)
    # Formula base: (Attacco A * Difesa B) / Media Lega
    xg_h_stats = (h_a_val * a_d_val) / avg_goals_league
    xg_a_stats = (a_a_val * h_d_val) / avg_goals_league
    
    # Correzione Elo (piccolo aggiustamento per qualità rosa)
    # Differenza Elo / 400 = Differenza di classe
    elo_diff = (h_elo + (100 if m_type=="Standard" else 0)) - a_elo
    elo_factor_h = 1 + (elo_diff / 1000.0) # Es. +0.10 gol se 100 pti in più
    elo_factor_a = 1 - (elo_diff / 1000.0)
    
    # LAMBDA FINALI (Expected Goals definitivi)
    # Aggiungiamo qui il Vantaggio Casa appreso dal ML
    f_xh = (xg_h_stats * elo_factor_h) + home_adv_goals
    f_xa = (xg_a_stats * elo_factor_a)
    
    # Tattica Big Match (Under performano spesso)
    if is_big_match:
        f_xh *= 0.90
        f_xa *= 0.90

    # 4. Generazione Probabilità (Dixon-Coles)
    p1, pX, p2, pGG = 0, 0, 0, 0
    matrix = np.zeros((10,10)); scores = []
    
    for h_g in range(10):
        for a_g in range(10):
            p = dixon_coles_probability(h_g, a_g, f_xh, f_xa, rho_val)
            matrix[h_g,a_g] = p
            if h_g > a_g: p1 += p
            elif h_g == a_g: pX += p
            else: p2 += p
            if h_g>0 and a_g>0: pGG += p
            if h_g<6 and a_g<6: scores.append({"Risultato": f"{h_g}-{a_g}", "Prob": p})
            
    # Normalizzazione (caso prob totali < 1)
    tot = np.sum(matrix)
    if tot > 0:
        matrix /= tot; p1 /= tot; pX /= tot; p2 /= tot; pGG /= tot

    # Doppia Chance
    p1X, pX2, p12 = p1+pX, pX+p2, p1+p2

    # Salvataggio Stato
    st.session_state.analyzed = True
    st.session_state.f_xh = f_xh; st.session_state.f_xa = f_xa
    st.session_state.h_name = h_name; st.session_state.a_name = a_name
    st.session_state.p1 = p1; st.session_state.pX = pX; st.session_state.p2 = p2
    st.session_state.p1X = p1X; st.session_state.pX2 = pX2; st.session_state.p12 = p12
    st.session_state.pGG = pGG
    st.session_state.matrix = matrix; st.session_state.scores = scores
    st.session_state.b1 = b1; st.session_state.bX = bX; st.session_state.b2 = b2

# --- OUTPUT ---
if st.session_state.analyzed:
    st.header(f"📊 Analisi: {st.session_state.h_name} vs {st.session_state.a_name}")
    st.metric("Expected Goals (xG)", f"{st.session_state.f_xh:.2f} - {st.session_state.f_xa:.2f}")

    tab1, tab2, tab3, tab4 = st.tabs(["🏆 Esito", "⚽ Gol & Handicap", "👤 Marcatori", "📝 Storico"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Probabilità 1X2")
            st.table(pd.DataFrame({
                "Esito": ["1", "X", "2", "1X", "X2", "12"],
                "Prob %": [f"{p1:.1%}", f"{pX:.1%}", f"{p2:.1%}", f"{p1X:.1%}", f"{pX2:.1%}", f"{p12:.1%}"],
                "Fair Odd": [f"{1/p1:.2f}", f"{1/pX:.2f}", f"{1/p2:.2f}", f"{1/p1X:.2f}", f"{1/pX2:.2f}", f"{1/p12:.2f}"],
                "Bookie": [b1, bX, b2, "-", "-", "-"],
                "Valore": [f"{(b1*p1-1):.1%}", f"{(bX*pX-1):.1%}", f"{(b2*p2-1):.1%}", "-", "-", "-"]
            }))
        with c2:
            st.subheader("Risultati Esatti")
            scores.sort(key=lambda x: x["Prob"], reverse=True)
            st.table(pd.DataFrame([{"Score": s["Risultato"], "Prob": f"{s['Prob']:.1%}", "Quota": f"{1/s['Prob']:.2f}"} for s in scores[:6]]))

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Under / Over")
            uo_res = []
            for l in [0.5, 1.5, 2.5, 3.5, 4.5]:
                p_pure = np.sum(matrix[np.indices((10,10))[0] + np.indices((10,10))[1] > l])
                trend = (h_uo_input.get(l,50) + a_uo_input.get(l,50))/200.0
                p_final = (p_pure * 0.7) + (trend * 0.3) # 70% Math, 30% Trend
                uo_res.append({"Linea": l, "Under %": f"{(1-p_final):.1%}", "Quota U": f"{1/(1-p_final):.2f}", "Over %": f"{p_final:.1%}", "Quota O": f"{1/p_final:.2f}"})
            st.table(pd.DataFrame(uo_res))
        with c2:
            st.subheader("Gol/NoGol & Handicap")
            st.table(pd.DataFrame([{"Esito": "GG", "Prob": f"{pGG:.1%}", "Fair": f"{1/pGG:.2f}"}, {"Esito": "NG", "Prob": f"{(1-pGG):.1%}", "Fair": f"{1/(1-pGG):.2f}"}]))
            h_hand = np.sum(matrix[np.indices((10,10))[0] - 1 > np.indices((10,10))[1]])
            st.write(f"**Handicap (-1) Casa:** {h_hand:.1%} (@{1/h_hand:.2f})")

    with tab3:
        st.subheader("Marcatore")
        c1, c2 = st.columns(2)
        pl_n = c1.text_input("Giocatore", "Lautaro")
        pl_xg = c1.number_input("xG/90", 0.01, 2.0, 0.50)
        pl_min = c2.number_input("Minuti", 1, 100, 85)
        t_xg = st.session_state.f_xh # Default casa
        p_goal = calculate_player_probability(pl_xg, pl_min, t_xg, 1.4)
        st.metric(f"Prob Goal {pl_n}", f"{p_goal:.1%}", f"Quota Reale: {1/p_goal:.2f}")

    with tab4:
        st.subheader("Storico & Backtesting")
        if st.button("💾 Salva Risultato"):
            st.session_state.history.append({"Match": f"{h_name}-{a_name}", "P1": p1, "PX": pX, "P2": p2, "Esito": "?"})
            st.success("Salvato")
        
        if st.session_state.history:
            ed = st.data_editor(pd.DataFrame(st.session_state.history), column_config={"Esito": st.column_config.SelectboxColumn("Reale", options=["1","X","2","?"])})
            val = ed[ed["Esito"] != "?"]
            if not val.empty:
                brier = []
                for _,r in val.iterrows():
                    o = [1 if r["Esito"]=="1" else 0, 1 if r["Esito"]=="X" else 0, 1 if r["Esito"]=="2" else 0]
                    brier.append((r["P1"]-o[0])**2 + (r["PX"]-o[1])**2 + (r["P2"]-o[2])**2)
                st.metric("Brier Score (Errore Medio)", f"{np.mean(brier):.3f}", help="Più è basso, meglio è (Target < 0.60)")
