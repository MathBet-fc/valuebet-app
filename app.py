import streamlit as st
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Mathbet fc Pro 2.0",
                   page_icon="🧠",
                   layout="wide")

# --- INIZIALIZZAZIONE SESSION STATE ---
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
    st.session_state.xg_h = 0
    st.session_state.xg_a = 0
    st.session_state.prob_1 = 0
    st.session_state.prob_X = 0
    st.session_state.prob_2 = 0
    st.session_state.all_scores = []

# --- DATABASE CAMPIONATI ---
# Nota: "w_elo" ora è un peso BASE, che verrà modificato dalla giornata
LEAGUES = {
    "🌐 Generico (Media)": { "avg": 1.35, "ha": 0.35, "w_elo_base": 0.40 }, 
    "🇮🇹 Serie A":          { "avg": 1.30, "ha": 0.25, "w_elo_base": 0.50 }, # Elo molto importante in A
    "🇮🇹 Serie B":          { "avg": 1.15, "ha": 0.30, "w_elo_base": 0.30 },
    "🇬🇧 Premier League":   { "avg": 1.55, "ha": 0.35, "w_elo_base": 0.55 },
    "🇩🇪 Bundesliga":       { "avg": 1.65, "ha": 0.40, "w_elo_base": 0.45 },
    "🇪🇸 La Liga":          { "avg": 1.25, "ha": 0.30, "w_elo_base": 0.55 },
    "🇫🇷 Ligue 1":          { "avg": 1.30, "ha": 0.28, "w_elo_base": 0.45 },
}

# Parametri Globali
SCALING_FACTOR = 400.0  # Standard Elo scaling divisor
KELLY_FRACTION = 0.25
RHO = -0.13 # Correlazione Dixon-Coles standard per punteggi bassi

# --- FUNZIONI MATEMATICHE AVANZATE ---

def calculate_dynamic_weights(matchday, base_w_elo):
    """
    Adatta i pesi in base alla fase della stagione.
    Giornate 1-8: Elo domina (le stats sono rumore).
    Giornate 9-25: Mix bilanciato.
    Giornate 26+: Le stats stagionali dominano (Elo conta meno).
    """
    if matchday <= 8:
        # Inizio stagione: Elo è Re
        w_elo = max(base_w_elo, 0.75)
        w_stats = 1.0 - w_elo
    elif matchday <= 19:
        # Metà girone andata: transizione
        w_elo = base_w_elo + 0.10
        w_stats = 1.0 - w_elo
    else:
        # Seconda parte stagione: le stats sono affidabili
        w_elo = base_w_elo
        w_stats = 1.0 - w_elo
        
    return w_elo, w_stats

def dixon_coles_probability(h_goals, a_goals, mu_h, mu_a, rho):
    """
    Calcola la probabilità di un punteggio esatto usando Dixon-Coles.
    Corregge la sottostima dei pareggi bassi (0-0, 1-1) tipica di Poisson.
    """
    prob = (math.exp(-mu_h) * (mu_h**h_goals) / math.factorial(h_goals)) * \
           (math.exp(-mu_a) * (mu_a**a_goals) / math.factorial(a_goals))
    
    correction = 1.0
    if h_goals == 0 and a_goals == 0:
        correction = 1.0 - (mu_h * mu_a * rho)
    elif h_goals == 0 and a_goals == 1:
        correction = 1.0 + (mu_h * rho)
    elif h_goals == 1 and a_goals == 0:
        correction = 1.0 + (mu_a * rho)
    elif h_goals == 1 and a_goals == 1:
        correction = 1.0 - rho
        
    return max(0.0, prob * correction)

def calculate_kelly(prob_true, odds_book):
    if odds_book <= 1.01 or prob_true <= 0: return 0.0
    b = odds_book - 1
    p = prob_true
    q = 1 - p
    kelly = ((b * p) - q) / b
    return max(0.0, (kelly * KELLY_FRACTION) * 100)

def normalize_elo(val):
    # Soft cap per Elo estremi
    if val < 1300: return 1300
    if val > 2100: return 2100
    return val

# --- INTERFACCIA UTENTE ---

with st.sidebar:
    st.title("⚙️ Configurazione 2.0")
    league_name = st.selectbox("Campionato", list(LEAGUES.keys()))
    L_DATA = LEAGUES[league_name]
    
    st.markdown("---")
    st.subheader("🗓️ Fase Stagione")
    matchday = st.slider("Giornata Attuale", 1, 38, 10, help="Le prime giornate danno più peso all'Elo, le ultime alle statistiche attuali.")
    
    # Calcolo pesi dinamici
    W_ELO_DYN, W_STATS_DYN = calculate_dynamic_weights(matchday, L_DATA["w_elo_base"])
    
    st.info(f"""
    ⚖️ **Pesi Dinamici:**
    Rating Storico (Elo): **{W_ELO_DYN*100:.0f}%**
    Statistiche Stagionali: **{W_STATS_DYN*100:.0f}%**
    """)

st.title("Mathbet fc Pro 🧠")
st.markdown(f"**Algoritmo di Precisione Dixon-Coles** | *Optimized for {league_name}*")

# --- INPUT DATI ---
col_h, col_a = st.columns(2)

with col_h:
    st.subheader("🏠 Squadra Casa")
    h_name = st.text_input("Nome", "Home")
    # ELO INPUT
    h_elo = st.number_input("ClubElo Rating", 1000, 2500, 1600, step=10, key="helo", help="Cerca su clubelo.com")
    # FORMAZIONE
    h_str = st.slider("Disponibilità Titolari %", 50, 100, 100, key="hstr", help="100% = Titolari. 80% = Manca bomber/difensore chiave.")
    
    with st.expander("📊 Statistiche Casa (Stagione + Ultime 5)", expanded=True):
        h_gf_s = st.number_input("GF Media Totale", 0.0, 5.0, 1.4, key="h1")
        h_gs_s = st.number_input("GS Media Totale", 0.0, 5.0, 1.0, key="h2")
        h_gf_h = st.number_input("GF Media in Casa", 0.0, 5.0, 1.6, key="h3")
        h_gs_h = st.number_input("GS Media in Casa", 0.0, 5.0, 0.8, key="h4")
        st.markdown("---")
        h_xg_l5 = st.number_input("xG Prodotti (Ultime 5)", 0.0, 20.0, 6.5, key="h5", help="Totale xG delle ultime 5 partite")
        h_xga_l5 = st.number_input("xG Subiti (Ultime 5)", 0.0, 20.0, 4.5, key="h6", help="Totale xG Concessi ultime 5 partite")

with col_a:
    st.subheader("✈️ Squadra Ospite")
    a_name = st.text_input("Nome", "Away")
    # ELO INPUT
    a_elo = st.number_input("ClubElo Rating", 1000, 2500, 1550, step=10, key="aelo")
    # FORMAZIONE
    a_str = st.slider("Disponibilità Titolari %", 50, 100, 100, key="astr")
    
    with st.expander("📊 Statistiche Ospite (Stagione + Ultime 5)", expanded=True):
        a_gf_s = st.number_input("GF Media Totale", 0.0, 5.0, 1.2, key="a1")
        a_gs_s = st.number_input("GS Media Totale", 0.0, 5.0, 1.3, key="a2")
        a_gf_a = st.number_input("GF Media Fuori", 0.0, 5.0, 1.0, key="a3")
        a_gs_a = st.number_input("GS Media Fuori", 0.0, 5.0, 1.5, key="a4")
        st.markdown("---")
        a_xg_l5 = st.number_input("xG Prodotti (Ultime 5)", 0.0, 20.0, 5.0, key="a5")
        a_xga_l5 = st.number_input("xG Subiti (Ultime 5)", 0.0, 20.0, 6.0, key="a6")

st.markdown("---")
st.subheader("💰 Quote Bookmaker (Per calcolo Valore)")
qc1, qc2, qc3 = st.columns(3)
b1 = qc1.number_input("Quota 1", 1.01, 100.0, 2.20)
bX = qc2.number_input("Quota X", 1.01, 100.0, 3.10)
b2 = qc3.number_input("Quota 2", 1.01, 100.0, 3.40)

# --- MOTORE DI CALCOLO ---
if st.button("🚀 CALCOLA PREVISIONI", type="primary", use_container_width=True):
    
    # 1. ELABORAZIONE STATISTICHE (Stats based xG)
    # Convertiamo i totali delle ultime 5 in medie
    h_form_att = h_xg_l5 / 5.0
    h_form_def = h_xga_l5 / 5.0
    a_form_att = a_xg_l5 / 5.0
    a_form_def = a_xga_l5 / 5.0
    
    # Pesi interni alle stats (Stagione vs Forma vs Casa/Fuori)
    # Casa conta di più per chi gioca in casa
    w_seas = 0.4
    w_venue = 0.35
    w_form = 0.25

    # Potenziale Offensivo/Difensivo ponderato
    h_att_val = (h_gf_s * w_seas) + (h_gf_h * w_venue) + (h_form_att * w_form)
    h_def_val = (h_gs_s * w_seas) + (h_gs_h * w_venue) + (h_form_def * w_form)
    
    a_att_val = (a_gf_s * w_seas) + (a_gf_a * w_venue) + (a_form_att * w_form)
    a_def_val = (a_gs_s * w_seas) + (a_gs_a * w_venue) + (a_form_def * w_form)
    
    # xG derivante dalle Stats (metodo classico: Attacco H * Difesa A / Media Lega)
    xg_stats_h = (h_att_val * a_def_val) / L_DATA["avg"]
    xg_stats_a = (a_att_val * h_def_val) / L_DATA["avg"]

    # 2. ELABORAZIONE ELO (Elo based xG)
    # Formula probabilità Elo: 1 / (1 + 10^((RB-RA)/400))
    # Convertiamo la probabilità Elo in aspettativa Gol
    
    # Home Advantage in Elo points (es. +100 punti elo per chi gioca in casa)
    elo_ha_points = L_DATA["ha"] * 400.0 # Convertiamo il fattore decimale in punti elo approx
    if league_name == "🇮🇹 Serie A": elo_ha_points = 80.0 
    
    diff_h = (h_elo + elo_ha_points) - a_elo
    
    # Probabilità vittoria attesa da Elo
    expected_score_elo_h = 1 / (1 + 10 ** (-diff_h / 400.0))
    
    # Convertiamo la % vittoria Elo in xG stimati (approssimazione euristica basata sulla media gol lega)
    # Se exp_score > 0.5, xG > media. 
    xg_elo_h = L_DATA["avg"] * (expected_score_elo_h / 0.5) ** 0.85 # esponente per smussare estremi
    xg_elo_a = L_DATA["avg"] * ((1 - expected_score_elo_h) / 0.5) ** 0.85

    # 3. FUSIONE (Weighted Average)
    final_xg_h = (xg_elo_h * W_ELO_DYN) + (xg_stats_h * W_STATS_DYN)
    final_xg_a = (xg_elo_a * W_ELO_DYN) + (xg_stats_a * W_STATS_DYN)
    
    # 4. CORREZIONE FORMAZIONE (Strength)
    # Non riduce linearmente, ma impatta esponenzialmente la capacità offensiva
    final_xg_h = final_xg_h * (h_str/100.0)
    final_xg_a = final_xg_a * (a_str/100.0)
    
    # Se mancano difensori (strength bassa), l'avversario guadagna xG
    if h_str < 90: final_xg_a *= 1.05 + ((100-h_str)/200.0)
    if a_str < 90: final_xg_h *= 1.05 + ((100-a_str)/200.0)

    # 5. DIXON-COLES SIMULATION
    prob_1, prob_X, prob_2 = 0, 0, 0
    prob_over_25 = 0
    prob_gg = 0
    score_matrix = np.zeros((7, 7))
    score_list = []
    
    for h in range(7):
        for a in range(7):
            p = dixon_coles_probability(h, a, final_xg_h, final_xg_a, RHO)
            
            if h > a: prob_1 += p
            elif h == a: prob_X += p
            else: prob_2 += p
            
            if (h+a) > 2.5: prob_over_25 += p
            if h > 0 and a > 0: prob_gg += p
            
            score_matrix[h, a] = p
            score_list.append({"Risultato": f"{h}-{a}", "Prob": p})

    # Normalizzazione finale (per sicurezza matematica)
    tot_prob = prob_1 + prob_X + prob_2
    prob_1 /= tot_prob
    prob_X /= tot_prob
    prob_2 /= tot_prob
    
    # Salvataggio
    st.session_state.xg_h = final_xg_h
    st.session_state.xg_a = final_xg_a
    score_list.sort(key=lambda x: x["Prob"], reverse=True)
    
    # --- DISPLAY OUTPUT ---
    st.balloons()
    
    # HEADER KPI
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("xG Casa Attesi", f"{final_xg_h:.2f}")
    kpi2.metric("xG Ospite Attesi", f"{final_xg_a:.2f}")
    tot_xg = final_xg_h + final_xg_a
    kpi3.metric("xG Totali Match", f"{tot_xg:.2f}", delta="High Volatility" if tot_xg > 3.0 else "Low Volatility")

    # 1X2 TABLE
    st.subheader("🏆 Analisi 1X2 & Valore")
    
    fair_1 = 1/prob_1 if prob_1 > 0 else 0
    fair_X = 1/prob_X if prob_X > 0 else 0
    fair_2 = 1/prob_2 if prob_2 > 0 else 0
    
    # Calcolo Kelly
    k1 = calculate_kelly(prob_1, b1)
    kX = calculate_kelly(prob_X, bX)
    k2 = calculate_kelly(prob_2, b2)
    
    # Evidenziare il valore
    def highlight_val(val):
        color = 'green' if float(val.strip('%')) > 0 else 'red'
        return f'color: {color}'

    df_1x2 = pd.DataFrame({
        "Esito": ["1 (Casa)", "X (Pareggio)", "2 (Ospite)"],
        "Probabilità": [f"{prob_1:.1%}", f"{prob_X:.1%}", f"{prob_2:.1%}"],
        "Quota Reale (Fair)": [f"{fair_1:.2f}", f"{fair_X:.2f}", f"{fair_2:.2f}"],
        "Quota Book": [b1, bX, b2],
        "Edge (Valore)": [f"{((b1/fair_1)-1)*100:.1f}%", f"{((bX/fair_X)-1)*100:.1f}%", f"{((b2/fair_2)-1)*100:.1f}%"],
        "Kelly Stake": [f"{k1:.1f}%", f"{kX:.1f}%", f"{k2:.1f}%"]
    })
    
    st.table(df_1x2)
    
    col_score, col_heat = st.columns([1, 2])
    
    with col_score:
        st.subheader("🎯 Top 5 Risultati")
        df_res = pd.DataFrame(score_list[:5])
        df_res["Prob"] = df_res["Prob"].apply(lambda x: f"{x:.1%}")
        st.dataframe(df_res, hide_index=True)
        
    with col_heat:
        st.subheader("🔥 Distribuzione Gol")
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.heatmap(score_matrix[:5, :5], annot=True, fmt=".0%", cmap="Blues", 
                    xticklabels=range(5), yticklabels=range(5))
        plt.xlabel(f"Gol {a_name}")
        plt.ylabel(f"Gol {h_name}")
        st.pyplot(fig)
        
    # SEZIONE GOAL
    st.markdown("---")
    g1, g2 = st.columns(2)
    with g1:
        st.subheader("📉 Under / Over 2.5")
        prob_under = 1.0 - prob_over_25
        fair_o = 1/prob_over_25
        fair_u = 1/prob_under
        st.write(f"**Over 2.5:** {prob_over_25:.1%} (Quota Fair: **{fair_o:.2f}**)")
        st.write(f"**Under 2.5:** {prob_under:.1%} (Quota Fair: **{fair_u:.2f}**)")
        st.progress(prob_over_25)
        
    with g2:
        st.subheader("⚽ Gol / No Gol")
        prob_ng = 1.0 - prob_gg
        fair_gg = 1/prob_gg
        fair_ng = 1/prob_ng
        st.write(f"**GG (Entrambe segnano):** {prob_gg:.1%} (Quota Fair: **{fair_gg:.2f}**)")
        st.write(f"**NG (No Gol):** {prob_ng:.1%} (Quota Fair: **{fair_ng:.2f}**)")
        st.progress(prob_gg)

# --- CSV DOWNLOAD ---
if st.session_state.xg_h > 0:
    csv_data = f"Home,Away,xG_H,xG_A,Prob1,ProbX,Prob2\n{h_name},{a_name},{st.session_state.xg_h:.2f},{st.session_state.xg_a:.2f},{st.session_state.prob_1:.2f},{st.session_state.prob_X:.2f},{st.session_state.prob_2:.2f}"
    st.download_button("💾 Scarica Dati", csv_data, "match_analysis.csv", "text/csv")
