import streamlit as st
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Mathbet fc Pro 3.1",
                   page_icon="⚽",
                   layout="wide")

# --- INIZIALIZZAZIONE SESSION STATE ---
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
    st.session_state.xg_h = 0
    st.session_state.xg_a = 0
    st.session_state.team_avg_h = 0 
    st.session_state.team_avg_a = 0 
    st.session_state.prob_1 = 0
    st.session_state.prob_X = 0
    st.session_state.prob_2 = 0
    st.session_state.all_scores = []

# --- DATABASE CAMPIONATI ---
LEAGUES = {
    "🌐 Generico (Media)": { "avg": 1.35, "ha": 0.35, "w_elo_base": 0.40 }, 
    "🇮🇹 Serie A":          { "avg": 1.30, "ha": 0.25, "w_elo_base": 0.50 },
    "🇮🇹 Serie B":          { "avg": 1.15, "ha": 0.30, "w_elo_base": 0.30 },
    "🇬🇧 Premier League":   { "avg": 1.55, "ha": 0.35, "w_elo_base": 0.55 },
    "🇩🇪 Bundesliga":       { "avg": 1.65, "ha": 0.40, "w_elo_base": 0.45 },
    "🇪🇸 La Liga":          { "avg": 1.25, "ha": 0.30, "w_elo_base": 0.55 },
    "🇫🇷 Ligue 1":          { "avg": 1.30, "ha": 0.28, "w_elo_base": 0.45 },
}

# Parametri Globali
SCALING_FACTOR = 400.0  
KELLY_FRACTION = 0.25
RHO = -0.13 # Correlazione Dixon-Coles

# --- FUNZIONI MATEMATICHE ---

def calculate_dynamic_weights(matchday, base_w_elo):
    """Calcola peso Elo vs Stats in base alla giornata."""
    if matchday <= 8:
        w_elo = max(base_w_elo, 0.75)
    elif matchday <= 19:
        w_elo = base_w_elo + 0.10
    else:
        w_elo = base_w_elo
    w_stats = 1.0 - w_elo
    return w_elo, w_stats

def dixon_coles_probability(h_goals, a_goals, mu_h, mu_a, rho):
    """Formula Dixon-Coles per correggere i pareggi bassi."""
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

def calculate_player_probability(metric_per90, expected_mins, team_match_xg, team_avg_xg):
    """Calcola probabilità Gol/Assist giocatore basandosi sugli xG di squadra."""
    base_lambda = (metric_per90 / 90.0) * expected_mins
    match_factor = 1.0
    if team_avg_xg > 0:
        match_factor = team_match_xg / team_avg_xg
    final_lambda = base_lambda * match_factor
    prob_at_least_one = 1 - math.exp(-final_lambda)
    return prob_at_least_one, final_lambda

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Configurazione")
    league_name = st.selectbox("Campionato", list(LEAGUES.keys()))
    L_DATA = LEAGUES[league_name]
    
    st.markdown("---")
    st.subheader("🗓️ Fase Stagione")
    matchday = st.slider("Giornata Attuale", 1, 38, 10)
    W_ELO_DYN, W_STATS_DYN = calculate_dynamic_weights(matchday, L_DATA["w_elo_base"])
    
    st.info(f"""
    ⚖️ **Pesi Dinamici Attivi:**
    Elo Storico: **{W_ELO_DYN*100:.0f}%**
    Stats Stagionali: **{W_STATS_DYN*100:.0f}%**
    """)

st.title("Mathbet fc Pro 3.1 ⚽")
st.markdown(f"**Core:** Dixon-Coles + Monte Carlo (Input: Gol Reali) | *League: {league_name}*")

# --- LINK UTILI ---
with st.expander("🔗 Link Utili (Clicca per aprire)", expanded=False):
    lc1, lc2, lc3 = st.columns(3)
    with lc1:
        st.caption("Rating")
        st.link_button("ClubElo", "http://clubelo.com")
    with lc2:
        st.caption("Stats Gol")
        st.link_button("Flashscore", "https://www.flashscore.it/")
    with lc3:
        st.caption("Giocatori")
        st.link_button("FBref", "https://fbref.com")

st.markdown("---")

# --- INPUT DATI ---
col_h, col_a = st.columns(2)

with col_h:
    st.subheader("🏠 Squadra Casa")
    h_name = st.text_input("Nome Casa", "Home")
    h_elo = st.number_input("ClubElo Rating", 1000, 2500, 1600, step=10, key="helo")
    h_str = st.slider("Disponibilità Titolari %", 50, 100, 100, key="hstr")
    
    with st.expander("📊 Stats Casa (Media)", expanded=True):
        h_gf_s = st.number_input("GF Media Stagione", 0.0, 5.0, 1.4, key="h1")
        h_gs_s = st.number_input("GS Media Stagione", 0.0, 5.0, 1.0, key="h2")
        h_gf_h = st.number_input("GF Media in Casa", 0.0, 5.0, 1.6, key="h3")
        h_gs_h = st.number_input("GS Media in Casa", 0.0, 5.0, 0.8, key="h4")
        st.markdown("---")
        # --- MODIFICA QUI: INPUT GOL REALI ---
        h_gf_l5 = st.number_input("GOL FATTI (Ultime 5)", 0, 25, 7, key="h5", help="Somma dei gol segnati nelle ultime 5 partite")
        h_gs_l5 = st.number_input("GOL SUBITI (Ultime 5)", 0, 25, 5, key="h6", help="Somma dei gol subiti nelle ultime 5 partite")

with col_a:
    st.subheader("✈️ Squadra Ospite")
    a_name = st.text_input("Nome Ospite", "Away")
    a_elo = st.number_input("ClubElo Rating", 1000, 2500, 1550, step=10, key="aelo")
    a_str = st.slider("Disponibilità Titolari %", 50, 100, 100, key="astr")
    
    with st.expander("📊 Stats Ospite (Media)", expanded=True):
        a_gf_s = st.number_input("GF Media Stagione", 0.0, 5.0, 1.2, key="a1")
        a_gs_s = st.number_input("GS Media Stagione", 0.0, 5.0, 1.3, key="a2")
        a_gf_a = st.number_input("GF Media Fuori", 0.0, 5.0, 1.0, key="a3")
        a_gs_a = st.number_input("GS Media Fuori", 0.0, 5.0, 1.5, key="a4")
        st.markdown("---")
        # --- MODIFICA QUI: INPUT GOL REALI ---
        a_gf_l5 = st.number_input("GOL FATTI (Ultime 5)", 0, 25, 5, key="a5")
        a_gs_l5 = st.number_input("GOL SUBITI (Ultime 5)", 0, 25, 6, key="a6")

st.markdown("---")
st.subheader("💰 Quote Bookmaker")
qc1, qc2, qc3 = st.columns(3)
b1 = qc1.number_input("Quota 1", 1.01, 100.0, 2.20)
bX = qc2.number_input("Quota X", 1.01, 100.0, 3.10)
b2 = qc3.number_input("Quota 2", 1.01, 100.0, 3.40)

# --- CALCOLO ---
if st.button("🚀 ANALIZZA PARTITA", type="primary", use_container_width=True):
    
    # 1. Stats Logic (Convertiamo i Gol totali in Media Form)
    # --- MODIFICA QUI: CALCOLO MEDIA DA GOL REALI ---
    h_form_att = h_gf_l5 / 5.0
    h_form_def = h_gs_l5 / 5.0
    a_form_att = a_gf_l5 / 5.0
    a_form_def = a_gs_l5 / 5.0
    
    w_seas, w_venue, w_form = 0.40, 0.35, 0.25

    h_att_val = (h_gf_s * w_seas) + (h_gf_h * w_venue) + (h_form_att * w_form)
    h_def_val = (h_gs_s * w_seas) + (h_gs_h * w_venue) + (h_form_def * w_form)
    a_att_val = (a_gf_s * w_seas) + (a_gf_a * w_venue) + (a_form_att * w_form)
    a_def_val = (a_gs_s * w_seas) + (a_gs_a * w_venue) + (a_form_def * w_form)
    
    xg_stats_h = (h_att_val * a_def_val) / L_DATA["avg"]
    xg_stats_a = (a_att_val * h_def_val) / L_DATA["avg"]

    # 2. Elo Logic (Stessa di prima)
    elo_ha_points = 80.0 if league_name == "🇮🇹 Serie A" else L_DATA["ha"] * 400.0
    diff_h = (h_elo + elo_ha_points) - a_elo
    expected_score_elo_h = 1 / (1 + 10 ** (-diff_h / 400.0))
    xg_elo_h = L_DATA["avg"] * (expected_score_elo_h / 0.5) ** 0.85
    xg_elo_a = L_DATA["avg"] * ((1 - expected_score_elo_h) / 0.5) ** 0.85

    # 3. Weighted Merge
    final_xg_h = (xg_elo_h * W_ELO_DYN) + (xg_stats_h * W_STATS_DYN)
    final_xg_a = (xg_elo_a * W_ELO_DYN) + (xg_stats_a * W_STATS_DYN)
    
    # 4. Strength Correction
    final_xg_h = final_xg_h * (h_str/100.0)
    final_xg_a = final_xg_a * (a_str/100.0)
    if h_str < 90: final_xg_a *= 1.05 + ((100-h_str)/200.0)
    if a_str < 90: final_xg_h *= 1.05 + ((100-a_str)/200.0)

    # 5. Dixon-Coles Loop
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

    # Normalization
    tot_prob = prob_1 + prob_X + prob_2
    prob_1 /= tot_prob
    prob_X /= tot_prob
    prob_2 /= tot_prob
    
    # Save State
    st.session_state.analyzed = True
    st.session_state.xg_h = final_xg_h
    st.session_state.xg_a = final_xg_a
    st.session_state.team_avg_h = h_gf_s
    st.session_state.team_avg_a = a_gf_s
    st.session_state.prob_1 = prob_1
    st.session_state.prob_X = prob_X
    st.session_state.prob_2 = prob_2
    score_list.sort(key=lambda x: x["Prob"], reverse=True)
    st.session_state.all_scores = score_list

    # --- OUTPUT ---
    st.balloons()
    
    # KPI + MONTE CARLO
    st.header(f"📊 Analisi: {h_name} vs {a_name}")
    c1, c2 = st.columns(2)
    c1.metric("xG Attesi (Stimati)", f"{final_xg_h:.2f} - {final_xg_a:.2f}")
    
    # Monte Carlo Sim
    n_sims = 10000
    sim_h = np.random.poisson(final_xg_h, n_sims)
    sim_a = np.random.poisson(final_xg_a, n_sims)
    sim_1 = np.sum(sim_h > sim_a) / n_sims
    sim_X = np.sum(sim_h == sim_a) / n_sims
    sim_2 = np.sum(sim_h < sim_a) / n_sims
    c2.info(f"🎲 Monte Carlo (10k Sim): 1: {sim_1:.1%} | X: {sim_X:.1%} | 2: {sim_2:.1%}")

    # 1X2 TABLE
    st.subheader("🏆 Esito Finale & Valore")
    k1, kX, k2 = calculate_kelly(prob_1, b1), calculate_kelly(prob_X, bX), calculate_kelly(prob_2, b2)
    df_1x2 = pd.DataFrame({
        "Esito": ["1", "X", "2"],
        "Prob %": [f"{prob_1*100:.1f}%", f"{prob_X*100:.1f}%", f"{prob_2*100:.1f}%"],
        "Fair": [f"{1/prob_1:.2f}", f"{1/prob_X:.2f}", f"{1/prob_2:.2f}"],
        "Book": [b1, bX, b2],
        "Value": [f"{((b1*prob_1)-1)*100:+.1f}%", f"{((bX*prob_X)-1)*100:+.1f}%", f"{((b2*prob_2)-1)*100:+.1f}%"],
        "Stake": [f"{k1:.1f}%", f"{kX:.1f}%", f"{k2:.1f}%"]
    })
    st.table(df_1x2)
    
    # SCORES & HEATMAP
    col_score, col_heat = st.columns([1, 2])
    with col_score:
        st.subheader("🎯 Top 5 Risultati")
        df_res = pd.DataFrame(score_list[:5])
        df_res["Prob"] = df_res["Prob"].apply(lambda x: f"{x:.1%}")
        st.dataframe(df_res, hide_index=True)
    with col_heat:
        st.subheader("🔥 Distribuzione Gol")
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.heatmap(score_matrix[:5, :5], annot=True, fmt=".0%", cmap="Blues", cbar=False)
        plt.xlabel(a_name); plt.ylabel(h_name)
        st.pyplot(fig)

    # U/O & GG
    st.markdown("---")
    g1, g2 = st.columns(2)
    with g1:
        st.subheader("📉 Under / Over 2.5")
        prob_under = 1.0 - prob_over_25
        st.write(f"**Over 2.5:** {prob_over_25:.1%} (Fair: **{1/prob_over_25:.2f}**)")
        st.write(f"**Under 2.5:** {prob_under:.1%} (Fair: **{1/prob_under:.2f}**)")
    with g2:
        st.subheader("⚽ Gol / No Gol")
        prob_ng = 1.0 - prob_gg
        st.write(f"**GG:** {prob_gg:.1%} (Fair: **{1/prob_gg:.2f}**)")
        st.write(f"**NG:** {prob_ng:.1%} (Fair: **{1/prob_ng:.2f}**)")

# --- PLAYER PROP SECTION ---
st.markdown("---")
st.header("👤 Marcatore / Assist")

if not st.session_state.analyzed:
    st.warning("⚠️ Prima analizza la partita cliccando il pulsante rosso!")
else:
    with st.expander("Apri Calcolatore Player Prop", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            player_name = st.text_input("Giocatore", "Nome")
            team_sel = st.radio("Squadra", [h_name, a_name], horizontal=True)
            mins = st.slider("Minuti previsti", 15, 95, 80)
        with c2:
            type_s = st.radio("Tipo", ["GOL", "ASSIST"], horizontal=True)
            val_90 = st.number_input(f"x{'G' if 'GOL' in type_s else 'A'}/90 (da FBref)", 0.00, 2.00, 0.35)
            b_odd = st.number_input("Quota Bookmaker", 1.0, 50.0, 3.00)

        # Logica contestuale
        ctx_xg = st.session_state.xg_h if team_sel == h_name else st.session_state.xg_a
        ctx_avg = st.session_state.team_avg_h if team_sel == h_name else st.session_state.team_avg_a

        p1, _ = calculate_player_probability(val_90, mins, ctx_xg, ctx_avg)
        fair = 1/p1 if p1>0 else 0
        edge = ((b_odd/fair)-1)*100 if fair>0 else -100
        
        c_res1, c_res2, c_res3 = st.columns(3)
        c_res1.metric(f"Prob {type_s}", f"{p1*100:.1f}%")
        c_res2.metric("Fair Odd", f"{fair:.2f}")
        c_res3.metric("Valore", f"{edge:+.1f}%", delta_color="normal" if edge<0 else "inverse")

# --- TOOLS EXTRA ---
st.markdown("---")
st.header("🛠️ Strumenti Extra")

with st.expander("🕵️ Reverse Engineering Quote"):
    q = st.number_input("Inserisci Quota Book", 1.01, 100.0, 1.90)
    st.write(f"Probabilità Implicita: **{1/q:.1%}**")

with st.expander("🧮 Calcolatore Manuale (Kelly)", expanded=False):
    k1, k2, k3 = st.columns(3)
    mp = k1.number_input("La tua stima (%)", 0.1, 100.0, 50.0) / 100
    mq = k2.number_input("Quota Book", 1.01, 100.0, 2.0)
    mb = k3.number_input("Bankroll", 0.0, 10000.0, 1000.0)
    
    mev = (mp * mq) - 1
    if mev > 0:
        mk = (((mq - 1) * mp) - (1 - mp)) / (mq - 1) * KELLY_FRACTION
        stake = mb * mk
        st.success(f"✅ VALUE! Stake: € {stake:.2f} ({mk*100:.1f}%)")
    else:
        st.error("❌ Nessun Valore")
