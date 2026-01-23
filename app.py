import streamlit as st
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Mathbet fc Pro",
                   page_icon="⚽",
                   layout="wide")

# --- INIZIALIZZAZIONE SESSION STATE (MEMORIA) ---
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
    st.session_state.xg_h = 0
    st.session_state.xg_a = 0
    st.session_state.h_avg_seas = 0
    st.session_state.a_avg_seas = 0
    st.session_state.prob_1 = 0
    st.session_state.prob_X = 0
    st.session_state.prob_2 = 0
    st.session_state.top_score = ""

# --- DATABASE CAMPIONATI ---
LEAGUES = {
    "🌐 Generico (Media)": { "avg": 1.35, "ha": 100, "w_elo": 0.35, "w_seas": 0.35, "w_form": 0.30 },
    "🇮🇹 Serie A":          { "avg": 1.30, "ha": 90,  "w_elo": 0.40, "w_seas": 0.35, "w_form": 0.25 },
    "🇮🇹 Serie B":          { "avg": 1.15, "ha": 85,  "w_elo": 0.20, "w_seas": 0.30, "w_form": 0.50 },
    "🇬🇧 Premier League":   { "avg": 1.48, "ha": 105, "w_elo": 0.45, "w_seas": 0.35, "w_form": 0.20 },
    "🇩🇪 Bundesliga":       { "avg": 1.58, "ha": 105, "w_elo": 0.35, "w_seas": 0.40, "w_form": 0.25 },
    "🇪🇸 La Liga":          { "avg": 1.25, "ha": 100, "w_elo": 0.50, "w_seas": 0.30, "w_form": 0.20 },
    "🇫🇷 Ligue 1":          { "avg": 1.28, "ha": 95,  "w_elo": 0.35, "w_seas": 0.35, "w_form": 0.30 },
}

# Parametri Globali
SCALING_FACTOR = 800  
W_VENUE = 0.20
WEIGHT_HISTORICAL_UO = 0.40
KELLY_FRACTION = 0.25
DRAW_BOOST = 1.10

# --- FUNZIONI MATEMATICHE ---
def normalize_elo(val):
    if val <= 100: return 1100 + (val * 9.5)
    if val > 2200: return val / 1.75
    return val

def poisson(k, lmbda):
    return (lmbda**k * math.exp(-lmbda)) / math.factorial(k)

def calculate_kelly(prob_true, odds_book):
    if odds_book <= 1.01 or prob_true <= 0: return 0.0
    b = odds_book - 1
    p = prob_true
    q = 1 - p
    kelly = ((b * p) - q) / b
    return max(0.0, (kelly * KELLY_FRACTION) * 100)

def calculate_player_probability(metric_per90, expected_mins, team_match_xg, team_avg_xg):
    base_lambda = (metric_per90 / 90.0) * expected_mins
    match_factor = 1.0
    if team_avg_xg > 0:
        match_factor = team_match_xg / team_avg_xg
    final_lambda = base_lambda * match_factor
    prob_at_least_one = 1 - math.exp(-final_lambda)
    prob_at_least_two = 1 - (math.exp(-final_lambda) + (final_lambda * math.exp(-final_lambda)))
    return prob_at_least_one, prob_at_least_two, final_lambda

# --- SIDEBAR & INPUT ---
with st.sidebar:
    st.title("⚙️ Configurazione")
    league_name = st.selectbox("Campionato", list(LEAGUES.keys()))
    L_DATA = LEAGUES[league_name]
    CURRENT_AVG_GOALS = L_DATA["avg"]
    CURRENT_HOME_ADV = L_DATA["ha"]
    W_ELO = L_DATA["w_elo"]
    W_SEASON = L_DATA["w_seas"]
    W_FORM = L_DATA["w_form"]

    st.info(f"""
    📊 **Parametri Attivi:**
    Avg Gol: {CURRENT_AVG_GOALS} | Home Adv: {CURRENT_HOME_ADV}
    
    ⚖️ **Pesi (Normalizzati):**
    Elo: {W_ELO*100:.0f}% | Stagione: {W_SEASON*100:.0f}% | Forma: {W_FORM*100:.0f}%
    """)

st.title("Mathbet fc Pro ⚽")
st.markdown("ℹ️ *Algoritmo Ibrido: Poisson Pesato + Elo Scaling (800) + Monte Carlo*")

with st.expander("🔗 Link Utili (Clicca per aprire)", expanded=False):
    lc1, lc2, lc3 = st.columns(3)
    with lc1:
        st.caption("Rating")
        st.link_button("ClubElo", "http://clubelo.com")
    with lc2:
        st.caption("Stats")
        st.link_button("FootyStats (Serie A)", "https://footystats.org/it/italy/serie-a/form-table")
    with lc3:
        st.caption("Giocatori")
        st.link_button("FBref (Players)", "https://fbref.com/en/comps/11/Serie-A-Stats")

st.markdown("---")

col_h, col_a = st.columns(2)
with col_h:
    st.subheader("🏠 Casa")
    h_name = st.text_input("Nome Casa", "Home Team")
    raw_h_elo = st.number_input("Rating (Elo/Opta)", 1500.0, step=10.0, key="h_elo")
    h_elo = normalize_elo(raw_h_elo)
    h_str = st.slider("Formazione Casa %", 50, 100, 100, key="h_str")
    with st.expander("📊 Stats Casa", expanded=True):
        h_gf_seas = st.number_input("GF Media Stagione", 0.0, 5.0, 1.5, key="h1")
        h_gs_seas = st.number_input("GS Media Stagione", 0.0, 5.0, 1.2, key="h2")
        h_gf_l5 = st.number_input("Gol fatti ultime 5", 0, 20, 7, key="h3")
        h_gs_l5 = st.number_input("Gol subiti ultime 5", 0, 20, 5, key="h4")
        h_gf_venue = st.number_input("GF Media Casa", 0.0, 5.0, 1.8, key="h5")
        h_gs_venue = st.number_input("GS Media Casa", 0.0, 5.0, 0.9, key="h6")
    with st.expander("📈 % Over Casa"):
        h_uo = {0.5: 95, 1.5: st.slider("% Over 1.5 H", 0, 100, 75, key="h15"), 2.5: st.slider("% Over 2.5 H", 0, 100, 50, key="h25")}

with col_a:
    st.subheader("✈️ Ospite")
    a_name = st.text_input("Nome Ospite", "Away Team")
    raw_a_elo = st.number_input("Rating (Elo/Opta)", 1450.0, step=10.0, key="a_elo")
    a_elo = normalize_elo(raw_a_elo)
    a_str = st.slider("Formazione Ospite %", 50, 100, 100, key="a_str")
    with st.expander("📊 Stats Ospite", expanded=True):
        a_gf_seas = st.number_input("GF Media Stagione", 0.0, 5.0, 1.2, key="a1")
        a_gs_seas = st.number_input("GS Media Stagione", 0.0, 5.0, 1.4, key="a2")
        a_gf_l5 = st.number_input("Gol fatti ultime 5", 0, 20, 5, key="a3")
        a_gs_l5 = st.number_input("Gol subiti ultime 5", 0, 20, 8, key="a4")
        a_gf_venue = st.number_input("GF Media Fuori", 0.0, 5.0, 1.0, key="a5")
        a_gs_venue = st.number_input("GS Media Fuori", 0.0, 5.0, 1.6, key="a6")
    with st.expander("📈 % Over Ospite"):
        a_uo = {0.5: 95, 1.5: st.slider("% Over 1.5 A", 0, 100, 70, key="a15"), 2.5: st.slider("% Over 2.5 A", 0, 100, 45, key="a25")}

st.markdown("---")
st.subheader("💰 Quote Bookmaker")
qc1, qc2, qc3, qc4, qc5 = st.columns(5)
b1 = qc1.number_input("1", 1.01, 100.0, 2.50)
bX = qc2.number_input("X", 1.01, 100.0, 3.20)
b2 = qc3.number_input("2", 1.01, 100.0, 2.90)
bGG = qc4.number_input("GG", 1.01, 100.0, 1.75)
bNG = qc5.number_input("NG", 1.01, 100.0, 2.05)

# --- CORE LOGIC ---
if st.button("🚀 ANALIZZA PARTITA", type="primary", use_container_width=True):
    
    # 1. FIX LOGICA: NORMALIZZAZIONE PESI
    tot_weight = W_SEASON + W_FORM + W_VENUE
    w_s_norm = W_SEASON / tot_weight
    w_f_norm = W_FORM / tot_weight
    w_v_norm = W_VENUE / tot_weight

    # 2. METRICHE
    h_avg_gf_form = h_gf_l5 / 5.0
    h_avg_gs_form = h_gs_l5 / 5.0
    a_avg_gf_form = a_gf_l5 / 5.0
    a_avg_gs_form = a_gs_l5 / 5.0

    att_h = (h_gf_seas * w_s_norm) + (h_avg_gf_form * w_f_norm) + (h_gf_venue * w_v_norm)
    def_h = (h_gs_seas * w_s_norm) + (h_avg_gs_form * w_f_norm) + (h_gs_venue * w_v_norm)
    att_a = (a_gf_seas * w_s_norm) + (a_avg_gf_form * w_f_norm) + (a_gf_venue * w_v_norm)
    def_a = (a_gs_seas * w_s_norm) + (a_avg_gs_form * w_f_norm) + (a_gs_venue * w_v_norm)

    # 3. ELO & XG COMBINATI
    w_stats_l = 1.0 - W_ELO
    diff_h = (h_elo + CURRENT_HOME_ADV) - a_elo
    diff_a = a_elo - (h_elo + CURRENT_HOME_ADV)
    
    xg_elo_h = CURRENT_AVG_GOALS * (1 + (diff_h / SCALING_FACTOR))
    xg_elo_a = CURRENT_AVG_GOALS * (1 + (diff_a / SCALING_FACTOR))
    
    xg_stats_h = (att_h * def_a) / CURRENT_AVG_GOALS
    xg_stats_a = (att_a * def_h) / CURRENT_AVG_GOALS

    xg_h = (xg_elo_h * W_ELO) + (xg_stats_h * w_stats_l)
    xg_a = (xg_elo_a * W_ELO) + (xg_stats_a * w_stats_l)
    
    xg_h = max(0.1, xg_h * (h_str / 100.0))
    xg_a = max(0.1, xg_a * (a_str / 100.0))

    # 4. CALCOLO PROBABILITÀ
    prob_1, prob_X, prob_2 = 0, 0, 0
    prob_GG, prob_NG = 0, 0
    total_goals_probs = {k: 0 for k in range(20)}
    uo_probs_math = {0.5: [0, 0], 1.5: [0, 0], 2.5: [0, 0], 3.5: [0, 0], 4.5: [0, 0]}
    score_matrix = np.zeros((6, 6))
    all_scores_list = []
    RHO = -0.13

    for h in range(10):
        for a in range(10):
            p = poisson(h, xg_h) * poisson(a, xg_a)
            correction = 1.0
            if h==0 and a==0: correction = 1.0 - (xg_h*xg_a*RHO)
            elif h==0 and a==1: correction = 1.0 + (xg_h*RHO)
            elif h==1 and a==0: correction = 1.0 + (xg_a*RHO)
            elif h==1 and a==1: correction = 1.0 - RHO
            p = max(0, p * correction)
            
            if h == a: p *= DRAW_BOOST

            if h > a: prob_1 += p
            elif h == a: prob_X += p
            else: prob_2 += p

            if h > 0 and a > 0: prob_GG += p
            else: prob_NG += p

            tot = h + a
            if tot in total_goals_probs: total_goals_probs[tot] += p
            for line in uo_probs_math:
                if tot < line: uo_probs_math[line][0] += p
                else: uo_probs_math[line][1] += p

            all_scores_list.append({"score": f"{h}-{a}", "prob": p})
            if h < 6 and a < 6: score_matrix[h, a] = p

    total_prob = prob_1 + prob_X + prob_2
    factor = 1.0 / total_prob if total_prob > 0 else 1.0
    p1, pX, p2 = prob_1 * factor, prob_X * factor, prob_2 * factor
    p_gg, p_ng = prob_GG * factor, prob_NG * factor

    for item in all_scores_list: item["prob"] *= factor
    all_scores_list.sort(key=lambda x: x["prob"], reverse=True)

    # 5. SALVATAGGIO IN SESSION STATE
    st.session_state.analyzed = True
    st.session_state.xg_h = xg_h
    st.session_state.xg_a = xg_a
    st.session_state.h_avg_seas = h_gf_seas
    st.session_state.a_avg_seas = a_gf_seas
    st.session_state.prob_1 = p1
    st.session_state.prob_X = pX
    st.session_state.prob_2 = p2
    st.session_state.top_score = all_scores_list[0]['score']

    # --- OUTPUT ---
    st.balloons()
    st.header(f"📊 {h_name} vs {a_name}")

    n_sims = 10000
    sim_h = np.random.poisson(xg_h, n_sims)
    sim_a = np.random.poisson(xg_a, n_sims)
    sim_1 = np.sum(sim_h > sim_a) / n_sims
    sim_X = np.sum(sim_h == sim_a) / n_sims
    sim_2 = np.sum(sim_h < sim_a) / n_sims

    c1, c2 = st.columns(2)
    c1.metric("xG Stimati (Corretti)", f"{xg_h:.2f} - {xg_a:.2f}")
    c2.info(f"🎲 Monte Carlo (10k Sim): 1: {sim_1:.1%} | X: {sim_X:.1%} | 2: {sim_2:.1%}")

    col_main, col_dc = st.columns(2)
    with col_main:
        st.subheader("🏆 Esito Finale")
        f1, fX, f2 = (1/p1 if p1>0 else 0), (1/pX if pX>0 else 0), (1/p2 if p2>0 else 0)
        k1, kX, k2 = calculate_kelly(p1, b1), calculate_kelly(pX, bX), calculate_kelly(p2, b2)
        df_1x2 = pd.DataFrame({
            "Esito": ["1", "X", "2"],
            "Prob %": [f"{p1*100:.1f}%", f"{pX*100:.1f}%", f"{p2*100:.1f}%"],
            "Fair": [f"{f1:.2f}", f"{fX:.2f}", f"{f2:.2f}"],
            "Book": [b1, bX, b2],
            "Value": [f"{((b1/f1)-1)*100:+.1f}%", f"{((bX/fX)-1)*100:+.1f}%", f"{((b2/f2)-1)*100:+.1f}%"],
            "Stake": [f"{k1:.1f}%", f"{kX:.1f}%", f"{k2:.1f}%"]
        })
        st.table(df_1x2)
    
    with col_dc:
        st.subheader("🛡️ Doppia Chance")
        p1X, pX2, p12 = p1+pX, pX+p2, p1+p2
        df_dc = pd.DataFrame({
            "Esito": ["1X", "X2", "12"],
            "Prob %": [f"{p1X*100:.1f}%", f"{pX2*100:.1f}%", f"{p12*100:.1f}%"],
            "Fair": [f"{1/p1X:.2f}", f"{1/pX2:.2f}", f"{1/p12:.2f}"]
        })
        st.table(df_dc)

    st.subheader("🎯 Risultati Esatti")
    cols = st.columns(3)
    for i in range(3):
        item = all_scores_list[i]
        cols[i].metric(f"Top {i+1}: {item['score']}", f"{item['prob']*100:.1f}%", f"Fair: {1/item['prob']:.2f}")

    

    with st.expander("🗺️ Heatmap Punteggi"):
        fig, ax = plt.subplots(figsize=(6,5))
        sns.heatmap(score_matrix, annot=True, fmt=".1%", cmap="YlGnBu", cbar=False)
        plt.xlabel(a_name); plt.ylabel(h_name)
        st.pyplot(fig)

    col_uo, col_gg = st.columns(2)
    with col_uo:
        st.subheader("📉 Under / Over")
        uo_list = []
        for line in [1.5, 2.5, 3.5]:
            u_math, o_math = uo_probs_math[line][0] * factor, uo_probs_math[line][1] * factor
            avg_hist_o = (h_uo.get(line,50)/100 + a_uo.get(line,50)/100)/2
            o_fin = (o_math * (1-WEIGHT_HISTORICAL_UO)) + (avg_hist_o * WEIGHT_HISTORICAL_UO)
            u_fin = 1.0 - o_fin
            uo_list.append({"Linea": line, "U %": f"{u_fin*100:.1f}%", "Fair U": f"{1/u_fin:.2f}", "O %": f"{o_fin*100:.1f}%", "Fair O": f"{1/o_fin:.2f}"})
        st.dataframe(pd.DataFrame(uo_list), hide_index=True)

    with col_gg:
        st.subheader("⚽ Gol / No Gol")
        st.dataframe(pd.DataFrame({
            "Esito": ["GOL", "NO GOL"],
            "Prob %": [f"{p_gg*100:.1f}%", f"{p_ng*100:.1f}%"],
            "Fair": [f"{1/p_gg:.2f}", f"{1/p_ng:.2f}"],
            "Book": [bGG, bNG]
        }), hide_index=True)

# --- PLAYER PROP ---
st.markdown("---")
st.header("👤 Marcatore / Assist")

if not st.session_state.analyzed:
    st.warning("⚠️ Prima analizza la partita cliccando il pulsante rosso sopra!")
else:
    with st.expander("Apri Calcolatore", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            player_name = st.text_input("Giocatore", "Nome")
            team_sel = st.radio("Squadra", [h_name, a_name], horizontal=True)
            mins = st.slider("Minuti previsti", 15, 95, 85)
        with c2:
            type_s = st.radio("Tipo", ["GOL", "ASSIST"], horizontal=True)
            val_90 = st.number_input(f"x{'G' if 'GOL' in type_s else 'A'}/90", 0.00, 2.00, 0.40)
            b_odd = st.number_input("Quota Bookmaker", 1.0, 50.0, 2.50)

        ctx_xg = st.session_state.xg_h if team_sel == h_name else st.session_state.xg_a
        ctx_avg = st.session_state.h_avg_seas if team_sel == h_name else st.session_state.a_avg_seas

        p1, p2, _ = calculate_player_probability(val_90, mins, ctx_xg, ctx_avg)
        fair = 1/p1 if p1>0 else 0
        edge = ((b_odd/fair)-1)*100 if fair>0 else -100
        
        c_res1, c_res2, c_res3 = st.columns(3)
        c_res1.metric(f"Prob {type_s}", f"{p1*100:.1f}%")
        c_res2.metric("Fair Odd", f"{fair:.2f}")
        c_res3.metric("Valore", f"{edge:+.1f}%", delta_color="normal" if edge<0 else "inverse")

# --- TOOLS EXTRA (CON CALCOLATORE MANUALE RIPRISTINATO) ---
st.markdown("---")
st.header("🛠️ Strumenti Extra")

with st.expander("🕵️ Reverse Engineering Quote"):
    q = st.number_input("Quota Book", 1.01, 100.0, 1.90)
    st.write(f"Prob Implicita: {1/q:.1%}")

with st.expander("🧮 Calcolatore Manuale (Value Bet)", expanded=False):
    k1, k2, k3 = st.columns(3)
    mp = k1.number_input("Probabilità Stimata (%)", 0.1, 100.0, 50.0) / 100
    mq = k2.number_input("Quota Bookmaker", 1.01, 100.0, 2.0)
    mb = k3.number_input("Bankroll", 0.0, 10000.0, 1000.0)
    
    mev = (mp * mq) - 1
    if mev > 0:
        mk = (((mq - 1) * mp) - (1 - mp)) / (mq - 1) * KELLY_FRACTION
        stake = mb * mk
        st.success(f"✅ VALUE BET! Edge: {mev*100:.1f}% | Stake (Kelly 1/4): € {stake:.2f}")
    else:
        st.error("❌ Nessun Valore")

# CSV DOWNLOAD
if st.session_state.analyzed:
    csv = pd.DataFrame([{
        "Home": h_name, "Away": a_name, 
        "xG_H": st.session_state.xg_h, "xG_A": st.session_state.xg_a,
        "Fair1": 1/st.session_state.prob_1, "Fair2": 1/st.session_state.prob_2
    }]).to_csv(index=False).encode('utf-8')
    st.download_button("💾 Scarica CSV", csv, "bet.csv", "text/csv")
