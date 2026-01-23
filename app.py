import streamlit as st
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Mathbet fc",
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
    st.session_state.score_matrix = None
    st.session_state.all_scores = []
    # Dati storici U/O salvati per calcoli successivi
    st.session_state.hist_uo_h = {}
    st.session_state.hist_uo_a = {}

# --- DATABASE CAMPIONATI (HOME ADVANTAGE RIDOTTO) ---
LEAGUES = {
    # Ho abbassato l'HA di circa 0.05 per tutti
    "🌐 Generico (Media)": { "avg": 1.35, "ha": 0.30, "w_elo_base": 0.40 }, 
    "🇮🇹 Serie A":          { "avg": 1.30, "ha": 0.20, "w_elo_base": 0.50 }, # Era 0.25
    "🇮🇹 Serie B":          { "avg": 1.15, "ha": 0.25, "w_elo_base": 0.30 }, # Era 0.30
    "🇬🇧 Premier League":   { "avg": 1.55, "ha": 0.30, "w_elo_base": 0.55 }, # Era 0.35
    "🇩🇪 Bundesliga":       { "avg": 1.65, "ha": 0.35, "w_elo_base": 0.45 }, # Era 0.40
    "🇪🇸 La Liga":          { "avg": 1.25, "ha": 0.25, "w_elo_base": 0.55 }, # Era 0.30
    "🇫🇷 Ligue 1":          { "avg": 1.30, "ha": 0.24, "w_elo_base": 0.45 }, # Era 0.28
}

# Parametri Globali
SCALING_FACTOR = 400.0  
KELLY_FRACTION = 0.25
RHO = -0.13 # Correlazione Dixon-Coles
WEIGHT_HIST_UO = 0.40 # Quanto pesano i dati storici inseriti dall'utente (40%)

# --- FUNZIONI MATEMATICHE ---

def calculate_dynamic_weights(matchday, base_w_elo):
    if matchday <= 8:
        w_elo = max(base_w_elo, 0.75)
    elif matchday <= 19:
        w_elo = base_w_elo + 0.10
    else:
        w_elo = base_w_elo
    w_stats = 1.0 - w_elo
    return w_elo, w_stats

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

def calculate_player_probability(metric_per90, expected_mins, team_match_xg, team_avg_xg):
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
    matchday = st.slider("Giornata Attuale", 1, 38, 10)
    W_ELO_DYN, W_STATS_DYN = calculate_dynamic_weights(matchday, L_DATA["w_elo_base"])

st.title("Mathbet fc ⚽")

# --- LINK UTILI ---
with st.expander("🔗 Link Utili (Clicca per aprire)", expanded=False):
    lc1, lc2, lc3 = st.columns(3)
    with lc1:
        st.caption("Rating")
        st.link_button("ClubElo", "http://clubelo.com")
    with lc2:
        st.caption("Stats Gol")
        st.link_button("FootyStats", "https://footystats.org/it/")
    with lc3:
        st.caption("Giocatori")
        st.link_button("FBref", "https://fbref.com")

st.markdown("---")

# --- INPUT DATI ---
col_h, col_a = st.columns(2)

# DIZIONARI PER DATI STORICI UTENTE
h_uo_input = {}
a_uo_input = {}

with col_h:
    st.subheader("🏠 Squadra Casa")
    h_name = st.text_input("Nome Casa", "Home")
    h_elo = st.number_input("ClubElo Rating", 1000, 2500, 1600, step=10, key="helo")
    h_str = st.slider("Disponibilità Titolari %", 50, 100, 100, key="hstr")
    
    with st.expander("📊 Stats & Gol (campionato)", expanded=True):
        h_gf_s = st.number_input("GF Media Stagione", 0.0, 5.0, 1.4, key="h1")
        h_gs_s = st.number_input("GS Media Stagione", 0.0, 5.0, 1.0, key="h2")
        h_gf_h = st.number_input("GF Media in Casa", 0.0, 5.0, 1.6, key="h3")
        h_gs_h = st.number_input("GS Media in Casa", 0.0, 5.0, 0.8, key="h4")
        st.markdown("---")
        h_gf_l5 = st.number_input("GOL FATTI (Ultime 5)", 0, 25, 7, key="h5")
        h_gs_l5 = st.number_input("GOL SUBITI (Ultime 5)", 0, 25, 5, key="h6")
    
    with st.expander("📈 Trend Under/Over (Storico %)"):
        st.caption("Inserisci la % di partite terminate Over nelle ultime 10/stagione")
        h_uo_input[0.5] = st.slider("% Over 0.5 Casa", 0, 100, 90, key="ho05")
        h_uo_input[1.5] = st.slider("% Over 1.5 Casa", 0, 100, 75, key="ho15")
        h_uo_input[2.5] = st.slider("% Over 2.5 Casa", 0, 100, 50, key="ho25")
        h_uo_input[3.5] = st.slider("% Over 3.5 Casa", 0, 100, 30, key="ho35")
        h_uo_input[4.5] = st.slider("% Over 4.5 Casa", 0, 100, 10, key="ho45")

with col_a:
    st.subheader("✈️ Squadra Ospite")
    a_name = st.text_input("Nome Ospite", "Away")
    a_elo = st.number_input("ClubElo Rating", 1000, 2500, 1550, step=10, key="aelo")
    a_str = st.slider("Disponibilità Titolari %", 50, 100, 100, key="astr")
    
    with st.expander("📊 Stats & Gol (campionato)", expanded=True):
        a_gf_s = st.number_input("GF Media Stagione", 0.0, 5.0, 1.2, key="a1")
        a_gs_s = st.number_input("GS Media Stagione", 0.0, 5.0, 1.3, key="a2")
        a_gf_a = st.number_input("GF Media Fuori", 0.0, 5.0, 1.0, key="a3")
        a_gs_a = st.number_input("GS Media Fuori", 0.0, 5.0, 1.5, key="a4")
        st.markdown("---")
        a_gf_l5 = st.number_input("GOL FATTI (Ultime 5)", 0, 25, 5, key="a5")
        a_gs_l5 = st.number_input("GOL SUBITI (Ultime 5)", 0, 25, 6, key="a6")
        
    with st.expander("📈 Trend Under/Over (Storico %)"):
        st.caption("Inserisci la % di partite terminate Over nelle ultime 10/stagione")
        a_uo_input[0.5] = st.slider("% Over 0.5 Ospite", 0, 100, 90, key="ao05")
        a_uo_input[1.5] = st.slider("% Over 1.5 Ospite", 0, 100, 70, key="ao15")
        a_uo_input[2.5] = st.slider("% Over 2.5 Ospite", 0, 100, 45, key="ao25")
        a_uo_input[3.5] = st.slider("% Over 3.5 Ospite", 0, 100, 25, key="ao35")
        a_uo_input[4.5] = st.slider("% Over 4.5 Ospite", 0, 100, 10, key="ao45")

st.markdown("---")
st.subheader("💰 Quote Bookmaker")
qc1, qc2, qc3 = st.columns(3)
b1 = qc1.number_input("Quota 1", 1.01, 100.0, 2.20)
bX = qc2.number_input("Quota X", 1.01, 100.0, 3.10)
b2 = qc3.number_input("Quota 2", 1.01, 100.0, 3.40)

# --- CALCOLO ---
if st.button("🚀 ANALIZZA PARTITA", type="primary", use_container_width=True):
    
    # 1. Stats Logic
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

    # 2. Elo Logic
    # Rimosso hardcode per Serie A, ora usa il valore del dizionario ridotto (0.20 * 400 = 80 punti)
    elo_ha_points = L_DATA["ha"] * 400.0
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

    # 5. Dixon-Coles Matrix Generation
    prob_1, prob_X, prob_2 = 0, 0, 0
    score_matrix = np.zeros((10, 10)) # Esteso a 10 gol per sicurezza calcoli
    score_list = []
    
    for h in range(10):
        for a in range(10):
            p = dixon_coles_probability(h, a, final_xg_h, final_xg_a, RHO)
            score_matrix[h, a] = p
            
            if h > a: prob_1 += p
            elif h == a: prob_X += p
            else: prob_2 += p
            
            if h < 7 and a < 7: # Limitiamo lista punteggi
                score_list.append({"Risultato": f"{h}-{a}", "Prob": p})

    # Normalization (Fix somma probabilità ~= 1)
    tot_prob = prob_1 + prob_X + prob_2
    prob_1 /= tot_prob
    prob_X /= tot_prob
    prob_2 /= tot_prob
    score_matrix /= tot_prob # Normalizzo matrice
    
    # Save State
    st.session_state.analyzed = True
    st.session_state.xg_h = final_xg_h
    st.session_state.xg_a = final_xg_a
    st.session_state.team_avg_h = h_gf_s
    st.session_state.team_avg_a = a_gf_s
    st.session_state.prob_1 = prob_1
    st.session_state.prob_X = prob_X
    st.session_state.prob_2 = prob_2
    st.session_state.score_matrix = score_matrix
    st.session_state.hist_uo_h = h_uo_input
    st.session_state.hist_uo_a = a_uo_input
    score_list.sort(key=lambda x: x["Prob"], reverse=True)
    st.session_state.all_scores = score_list

    # --- OUTPUT ---
    st.balloons()
    
    st.header(f"📊 Analisi: {h_name} vs {a_name}")
    c1, c2 = st.columns(2)
    c1.metric("xG Attesi (Stimati)", f"{final_xg_h:.2f} - {final_xg_a:.2f}")
    
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
    
    # --- SEZIONE UNDER / OVER (CON INPUT UTENTE) ---
    st.markdown("---")
    st.subheader("📉 Under / Over (Stats + Storia)")
    
    uo_data = []
    lines = [0.5, 1.5, 2.5, 3.5, 4.5]
    
    # Loop Calcolo Probabilità Matematica dalla Matrice
    for line in lines:
        prob_over_math = 0
        for h in range(10):
            for a in range(10):
                if (h + a) > line:
                    prob_over_math += score_matrix[h, a]
        
        # MIX con Dati Storici Utente
        hist_avg = (h_uo_input[line] + a_uo_input[line]) / 200.0 # /200 per avere decimale
        
        # Formula Ibrida: Math * (1-W) + Hist * W
        final_prob_over = (prob_over_math * (1 - WEIGHT_HIST_UO)) + (hist_avg * WEIGHT_HIST_UO)
        final_prob_under = 1.0 - final_prob_over
        
        # Fair Odds
        fair_o = 1/final_prob_over if final_prob_over > 0 else 0
        fair_u = 1/final_prob_under if final_prob_under > 0 else 0
        
        uo_data.append({
            "Linea": line,
            "Under %": f"{final_prob_under*100:.1f}%",
            "Fair U": f"{fair_u:.2f}",
            "Over %": f"{final_prob_over*100:.1f}%",
            "Fair O": f"{fair_o:.2f}"
        })
    
    st.table(pd.DataFrame(uo_data))

    # --- HANDICAP & MULTIGOL ---
    col_handicap, col_multi = st.columns(2)
    
    with col_handicap:
        st.subheader("🏁 Handicap")
        
        # 1. Handicap Europeo (3 esiti)
        eh_data = []
        
        # EH -1 (Casa parte da -1)
        eh_minus1_1 = 0
        eh_minus1_X = 0
        eh_minus1_2 = 0
        
        # EH +1 (Casa parte da +1)
        eh_plus1_1 = 0
        eh_plus1_X = 0
        eh_plus1_2 = 0

        # Asian Handicap (Principali)
        ah_home_minus_05 = prob_1 # Vittoria Casa
        ah_home_00 = prob_1 / (prob_1 + prob_2) if (prob_1+prob_2)>0 else 0 # Draw No Bet
        
        for h in range(10):
            for a in range(10):
                p = score_matrix[h, a]
                
                # EH -1 Calcoli
                if (h - 1) > a: eh_minus1_1 += p
                elif (h - 1) == a: eh_minus1_X += p
                else: eh_minus1_2 += p
                
                # EH +1 Calcoli
                if (h + 1) > a: eh_plus1_1 += p
                elif (h + 1) == a: eh_plus1_X += p
                else: eh_plus1_2 += p
        
        # Tabella Handicap Europeo
        st.write("**Handicap Europeo**")
        eh_df = pd.DataFrame([
            {"Tipo": f"{h_name} (-1)", "1 (Win >1)": f"{eh_minus1_1*100:.1f}%", "Fair 1": f"{1/eh_minus1_1:.2f}", "X (Win =1)": f"{eh_minus1_X*100:.1f}%", "Fair X": f"{1/eh_minus1_X:.2f}"},
            {"Tipo": f"{h_name} (+1)", "1 (1X)": f"{eh_plus1_1*100:.1f}%", "Fair 1": f"{1/eh_plus1_1:.2f}", "2 (Lose >1)": f"{eh_plus1_2*100:.1f}%", "Fair 2": f"{1/eh_plus1_2:.2f}"}
        ])
        st.dataframe(eh_df, hide_index=True)

        # Tabella Asian (Semplificata)
        st.write("**Asian Handicap (Casa)**")
        ah_df = pd.DataFrame([
            {"Linea": "AH 0.0 (DNB)", "Prob %": f"{ah_home_00*100:.1f}%", "Fair": f"{1/ah_home_00:.2f}"},
            {"Linea": "AH -0.5 (Win)", "Prob %": f"{ah_home_minus_05*100:.1f}%", "Fair": f"{1/ah_home_minus_05:.2f}"}
        ])
        st.dataframe(ah_df, hide_index=True)

    with col_multi:
        st.subheader("🔢 Multigol")
        mg_ranges = [(1,2), (1,3), (2,3), (2,4), (3,5)]
        mg_data = []
        
        for r_min, r_max in mg_ranges:
            prob_mg = 0
            for h in range(10):
                for a in range(10):
                    tot = h + a
                    if tot >= r_min and tot <= r_max:
                        prob_mg += score_matrix[h, a]
            
            mg_data.append({
                "Range": f"{r_min}-{r_max}",
                "Prob %": f"{prob_mg*100:.1f}%",
                "Fair": f"{1/prob_mg:.2f}"
            })
            
        st.dataframe(pd.DataFrame(mg_data), hide_index=True)

    # --- PLAYER PROP ---
    st.markdown("---")
    st.header("👤 Marcatore / Assist")

    with st.expander("Apri Calcolatore Player Prop", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            player_name = st.text_input("Giocatore", "Nome")
            team_sel = st.radio("Squadra", [h_name, a_name], horizontal=True)
            mins = st.slider("Minuti previsti", 15, 95, 80)
        with c2:
            type_s = st.radio("Tipo", ["GOL", "ASSIST"], horizontal=True)
            val_90 = st.number_input(f"x{'G' if 'GOL' in type_s else 'A'}/90 (da FBref)", 0.00, 2.00, 0.35)
            b_odd = st.number_input("Quota Bookmaker", 1.0, 50.0, 3.00)

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
