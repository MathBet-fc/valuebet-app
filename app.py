import streamlit as st
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Mathbet fc", page_icon="⚽", layout="wide")

# --- CLUBELO API (Leggera e Affidabile) ---
@st.cache_data(ttl=3600) 
def get_clubelo_database():
    try:
        date_str = date.today().strftime("%Y-%m-%d")
        url = f"http://api.clubelo.com/{date_str}"
        df = pd.read_csv(url)
        return dict(zip(df.Club, df.Elo))
    except: return {}

ELO_DB = get_clubelo_database()

# --- DATABASE PARAMETRI CAMPIONATI ---
LEAGUES = {
    "🌐 Generico (Media)": { "avg": 1.35, "ha": 0.30, "w_elo_base": 0.40, "rho": -0.13 }, 
    "🇮🇹 Serie A":          { "avg": 1.30, "ha": 0.20, "w_elo_base": 0.50, "rho": -0.14 },
    "🇮🇹 Serie B":          { "avg": 1.15, "ha": 0.25, "w_elo_base": 0.30, "rho": -0.18 },
    "🇬🇧 Premier League":   { "avg": 1.55, "ha": 0.30, "w_elo_base": 0.55, "rho": -0.12 },
    "🇩🇪 Bundesliga":       { "avg": 1.65, "ha": 0.35, "w_elo_base": 0.45, "rho": -0.10 },
    "🇪🇸 La Liga":          { "avg": 1.25, "ha": 0.25, "w_elo_base": 0.55, "rho": -0.14 },
    "🇫🇷 Ligue 1":          { "avg": 1.30, "ha": 0.24, "w_elo_base": 0.45, "rho": -0.15 },
}

# --- FUNZIONI MATEMATICHE CORE ---
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

# --- INIZIALIZZAZIONE ---
if 'history' not in st.session_state: st.session_state.history = []
if 'analyzed' not in st.session_state: st.session_state.analyzed = False

# --- SIDEBAR (CONFIGURAZIONE) ---
with st.sidebar:
    st.title("⚙️ Configurazione")
    league_name = st.selectbox("Campionato", list(LEAGUES.keys()))
    L_DATA = LEAGUES[league_name]
    
    matchday = st.slider("Giornata", 1, 38, 10)
    w_elo = (L_DATA["w_elo_base"] + 0.10) if 8 < matchday <= 19 else (max(L_DATA["w_elo_base"], 0.75) if matchday <= 8 else L_DATA["w_elo_base"])
    
    st.markdown("---")
    st.subheader("🏟️ Contesto Partita")
    m_type = st.radio("Tipo Incontro", ["Standard", "Derby (Stesso Stadio)", "Campo Neutro"])
    is_big_match = st.checkbox("🔥 Big Match (Scontro Diretto)")
    use_xg_mode = st.toggle("📊 Usa Modalità xG (Expected Goals)", value=True)
    
    CURRENT_RHO = L_DATA.get("rho", -0.13)

st.title("Mathbet fc - Ultimate Analysis ⚽")

# --- INPUT PRINCIPALE ---
col_h, col_a = st.columns(2)
h_uo_input, a_uo_input = {}, {}
sos_options = ["Media (Standard)", "Difficili (Top Team)", "Facili (Bassa Classifica)"]

# --- SQUADRA CASA ---
with col_h:
    st.subheader("🏠 Squadra Casa")
    h_name = st.text_input("Nome", "Inter", key="hn")
    auto_elo_h = float(ELO_DB.get(h_name, 1600.0))
    h_elo = st.number_input("Rating Elo", 1000.0, 2500.0, value=auto_elo_h, step=10.0, key="helo")
    
    with st.expander("📊 Statistiche (Stagione & Forma)", expanded=True):
        st.caption("Inserisci GF/GS o xG/xGA medi")
        c1, c2 = st.columns(2)
        h_gf_s = c1.number_input("GF Totali (Media)", 0.0, 5.0, 1.85, 0.05)
        h_gs_s = c2.number_input("GS Totali (Media)", 0.0, 5.0, 0.95, 0.05)
        h_gf_h = c1.number_input("GF Casa (Media)", 0.0, 5.0, 2.10, 0.05)
        h_gs_h = c2.number_input("GS Casa (Media)", 0.0, 5.0, 0.70, 0.05)
        
        st.markdown("---")
        h_sos = st.selectbox("Avversari recenti (SoS)", sos_options, key="hsos")
        h_gf_l5 = st.number_input("Gol Fatti (Ultime 5)", 0.0, 25.0, 8.0, 1.0)
        h_gs_l5 = st.number_input("Gol Subiti (Ultime 5)", 0.0, 25.0, 4.0, 1.0)

    with st.expander("🩺 Condizione & Tattica"):
        h_str = st.slider("Titolari %", 50, 100, 100, key="hs")
        h_rest = st.slider("Giorni di Riposo", 2, 10, 7, key="hr")
        h_m_a = st.checkbox("Manca Bomber", key="hma")
        h_m_d = st.checkbox("Manca Difensore Chiave", key="hmd")

    with st.expander("📈 Trend Over/Under"):
        for l in [0.5, 1.5, 2.5, 3.5, 4.5]: h_uo_input[l] = st.slider(f"Over {l} % (Casa)", 0, 100, 50, key=f"ho{l}")

# --- SQUADRA OSPITE ---
with col_a:
    st.subheader("✈️ Squadra Ospite")
    a_name = st.text_input("Nome", "Juventus", key="an")
    auto_elo_a = float(ELO_DB.get(a_name, 1550.0))
    a_elo = st.number_input("Rating Elo", 1000.0, 2500.0, value=auto_elo_a, step=10.0, key="aelo")

    with st.expander("📊 Statistiche (Stagione & Forma)", expanded=True):
        st.caption("Inserisci GF/GS o xG/xGA medi")
        c3, c4 = st.columns(2)
        a_gf_s = c3.number_input("GF Totali (Media)", 0.0, 5.0, 1.45, 0.05, key="ags")
        a_gs_s = c4.number_input("GS Totali (Media)", 0.0, 5.0, 0.85, 0.05, key="agss")
        a_gf_a = c3.number_input("GF Fuori (Media)", 0.0, 5.0, 1.20, 0.05, key="agfa")
        a_gs_a = c4.number_input("GS Fuori (Media)", 0.0, 5.0, 0.90, 0.05, key="agsa")

        st.markdown("---")
        a_sos = st.selectbox("Avversari recenti (SoS)", sos_options, key="asos")
        a_gf_l5 = st.number_input("Gol Fatti (Ultime 5)", 0.0, 25.0, 6.0, 1.0, key="agl5")
        a_gs_l5 = st.number_input("Gol Subiti (Ultime 5)", 0.0, 25.0, 3.0, 1.0, key="agsl5")

    with st.expander("🩺 Condizione & Tattica"):
        a_str = st.slider("Titolari %", 50, 100, 100, key="as")
        a_rest = st.slider("Giorni di Riposo", 2, 10, 7, key="ar")
        a_m_a = st.checkbox("Manca Bomber", key="ama")
        a_m_d = st.checkbox("Manca Difensore Chiave", key="amd")

    with st.expander("📈 Trend Over/Under"):
        for l in [0.5, 1.5, 2.5, 3.5, 4.5]: a_uo_input[l] = st.slider(f"Over {l} % (Ospite)", 0, 100, 50, key=f"ao{l}")

st.subheader("💰 Quote Bookmaker")
qc1, qc2, qc3 = st.columns(3)
b1 = qc1.number_input("Quota 1", 1.01, 100.0, 2.10)
bX = qc2.number_input("Quota X", 1.01, 100.0, 3.20)
b2 = qc3.number_input("Quota 2", 1.01, 100.0, 3.60)

# --- MOTORE DI CALCOLO ---
if st.button("🚀 ANALIZZA PARTITA", type="primary", use_container_width=True):
    
    # 1. Ajustment Home Advantage
    ha_val = L_DATA["ha"]
    if m_type == "Campo Neutro": ha_val = 0.0
    elif m_type == "Derby (Stesso Stadio)": ha_val *= 0.5
    
    # 2. SoS Adjustment (Ultime 5 partite)
    h_gf_l5_c, h_gs_l5_c = h_gf_l5, h_gs_l5
    a_gf_l5_c, a_gs_l5_c = a_gf_l5, a_gs_l5

    if h_sos == "Difficili (Top Team)": h_gf_l5_c *= 1.25; h_gs_l5_c *= 0.85
    elif h_sos == "Facili (Bassa Classifica)": h_gf_l5_c *= 0.85; h_gs_l5_c *= 1.20 
    
    if a_sos == "Difficili (Top Team)": a_gf_l5_c *= 1.25; a_gs_l5_c *= 0.85
    elif a_sos == "Facili (Bassa Classifica)": a_gf_l5_c *= 0.85; a_gs_l5_c *= 1.20

    # 3. Pesi Statistici
    if use_xg_mode:
        w_seas, w_ha, w_l5 = 0.50, 0.35, 0.15 
    else:
        w_seas, w_ha, w_l5 = 0.40, 0.35, 0.25 

    # 4. Calcolo Forza Attacco/Difesa
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
    
    # Expected Goals Statistici
    xg_s_h = (h_att_val * a_def_val) / L_DATA["avg"]
    xg_s_a = (a_att_val * h_def_val) / L_DATA["avg"]
    
    # 5. Expected Goals da Elo
    exp_h = 1 / (1 + 10 ** (-((h_elo + ha_val*400) - a_elo)/400.0))
    xg_e_h = L_DATA["avg"] * (exp_h / 0.5) ** 0.85
    xg_e_a = L_DATA["avg"] * ((1 - exp_h) / 0.5) ** 0.85
    
    # 6. Fusione & Malus
    f_xh = ((xg_e_h * w_elo) + (xg_s_h * (1-w_elo))) * (h_str/100.0)
    f_xa = ((xg_e_a * w_elo) + (xg_s_a * (1-w_elo))) * (a_str/100.0)
    
    # Fatica
    fatigue_malus = 0.05 
    if h_rest <= 3: f_xh *= (1 - fatigue_malus); f_xa *= (1 + fatigue_malus) 
    if a_rest <= 3: f_xa *= (1 - fatigue_malus); f_xh *= (1 + fatigue_malus)
    
    # Big Match (Tattica conservativa)
    if is_big_match: f_xh *= 0.90; f_xa *= 0.90
    
    # Assenze chiave
    if h_m_a: f_xh *= 0.85
    if h_m_d: f_xa *= 1.20
    if a_m_a: f_xa *= 0.85
    if a_m_d: f_xh *= 1.20

    # 7. Generazione Matrice Poisson (Dixon-Coles)
    p1, pX, p2, pGG = 0, 0, 0, 0
    matrix = np.zeros((10,10))
    scores = []
    
    for h_g in range(10):
        for a_g in range(10):
            p = dixon_coles_probability(h_g, a_g, f_xh, f_xa, CURRENT_RHO)
            matrix[h_g,a_g] = p
            if h_g > a_g: p1 += p
            elif h_g == a_g: pX += p
            else: p2 += p
            if h_g>0 and a_g>0: pGG += p
            if h_g<6 and a_g<6: scores.append({"Risultato": f"{h_g}-{a_g}", "Prob": p})
    
    # Normalizzazione
    total_prob = np.sum(matrix)
    if total_prob > 0:
        matrix /= total_prob
        p1 /= total_prob; pX /= total_prob; p2 /= total_prob; pGG /= total_prob
    
    # Stabilità (Simulazione Monte Carlo)
    sim = []
    for _ in range(5000):
        gh = np.random.poisson(max(0.1, np.random.normal(f_xh, 0.15*f_xh)))
        ga = np.random.poisson(max(0.1, np.random.normal(f_xa, 0.15*f_xa)))
        sim.append(1 if gh>ga else (0 if gh==ga else 2))
    s1, sX, s2 = sim.count(1)/5000, sim.count(0)/5000, sim.count(2)/5000
    stability = max(0, 100 - ((abs(p1-s1)+abs(pX-sX)+abs(p2-s2))/3*400))

    # Salvataggio Stato
    st.session_state.analyzed = True
    st.session_state.f_xh = f_xh; st.session_state.f_xa = f_xa
    st.session_state.h_name = h_name; st.session_state.a_name = a_name
    st.session_state.p1 = p1; st.session_state.pX = pX; st.session_state.p2 = p2
    st.session_state.pGG = pGG; st.session_state.stability = stability
    st.session_state.matrix = matrix; st.session_state.scores = scores
    st.session_state.b1 = b1; st.session_state.bX = bX; st.session_state.b2 = b2

# --- OUTPUT (TABELLE E GRAFICI) ---
if st.session_state.analyzed:
    st.header(f"📊 {st.session_state.h_name} vs {st.session_state.a_name}")
    st.metric("Expected Goals (xG) Previsti", f"{st.session_state.f_xh:.2f} - {st.session_state.f_xa:.2f}", delta=f"Stabilità: {st.session_state.stability:.1f}%")

    tab1, tab2, tab3, tab4 = st.tabs(["🏆 Esito 1X2", "⚽ Gol & Handicap", "👤 Marcatori", "📝 Storico & Tools"])

    # TAB 1: ESITO
    with tab1:
        c_prob, c_chart = st.columns([1, 1])
        with c_prob:
            st.subheader("Probabilità & Valore")
            p1, pX, p2 = st.session_state.p1, st.session_state.pX, st.session_state.p2
            b1, bX, b2 = st.session_state.b1, st.session_state.bX, st.session_state.b2
            
            res_data = {
                "Esito": ["1 (Casa)", "X (Pareggio)", "2 (Ospite)"],
                "Prob %": [f"{p1:.1%}", f"{pX:.1%}", f"{p2:.1%}"],
                "Fair Odd": [f"{1/p1:.2f}", f"{1/pX:.2f}", f"{1/p2:.2f}"],
                "Bookmaker": [b1, bX, b2],
                "Valore %": [f"{(b1*p1-1):.1%}", f"{(bX*pX-1):.1%}", f"{(b2*p2-1):.1%}"],
                "Kelly Stake": [f"{calculate_kelly(p1,b1):.1f}%", f"{calculate_kelly(pX,bX):.1f}%", f"{calculate_kelly(p2,b2):.1f}%"]
            }
            st.table(pd.DataFrame(res_data))

            scores = st.session_state.scores
            scores.sort(key=lambda x: x["Prob"], reverse=True)
            st.subheader("Risultati Esatti Top 5")
            st.table(pd.DataFrame([{"Risultato": s["Risultato"], "Probabilità": f"{s['Prob']:.1%}", "Quota Reale": f"{1/s['Prob']:.2f}"} for s in scores[:5]]))

        with c_chart:
            st.subheader("Heatmap Probabilità")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(st.session_state.matrix[:6,:6], annot=True, fmt=".0%", cmap="Greens", cbar=False)
            plt.xlabel("Gol Ospite"); plt.ylabel("Gol Casa")
            st.pyplot(fig)

    # TAB 2: GOL & HANDICAP
    with tab2:
        col_g1, col_g2 = st.columns(2)
        matrix = st.session_state.matrix
        
        with col_g1:
            st.subheader("📉 Under / Over")
            uo_res = []
            for l in [0.5, 1.5, 2.5, 3.5, 4.5]:
                # Calcolo probabilità pura dalla matrice
                p_over_pure = np.sum(matrix[np.indices((10,10))[0] + np.indices((10,10))[1] > l])
                # Fusione con il trend manuale inserito dall'utente (65% math, 35% trend)
                trend_avg = (h_uo_input[l] + a_uo_input[l]) / 200.0
                p_over = (p_over_pure * 0.65) + (trend_avg * 0.35)
                
                uo_res.append({
                    "Linea": l,
                    "Under %": f"{(1-p_over):.1%}", "Quota U": f"{1/(1-p_over):.2f}",
                    "Over %": f"{p_over:.1%}", "Quota O": f"{1/p_over:.2f}"
                })
            st.table(pd.DataFrame(uo_res))

            st.subheader("Gol / NoGol")
            pGG = st.session_state.pGG
            st.table(pd.DataFrame([
                {"Esito": "Gol (GG)", "Prob": f"{pGG:.1%}", "Quota": f"{1/pGG:.2f}"},
                {"Esito": "NoGol (NG)", "Prob": f"{(1-pGG):.1%}", "Quota": f"{1/(1-pGG):.2f}"}
            ]))

        with col_g2:
            st.subheader("🔢 Multigol")
            mg_res = []
            for r in [(1,2), (1,3), (2,3), (2,4), (3,5)]:
                mask = (np.indices((10,10))[0] + np.indices((10,10))[1] >= r[0]) & \
                       (np.indices((10,10))[0] + np.indices((10,10))[1] <= r[1])
                pm = np.sum(matrix[mask])
                mg_res.append({"Range": f"{r[0]}-{r[1]}", "Prob": f"{pm:.1%}", "Quota": f"{1/pm:.2f}"})
            st.table(pd.DataFrame(mg_res))
            
            st.subheader("🏁 Handicap & Asian")
            # Handicap Europeo -1 (Casa vince con 2+ gol di scarto)
            h1_minus1 = np.sum(matrix[np.indices((10,10))[0] - 1 > np.indices((10,10))[1]])
            # Draw No Bet (Rimborsato in caso di pareggio)
            dnb_1 = st.session_state.p1 / (st.session_state.p1 + st.session_state.p2) if (st.session_state.p1 + st.session_state.p2) > 0 else 0
            
            st.write(f"**Handicap Europeo (-1) Casa:** {h1_minus1:.1%} (@{1/h1_minus1:.2f})")
            st.write(f"**Draw No Bet (1 DNB):** {dnb_1:.1%} (@{1/dnb_1:.2f})")

    # TAB 3: MARCATORI (CALCOLATORE)
    with tab3:
        st.subheader("Calcolatore Probabilità Marcatore")
        st.info("Inserisci l'xG/90 del giocatore (trovabile su FBref o Sofascore) per calcolare la sua probabilità.")
        
        c_p1, c_p2 = st.columns(2)
        pl_name = c_p1.text_input("Nome Giocatore", "Vlahovic")
        pl_team = c_p2.radio("Squadra", [st.session_state.h_name, st.session_state.a_name])
        
        pl_xg90 = c_p1.number_input("xG per 90 min", 0.01, 2.00, 0.45)
        pl_min = c_p2.number_input("Minuti previsti", 10, 100, 85)
        pl_odd = st.number_input("Quota Goal Bookmaker", 1.01, 100.0, 2.50)
        
        # Selezione xG squadra corretti
        if pl_team == st.session_state.h_name:
            team_xg_match = st.session_state.f_xh
            team_avg = 1.50 # Valore medio placeholder
        else:
            team_xg_match = st.session_state.f_xa
            team_avg = 1.20 # Valore medio placeholder
            
        prob_score, _ = calculate_player_probability(pl_xg90, pl_min, team_xg_match, team_avg)
        
        st.metric(f"Probabilità Goal {pl_name}", f"{prob_score:.1%}", delta=f"Valore: {((pl_odd*prob_score)-1)*100:.1f}%")
        st.write(f"Quota Reale (Fair Odd): **{1/prob_score:.2f}**")

    # TAB 4: STORICO E TOOLS
    with tab4:
        st.subheader("Storico Analisi (Backtesting)")
        if st.button("💾 Salva Partita nello Storico"):
            st.session_state.history.append({
                "Data": date.today().strftime("%d/%m"),
                "Match": f"{st.session_state.h_name} - {st.session_state.a_name}",
                "P1": st.session_state.p1, "PX": st.session_state.pX, "P2": st.session_state.p2,
                "Esito Reale": "In attesa"
            })
            st.success("Salvato!")
        
        if st.session_state.history:
            df_hist = pd.DataFrame(st.session_state.history)
            edited_df = st.data_editor(df_hist, num_rows="dynamic")
            
            # Calcolo Brier Score (Precisione)
            valid_rows = edited_df[edited_df["Esito Reale"] != "In attesa"]
            if not valid_rows.empty:
                brier_scores = []
                for _, r in valid_rows.iterrows():
                    outcome = [1 if r["Esito Reale"]=="1" else 0, 1 if r["Esito Reale"]=="X" else 0, 1 if r["Esito Reale"]=="2" else 0]
                    bs = (r["P1"]-outcome[0])**2 + (r["PX"]-outcome[1])**2 + (r["P2"]-outcome[2])**2
                    brier_scores.append(bs)
                st.metric("Brier Score Medio (0=Perfetto, 2=Pessimo)", f"{np.mean(brier_scores):.3f}")

        st.markdown("---")
        st.subheader("🧮 Calcolatrici Rapide")
        mc1, mc2 = st.columns(2)
        with mc1:
            st.caption("Convertitore Quota -> Probabilità")
            q_in = st.number_input("Inserisci Quota", 1.01, 100.0, 2.00)
            st.write(f"Probabilità Implicita: **{1/q_in:.1%}**")
        with mc2:
            st.caption("Calcolatore Valore Manuale")
            m_prob = st.number_input("Tua Probabilità %", 0.1, 100.0, 50.0) / 100
            m_quota = st.number_input("Quota Book", 1.01, 100.0, 2.10)
            valore = (m_quota * m_prob) - 1
            st.write(f"Valore: **{valore:.1%}**")
            if valore > 0: st.success("✅ VALUE BET")
            else: st.error("❌ NO VALUE")
