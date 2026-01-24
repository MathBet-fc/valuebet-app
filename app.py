import streamlit as st
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date
import cloudscraper # <--- QUESTA è la chiave per sbloccare il 403
from io import StringIO

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Mathbet fc - Ultimate Full", page_icon="⚽", layout="wide")

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

# --- FUNZIONI MATEMATICHE ---
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

# --- AUTOMAZIONE FBREF (VERSIONE CLOUDSCRAPER ANTI-BLOCCO) ---
@st.cache_data(ttl=3600)
def load_fbref_data(url):
    try:
        # Crea uno scraper che si comporta come un utente reale
        scraper = cloudscraper.create_scraper() 
        response = scraper.get(url)
        
        if response.status_code != 200:
            st.error(f"Errore connessione FBref (Codice {response.status_code}). Riprova o controlla il link.")
            return None

        # Parsing HTML
        dfs = pd.read_html(StringIO(response.text), header=[0, 1])
        
        df = None
        for table in dfs:
            cols_flat = [c[1].lower() if isinstance(c, tuple) else str(c).lower() for c in table.columns]
            top_level = [c[0].lower() if isinstance(c, tuple) else "" for c in table.columns]
            if ('squadra' in cols_flat or 'squad' in cols_flat) and ('home' in top_level or 'casa' in top_level):
                df = table
                break
        
        if df is None: return None

        new_data = []
        for _, row in df.iterrows():
            team_name = ""
            for col in df.columns:
                if col[1].lower() in ['squad', 'squadra']:
                    team_name = str(row[col])
                    break
            if not team_name or team_name == "nan": continue

            def get_val(lbl_top_list, lbl_sub):
                for top in lbl_top_list:
                    if (top, lbl_sub) in df.columns: return float(row[(top, lbl_sub)])
                return 0.0

            k_h, k_a = ['Home', 'Casa'], ['Away', 'Fuori']
            mp_h = max(1, get_val(k_h, 'MP') or get_val(k_h, 'PG'))
            mp_a = max(1, get_val(k_a, 'MP') or get_val(k_a, 'PG'))

            stats = {
                'common_name': team_name,
                'gf_home': get_val(k_h, 'GF') / mp_h,
                'gs_home': (get_val(k_h, 'GA') or get_val(k_h, 'GS')) / mp_h,
                'xg_home': get_val(k_h, 'xG') / mp_h,
                'xga_home': (get_val(k_h, 'xGA') or get_val(k_h, 'xG.1')) / mp_h,
                'gf_away': get_val(k_a, 'GF') / mp_a,
                'gs_away': (get_val(k_a, 'GA') or get_val(k_a, 'GS')) / mp_a,
                'xg_away': get_val(k_a, 'xG') / mp_a,
                'xga_away': (get_val(k_a, 'xGA') or get_val(k_a, 'xG.1')) / mp_a,
            }
            # Fallback xG totali
            stats['xg_tot'] = (stats['xg_home']*mp_h + stats['xg_away']*mp_a) / (mp_h+mp_a)
            stats['xga_tot'] = (stats['xga_home']*mp_h + stats['xga_away']*mp_a) / (mp_h+mp_a)
            stats['gf_tot'] = (stats['gf_home']*mp_h + stats['gf_away']*mp_a) / (mp_h+mp_a)
            stats['gs_tot'] = (stats['gs_home']*mp_h + stats['gs_away']*mp_a) / (mp_h+mp_a)
            
            new_data.append(stats)
        return pd.DataFrame(new_data)
    except Exception as e:
        st.error(f"Errore tecnico: {e}") 
        return None

# --- INIZIALIZZAZIONE SESSION STATE ---
if 'history' not in st.session_state: st.session_state.history = []
if 'analyzed' not in st.session_state: st.session_state.analyzed = False

# --- SIDEBAR ---
with st.sidebar:
    st.title("Mathbet fc Ultimate 🚀")
    fbref_url = st.text_input("🔗 Link FBref (Stats)", placeholder="https://fbref.com/it/comp/11/Stats-Serie-A")
    fs_df = None
    if fbref_url:
        fs_df = load_fbref_data(fbref_url)
        if fs_df is not None: st.success(f"✅ Dati ok: {len(fs_df)} squadre")
        else: st.warning("Tabella non trovata o connessione bloccata.")

    st.markdown("---")
    league_name = st.selectbox("Campionato", list(LEAGUES.keys()))
    L_DATA = LEAGUES[league_name]
    matchday = st.slider("Giornata", 1, 38, 10)
    w_elo = (L_DATA["w_elo_base"] + 0.10) if 8 < matchday <= 19 else (max(L_DATA["w_elo_base"], 0.75) if matchday <= 8 else L_DATA["w_elo_base"])
    
    st.markdown("---")
    m_type = st.radio("Contesto", ["Standard", "Derby", "Campo Neutro"])
    is_big_match = st.checkbox("🔥 Big Match")
    use_xg_mode = st.toggle("📊 Usa xG", value=True)
    CURRENT_RHO = L_DATA.get("rho", -0.13)

st.title("Mathbet fc - Ultimate Analysis ⚽")

# --- INPUT SQUADRE ---
col_h, col_a = st.columns(2)
h_uo_input, a_uo_input = {}, {}
sos_options = ["Media (Standard)", "Difficili (Top Team)", "Facili (Bassa Classifica)"]

with col_h:
    st.subheader("🏠 Squadra Casa")
    d_h_gf_s, d_h_gs_s, d_h_gf_h, d_h_gs_h = 1.45, 1.05, 1.65, 0.85
    h_name_in = "Inter"

    if fs_df is not None:
        h_name = st.selectbox("Seleziona Casa", fs_df['common_name'].unique(), key="sel_h")
        row = fs_df[fs_df['common_name'] == h_name].iloc[0]
        if use_xg_mode:
            d_h_gf_s, d_h_gs_s = row['xg_tot'], row['xga_tot']
            d_h_gf_h, d_h_gs_h = row['xg_home'], row['xga_home']
        else:
            d_h_gf_s, d_h_gs_s = row['gf_tot'], row['gs_tot']
            d_h_gf_h, d_h_gs_h = row['gf_home'], row['gs_home']
    else:
        h_name = st.text_input("Nome Casa", h_name_in, key="h_n")
    
    h_elo = st.number_input("Rating Casa", 1000.0, 2500.0, value=float(ELO_DB.get(h_name, 1600.0)), key="helo")
    h_str = st.slider("Titolari Casa %", 50, 100, 100, key="hs")
    h_rest = st.slider("Giorni Riposo (C)", 2, 10, 7, key="h_rest")
    c_h1, c_h2 = st.columns(2)
    h_m_a, h_m_d = c_h1.checkbox("No Bomber (C)", key="h_ma"), c_h2.checkbox("No Difesa (C)", key="h_md")
    
    with st.expander("📊 Dati", expanded=True):
        h_gf_s = st.number_input("GF/xG Tot", 0.0, 5.0, value=float(d_h_gf_s), step=0.01, key="h1")
        h_gs_s = st.number_input("GS/xG Tot", 0.0, 5.0, value=float(d_h_gs_s), step=0.01, key="h2")
        h_gf_h = st.number_input("GF/xG Casa", 0.0, 5.0, value=float(d_h_gf_h), step=0.01, key="h3")
        h_gs_h = st.number_input("GS/xG Casa", 0.0, 5.0, value=float(d_h_gs_h), step=0.01, key="h4")
        h_sos = st.selectbox("SoS L5 Casa", sos_options, key="h_sos")
        h_gf_l5 = st.number_input("GF Reali L5", 0.0, 25.0, 7.0, step=0.5, key="h5")
        h_gs_l5 = st.number_input("GS Reali L5", 0.0, 25.0, 5.0, step=0.5, key="h6")
    
    with st.expander("📈 Over % Casa"):
        for l in [0.5, 1.5, 2.5, 3.5, 4.5]: h_uo_input[l] = st.slider(f"O{l} H", 0, 100, 50, key=f"ho{l}")

with col_a:
    st.subheader("✈️ Squadra Ospite")
    d_a_gf_s, d_a_gs_s, d_a_gf_a, d_a_gs_a = 1.25, 1.35, 1.10, 1.55
    a_name_in = "Milan"

    if fs_df is not None:
        a_name = st.selectbox("Seleziona Ospite", fs_df['common_name'].unique(), key="sel_a")
        row = fs_df[fs_df['common_name'] == a_name].iloc[0]
        if use_xg_mode:
            d_a_gf_s, d_a_gs_s = row['xg_tot'], row['xga_tot']
            d_a_gf_a, d_a_gs_a = row['xg_away'], row['xga_away']
        else:
            d_a_gf_s, d_a_gs_s = row['gf_tot'], row['gs_tot']
            d_a_gf_a, d_a_gs_a = row['gf_away'], row['gs_away']
    else:
        a_name = st.text_input("Nome Ospite", a_name_in, key="a_n")

    a_elo = st.number_input("Rating Ospite", 1000.0, 2500.0, value=float(ELO_DB.get(a_name, 1550.0)), key="aelo")
    a_str = st.slider("Titolari Ospite %", 50, 100, 100, key="as")
    a_rest = st.slider("Giorni Riposo (O)", 2, 10, 7, key="a_rest")
    c_a1, c_a2 = st.columns(2)
    a_m_a, a_m_d = c_a1.checkbox("No Bomber (O)", key="a_ma"), c_a2.checkbox("No Difesa (O)", key="a_md")
    
    with st.expander("📊 Dati", expanded=True):
        a_gf_s = st.number_input("GF/xG Tot ", 0.0, 5.0, value=float(d_a_gf_s), step=0.01, key="a1")
        a_gs_s = st.number_input("GS/xG Tot ", 0.0, 5.0, value=float(d_a_gs_s), step=0.01, key="a2")
        a_gf_a = st.number_input("GF/xG Fuori", 0.0, 5.0, value=float(d_a_gf_a), step=0.01, key="a3")
        a_gs_a = st.number_input("GS/xG Fuori", 0.0, 5.0, value=float(d_a_gs_a), step=0.01, key="a4")
        a_sos = st.selectbox("SoS L5 Ospite", sos_options, key="a_sos")
        a_gf_l5 = st.number_input("GF Reali L5 ", 0.0, 25.0, 5.0, step=0.5, key="a5")
        a_gs_l5 = st.number_input("GS Reali L5 ", 0.0, 25.0, 6.0, step=0.5, key="a6")

    with st.expander("📈 Over % Ospite"):
        for l in [0.5, 1.5, 2.5, 3.5, 4.5]: a_uo_input[l] = st.slider(f"O{l} A", 0, 100, 50, key=f"ao{l}")

st.subheader("💰 Quote")
qc1, qc2, qc3 = st.columns(3)
b1 = qc1.number_input("Q1", 1.01, 100.0, 2.20, key="b1")
bX = qc2.number_input("QX", 1.01, 100.0, 3.10, key="bX")
b2 = qc3.number_input("Q2", 1.01, 100.0, 3.40, key="b2")

# --- CALCOLO ---
if st.button("🚀 ANALIZZA PARTITA", type="primary", use_container_width=True):
    ha_val = L_DATA["ha"]
    if m_type == "Campo Neutro": ha_val = 0.0
    elif m_type == "Derby": ha_val *= 0.5
    
    # SoS Adjust
    h_gf_l5_c, h_gs_l5_c = h_gf_l5, h_gs_l5
    a_gf_l5_c, a_gs_l5_c = a_gf_l5, a_gs_l5
    if h_sos == "Difficili (Top Team)": h_gf_l5_c *= 1.25; h_gs_l5_c *= 0.85
    elif h_sos == "Facili (Bassa Classifica)": h_gf_l5_c *= 0.85; h_gs_l5_c *= 1.20 
    if a_sos == "Difficili (Top Team)": a_gf_l5_c *= 1.25; a_gs_l5_c *= 0.85
    elif a_sos == "Facili (Bassa Classifica)": a_gf_l5_c *= 0.85; a_gs_l5_c *= 1.20

    # Att/Def Weights
    if use_xg_mode: w_seas, w_ha, w_l5 = 0.50, 0.35, 0.15
    else: w_seas, w_ha, w_l5 = 0.40, 0.35, 0.25

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
    
    xg_s_h = (h_att_val * a_def_val) / L_DATA["avg"]
    xg_s_a = (a_att_val * h_def_val) / L_DATA["avg"]
    
    # Elo
    exp_h = 1 / (1 + 10 ** (-((h_elo + ha_val*400) - a_elo)/400.0))
    xg_e_h, xg_e_a = L_DATA["avg"]*(exp_h/0.5)**0.85, L_DATA["avg"]*((1-exp_h)/0.5)**0.85
    
    # Final Lambda
    f_xh = ((xg_e_h * w_elo) + (xg_s_h * (1-w_elo))) * (h_str/100.0)
    f_xa = ((xg_e_a * w_elo) + (xg_s_a * (1-w_elo))) * (a_str/100.0)
    
    # Fatigue/Tactics
    fatigue_malus = 0.05 
    if h_rest <= 3: f_xh *= (1 - fatigue_malus); f_xa *= (1 + fatigue_malus) 
    if a_rest <= 3: f_xa *= (1 - fatigue_malus); f_xh *= (1 + fatigue_malus)
    if is_big_match: f_xh *= 0.90; f_xa *= 0.90
    if h_m_a: f_xh *= 0.85
    if h_m_d: f_xa *= 1.20
    if a_m_a: f_xa *= 0.85
    if a_m_d: f_xh *= 1.20

    # Poisson
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
        p1 /= total_prob; pX /= total_prob; p2 /= total_prob; pGG /= total_prob
    
    # Stability
    sim = []
    for _ in range(5000):
        gh = np.random.poisson(max(0.1, np.random.normal(f_xh, 0.15*f_xh)))
        ga = np.random.poisson(max(0.1, np.random.normal(f_xa, 0.15*f_xa)))
        sim.append(1 if gh>ga else (0 if gh==ga else 2))
    s1, sX, s2 = sim.count(1)/5000, sim.count(0)/5000, sim.count(2)/5000
    stability = max(0, 100 - ((abs(p1-s1)+abs(pX-sX)+abs(p2-s2))/3*400))

    # Save State
    st.session_state.analyzed = True
    st.session_state.f_xh = f_xh
    st.session_state.f_xa = f_xa
    st.session_state.home_name_display = h_name
    st.session_state.away_name_display = a_name
    st.session_state.p1, st.session_state.pX, st.session_state.p2 = p1, pX, p2
    st.session_state.stability = stability
    st.session_state.pGG = pGG
    st.session_state.matrix = matrix
    st.session_state.scores = scores
    st.session_state.b1, st.session_state.bX, st.session_state.b2 = b1, bX, b2

# --- OUTPUT TABS (ORGANIZZAZIONE) ---
if st.session_state.analyzed:
    st.header(f"📊 {st.session_state.home_name_display} vs {st.session_state.away_name_display}")
    st.metric("Expected Goals Match", f"{st.session_state.f_xh:.2f} - {st.session_state.f_xa:.2f}", delta=f"Stabilità: {st.session_state.stability:.1f}%")

    tab1, tab2, tab3, tab4 = st.tabs(["🏆 Esito & Risultati", "⚽ Gol & Under/Over", "👤 Marcatori", "📝 Storico & Tools"])

    with tab1:
        st.subheader("Probabilità 1X2")
        p1, pX, p2 = st.session_state.p1, st.session_state.pX, st.session_state.p2
        b1, bX, b2 = st.session_state.b1, st.session_state.bX, st.session_state.b2
        
        st.table(pd.DataFrame({
            "Esito":["1","X","2"], 
            "Prob %":[f"{p1:.1%}",f"{pX:.1%}",f"{p2:.1%}"], 
            "Fair Odd":[f"{1/p1:.2f}",f"{1/pX:.2f}",f"{1/p2:.2f}"], 
            "Bookie":[b1, bX, b2],
            "Value %":[f"{(b1*p1-1):.1%}",f"{(bX*pX-1):.1%}",f"{(b2*p2-1):.1%}"], 
            "Stake (Kelly)":[f"{calculate_kelly(p1,b1):.1f}%",f"{calculate_kelly(pX,bX):.1f}%",f"{calculate_kelly(p2,b2):.1f}%"]
        }))

        c1, c2 = st.columns([1,2])
        scores = st.session_state.scores
        scores.sort(key=lambda x: x["Prob"], reverse=True)
        with c1:
            st.subheader("Risultati Esatti")
            st.table(pd.DataFrame([{"Score": s["Risultato"], "%": f"{s['Prob']:.1%}", "Fair": f"{1/s['Prob']:.2f}"} for s in scores[:5]]))
        with c2: 
            fig, ax = plt.subplots(figsize=(5,3))
            sns.heatmap(st.session_state.matrix[:5,:5], annot=True, fmt=".0%", cmap="Greens", cbar=False)
            plt.xlabel("Ospite"); plt.ylabel("Casa")
            st.pyplot(fig)

    with tab2:
        st.subheader("📉 Under / Over")
        matrix = st.session_state.matrix
        uo_data = []
        for l in [0.5, 1.5, 2.5, 3.5, 4.5]:
            p_over = (np.sum(matrix[np.indices((10,10))[0] + np.indices((10,10))[1] > l]) * 0.65) + ((h_uo_input[l] + a_uo_input[l])/200.0 * 0.35)
            uo_data.append({"Linea": l, "Under %": f"{(1-p_over):.1%}", "Fair U": f"{1/(1-p_over):.2f}", "Over %": f"{p_over:.1%}", "Fair O": f"{1/p_over:.2f}"})
        st.table(pd.DataFrame(uo_data))

        cm1, cm2 = st.columns(2)
        with cm1:
            pGG = st.session_state.pGG
            st.subheader("Gol / NoGol")
            st.table(pd.DataFrame([{"Esito": "GG", "Prob": f"{pGG:.1%}", "Fair": f"{1/pGG:.2f}"}, {"Esito": "NG", "Prob": f"{(1-pGG):.1%}", "Fair": f"{1/(1-pGG):.2f}"}]))
            
            st.subheader("Multigol")
            mg_res = []
            for r in [(1,2), (1,3), (2,3), (2,4), (3,5)]:
                pm = np.sum(matrix[(np.indices((10,10))[0] + np.indices((10,10))[1] >= r[0]) & (np.indices((10,10))[0] + np.indices((10,10))[1] <= r[1])])
                mg_res.append({"Range": f"{r[0]}-{r[1]}", "Prob": f"{pm:.1%}", "Fair": f"{1/pm:.2f}"})
            st.table(pd.DataFrame(mg_res))
            
        with cm2:
            st.subheader("Handicap & Asian")
            h1_1 = np.sum(matrix[np.indices((10,10))[0] - 1 > np.indices((10,10))[1]])
            st.write(f"**Handicap Europeo (-1):** Prob 1: {h1_1:.1%} | Fair: {1/h1_1:.2f}")
            dnb_p = (p1/(p1+p2)) if (p1+p2)>0 else 0
            st.write(f"**Asian DNB (0.0):** Prob 1: {dnb_p:.1%} | Fair: {1/dnb_p:.2f}")

    with tab3:
        st.header("Calcolatore Marcatore")
        pcol1, pcol2 = st.columns(2)
        n_h = st.session_state.home_name_display
        n_a = st.session_state.away_name_display
        
        p_t = pcol1.radio("Squadra", [n_h, n_a], horizontal=True)
        p_v = pcol2.number_input("xG/90 (da FBref)", 0.01, 2.0, 0.40)
        p_m = pcol1.number_input("Minuti previsti", 1, 100, 80)
        p_b = pcol2.number_input("Quota Goal", 1.01, 100.0, 2.50)
        
        ctx_xg = st.session_state.f_xh if p_t == n_h else st.session_state.f_xa
        # Usa il dato stagionale
        ctx_avg = 1.45 # Fallback
        
        prob_p, _ = calculate_player_probability(p_v, p_m, ctx_xg, ctx_avg)
        
        r1, r2, r3 = st.columns(3)
        r1.metric("Probabilità", f"{prob_p:.1%}")
        r2.metric("Fair Odd", f"{1/prob_p:.2f}")
        r3.metric("Valore %", f"{((p_b*prob_p)-1)*100:+.1f}%")

    with tab4:
        st.header("Backtesting & Tools")
        if st.button("💾 SALVA IN STORICO"):
            st.session_state.history.append({
                "Data": date.today().strftime("%d/%m"), 
                "Match": f"{n_h}-{n_a}", 
                "P1": p1, "PX": pX, "P2": p2, 
                "Risultato": "In attesa"
            })
            st.toast("Salvato!")

        if st.session_state.history:
            df_h = pd.DataFrame(st.session_state.history)
            ed_df = st.data_editor(df_h, column_config={"Risultato": st.column_config.SelectboxColumn("Esito Reale", options=["1", "X", "2", "In attesa"])})
            val = ed_df[ed_df["Risultato"] != "In attesa"]
            if not val.empty:
                brier = []
                for _, r in val.iterrows():
                    o = [1 if r["Risultato"]=="1" else 0, 1 if r["Risultato"]=="X" else 0, 1 if r["Risultato"]=="2" else 0]
                    brier.append((r["P1"]-o[0])**2 + (r["PX"]-o[1])**2 + (r["P2"]-o[2])**2)
                st.metric("Brier Score Medio", f"{np.mean(brier):.3f}", help="0=Perfetto, 2=Pessimo.")
            if st.button("🗑️ Reset"): st.session_state.history = []; st.rerun()

        st.markdown("---")
        st.subheader("Calcolatrice Manuale")
        cM1, cM2 = st.columns(2)
        with cM1:
            q_in = st.number_input("Reverse Quota", 1.01, 100.0, 2.0)
            st.write(f"Prob: **{1/q_in:.1%}**")
        with cM2:
            mp = st.number_input("Tua Stima %", 1.0, 100.0, 50.0) / 100
            mq = st.number_input("Quota Book", 1.01, 100.0, 2.0)
            if (mp * mq) - 1 > 0: st.success(f"Valore! Stake: **{((((mq-1)*mp)-(1-mp))/(mq-1)*25):.1f}%**")
            else: st.error("No Value")
