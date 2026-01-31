import streamlit as st
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import json
import os
import requests
from datetime import date

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Mathbet fc - Ultimate Pro", page_icon="⚽", layout="wide")

# ==============================================================================
# 📂 CONFIGURAZIONE FILE CSV (MAPPING 4 FILE)
# ==============================================================================
LEAGUE_FILES = {
    "🇮🇹 Serie A": {
        "total": "serie_a_total.csv",
        "home": "serie_a_home.csv",
        "away": "serie_a_away.csv",
        "form": "serie_a_form.csv"
    },
    "🇬🇧 Premier League": {
        "total": "premier_total.csv",
        "home": "premier_home.csv",
        "away": "premier_away.csv",
        "form": "premier_form.csv"
    },
    "🇪🇸 La Liga": {
        "total": "liga_total.csv",
        "home": "liga_home.csv",
        "away": "liga_away.csv",
        "form": "liga_form.csv"
    },
    "🇩🇪 Bundesliga": {
        "total": "bundesliga_total.csv",
        "home": "bundesliga_home.csv",
        "away": "bundesliga_away.csv",
        "form": "bundesliga_form.csv"
    },
    "🇫🇷 Ligue 1": {
        "total": "ligue1_total.csv",
        "home": "ligue1_home.csv",
        "away": "ligue1_away.csv",
        "form": "ligue1_form.csv"
    }
}

# --- PARAMETRI ML ---
LEAGUES = {
    "🌐 Generico (Default)": { "avg": 1.35, "ha": 0.25, "rho": -0.10 },
    "🇮🇹 Serie A":          { "avg": 1.28, "ha": 0.059, "rho": -0.032 },
    "🇬🇧 Premier League":   { "avg": 1.47, "ha": 0.046, "rho": 0.006 },
    "🇪🇸 La Liga":          { "avg": 1.31, "ha": 0.143, "rho": 0.060 },
    "🇩🇪 Bundesliga":       { "avg": 1.57, "ha": 0.066, "rho": -0.091 },
    "🇫🇷 Ligue 1":          { "avg": 1.49, "ha": 0.120, "rho": -0.026 },
}

# ==============================================================================
# 🌍 ELO RATING AUTOMATICO
# ==============================================================================
@st.cache_data(ttl=3600)
def get_clubelo_ratings():
    try:
        date_str = date.today().strftime("%Y-%m-%d")
        url = f"http://api.clubelo.com/{date_str}"
        df = pd.read_csv(url)
        return dict(zip(df.Club, df.Elo))
    except: return {}

ELO_DB = get_clubelo_ratings()

def find_elo(team_name):
    if not ELO_DB: return 1600.0
    if team_name in ELO_DB: return float(ELO_DB[team_name])
    # Ricerca parziale
    team_clean = team_name.lower().replace("ac ", "").replace("fc ", "")
    for k, v in ELO_DB.items():
        if team_clean in k.lower(): return float(v)
    return 1600.0

# ==============================================================================
# 🧠 FUNZIONI CARICAMENTO DATI (4 CSV)
# ==============================================================================
@st.cache_data
def load_league_stats(league_name):
    if league_name not in LEAGUE_FILES: return {}, []
    files = LEAGUE_FILES[league_name]
    stats_db = {}
    team_list = []

    def read_csv_safe(filename):
        if os.path.exists(filename):
            try:
                # Gestione flessibile separatori e nomi colonne
                df = pd.read_csv(filename, sep=None, engine='python')
                df.columns = [c.strip().lower() for c in df.columns]
                return df
            except: return None
        return None

    def safe_avg(val, n): return float(val) / n if n > 0 else 0.0

    def extract_stats(row):
        # Supporta varie nomenclature (goals/gf, ga/gs, matches/played)
        goals = row.get('goals', row.get('gf', row.get('fthg', 0)))
        ga = row.get('ga', row.get('gs', row.get('ftag', 0)))
        xg = row.get('xg', 0)
        xga = row.get('xga', 0)
        matches = row.get('matches', row.get('played', 0))
        
        return {
            "gf": safe_avg(goals, matches), "gs": safe_avg(ga, matches),
            "xg": safe_avg(xg, matches), "xga": safe_avg(xga, matches),
            "matches": matches
        }

    # Caricamento DataFrame
    df_total = read_csv_safe(files["total"])
    df_home = read_csv_safe(files["home"])
    df_away = read_csv_safe(files["away"])
    df_form = read_csv_safe(files["form"])

    if df_total is None: return {}, []

    for _, row in df_total.iterrows():
        # Trova la colonna del nome squadra
        team_col = next((c for c in row.index if 'team' in c or 'squad' in c or 'club' in c), None)
        if not team_col: continue
        
        team_name = str(row[team_col]).strip()
        total_stats = extract_stats(row)
        
        if total_stats["matches"] > 0:
            stats_db[team_name] = {
                "matches": total_stats["matches"], "total": total_stats,
                "home": {"gf": 0, "gs": 0, "xg": 0, "xga": 0, "matches": 0},
                "away": {"gf": 0, "gs": 0, "xg": 0, "xga": 0, "matches": 0},
                "form": {"gf": 0, "gs": 0, "xg": 0, "xga": 0, "matches": 0}
            }
            team_list.append(team_name)

    # Arricchimento dati
    for df, key in [(df_home, "home"), (df_away, "away"), (df_form, "form")]:
        if df is not None:
            for _, row in df.iterrows():
                team_col = next((c for c in row.index if 'team' in c or 'squad' in c), None)
                if team_col:
                    t = str(row[team_col]).strip()
                    if t in stats_db: stats_db[t][key] = extract_stats(row)

    team_list.sort()
    return stats_db, team_list

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
    return max(0.0, kelly * 0.25 * 100) # Kelly Frazionato 25%

def calculate_player_probability(metric_per90, expected_mins, team_match_xg, team_avg_xg):
    base_lambda = (metric_per90 / 90.0) * expected_mins
    if team_avg_xg <= 0: team_avg_xg = 0.01
    match_factor = team_match_xg / team_avg_xg
    final_lambda = base_lambda * match_factor
    return 1 - math.exp(-final_lambda)

@st.cache_data
def monte_carlo_simulation(f_xh, f_xa, n_sims=5000):
    sim = []
    for _ in range(n_sims):
        gh = np.random.poisson(max(0.1, np.random.normal(f_xh, 0.15*f_xh)))
        ga = np.random.poisson(max(0.1, np.random.normal(f_xa, 0.15*f_xa)))
        sim.append(1 if gh>ga else (0 if gh==ga else 2))
    return sim

def calcola_forza_squadra(att_season, def_season, att_form, def_form, w_season):
    # att_form e def_form sono già MEDIE
    att = (att_season * w_season) + (att_form * (1-w_season))
    def_ = (def_season * w_season) + (def_form * (1-w_season))
    return att, def_

def salva_storico_json():
    try:
        with open('mathbet_history.json', 'w') as f:
            json.dump(st.session_state.history, f, indent=2, default=str)
        return True
    except: return False

def carica_storico_json():
    try:
        with open('mathbet_history.json', 'r') as f: return json.load(f)
    except: return []

# --- INIZIALIZZAZIONE ---
if 'history' not in st.session_state: st.session_state.history = carica_storico_json()
if 'analyzed' not in st.session_state: st.session_state.analyzed = False

# --- SIDEBAR ---
with st.sidebar:
    st.title("🧠 Configurazione")
    league_name = st.selectbox("Campionato", list(LEAGUES.keys()))
    L_DATA = LEAGUES[league_name]
    
    STATS_DB, TEAM_LIST = load_league_stats(league_name)
    
    if STATS_DB:
        st.success(f"✅ Dati {league_name} caricati!")
        files = LEAGUE_FILES.get(league_name, {})
        missing = [k for k, v in files.items() if not os.path.exists(v)]
        if missing: st.warning(f"⚠️ Mancanti: {', '.join(missing)}")
    elif league_name != "🌐 Generico (Default)":
        st.error(f"❌ CSV non trovati.")

    st.markdown("---")
    data_mode = st.radio("Dati Analisi", ["Solo Gol Reali", "Solo xG", "Ibrido (Consigliato)"], index=2)
    st.markdown("---")
    matchday = st.slider("Giornata", 1, 38, 22)
    w_seas = min(0.90, 0.30 + (matchday * 0.02)) 
    st.markdown("---")
    m_type = st.radio("Contesto", ["Standard", "Derby", "Campo Neutro"])
    is_big_match = st.checkbox("🔥 Big Match")

st.title("Mathbet fc - Ultimate Pro 🚀")

col_h, col_a = st.columns(2)
h_uo_input, a_uo_input = {}, {}

def get_val(stats_dict, metric, mode):
    val_gol = stats_dict.get(metric, 0.0)
    xg_key = 'xg' if metric == 'gf' else 'xga'
    val_xg = stats_dict.get(xg_key, 0.0)
    if mode == "Solo Gol Reali": return val_gol
    if mode == "Solo xG": return val_xg
    return (val_gol + val_xg) / 2.0

def get_form_val(stats_dict, metric, mode):
    form_data = stats_dict.get("form", {})
    if form_data.get("matches", 0) > 0:
        val_gol = form_data.get('gf' if metric == 'gf' else 'gs', 0.0)
        val_xg = form_data.get('xg' if metric == 'gf' else 'xga', 0.0)
        if mode == "Solo Gol Reali": return val_gol
        if mode == "Solo xG": return val_xg
        return (val_gol + val_xg) / 2.0
    # Fallback su stagionale
    return get_val(stats_dict["total"], metric, mode)

# --- INPUT SQUADRE ---
with col_h:
    st.subheader("🏠 Squadra Casa")
    if TEAM_LIST:
        h_idx = 0
        h_name = st.selectbox("Seleziona Casa", TEAM_LIST, index=h_idx, key="h_sel")
        h_stats = STATS_DB[h_name]
        auto_elo_h = find_elo(h_name)
    else:
        h_name = st.text_input("Nome Casa", "Inter")
        h_stats = None
        auto_elo_h = 1600.0

    h_elo = st.number_input("Rating Elo Casa", 1000.0, 2500.0, auto_elo_h, step=10.0)
    
    with st.expander("📊 Stats & Forma", expanded=True):
        def_att_s, def_def_s = 1.85, 0.95
        def_att_h, def_def_h = 1.95, 0.85
        def_form_att, def_form_def = 1.8, 0.8
        
        if h_stats:
            def_att_s = get_val(h_stats["total"], 'gf', data_mode)
            def_def_s = get_val(h_stats["total"], 'gs', data_mode)
            if h_stats["home"]["matches"] > 0:
                def_att_h = get_val(h_stats["home"], 'gf', data_mode)
                def_def_h = get_val(h_stats["home"], 'gs', data_mode)
            else:
                def_att_h = def_att_s * 1.15; def_def_h = def_def_s * 0.85
            def_form_att = get_form_val(h_stats, 'gf', data_mode)
            def_form_def = get_form_val(h_stats, 'gs', data_mode)

        c1, c2 = st.columns(2)
        h_att = c1.number_input("Attacco Tot (C)", 0.0, 5.0, float(def_att_s), 0.01)
        h_def = c2.number_input("Difesa Tot (C)", 0.0, 5.0, float(def_def_s), 0.01)
        st.markdown("---")
        c3, c4 = st.columns(2)
        h_att_home = c3.number_input("Attacco Casa", 0.0, 5.0, float(def_att_h), 0.01)
        h_def_home = c4.number_input("Difesa Casa", 0.0, 5.0, float(def_def_h), 0.01)
        st.markdown("---")
        h_form_att = st.number_input("Attacco Forma (C)", 0.0, 5.0, float(def_form_att), 0.1)
        h_form_def = st.number_input("Difesa Forma (C)", 0.0, 5.0, float(def_form_def), 0.1)

    with st.expander("Trend Over"):
        for l in [0.5, 1.5, 2.5, 3.5, 4.5]: h_uo_input[l] = st.slider(f"O{l} H", 0, 100, 50, key=f"ho{l}")

with col_a:
    st.subheader("✈️ Squadra Ospite")
    if TEAM_LIST:
        a_name = st.selectbox("Seleziona Ospite", TEAM_LIST, index=1 if len(TEAM_LIST)>1 else 0, key="a_sel")
        a_stats = STATS_DB[a_name]
        auto_elo_a = find_elo(a_name)
    else:
        a_name = st.text_input("Nome Ospite", "Juventus")
        a_stats = None
        auto_elo_a = 1550.0

    a_elo = st.number_input("Rating Elo Ospite", 1000.0, 2500.0, auto_elo_a, step=10.0)

    with st.expander("📊 Stats & Forma", expanded=True):
        def_att_s_a, def_def_s_a = 1.45, 0.85
        def_att_a, def_def_a = 1.25, 1.05
        def_form_att_a, def_form_def_a = 1.4, 0.8
        
        if a_stats:
            def_att_s_a = get_val(a_stats["total"], 'gf', data_mode)
            def_def_s_a = get_val(a_stats["total"], 'gs', data_mode)
            if a_stats["away"]["matches"] > 0:
                def_att_a = get_val(a_stats["away"], 'gf', data_mode)
                def_def_a = get_val(a_stats["away"], 'gs', data_mode)
            else:
                def_att_a = def_att_s_a * 0.85; def_def_a = def_def_s_a * 1.15
            def_form_att_a = get_form_val(a_stats, 'gf', data_mode)
            def_form_def_a = get_form_val(a_stats, 'gs', data_mode)

        c5, c6 = st.columns(2)
        a_att = c5.number_input("Attacco Tot (O)", 0.0, 5.0, float(def_att_s_a), 0.01)
        a_def = c6.number_input("Difesa Tot (O)", 0.0, 5.0, float(def_def_s_a), 0.01)
        st.markdown("---")
        c7, c8 = st.columns(2)
        a_att_away = c7.number_input("Attacco Fuori", 0.0, 5.0, float(def_att_a), 0.01)
        a_def_away = c8.number_input("Difesa Fuori", 0.0, 5.0, float(def_def_a), 0.01)
        st.markdown("---")
        a_form_att = st.number_input("Attacco Forma (O)", 0.0, 5.0, float(def_form_att_a), 0.1)
        a_form_def = st.number_input("Difesa Forma (O)", 0.0, 5.0, float(def_form_def_a), 0.1)

    with st.expander("Trend Over"):
        for l in [0.5, 1.5, 2.5, 3.5, 4.5]: a_uo_input[l] = st.slider(f"O{l} A", 0, 100, 50, key=f"ao{l}")

st.subheader("💰 Quote Bookmaker")
qc1, qc2, qc3 = st.columns(3)
b1 = qc1.number_input("Quota 1", 1.01, 100.0, 2.10)
bX = qc2.number_input("Quota X", 1.01, 100.0, 3.20)
b2 = qc3.number_input("Quota 2", 1.01, 100.0, 3.60)

with st.expander("⚙️ Fine Tuning"):
    c_str1, c_str2 = st.columns(2)
    h_str = c_str1.slider("Titolari % Casa", 50, 100, 100)
    a_str = c_str2.slider("Titolari % Ospite", 50, 100, 100)
    h_rest = c_str1.slider("Riposo Casa", 2, 10, 7)
    a_rest = c_str2.slider("Riposo Ospite", 2, 10, 7)
    h_m_a = c_str1.checkbox("No Bomber Casa")
    a_m_a = c_str2.checkbox("No Bomber Ospite")
    h_m_d = c_str1.checkbox("No Difensore Casa")
    a_m_d = c_str2.checkbox("No Difensore Ospite")

# --- CALCOLO CORE ---
if st.button("🚀 ANALIZZA", type="primary", use_container_width=True):
    with st.spinner("Calcolo in corso..."):
        home_adv = L_DATA["ha"]
        rho = L_DATA["rho"]
        avg = L_DATA["avg"]
        
        if m_type == "Campo Neutro": home_adv = 0.0
        elif m_type == "Derby": home_adv *= 0.5

        w_split = 0.60 
        h_base_att = (h_att*(1-w_split)) + (h_att_home*w_split)
        h_base_def = (h_def*(1-w_split)) + (h_def_home*w_split)
        a_base_att = (a_att*(1-w_split)) + (a_att_away*w_split)
        a_base_def = (a_def*(1-w_split)) + (a_def_away*w_split)

        h_final_att, h_final_def = calcola_forza_squadra(h_base_att, h_base_def, h_form_att, h_form_def, w_seas)
        a_final_att, a_final_def = calcola_forza_squadra(a_base_att, a_base_def, a_form_att, a_form_def, w_seas)
        
        xg_h = (h_final_att * a_final_def) / avg
        xg_a = (a_final_att * h_final_def) / avg
        
        elo_fac_h = 1 + ((h_elo + (100 if m_type=="Standard" else 0) - a_elo)/1000.0)
        elo_fac_a = 1 - ((h_elo + (100 if m_type=="Standard" else 0) - a_elo)/1000.0)
        
        f_xh = (xg_h * elo_fac_h) + home_adv
        f_xa = (xg_a * elo_fac_a)
        
        ft_malus = 0.05 
        if h_rest <= 3: f_xh *= (1-ft_malus); f_xa *= (1+ft_malus) 
        if a_rest <= 3: f_xa *= (1-ft_malus); f_xh *= (1+ft_malus)
        f_xh *= (h_str/100.0); f_xa *= (a_str/100.0)
        if is_big_match: f_xh *= 0.90; f_xa *= 0.90
        if h_m_a: f_xh *= 0.85
        if h_m_d: f_xa *= 1.20
        if a_m_a: f_xa *= 0.85
        if a_m_d: f_xh *= 1.20

        p1, pX, p2, pGG = 0, 0, 0, 0
        matrix = np.zeros((10,10)); scores = []
        for h in range(10):
            for a in range(10):
                p = dixon_coles_probability(h, a, f_xh, f_xa, rho)
                matrix[h,a] = p
                if h > a: p1 += p
                elif h == a: pX += p
                else: p2 += p
                if h>0 and a>0: pGG += p
                if h<6 and a<6: scores.append({"Risultato": f"{h}-{a}", "Prob": p})
        
        tot = np.sum(matrix)
        if tot > 0: matrix /= tot; p1 /= tot; pX /= tot; p2 /= tot; pGG /= tot
        
        sim = monte_carlo_simulation(f_xh, f_xa)
        s1, sX, s2 = sim.count(1)/5000, sim.count(0)/5000, sim.count(2)/5000
        stability = max(0, 100 - ((abs(p1-s1)+abs(pX-sX)+abs(p2-s2))/3*400))

        st.session_state.analyzed = True
        st.session_state.f_xh = f_xh; st.session_state.f_xa = f_xa
        st.session_state.h_name = h_name; st.session_state.a_name = a_name
        st.session_state.p1 = p1; st.session_state.pX = pX; st.session_state.p2 = p2
        st.session_state.pGG = pGG; st.session_state.stability = stability
        st.session_state.matrix = matrix; st.session_state.scores = scores
        st.session_state.b1 = b1; st.session_state.bX = bX; st.session_state.b2 = b2

# --- OUTPUT TABS ---
if st.session_state.analyzed:
    st.markdown("---")
    st.header(f"📊 {st.session_state.h_name} vs {st.session_state.a_name}")
    c_m1, c_m2 = st.columns(2)
    c_m1.metric("xG Previsti", f"{st.session_state.f_xh:.2f} - {st.session_state.f_xa:.2f}")
    c_m2.metric("Stabilità", f"{st.session_state.stability:.1f}%")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏆 Esito", "⚽ Gol & Handicap", "👤 Player Props", "📊 Grafici", "📝 Storico"])

    with tab1:
        st.subheader("Probabilità 1X2")
        df_1x2 = pd.DataFrame({
            "Esito": ["1", "X", "2"],
            "Prob %": [f"{st.session_state.p1:.1%}", f"{st.session_state.pX:.1%}", f"{st.session_state.p2:.1%}"],
            "Fair Odd": [f"{1/st.session_state.p1:.2f}", f"{1/st.session_state.pX:.2f}", f"{1/st.session_state.p2:.2f}"],
            "Bookie": [st.session_state.b1, st.session_state.bX, st.session_state.b2],
            "Kelly Stake %": [
                f"{calculate_kelly(st.session_state.p1, st.session_state.b1):.1f}%",
                f"{calculate_kelly(st.session_state.pX, st.session_state.bX):.1f}%",
                f"{calculate_kelly(st.session_state.p2, st.session_state.b2):.1f}%"
            ]
        })
        st.dataframe(df_1x2, hide_index=True, use_container_width=True)
        
        c_res1, c_res2 = st.columns(2)
        with c_res1:
            st.subheader("Risultati Esatti")
            sc = sorted(st.session_state.scores, key=lambda x: x["Prob"], reverse=True)[:6]
            st.dataframe(pd.DataFrame(sc), hide_index=True)
        with c_res2:
            fig, ax = plt.subplots(figsize=(4,3))
            sns.heatmap(st.session_state.matrix[:5,:5], annot=True, fmt=".0%", cmap="Greens", cbar=False)
            plt.xlabel("Ospite"); plt.ylabel("Casa"); st.pyplot(fig)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Over / Under")
            uo_res = []
            for l in [0.5, 1.5, 2.5, 3.5, 4.5]:
                p_pure = np.sum(st.session_state.matrix[np.indices((10,10))[0] + np.indices((10,10))[1] > l])
                trend = (h_uo_input.get(l,50) + a_uo_input.get(l,50))/200.0
                p_final = (p_pure * 0.7) + (trend * 0.3)
                uo_res.append({"Linea": l, "Over": f"{p_final:.1%}", "Fair O": f"{1/p_final:.2f}", "Under": f"{(1-p_final):.1%}"})
            st.dataframe(pd.DataFrame(uo_res), hide_index=True)
        
        with c2:
            st.subheader("Handicap & Multigol")
            # Handicap
            h_h1 = np.sum(st.session_state.matrix[np.indices((10,10))[0] - 1 > np.indices((10,10))[1]])
            h_h2 = np.sum(st.session_state.matrix[np.indices((10,10))[0] + 1 < np.indices((10,10))[1]])
            st.write(f"🏠 **Casa (-1):** {h_h1:.1%} (@{1/h_h1:.2f})")
            st.write(f"✈️ **Ospite (-1):** {h_h2:.1%} (@{1/h_h2:.2f})")
            
            # Multigol
            mg_1_3 = np.sum(st.session_state.matrix[(np.indices((10,10))[0] + np.indices((10,10))[1] >= 1) & (np.indices((10,10))[0] + np.indices((10,10))[1] <= 3)])
            mg_2_4 = np.sum(st.session_state.matrix[(np.indices((10,10))[0] + np.indices((10,10))[1] >= 2) & (np.indices((10,10))[0] + np.indices((10,10))[1] <= 4)])
            st.write(f"🔢 **Multigol 1-3:** {mg_1_3:.1%} (@{1/mg_1_3:.2f})")
            st.write(f"🔢 **Multigol 2-4:** {mg_2_4:.1%} (@{1/mg_2_4:.2f})")
            
            st.write("---")
            st.write(f"🥅 **Goal (GG):** {st.session_state.pGG:.1%} (@{1/st.session_state.pGG:.2f})")

    with tab3:
        st.subheader("Player Props (Marcatore/Assist)")
        c_p1, c_p2 = st.columns(2)
        pl_name = c_p1.text_input("Nome", "Vlahovic")
        pl_type = c_p2.selectbox("Tipo", ["Gol (xG)", "Assist (xA)"])
        
        c_p3, c_p4 = st.columns(2)
        pl_val = c_p3.number_input("Valore xG/xA per 90'", 0.01, 2.0, 0.45)
        pl_min = c_p4.number_input("Minuti previsti", 1, 100, 85)
        
        team_xg = st.session_state.f_xh if st.checkbox("Gioca in Casa?", value=True) else st.session_state.f_xa
        
        # Logica adattata per Assist: usiamo lo stesso motore Poisson ma scalato
        prob = calculate_player_probability(pl_val, pl_min, team_xg, 1.4)
        
        st.metric(f"Probabilità {pl_type} {pl_name}", f"{prob:.1%}", f"Quota Fair: {1/prob:.2f}")

    with tab4:
        st.subheader("Analisi Grafica")
        c_g1, c_g2 = st.columns(2)
        with c_g1:
            fig_bar = px.bar(x=["1","X","2"], y=[st.session_state.p1, st.session_state.pX, st.session_state.p2], 
                             title="Probabilità Esito", labels={'y':'Probabilità', 'x':'Esito'}, color_discrete_sequence=['#4CAF50'])
            st.plotly_chart(fig_bar, use_container_width=True)
        with c_g2:
            # Distribuzione Gol
            gol_probs = [np.sum(st.session_state.matrix[np.indices((10,10))[0] + np.indices((10,10))[1] == i]) for i in range(7)]
            fig_gol = px.line(x=list(range(7)), y=gol_probs, title="Distribuzione Gol Totali", markers=True)
            st.plotly_chart(fig_gol, use_container_width=True)

    with tab5:
        st.subheader("Storico")
        if st.button("💾 Salva Analisi"):
            st.session_state.history.append({
                "Data": date.today().strftime("%d/%m"),
                "Match": f"{st.session_state.h_name}-{st.session_state.a_name}",
                "P1": round(st.session_state.p1, 2),
                "Esito": "?"
            })
            salva_storico_json()
            st.success("Salvato!")
        
        if st.session_state.history:
            st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
            if st.button("🗑️ Reset"):
                st.session_state.history = []
                salva_storico_json()
                st.rerun()

# --- MANUAL TOOLS (RIPRISTINATO) ---
st.markdown("---")
with st.expander("🛠️ Strumenti Manuali (Calcolatrice Fair Odd & Kelly)"):
    c_man1, c_man2 = st.columns(2)
    with c_man1:
        st.caption("Calcola Probabilità da Quota")
        q_in = st.number_input("Inserisci Quota", 1.01, 100.0, 2.0)
        st.write(f"Probabilità Implicita: **{1/q_in:.1%}**")
    with c_man2:
        st.caption("Calcolatore Kelly")
        my_prob = st.number_input("Tua Probabilità (%)", 1.0, 100.0, 50.0) / 100
        my_odd = st.number_input("Quota Bookmaker", 1.01, 100.0, 2.0)
        kelly_res = calculate_kelly(my_prob, my_odd)
        if kelly_res > 0: st.success(f"Stake Consigliato: {kelly_res:.1f}% del Bankroll")
        else: st.error("Nessun Valore (Stake 0%)")
