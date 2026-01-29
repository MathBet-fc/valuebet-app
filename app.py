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
from datetime import datetime

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Mathbet fc - ML Ultimate Pro", page_icon="🧠", layout="wide")

# ==============================================================================
# 📂 CONFIGURAZIONE FILE CSV (MAPPING 4 FILE PER CAMPIONATO)
# ==============================================================================
LEAGUE_FILES = {
    "🇮🇹 Serie A": {
        "total": "serie_a_total.csv",
        "home": "serie_a_home.csv",
        "away": "serie_a_away.csv",
        "form": "serie_a_form.csv"  # File Ultimo Mese
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
# 🧠 FUNZIONI DI CARICAMENTO DATI (AGGIORNATA PER 4 CSV)
# ==============================================================================
@st.cache_data
def load_league_stats(league_name):
    """
    Carica 4 CSV: Totale, Casa, Fuori, Forma (Ultimo Mese).
    Divide sempre i valori cumulativi per il numero di partite giocate in quel contesto.
    """
    if league_name not in LEAGUE_FILES:
        return {}, []

    files = LEAGUE_FILES[league_name]
    stats_db = {}
    team_list = []

    def read_csv_safe(filename):
        if os.path.exists(filename):
            try:
                # Supporta separatori diversi e pulisce i nomi delle colonne
                df = pd.read_csv(filename, sep=None, engine='python')
                df.columns = [c.strip().lower() for c in df.columns] # Colonne minuscole
                return df
            except:
                return None
        return None

    # Helper per dividere Gol/xG per Partite
    def safe_avg(val, n):
        return float(val) / n if n > 0 else 0.0

    # Helper per estrarre i dati da una riga
    def extract_stats(row):
        # Cerca colonne standard (goals, ga, xg, xga)
        # Alcuni CSV usano 'goals', altri 'fthg', cerchiamo di essere flessibili
        goals = row.get('goals', row.get('gf', 0))
        ga = row.get('ga', row.get('gs', 0))
        xg = row.get('xg', 0)
        xga = row.get('xga', 0)
        matches = row.get('matches', 0)
        
        return {
            "gf": safe_avg(goals, matches),
            "gs": safe_avg(ga, matches),
            "xg": safe_avg(xg, matches),
            "xga": safe_avg(xga, matches),
            "matches": matches
        }

    # 1. Carica i DataFrame
    df_total = read_csv_safe(files["total"])
    df_home = read_csv_safe(files["home"])
    df_away = read_csv_safe(files["away"])
    df_form = read_csv_safe(files["form"]) # NUOVO FILE

    if df_total is None:
        return {}, []

    # 2. Inizializza DB con i dati Totali
    for _, row in df_total.iterrows():
        # Cerca la colonna del nome squadra (team, squad, club...)
        team_col = next((c for c in row.index if 'team' in c or 'squad' in c), None)
        if not team_col: continue
        
        team_name = str(row[team_col]).strip()
        
        # Estrai stats totali
        total_stats = extract_stats(row)
        
        if total_stats["matches"] > 0:
            stats_db[team_name] = {
                "matches": total_stats["matches"],
                "total": total_stats,
                # Placeholder per gli altri contesti
                "home": {"gf": 0, "gs": 0, "xg": 0, "xga": 0, "matches": 0},
                "away": {"gf": 0, "gs": 0, "xg": 0, "xga": 0, "matches": 0},
                "form": {"gf": 0, "gs": 0, "xg": 0, "xga": 0, "matches": 0}
            }
            team_list.append(team_name)

    # 3. Arricchisci con dati Casa
    if df_home is not None:
        for _, row in df_home.iterrows():
            team_col = next((c for c in row.index if 'team' in c or 'squad' in c), None)
            if team_col:
                t = str(row[team_col]).strip()
                if t in stats_db:
                    stats_db[t]["home"] = extract_stats(row)

    # 4. Arricchisci con dati Fuori
    if df_away is not None:
        for _, row in df_away.iterrows():
            team_col = next((c for c in row.index if 'team' in c or 'squad' in c), None)
            if team_col:
                t = str(row[team_col]).strip()
                if t in stats_db:
                    stats_db[t]["away"] = extract_stats(row)
    
    # 5. Arricchisci con dati Forma (Ultimo Mese)
    if df_form is not None:
        for _, row in df_form.iterrows():
            team_col = next((c for c in row.index if 'team' in c or 'squad' in c), None)
            if team_col:
                t = str(row[team_col]).strip()
                if t in stats_db:
                    # Qui calcoliamo le medie basate SOLO sulle partite dell'ultimo mese
                    stats_db[t]["form"] = extract_stats(row)

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
    # att_form e def_form arrivano qui come MEDIE per partita.
    # La formula originale si aspetta un TOTALE su 5 partite o simile per pesarlo col 25-30%
    # Quindi, se att_form è una media (es. 1.8 gol/partita), non dobbiamo dividerla ancora per 5.0
    # Ma la formula originale era: ((att_form_totale / 5.0) * (1-w)).
    # Dato che noi abbiamo già la MEDIA (att_form = att_form_totale / partite_giocate), usiamo direttamente quella.
    
    att = (att_season * w_season) + (att_form * (1-w_season))
    def_ = (def_season * w_season) + (def_form * (1-w_season))
    return att, def_

def salva_storico_json():
    try:
        with open('mathbet_history.json', 'w') as f:
            json.dump(st.session_state.history, f, indent=2, default=str)
        return True
    except Exception as e:
        return False

def carica_storico_json():
    try:
        with open('mathbet_history.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def export_risultati_csv(data):
    df = pd.DataFrame([data])
    return df.to_csv(index=False).encode('utf-8')

# --- INIZIALIZZAZIONE ---
if 'history' not in st.session_state: st.session_state.history = carica_storico_json()
if 'analyzed' not in st.session_state: st.session_state.analyzed = False

# --- SIDEBAR ---
with st.sidebar:
    st.title("🧠 Configurazione")
    league_name = st.selectbox("Campionato", list(LEAGUES.keys()))
    L_DATA = LEAGUES[league_name]
    
    # CARICAMENTO DATI
    STATS_DB, TEAM_LIST = load_league_stats(league_name)
    
    if STATS_DB:
        st.success(f"✅ Dati {league_name} caricati! ({len(TEAM_LIST)} squadre)")
        files = LEAGUE_FILES.get(league_name, {})
        missing = []
        if not os.path.exists(files.get("home", "")): missing.append("Casa")
        if not os.path.exists(files.get("away", "")): missing.append("Fuori")
        if not os.path.exists(files.get("form", "")): missing.append("Forma")
        if missing:
            st.warning(f"⚠️ File mancanti: {', '.join(missing)}. Uso stime.")
    elif league_name != "🌐 Generico (Default)":
        st.error(f"❌ File CSV non trovati per {league_name}. Controlla i nomi.")

    st.markdown("---")
    
    # SELETTORE MODALITÀ DATI
    data_mode = st.radio(
        "Modalità Analisi Dati",
        ["Solo Gol Reali", "Solo xG (Expected Goals)", "Ibrido (Consigliato)"],
        index=2,
        help="Ibrido: Media 50/50 tra Gol Reali e xG."
    )
    
    st.markdown("---")
    matchday = st.slider("Giornata", 1, 38, 22)
    w_seas = min(0.90, 0.30 + (matchday * 0.02)) 
    
    st.markdown("---")
    m_type = st.radio("Contesto", ["Standard", "Derby", "Campo Neutro"])
    is_big_match = st.checkbox("🔥 Big Match")

st.title("Mathbet fc - ML Ultimate Edition 🚀")

# --- SELEZIONE SQUADRE ---
col_h, col_a = st.columns(2)
h_uo_input, a_uo_input = {}, {}

# Helper per estrarre valori
def get_val(stats_dict, metric, mode):
    # metric = 'gf' o 'gs'
    val_gol = stats_dict.get(metric, 0.0)
    # xg key
    xg_key = 'xg' if metric == 'gf' else 'xga'
    val_xg = stats_dict.get(xg_key, 0.0)
    
    if mode == "Solo Gol Reali": return val_gol
    if mode == "Solo xG (Expected Goals)": return val_xg
    return (val_gol + val_xg) / 2.0

# SQUADRA CASA
with col_h:
    st.subheader("🏠 Squadra Casa")
    
    if TEAM_LIST:
        h_idx = TEAM_LIST.index("Inter") if "Inter" in TEAM_LIST else 0
        h_name = st.selectbox("Seleziona Casa", TEAM_LIST, index=h_idx, key="h_sel")
        h_stats = STATS_DB[h_name]
    else:
        h_name = st.text_input("Nome Casa", "Inter")
        h_stats = None

    h_elo = st.number_input("Rating Elo Casa", 1000.0, 2500.0, 1600.0, step=10.0)
    
    with st.expander("📊 Dati & Forma", expanded=True):
        # Default se mancano i dati
        def_att_s, def_def_s = 1.85, 0.95
        def_att_h, def_def_h = 1.95, 0.85
        def_form_att, def_form_def = 1.8, 1.0 # Ora sono medie partita!
        
        if h_stats:
            # 1. Stagionali (Totale)
            def_att_s = get_val(h_stats["total"], 'gf', data_mode)
            def_def_s = get_val(h_stats["total"], 'gs', data_mode)
            
            # 2. Casa
            if h_stats["home"]["matches"] > 0:
                def_att_h = get_val(h_stats["home"], 'gf', data_mode)
                def_def_h = get_val(h_stats["home"], 'gs', data_mode)
            else:
                def_att_h = def_att_s * 1.15
                def_def_h = def_def_s * 0.85
            
            # 3. Forma (Ultimo Mese da CSV dedicato)
            if h_stats["form"]["matches"] > 0:
                def_form_att = get_val(h_stats["form"], 'gf', data_mode)
                def_form_def = get_val(h_stats["form"], 'gs', data_mode)
            else:
                # Fallback se manca il file form: usa la media stagionale
                def_form_att = def_att_s
                def_form_def = def_def_s

        st.caption(f"Media Stagionale ({data_mode})")
        c1, c2 = st.columns(2)
        h_att = c1.number_input("Attacco Totale (C)", 0.0, 5.0, float(def_att_s), 0.01)
        h_def = c2.number_input("Difesa Totale (C)", 0.0, 5.0, float(def_def_s), 0.01)
        
        st.markdown("---")
        st.caption(f"Rendimento in Casa ({data_mode})")
        c3, c4 = st.columns(2)
        h_att_home = c3.number_input("Attacco Casa", 0.0, 5.0, float(def_att_h), 0.01)
        h_def_home = c4.number_input("Difesa Casa", 0.0, 5.0, float(def_def_h), 0.01)

        st.markdown("---")
        st.caption(f"Forma Recente (Media Ultimo Mese)")
        h_form_att = st.number_input("Attacco Forma (C)", 0.0, 5.0, float(def_form_att), 0.1, help="Media Gol/xG dell'ultimo mese")
        h_form_def = st.number_input("Difesa Forma (C)", 0.0, 5.0, float(def_form_def), 0.1)

    with st.expander("📈 Trend Over"):
        for l in [0.5, 1.5, 2.5, 3.5, 4.5]: h_uo_input[l] = st.slider(f"Over {l} % H", 0, 100, 50, key=f"ho{l}")

# SQUADRA OSPITE
with col_a:
    st.subheader("✈️ Squadra Ospite")
    
    if TEAM_LIST:
        a_idx = 1 if len(TEAM_LIST) > 1 else 0
        a_name = st.selectbox("Seleziona Ospite", TEAM_LIST, index=a_idx, key="a_sel")
        a_stats = STATS_DB[a_name]
    else:
        a_name = st.text_input("Nome Ospite", "Juventus")
        a_stats = None

    a_elo = st.number_input("Rating Elo Ospite", 1000.0, 2500.0, 1550.0, step=10.0)

    with st.expander("📊 Dati & Forma", expanded=True):
        def_att_s_a, def_def_s_a = 1.45, 0.85
        def_att_a, def_def_a = 1.25, 1.05
        def_form_att_a, def_form_def_a = 1.5, 0.8
        
        if a_stats:
            def_att_s_a = get_val(a_stats["total"], 'gf', data_mode)
            def_def_s_a = get_val(a_stats["total"], 'gs', data_mode)
            
            if a_stats["away"]["matches"] > 0:
                def_att_a = get_val(a_stats["away"], 'gf', data_mode)
                def_def_a = get_val(a_stats["away"], 'gs', data_mode)
            else:
                def_att_a = def_att_s_a * 0.85
                def_def_a = def_def_s_a * 1.15
            
            if a_stats["form"]["matches"] > 0:
                def_form_att_a = get_val(a_stats["form"], 'gf', data_mode)
                def_form_def_a = get_val(a_stats["form"], 'gs', data_mode)
            else:
                def_form_att_a = def_att_s_a
                def_form_def_a = def_def_s_a

        st.caption(f"Media Stagionale ({data_mode})")
        c5, c6 = st.columns(2)
        a_att = c5.number_input("Attacco Totale (O)", 0.0, 5.0, float(def_att_s_a), 0.01)
        a_def = c6.number_input("Difesa Totale (O)", 0.0, 5.0, float(def_def_s_a), 0.01)
        
        st.markdown("---")
        st.caption(f"Rendimento in Trasferta ({data_mode})")
        c7, c8 = st.columns(2)
        a_att_away = c7.number_input("Attacco Fuori", 0.0, 5.0, float(def_att_a), 0.01)
        a_def_away = c8.number_input("Difesa Fuori", 0.0, 5.0, float(def_def_a), 0.01)

        st.markdown("---")
        st.caption(f"Forma Recente (Media Ultimo Mese)")
        a_form_att = st.number_input("Attacco Forma (O)", 0.0, 5.0, float(def_form_att_a), 0.1, help="Media Gol/xG dell'ultimo mese")
        a_form_def = st.number_input("Difesa Forma (O)", 0.0, 5.0, float(def_form_def_a), 0.1)

    with st.expander("📈 Trend Over"):
        for l in [0.5, 1.5, 2.5, 3.5, 4.5]: a_uo_input[l] = st.slider(f"Over {l} % A", 0, 100, 50, key=f"ao{l}")

st.subheader("💰 Quote Bookmaker")
qc1, qc2, qc3 = st.columns(3)
b1 = qc1.number_input("Quota 1", 1.01, 100.0, 2.10)
bX = qc2.number_input("Quota X", 1.01, 100.0, 3.20)
b2 = qc3.number_input("Quota 2", 1.01, 100.0, 3.60)

# --- OPZIONI AVANZATE ---
with st.expander("⚙️ Fine Tuning (Stanchezza & Assenze)"):
    c_str1, c_str2 = st.columns(2)
    h_str = c_str1.slider("Titolari % Casa", 50, 100, 100)
    a_str = c_str2.slider("Titolari % Ospite", 50, 100, 100)
    h_rest = c_str1.slider("Riposo Casa (gg)", 2, 10, 7)
    a_rest = c_str2.slider("Riposo Ospite (gg)", 2, 10, 7)
    h_m_a = c_str1.checkbox("No Bomber Casa")
    a_m_a = c_str2.checkbox("No Bomber Ospite")
    h_m_d = c_str1.checkbox("No Difensore Casa")
    a_m_d = c_str2.checkbox("No Difensore Ospite")

# --- CALCOLO ---
if st.button("🚀 ANALIZZA CON ML", type="primary", use_container_width=True):
    with st.spinner("Analisi in corso..."):
        
        home_adv_goals = L_DATA["ha"]
        rho_val = L_DATA["rho"]
        avg_goals_league = L_DATA["avg"]
        
        if m_type == "Campo Neutro": home_adv_goals = 0.0
        elif m_type == "Derby": home_adv_goals *= 0.5

        w_split = 0.60 
        
        h_base_att = (h_att * (1-w_split)) + (h_att_home * w_split)
        h_base_def = (h_def * (1-w_split)) + (h_def_home * w_split)
        
        a_base_att = (a_att * (1-w_split)) + (a_att_away * w_split)
        a_base_def = (a_def * (1-w_split)) + (a_def_away * w_split)

        # Integrazione Forma (Ora h_form_att è una MEDIA, quindi il calcolo forza è coerente)
        h_final_att, h_final_def = calcola_forza_squadra(h_base_att, h_base_def, h_form_att, h_form_def, w_seas)
        a_final_att, a_final_def = calcola_forza_squadra(a_base_att, a_base_def, a_form_att, a_form_def, w_seas)
        
        xg_h_stats = (h_final_att * a_final_def) / avg_goals_league
        xg_a_stats = (a_final_att * h_final_def) / avg_goals_league
        
        elo_diff = (h_elo + (100 if m_type=="Standard" else 0)) - a_elo
        elo_factor_h = 1 + (elo_diff / 1000.0)
        elo_factor_a = 1 - (elo_diff / 1000.0)
        
        f_xh = (xg_h_stats * elo_factor_h) + home_adv_goals
        f_xa = (xg_a_stats * elo_factor_a)
        
        fatigue_malus = 0.05 
        if h_rest <= 3: f_xh *= (1 - fatigue_malus); f_xa *= (1 + fatigue_malus) 
        if a_rest <= 3: f_xa *= (1 - fatigue_malus); f_xh *= (1 + fatigue_malus)
        f_xh *= (h_str/100.0); f_xa *= (a_str/100.0)
        
        if is_big_match: f_xh *= 0.90; f_xa *= 0.90
        if h_m_a: f_xh *= 0.85
        if h_m_d: f_xa *= 1.20
        if a_m_a: f_xa *= 0.85
        if a_m_d: f_xh *= 1.20

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

# --- OUTPUT VISIVO ---
if st.session_state.analyzed:
    st.markdown("---")
    st.header(f"📊 {st.session_state.h_name} vs {st.session_state.a_name}")
    
    col_m1, col_m2 = st.columns(2)
    col_m1.metric("Expected Goals (xG)", f"{st.session_state.f_xh:.2f} - {st.session_state.f_xa:.2f}")
    col_m2.metric("Stabilità", f"{st.session_state.stability:.1f}%")

    tab1, tab2, tab3, tab4 = st.tabs(["🏆 Esito", "⚽ Gol", "👤 Marcatori", "📝 Storico"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("1X2")
            df_res = pd.DataFrame({
                "Esito": ["1", "X", "2"],
                "Prob %": [f"{st.session_state.p1:.1%}", f"{st.session_state.pX:.1%}", f"{st.session_state.p2:.1%}"],
                "Fair Odd": [f"{1/st.session_state.p1:.2f}", f"{1/st.session_state.pX:.2f}", f"{1/st.session_state.p2:.2f}"],
                "Bookie": [st.session_state.b1, st.session_state.bX, st.session_state.b2],
                "Value": [f"{(st.session_state.b1*st.session_state.p1-1):.1%}", 
                          f"{(st.session_state.bX*st.session_state.pX-1):.1%}", 
                          f"{(st.session_state.b2*st.session_state.p2-1):.1%}"]
            })
            st.dataframe(df_res, hide_index=True)
        with c2:
            st.subheader("Risultati Esatti")
            scores = st.session_state.scores
            scores.sort(key=lambda x: x["Prob"], reverse=True)
            df_sc = pd.DataFrame([{ "Score": s["Risultato"], "Prob": f"{s['Prob']:.1%}", "Quota": f"{1/s['Prob']:.2f}"} for s in scores[:6]])
            st.dataframe(df_sc, hide_index=True)
            
            fig, ax = plt.subplots(figsize=(4,3))
            sns.heatmap(st.session_state.matrix[:5,:5], annot=True, fmt=".0%", cmap="Greens", cbar=False)
            plt.xlabel(st.session_state.a_name); plt.ylabel(st.session_state.h_name)
            st.pyplot(fig)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Under / Over")
            uo_res = []
            for l in [0.5, 1.5, 2.5, 3.5, 4.5]:
                p_pure = np.sum(st.session_state.matrix[np.indices((10,10))[0] + np.indices((10,10))[1] > l])
                trend = (h_uo_input.get(l,50) + a_uo_input.get(l,50))/200.0
                p_final = (p_pure * 0.7) + (trend * 0.3) 
                uo_res.append({"Linea": l, "Over %": f"{p_final:.1%}", "Quota O": f"{1/p_final:.2f}", "Under %": f"{(1-p_final):.1%}", "Quota U": f"{1/(1-p_final):.2f}"})
            st.dataframe(pd.DataFrame(uo_res), hide_index=True)
        with c2:
            st.subheader("Goal / No Goal")
            st.dataframe(pd.DataFrame([
                {"Esito": "Goal (GG)", "Prob": f"{st.session_state.pGG:.1%}", "Quota": f"{1/st.session_state.pGG:.2f}"},
                {"Esito": "No Goal (NG)", "Prob": f"{(1-st.session_state.pGG):.1%}", "Quota": f"{1/(1-st.session_state.pGG):.2f}"}
            ]), hide_index=True)

    with tab3:
        st.subheader("Marcatori")
        pl_n = st.text_input("Giocatore", "Lautaro")
        c_p1, c_p2 = st.columns(2)
        pl_xg = c_p1.number_input("xG/90", 0.01, 2.0, 0.50)
        pl_min = c_p2.number_input("Minuti", 1, 100, 85)
        # Usa xG della squadra calcolati
        team_xg = st.session_state.f_xh if st.checkbox("È squadra di casa?", value=True) else st.session_state.f_xa
        p_goal = calculate_player_probability(pl_xg, pl_min, team_xg, 1.4)
        st.metric(f"Prob Goal {pl_n}", f"{p_goal:.1%}", f"Quota Fair: {1/p_goal:.2f}")

    with tab4:
        st.subheader("Storico")
        if st.button("💾 Salva"):
            st.session_state.history.append({"Match": f"{st.session_state.h_name}-{st.session_state.a_name}", "P1": st.session_state.p1, "Esito": "?"})
            salva_storico_json()
            st.success("Salvato!")
