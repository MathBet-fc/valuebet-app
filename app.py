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
import io
from datetime import datetime

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Mathbet fc - ML Ultimate Pro", page_icon="🧠", layout="wide")

# ==============================================================================
# 📂 CONFIGURAZIONE FILE CSV
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
    "🌐 Generico (Default)": { "avg": 1.35, "ha": 0.25, "rho": -0.08 },
    "🇮🇹 Serie A":          { "avg": 1.28, "ha": 0.059, "rho": -0.025 },
    "🇬🇧 Premier League":   { "avg": 1.47, "ha": 0.046, "rho": 0.005 },
    "🇪🇸 La Liga":          { "avg": 1.31, "ha": 0.143, "rho": 0.050 },
    "🇩🇪 Bundesliga":       { "avg": 1.57, "ha": 0.066, "rho": -0.080 },
    "🇫🇷 Ligue 1":          { "avg": 1.49, "ha": 0.120, "rho": -0.020 },
}

# ==============================================================================
# 🧠 FUNZIONI DATI & CALCOLO
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
                df = pd.read_csv(filename, sep=None, engine='python')
                df.columns = [c.strip().lower() for c in df.columns]
                return df
            except: return None
        return None

    def safe_avg(val, n): return float(val) / n if n > 0 else 0.0

    def extract_stats(row):
        goals = row.get('goals', row.get('gf', 0))
        ga = row.get('ga', row.get('gs', 0))
        xg = row.get('xg', 0)
        xga = row.get('xga', 0)
        matches = row.get('matches', 0)
        return {"gf": safe_avg(goals, matches), "gs": safe_avg(ga, matches), 
                "xg": safe_avg(xg, matches), "xga": safe_avg(xga, matches), "matches": matches}

    df_total = read_csv_safe(files["total"])
    df_home = read_csv_safe(files["home"])
    df_away = read_csv_safe(files["away"])
    df_form = read_csv_safe(files["form"])

    if df_total is None: return {}, []

    for _, row in df_total.iterrows():
        team_col = next((c for c in row.index if 'team' in c or 'squad' in c), None)
        if not team_col: continue
        team_name = str(row[team_col]).strip()
        total_stats = extract_stats(row)
        if total_stats["matches"] > 0:
            stats_db[team_name] = {"matches": total_stats["matches"], "total": total_stats, 
                                   "home": {"gf":0,"gs":0,"xg":0,"xga":0,"matches":0},
                                   "away": {"gf":0,"gs":0,"xg":0,"xga":0,"matches":0},
                                   "form": {"gf":0,"gs":0,"xg":0,"xga":0,"matches":0}}
            team_list.append(team_name)

    if df_home is not None:
        for _, row in df_home.iterrows():
            team_col = next((c for c in row.index if 'team' in c or 'squad' in c), None)
            if team_col:
                t = str(row[team_col]).strip()
                if t in stats_db: stats_db[t]["home"] = extract_stats(row)
    if df_away is not None:
        for _, row in df_away.iterrows():
            team_col = next((c for c in row.index if 'team' in c or 'squad' in c), None)
            if team_col:
                t = str(row[team_col]).strip()
                if t in stats_db: stats_db[t]["away"] = extract_stats(row)
    if df_form is not None:
        for _, row in df_form.iterrows():
            team_col = next((c for c in row.index if 'team' in c or 'squad' in c), None)
            if team_col:
                t = str(row[team_col]).strip()
                if t in stats_db: stats_db[t]["form"] = extract_stats(row)

    team_list.sort()
    return stats_db, team_list

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

# --- NUOVE FUNZIONI STATS EXTRA ---
def poisson_probability(k, lamb):
    return (math.exp(-lamb) * (lamb**k)) / math.factorial(k)

def calculate_stats_probs(avg_h, avg_a):
    p1, pX, p2 = 0, 0, 0
    limit = int((avg_h + avg_a) * 3) + 5
    for h in range(limit):
        for a in range(limit):
            prob = poisson_probability(h, avg_h) * poisson_probability(a, avg_a)
            if h > a: p1 += prob
            elif h == a: pX += prob
            else: p2 += prob
    
    avg_tot = avg_h + avg_a
    lines = {}
    base_line = round(avg_tot)
    for line in [base_line-1.5, base_line-0.5, base_line+0.5, base_line+1.5]:
        if line < 0: continue
        p_under = sum(poisson_probability(k, avg_tot) for k in range(int(line) + 1))
        p_over = 1 - p_under
        lines[f"Over {line}"] = {"prob": p_over, "odd": 1/p_over if p_over > 0 else 999}
    return p1, pX, p2, lines

@st.cache_data
def monte_carlo_simulation(f_xh, f_xa, n_sims=5000):
    sim = []
    for _ in range(n_sims):
        gh = np.random.poisson(max(0.1, np.random.normal(f_xh, 0.20*f_xh)))
        ga = np.random.poisson(max(0.1, np.random.normal(f_xa, 0.20*f_xa)))
        sim.append(1 if gh>ga else (0 if gh==ga else 2))
    return sim

def calcola_forza_squadra(att_season, def_season, att_form, def_form, w_season):
    att = (att_season * w_season) + (att_form * (1-w_season))
    def_ = (def_season * w_season) + (def_form * (1-w_season))
    return att, def_

def carica_storico_json():
    try:
        with open('mathbet_history.json', 'r') as f: return json.load(f)
    except FileNotFoundError: return []

def salva_storico_json():
    try:
        with open('mathbet_history.json', 'w') as f: json.dump(st.session_state.history, f, indent=2, default=str)
        return True
    except Exception: return False

def generate_excel_report():
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        if st.session_state.analyzed:
            data_main = {
                "Parametro": ["Partita", "Data", "Campionato", "xG Casa", "xG Ospite", "Stabilità", "Prob 1", "Prob X", "Prob 2"],
                "Valore": [
                    f"{st.session_state.h_name} - {st.session_state.a_name}",
                    datetime.now().strftime("%Y-%m-%d"),
                    st.session_state.get('league_name', 'N/A'),
                    round(st.session_state.f_xh, 2),
                    round(st.session_state.f_xa, 2),
                    f"{st.session_state.stability:.1f}%",
                    f"{st.session_state.p1:.1%}",
                    f"{st.session_state.pX:.1%}",
                    f"{st.session_state.p2:.1%}"
                ]
            }
            pd.DataFrame(data_main).to_excel(writer, sheet_name='Analisi Match', index=False)
            data_odds = {
                "Esito": ["1", "X", "2"],
                "Prob %": [st.session_state.p1, st.session_state.pX, st.session_state.p2],
                "Quota Fair": [1/st.session_state.p1, 1/st.session_state.pX, 1/st.session_state.p2],
                "Quota Bookie": [st.session_state.b1, st.session_state.bX, st.session_state.b2],
                "Value Bet %": [(st.session_state.b1 * st.session_state.p1) - 1, (st.session_state.bX * st.session_state.pX) - 1, (st.session_state.b2 * st.session_state.p2) - 1]
            }
            pd.DataFrame(data_odds).to_excel(writer, sheet_name='Analisi Match', startrow=12, index=False)
            scores = st.session_state.scores.copy()
            scores.sort(key=lambda x: x["Prob"], reverse=True)
            pd.DataFrame(scores[:10]).to_excel(writer, sheet_name='Analisi Match', startrow=18, index=False)
        if st.session_state.history:
            df_hist = pd.DataFrame(st.session_state.history)
            df_hist.to_excel(writer, sheet_name='Storico Completo', index=False)
    return output.getvalue()

# --- INIZIALIZZAZIONE ---
if 'history' not in st.session_state: st.session_state.history = carica_storico_json()
if 'analyzed' not in st.session_state: st.session_state.analyzed = False

# --- UI ---
with st.sidebar:
    st.title("🧠 Configurazione")
    league_name = st.selectbox("Campionato", list(LEAGUES.keys()))
    st.session_state.league_name = league_name
    L_DATA = LEAGUES[league_name]
    
    STATS_DB, TEAM_LIST = load_league_stats(league_name)
    if STATS_DB: st.success(f"✅ Dati caricati! ({len(TEAM_LIST)} squadre)")
    else: st.error("❌ CSV non trovati.")
    st.markdown("---")
    
    data_mode = st.radio("Dati Analisi", ["Solo Gol Reali", "Solo xG (Expected Goals)", "Ibrido (Consigliato)"], index=2)
    st.markdown("---")
    st.subheader("⚡ Dinamica")
    volatility = st.slider("Volatilità", 0.8, 1.4, 1.0, 0.05)
    matchday = st.slider("Giornata", 1, 38, 22)
    w_seas = min(0.90, 0.30 + (matchday * 0.02)) 
    st.markdown("---")
    m_type = st.radio("Contesto", ["Standard", "Derby", "Campo Neutro"])
    is_big_match = st.checkbox("🔥 Big Match")
    
    # --- NUOVO INPUT STATS EXTRA ---
    st.markdown("---")
    with st.expander("🔢 Dati Extra Manuali (Stats)", expanded=False):
        st.caption("Inserisci le medie previste (Casa vs Ospite)")
        c_corn_h = st.number_input("Angoli Casa (Avg)", 0.0, 20.0, 5.5, 0.1)
        c_corn_a = st.number_input("Angoli Ospite (Avg)", 0.0, 20.0, 4.5, 0.1)
        st.divider()
        c_card_h = st.number_input("Cartellini Casa (Avg)", 0.0, 10.0, 2.0, 0.1)
        c_card_a = st.number_input("Cartellini Ospite (Avg)", 0.0, 10.0, 2.5, 0.1)
        st.divider()
        c_shot_h = st.number_input("Tiri Totali Casa", 0.0, 50.0, 12.0, 0.5)
        c_shot_a = st.number_input("Tiri Totali Ospite", 0.0, 50.0, 10.0, 0.5)
        st.divider()
        c_sot_h = st.number_input("Tiri in Porta Casa", 0.0, 20.0, 4.5, 0.1)
        c_sot_a = st.number_input("Tiri in Porta Ospite", 0.0, 20.0, 3.5, 0.1)
        st.divider()
        c_foul_h = st.number_input("Falli Casa", 0.0, 30.0, 11.5, 0.5)
        c_foul_a = st.number_input("Falli Ospite", 0.0, 30.0, 12.5, 0.5)

st.title("Mathbet fc - ML Ultimate Edition 🚀")

col_h, col_a = st.columns(2)
h_uo_input, a_uo_input = {}, {}

def get_val(stats_dict, metric, mode):
    val_gol = stats_dict.get('gf' if metric == 'gf' else 'gs', 0.0)
    val_xg = stats_dict.get('xg' if metric == 'gf' else 'xga', 0.0)
    if mode == "Solo Gol Reali": return val_gol
    if mode == "Solo xG (Expected Goals)": return val_xg
    return (val_gol + val_xg) / 2.0

def get_form_val(stats_dict, metric, mode):
    form_data = stats_dict.get("form", {})
    matches_l5 = form_data.get("matches", 0)
    if matches_l5 > 0:
        val_gol_avg = form_data.get('gf_avg' if metric == 'gf' else 'gs_avg', 0.0)
        val_xg_avg = form_data.get('xg_avg' if metric == 'gf' else 'xga_avg', 0.0)
        if mode == "Solo Gol Reali": avg = val_gol_avg
        elif mode == "Solo xG (Expected Goals)": avg = val_xg_avg
        else: avg = (val_gol_avg + val_xg_avg) / 2.0
        return avg * 5.0
    season_avg_gol = stats_dict["total"].get('gf' if metric == 'gf' else 'gs', 0.0)
    season_avg_xg = stats_dict["total"].get('xg' if metric == 'gf' else 'xga', 0.0)
    if mode == "Solo Gol Reali": base_val = season_avg_gol
    elif mode == "Solo xG (Expected Goals)": base_val = season_avg_xg
    else: base_val = (season_avg_gol + season_avg_xg) / 2.0
    return base_val * 5.0

with col_h:
    st.subheader("🏠 Squadra Casa")
    h_idx = TEAM_LIST.index("Inter") if TEAM_LIST and "Inter" in TEAM_LIST else 0
    h_name = st.selectbox("Seleziona Casa", TEAM_LIST, index=h_idx, key="h_sel") if TEAM_LIST else st.text_input("Nome Casa", "Inter")
    h_stats = STATS_DB.get(h_name) if STATS_DB else None
    h_elo = st.number_input("Rating Elo Casa", 1000.0, 2500.0, 1600.0, step=10.0)
    
    with st.expander("📊 Dati", expanded=True):
        def_att_s, def_def_s, def_att_h, def_def_h, def_form_att, def_form_def = 1.85, 0.95, 1.95, 0.85, 9.0, 4.0
        if h_stats:
            def_att_s = get_val(h_stats["total"], 'gf', data_mode); def_def_s = get_val(h_stats["total"], 'gs', data_mode)
            if h_stats["home"]["matches"] > 0: def_att_h = get_val(h_stats["home"], 'gf', data_mode); def_def_h = get_val(h_stats["home"], 'gs', data_mode)
            else: def_att_h = def_att_s * 1.15; def_def_h = def_def_s * 0.85
            def_form_att = get_form_val(h_stats, 'gf', data_mode); def_form_def = get_form_val(h_stats, 'gs', data_mode)

        c1, c2 = st.columns(2)
        h_att = c1.number_input("Attacco Totale (C)", 0.0, 5.0, float(def_att_s), 0.01)
        h_def = c2.number_input("Difesa Totale (C)", 0.0, 5.0, float(def_def_s), 0.01)
        c3, c4 = st.columns(2)
        h_att_home = c3.number_input("Attacco Casa", 0.0, 5.0, float(def_att_h), 0.01)
        h_def_home = c4.number_input("Difesa Casa", 0.0, 5.0, float(def_def_h), 0.01)
        h_form_att = st.number_input("Attacco L5 (C)", 0.0, 25.0, float(def_form_att), 0.1)
        h_form_def = st.number_input("Difesa L5 (C)", 0.0, 25.0, float(def_form_def), 0.1)

    with st.expander("Over Trend"):
        for l in [0.5, 1.5, 2.5, 3.5, 4.5]: h_uo_input[l] = st.slider(f"Over {l} % H", 0, 100, 50, key=f"ho{l}")

with col_a:
    st.subheader("✈️ Squadra Ospite")
    a_idx = 1 if TEAM_LIST and len(TEAM_LIST) > 1 else 0
    a_name = st.selectbox("Seleziona Ospite", TEAM_LIST, index=a_idx, key="a_sel") if TEAM_LIST else st.text_input("Nome Ospite", "Juve")
    a_stats = STATS_DB.get(a_name) if STATS_DB else None
    a_elo = st.number_input("Rating Elo Ospite", 1000.0, 2500.0, 1550.0, step=10.0)

    with st.expander("📊 Dati", expanded=True):
        def_att_s_a, def_def_s_a, def_att_a, def_def_a, def_form_att_a, def_form_def_a = 1.45, 0.85, 1.25, 1.05, 7.0, 4.0
        if a_stats:
            def_att_s_a = get_val(a_stats["total"], 'gf', data_mode); def_def_s_a = get_val(a_stats["total"], 'gs', data_mode)
            if a_stats["away"]["matches"] > 0: def_att_a = get_val(a_stats["away"], 'gf', data_mode); def_def_a = get_val(a_stats["away"], 'gs', data_mode)
            else: def_att_a = def_att_s_a * 0.85; def_def_a = def_def_s_a * 1.15
            def_form_att_a = get_form_val(a_stats, 'gf', data_mode); def_form_def_a = get_form_val(a_stats, 'gs', data_mode)

        c5, c6 = st.columns(2)
        a_att = c5.number_input("Attacco Totale (O)", 0.0, 5.0, float(def_att_s_a), 0.01)
        a_def = c6.number_input("Difesa Totale (O)", 0.0, 5.0, float(def_def_s_a), 0.01)
        c7, c8 = st.columns(2)
        a_att_away = c7.number_input("Attacco Fuori", 0.0, 5.0, float(def_att_a), 0.01)
        a_def_away = c8.number_input("Difesa Fuori", 0.0, 5.0, float(def_def_a), 0.01)
        a_form_att = st.number_input("Attacco L5 (O)", 0.0, 25.0, float(def_form_att_a), 0.1)
        a_form_def = st.number_input("Difesa L5 (O)", 0.0, 25.0, float(def_form_def_a), 0.1)

    with st.expander("Over Trend"):
        for l in [0.5, 1.5, 2.5, 3.5, 4.5]: a_uo_input[l] = st.slider(f"Over {l} % A", 0, 100, 50, key=f"ao{l}")

st.subheader("💰 Quote")
q1, qx, q2 = st.columns(3)
b1 = q1.number_input("Q1", 1.01, 100.0, 2.10)
bX = qx.number_input("QX", 1.01, 100.0, 3.20)
b2 = q2.number_input("Q2", 1.01, 100.0, 3.60)

with st.expander("⚙️ Fine Tuning"):
    c1, c2 = st.columns(2)
    h_str = c1.slider("Titolari % Casa", 50, 100, 100); a_str = c2.slider("Titolari % Ospite", 50, 100, 100)
    h_rest = c1.slider("Riposo Casa", 2, 10, 7); a_rest = c2.slider("Riposo Ospite", 2, 10, 7)
    h_m_a = c1.checkbox("No Bomber Casa"); a_m_a = c2.checkbox("No Bomber Ospite")
    h_m_d = c1.checkbox("No Difensore Casa"); a_m_d = c2.checkbox("No Difensore Ospite")

if st.button("🚀 ANALIZZA", type="primary", use_container_width=True):
    with st.spinner("Calcolo in corso..."):
        home_adv = L_DATA["ha"] if m_type == "Standard" else (0.0 if m_type == "Campo Neutro" else L_DATA["ha"]*0.5)
        w_split = 0.60
        h_base_att = (h_att*(1-w_split)) + (h_att_home*w_split); h_base_def = (h_def*(1-w_split)) + (h_def_home*w_split)
        a_base_att = (a_att*(1-w_split)) + (a_att_away*w_split); a_base_def = (a_def*(1-w_split)) + (a_def_away*w_split)
        h_fin_att, h_fin_def = calcola_forza_squadra(h_base_att, h_base_def, h_form_att, h_form_def, w_seas)
        a_fin_att, a_fin_def = calcola_forza_squadra(a_base_att, a_base_def, a_form_att, a_form_def, w_seas)
        xg_h = (h_fin_att * a_fin_def) / L_DATA["avg"]; xg_a = (a_fin_att * h_fin_def) / L_DATA["avg"]
        elo_diff = (h_elo + (100 if m_type=="Standard" else 0)) - a_elo
        f_xh = (xg_h * (1 + elo_diff/1000.0)) + home_adv
        f_xa = (xg_a * (1 - elo_diff/1000.0))
        f_xh *= volatility; f_xa *= volatility
        if h_rest <= 3: f_xh*=0.95; f_xa*=1.05
        if a_rest <= 3: f_xa*=0.95; f_xh*=1.05
        f_xh *= h_str/100.0; f_xa *= a_str/100.0
        if is_big_match: f_xh*=0.9; f_xa*=0.9
        if h_m_a: f_xh*=0.85; 
        if h_m_d: f_xa*=1.20
        if a_m_a: f_xa*=0.85; 
        if a_m_d: f_xh*=1.20

        matrix = np.zeros((10,10)); scores = []
        p1, pX, p2, pGG = 0,0,0,0
        for h in range(10):
            for a in range(10):
                p = dixon_coles_probability(h, a, f_xh, f_xa, L_DATA["rho"])
                matrix[h,a] = p
                if h>a: p1+=p
                elif h==a: pX+=p
                else: p2+=p
                if h>0 and a>0: pGG+=p
                if h<6 and a<6: scores.append({"Risultato": f"{h}-{a}", "Prob": p})
        tot = np.sum(matrix); matrix/=tot; p1/=tot; pX/=tot; p2/=tot; pGG/=tot
        sim = monte_carlo_simulation(f_xh, f_xa)
        s1, sX, s2 = sim.count(1)/5000, sim.count(0)/5000, sim.count(2)/5000
        stability = max(0, 100 - ((abs(p1-s1)+abs(pX-sX)+abs(p2-s2))/3*400))

        # --- CALCOLO STATS EXTRA ---
        corn_1, corn_X, corn_2, corn_lines = calculate_stats_probs(c_corn_h, c_corn_a)
        card_1, card_X, card_2, card_lines = calculate_stats_probs(c_card_h, c_card_a)
        shot_1, shot_X, shot_2, shot_lines = calculate_stats_probs(c_shot_h, c_shot_a)
        sot_1, sot_X, sot_2, sot_lines = calculate_stats_probs(c_sot_h, c_sot_a)
        foul_1, foul_X, foul_2, foul_lines = calculate_stats_probs(c_foul_h, c_foul_a)

        st.session_state.analyzed = True
        st.session_state.f_xh = f_xh; st.session_state.f_xa = f_xa
        st.session_state.h_name = h_name; st.session_state.a_name = a_name
        st.session_state.p1 = p1; st.session_state.pX = pX; st.session_state.p2 = p2
        st.session_state.pGG = pGG; st.session_state.stability = stability
        st.session_state.matrix = matrix; st.session_state.scores = scores
        st.session_state.b1 = b1; st.session_state.bX = bX; st.session_state.b2 = b2
        st.session_state.stats = {
            "corners": {"1": corn_1, "X": corn_X, "2": corn_2, "lines": corn_lines},
            "cards": {"1": card_1, "X": card_X, "2": card_2, "lines": card_lines},
            "shots": {"1": shot_1, "X": shot_X, "2": shot_2, "lines": shot_lines},
            "sot": {"1": sot_1, "X": sot_X, "2": sot_2, "lines": sot_lines},
            "fouls": {"1": foul_1, "X": foul_X, "2": foul_2, "lines": foul_lines}
        }

if st.session_state.analyzed:
    st.markdown("---")
    st.header(f"📊 {st.session_state.h_name} vs {st.session_state.a_name}")
    c1, c2 = st.columns(2)
    c1.metric("xG Previsti", f"{st.session_state.f_xh:.2f} - {st.session_state.f_xa:.2f}")
    c2.metric("Affidabilità", f"{st.session_state.stability:.1f}%")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏆 Esito", "⚽ Gol/Multigol", "👤 Player", "⛳ Stats Extra", "📝 Storico"])
    
    with tab1:
        c_1, c_2 = st.columns(2)
        with c_1:
            st.subheader("1X2 & Doppia Chance")
            res_df = pd.DataFrame({
                "Esito": ["1", "X", "2", "1X", "X2", "12"],
                "Prob %": [f"{st.session_state.p1:.1%}", f"{st.session_state.pX:.1%}", f"{st.session_state.p2:.1%}", f"{(st.session_state.p1+st.session_state.pX):.1%}", f"{(st.session_state.pX+st.session_state.p2):.1%}", f"{(st.session_state.p1+st.session_state.p2):.1%}"],
                "Quota": [f"{1/st.session_state.p1:.2f}", f"{1/st.session_state.pX:.2f}", f"{1/st.session_state.p2:.2f}", f"{1/(st.session_state.p1+st.session_state.pX):.2f}", f"{1/(st.session_state.pX+st.session_state.p2):.2f}", f"{1/(st.session_state.p1+st.session_state.p2):.2f}"]
            })
            st.dataframe(res_df, hide_index=True)
            
            st.subheader("Value Bet (vs Bookie)")
            val_df = pd.DataFrame({
                "Esito": ["1", "X", "2"],
                "Tua Quota": [f"{1/st.session_state.p1:.2f}", f"{1/st.session_state.pX:.2f}", f"{1/st.session_state.p2:.2f}"],
                "Bookie": [st.session_state.b1, st.session_state.bX, st.session_state.b2],
                "Valore": [f"{(st.session_state.b1*st.session_state.p1-1):.1%}", f"{(st.session_state.bX*st.session_state.pX-1):.1%}", f"{(st.session_state.b2*st.session_state.p2-1):.1%}"]
            })
            st.dataframe(val_df.style.applymap(lambda x: "background-color: #d4edda" if "%" in str(x) and "-" not in str(x) and str(x) != "0.0%" else "", subset=["Valore"]), hide_index=True)

        with c_2:
            st.subheader("Risultati Esatti")
            scores = st.session_state.scores; scores.sort(key=lambda x: x["Prob"], reverse=True)
            st.dataframe(pd.DataFrame([{"Risultato": s["Risultato"], "Prob": f"{s['Prob']:.1%}"} for s in scores[:6]]), hide_index=True)
            
            st.subheader("Heatmap Punteggi")
            fig, ax = plt.subplots(figsize=(5,4))
            sns.heatmap(st.session_state.matrix[:6,:6], annot=True, fmt=".0%", cmap="Greens", cbar=False,
                        xticklabels=[x for x in range(6)], yticklabels=[y for y in range(6)])
            plt.xlabel(st.session_state.a_name)
            plt.ylabel(st.session_state.h_name)
            st.pyplot(fig)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Under / Over")
            uo_list = []
            for l in [0.5, 1.5, 2.5, 3.5]:
                p_pure = np.sum(st.session_state.matrix[np.indices((10,10))[0] + np.indices((10,10))[1] > l])
                trend = (h_uo_input.get(l,50) + a_uo_input.get(l,50))/200.0
                pf = (p_pure*0.7)+(trend*0.3)
                uo_list.append({"Linea": f"Over {l}", "Prob %": f"{pf:.1%}", "Quota": f"{1/pf:.2f}"})
            st.dataframe(pd.DataFrame(uo_list), hide_index=True)
            st.write(f"**Goal / Goal:** {st.session_state.pGG:.1%} (@{1/st.session_state.pGG:.2f})")
            
            st.subheader("Handicap (-1)")
            h_hand = np.sum(st.session_state.matrix[np.indices((10,10))[0] - 1 > np.indices((10,10))[1]])
            a_hand = np.sum(st.session_state.matrix[np.indices((10,10))[0] + 1 < np.indices((10,10))[1]])
            st.write(f"**1H (-1):** {h_hand:.1%} (@{1/h_hand:.2f})")
            st.write(f"**2H (-1):** {a_hand:.1%} (@{1/a_hand:.2f})")

        with c2:
            st.subheader("Multigol")
            mg_res = []
            for r in [(1,2), (1,3), (2,3), (2,4), (3,4), (3,5)]:
                pm = np.sum(st.session_state.matrix[(np.indices((10,10))[0] + np.indices((10,10))[1] >= r[0]) & (np.indices((10,10))[0] + np.indices((10,10))[1] <= r[1])])
                mg_res.append({"Range": f"{r[0]}-{r[1]}", "Prob %": f"{pm:.1%}", "Quota": f"{1/pm:.2f}"})
            st.dataframe(pd.DataFrame(mg_res), hide_index=True)

    with tab3:
        pl_n = st.text_input("Nome", "Vlahovic")
        c1, c2 = st.columns(2)
        pxg = c1.number_input("xG/90", 0.0, 2.0, 0.5); pmin = c2.number_input("Minuti", 1, 100, 90)
        txg = st.session_state.f_xh if st.checkbox("Casa?") else st.session_state.f_xa
        pprob = calculate_player_probability(pxg, pmin, txg, 1.4)
        st.metric(f"Gol {pl_n}", f"{pprob:.1%}", f"Quota: {1/pprob:.2f}")

    with tab4:
        st.subheader("⛳ Angoli, Cartellini e Tiri")
        c_stats_1, c_stats_2 = st.columns(2)
        with c_stats_1:
            st.markdown("### 🚩 Angoli")
            corn = st.session_state.stats["corners"]
            st.table(pd.DataFrame({"Esito": ["1", "X", "2"], "Prob %": [f"{corn['1']:.1%}", f"{corn['X']:.1%}", f"{corn['2']:.1%}"], "Fair Odd": [f"{1/corn['1']:.2f}", f"{1/corn['X']:.2f}", f"{1/corn['2']:.2f}"]}))
            st.write("**Totale Angoli**")
            st.dataframe(pd.DataFrame([{"Linea": k, "Over %": f"{v['prob']:.1%}", "Quota": f"{v['odd']:.2f}"} for k,v in corn["lines"].items()]), hide_index=True)

            st.markdown("### 🟨 Cartellini")
            card = st.session_state.stats["cards"]
            st.table(pd.DataFrame({"Esito": ["1", "X", "2"], "Prob %": [f"{card['1']:.1%}", f"{card['X']:.1%}", f"{card['2']:.1%}"], "Fair Odd": [f"{1/card['1']:.2f}", f"{1/card['X']:.2f}", f"{1/card['2']:.2f}"]}))
            st.dataframe(pd.DataFrame([{"Linea": k, "Over %": f"{v['prob']:.1%}", "Quota": f"{v['odd']:.2f}"} for k,v in card["lines"].items()]), hide_index=True)

        with c_stats_2:
            st.markdown("### 🥅 Tiri Totali")
            shot = st.session_state.stats["shots"]
            st.table(pd.DataFrame({"Esito": ["1", "X", "2"], "Prob %": [f"{shot['1']:.1%}", f"{shot['X']:.1%}", f"{shot['2']:.1%}"], "Quota": [f"{1/shot['1']:.2f}", f"{1/shot['X']:.2f}", f"{1/shot['2']:.2f}"]}))
            st.dataframe(pd.DataFrame([{"Linea": k, "Over %": f"{v['prob']:.1%}", "Quota": f"{v['odd']:.2f}"} for k,v in shot["lines"].items()]), hide_index=True)

            st.markdown("### 🎯 Tiri in Porta")
            sot = st.session_state.stats["sot"]
            st.dataframe(pd.DataFrame([{"Linea": k, "Over %": f"{v['prob']:.1%}", "Quota": f"{v['odd']:.2f}"} for k,v in sot["lines"].items()]), hide_index=True)
            
            st.markdown("### 🛑 Falli")
            foul = st.session_state.stats["fouls"]
            st.dataframe(pd.DataFrame([{"Linea": k, "Over %": f"{v['prob']:.1%}", "Quota": f"{v['odd']:.2f}"} for k,v in foul["lines"].items()]), hide_index=True)

    with tab5:
        c1, c2 = st.columns(2)
        if c1.button("💾 Salva in Storico"):
            st.session_state.history.append({"Match": f"{st.session_state.h_name}-{st.session_state.a_name}", "xG": f"{st.session_state.f_xh:.2f}-{st.session_state.f_xa:.2f}", "P1": st.session_state.p1})
            salva_storico_json()
            st.success("Salvato!")
        
        excel_data = generate_excel_report()
        c2.download_button("📥 Scarica Report Excel", excel_data, f"Mathbet_{datetime.now().strftime('%H%M')}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
