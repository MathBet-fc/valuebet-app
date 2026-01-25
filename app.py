import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import math
import re
from bs4 import BeautifulSoup
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Mathbet fc - Understat Live", page_icon="📊", layout="wide")

# ==============================================================================
# 🕵️‍♂️ UNDERSTAT SCRAPER ENGINE
# ==============================================================================
# Mappatura dei campionati sugli URL di Understat
UNDERSTAT_LEAGUES = {
    "🇮🇹 Serie A": "Serie_A",
    "🇬🇧 Premier League": "EPL",
    "🇪🇸 La Liga": "La_liga",
    "🇩🇪 Bundesliga": "Bundesliga",
    "🇫🇷 Ligue 1": "Ligue_1"
}

@st.cache_data(ttl=3600) # Aggiorna ogni ora
def scrape_understat(league_name):
    """
    Si collega a Understat, scarica i dati JSON delle partite e calcola le statistiche.
    """
    if league_name not in UNDERSTAT_LEAGUES:
        return {}, []

    league_slug = UNDERSTAT_LEAGUES[league_name]
    # Determina l'anno della stagione (es. Agosto 2024 -> stagione 2024)
    season_year = datetime.now().year
    if datetime.now().month < 7: # Se siamo a Gennaio-Giugno, la stagione è iniziata l'anno prima
        season_year -= 1
        
    url = f"https://understat.com/league/{league_slug}/{season_year}"
    
    try:
        response = requests.get(url)
        if response.status_code != 200:
            st.error(f"Errore connessione Understat: {response.status_code}")
            return {}, []
            
        soup = BeautifulSoup(response.content, 'lxml')
        scripts = soup.find_all('script')
        
        # Cerca lo script che contiene "datesData" (i dati delle partite)
        matches_data = []
        for script in scripts:
            if 'datesData' in script.text:
                # Estrae il JSON dalla stringa JavaScript
                match = re.search(r"JSON\.parse\('([^']+)'\)", script.text)
                if match:
                    json_str = match.group(1).encode('utf-8').decode('unicode_escape')
                    matches_data = json.loads(json_str)
                    break
        
        if not matches_data:
            return {}, []

        # --- ELABORAZIONE DATI ---
        stats_db = {}
        team_names = set()
        
        # Filtra solo le partite giocate (quelle che hanno i gol segnati)
        played_matches = [m for m in matches_data if m['isResult'] == True]
        
        # Crea struttura dati per ogni squadra
        for m in played_matches:
            h_team = m['h']['title']
            a_team = m['a']['title']
            team_names.add(h_team)
            team_names.add(a_team)
            
            # Inizializza se non esiste
            if h_team not in stats_db: stats_db[h_team] = {'home': [], 'away': [], 'all': []}
            if a_team not in stats_db: stats_db[a_team] = {'home': [], 'away': [], 'all': []}
            
            # Estrai dati numerici
            h_goals = int(m['goals']['h'])
            a_goals = int(m['goals']['a'])
            h_xg = float(m['xG']['h'])
            a_xg = float(m['xG']['a'])
            date_match = m['datetime']
            
            # Aggiungi a Casa
            stats_db[h_team]['home'].append({'gf': h_goals, 'gs': a_goals, 'xg': h_xg, 'xga': a_xg, 'date': date_match})
            stats_db[h_team]['all'].append({'gf': h_goals, 'gs': a_goals, 'xg': h_xg, 'xga': a_xg, 'date': date_match})
            
            # Aggiungi a Ospite
            stats_db[a_team]['away'].append({'gf': a_goals, 'gs': h_goals, 'xg': a_xg, 'xga': h_xg, 'date': date_match})
            stats_db[a_team]['all'].append({'gf': a_goals, 'gs': h_goals, 'xg': a_xg, 'xga': h_xg, 'date': date_match})

        # Calcola le medie finali
        final_db = {}
        for team, data in stats_db.items():
            # Ordina tutte le partite per data per calcolare la forma
            all_sorted = sorted(data['all'], key=lambda x: x['date'])
            last_5 = all_sorted[-5:]
            
            # Helper per calcolare medie
            def get_mean(matches, key):
                if not matches: return 0.0
                return sum(m[key] for m in matches) / len(matches)
            
            def get_sum(matches, key):
                return sum(m[key] for m in matches)

            final_db[team] = {
                "matches": len(all_sorted),
                "total": {
                    "gf": get_mean(data['all'], 'gf'),
                    "gs": get_mean(data['all'], 'gs'),
                    "xg": get_mean(data['all'], 'xg'),
                    "xga": get_mean(data['all'], 'xga')
                },
                "home": {
                    "gf": get_mean(data['home'], 'gf'),
                    "gs": get_mean(data['home'], 'gs'),
                    "xg": get_mean(data['home'], 'xg'),
                    "xga": get_mean(data['home'], 'xga')
                },
                "away": {
                    "gf": get_mean(data['away'], 'gf'),
                    "gs": get_mean(data['away'], 'gs'),
                    "xg": get_mean(data['away'], 'xg'),
                    "xga": get_mean(data['away'], 'xga')
                },
                "form": {
                    "gf_l5": get_sum(last_5, 'gf'),
                    "gs_l5": get_sum(last_5, 'gs'),
                    "xg_l5": get_sum(last_5, 'xg'),
                    "xga_l5": get_sum(last_5, 'xga')
                }
            }
            
        return final_db, sorted(list(team_names))
        
    except Exception as e:
        st.error(f"Errore nello scraping: {e}")
        return {}, []

# --- PARAMETRI ML ---
LEAGUES = {
    "🌐 Generico (Default)": { "avg": 1.35, "ha": 0.25, "rho": -0.10 },
    "🇮🇹 Serie A":          { "avg": 1.28, "ha": 0.059, "rho": -0.032 },
    "🇬🇧 Premier League":   { "avg": 1.47, "ha": 0.046, "rho": 0.006 },
    "🇪🇸 La Liga":          { "avg": 1.31, "ha": 0.143, "rho": 0.060 },
    "🇩🇪 Bundesliga":       { "avg": 1.57, "ha": 0.066, "rho": -0.091 },
    "🇫🇷 Ligue 1":          { "avg": 1.49, "ha": 0.120, "rho": -0.026 },
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
    att = (att_season * w_season) + ((att_form/5.0) * (1-w_season))
    def_ = (def_season * w_season) + ((def_form/5.0) * (1-w_season))
    return att, def_

# --- INIZIALIZZAZIONE ---
if 'history' not in st.session_state: st.session_state.history = []
if 'analyzed' not in st.session_state: st.session_state.analyzed = False

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚡ Understat Auto")
    league_name = st.selectbox("Campionato", list(LEAGUES.keys()))
    L_DATA = LEAGUES[league_name]
    
    # 🔄 CARICAMENTO DATI LIVE DA UNDERSTAT
    if league_name in UNDERSTAT_LEAGUES:
        with st.spinner(f"Scaricamento dati {league_name} da Understat..."):
            STATS_DB, TEAM_LIST = scrape_understat(league_name)
        if STATS_DB:
            st.success(f"✅ Dati xG scaricati! ({len(TEAM_LIST)} squadre)")
        else:
            st.warning("⚠️ Impossibile scaricare dati. Controlla connessione.")
    else:
        STATS_DB, TEAM_LIST = {}, []

    st.markdown("---")
    
    # SELETTORE MODALITÀ DATI
    data_mode = st.radio(
        "Modalità Analisi Dati",
        ["Solo Gol Reali", "Solo xG (Expected Goals)", "Ibrido (50/50)"],
        index=2,
        help="Scegli se usare i Gol segnati veramente, gli Expected Goals (qualità tiri) o una media."
    )
    
    matchday = st.slider("Giornata (Peso Stagione)", 1, 38, 25)
    w_seas = min(0.90, 0.30 + (matchday * 0.02)) 
    
    st.markdown("---")
    m_type = st.radio("Contesto", ["Standard", "Derby", "Campo Neutro"])
    is_big_match = st.checkbox("🔥 Big Match")

st.title("Mathbet fc - Understat Live Edition 🚀")

# --- SELEZIONE SQUADRE ---
col_h, col_a = st.columns(2)
h_uo_input, a_uo_input = {}, {}

# Funzione helper per estrarre il valore giusto in base alla modalità
def get_val(stats_dict, metric, mode):
    # metric è 'gf', 'gs'
    val_gol = stats_dict.get(metric, 0.0)
    # metric diventa 'xg', 'xga'
    xg_key = 'xg' if metric == 'gf' else 'xga'
    val_xg = stats_dict.get(xg_key, 0.0)
    
    if mode == "Solo Gol Reali": return val_gol
    if mode == "Solo xG (Expected Goals)": return val_xg
    # Ibrido
    return (val_gol + val_xg) / 2.0

# SQUADRA CASA
with col_h:
    st.subheader("🏠 Squadra Casa")
    
    if TEAM_LIST:
        h_idx = 0
        # Cerca default
        for i, t in enumerate(TEAM_LIST):
            if "Inter" in t: h_idx = i; break
        h_name = st.selectbox("Seleziona Casa", TEAM_LIST, index=h_idx, key="h_sel")
        h_stats = STATS_DB[h_name]
    else:
        h_name = st.text_input("Nome Casa", "Inter")
        h_stats = None

    h_elo = st.number_input("Rating Elo Casa", 1000.0, 2500.0, 1600.0, step=10.0)
    
    with st.expander("📊 Dati Auto-Calcolati", expanded=True):
        def_att_s, def_def_s = 1.5, 1.0
        def_att_h, def_def_h = 1.6, 0.9
        def_form_att, def_form_def = 7.0, 4.0
        
        if h_stats:
            def_att_s = get_val(h_stats["total"], 'gf', data_mode)
            def_def_s = get_val(h_stats["total"], 'gs', data_mode)
            def_att_h = get_val(h_stats["home"], 'gf', data_mode)
            def_def_h = get_val(h_stats["home"], 'gs', data_mode)
            
            # Per la forma, possiamo usare Gol Reali o xG a seconda della modalità
            l5_gf_real = h_stats["form"]["gf_l5"]
            l5_xg = h_stats["form"]["xg_l5"]
            l5_gs_real = h_stats["form"]["gs_l5"]
            l5_xga = h_stats["form"]["xga_l5"]
            
            if data_mode == "Solo Gol Reali":
                def_form_att, def_form_def = l5_gf_real, l5_gs_real
            elif data_mode == "Solo xG (Expected Goals)":
                def_form_att, def_form_def = l5_xg, l5_xga
            else:
                def_form_att = (l5_gf_real + l5_xg) / 2
                def_form_def = (l5_gs_real + l5_xga) / 2

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
        st.caption("Forma Recente (Ultime 5)")
        h_form_att = st.number_input("Attacco L5 (C)", 0.0, 25.0, float(def_form_att), 0.5)
        h_form_def = st.number_input("Difesa L5 (C)", 0.0, 25.0, float(def_form_def), 0.5)

    with st.expander("📈 Trend Over"):
        for l in [0.5, 1.5, 2.5, 3.5, 4.5]: h_uo_input[l] = st.slider(f"Over {l} % H", 0, 100, 50, key=f"ho{l}")

# SQUADRA OSPITE
with col_a:
    st.subheader("✈️ Squadra Ospite")
    
    if TEAM_LIST:
        def_a_idx = 1 if len(TEAM_LIST) > 1 else 0
        for i, t in enumerate(TEAM_LIST):
            if "Juv" in t or "Mil" in t:
                def_a_idx = i; break
        
        a_name = st.selectbox("Seleziona Ospite", TEAM_LIST, index=def_a_idx, key="a_sel")
        a_stats = STATS_DB[a_name]
    else:
        a_name = st.text_input("Nome Ospite", "Juventus")
        a_stats = None

    a_elo = st.number_input("Rating Elo Ospite", 1000.0, 2500.0, 1550.0, step=10.0)

    with st.expander("📊 Dati Auto-Calcolati", expanded=True):
        def_att_s_a, def_def_s_a = 1.3, 1.0
        def_att_a, def_def_a = 1.2, 1.1
        def_form_att_a, def_form_def_a = 6.0, 5.0
        
        if a_stats:
            def_att_s_a = get_val(a_stats["total"], 'gf', data_mode)
            def_def_s_a = get_val(a_stats["total"], 'gs', data_mode)
            def_att_a = get_val(a_stats["away"], 'gf', data_mode)
            def_def_a = get_val(a_stats["away"], 'gs', data_mode)
            
            l5_gf_real_a = a_stats["form"]["gf_l5"]
            l5_xg_a = a_stats["form"]["xg_l5"]
            l5_gs_real_a = a_stats["form"]["gs_l5"]
            l5_xga_a = a_stats["form"]["xga_l5"]
            
            if data_mode == "Solo Gol Reali":
                def_form_att_a, def_form_def_a = l5_gf_real_a, l5_gs_real_a
            elif data_mode == "Solo xG (Expected Goals)":
                def_form_att_a, def_form_def_a = l5_xg_a, l5_xga_a
            else:
                def_form_att_a = (l5_gf_real_a + l5_xg_a) / 2
                def_form_def_a = (l5_gs_real_a + l5_xga_a) / 2

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
        st.caption("Forma Recente (Ultime 5)")
        a_form_att = st.number_input("Attacco L5 (O)", 0.0, 25.0, float(def_form_att_a), 0.5)
        a_form_def = st.number_input("Difesa L5 (O)", 0.0, 25.0, float(def_form_def_a), 0.5)

    with st.expander("📈 Trend Over"):
        for l in [0.5, 1.5, 2.5, 3.5, 4.5]: a_uo_input[l] = st.slider(f"Over {l} % A", 0, 100, 50, key=f"ao{l}")

st.subheader("💰 Quote Bookmaker")
qc1, qc2, qc3 = st.columns(3)
b1 = qc1.number_input("Quota 1", 1.01, 100.0, 2.10)
bX = qc2.number_input("Quota X", 1.01, 100.0, 3.20)
b2 = qc3.number_input("Quota 2", 1.01, 100.0, 3.60)

# --- OPZIONI AVANZATE ---
with st.expander("⚙️ Fine Tuning"):
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

        # 1. Calcolo Forza (Split Totale/Casa-Fuori)
        w_split = 0.60 
        
        h_base_att = (h_att * (1-w_split)) + (h_att_home * w_split)
        h_base_def = (h_def * (1-w_split)) + (h_def_home * w_split)
        
        a_base_att = (a_att * (1-w_split)) + (a_att_away * w_split)
        a_base_def = (a_def * (1-w_split)) + (a_def_away * w_split)

        # 2. Integrazione Forma
        h_final_att, h_final_def = calcola_forza_squadra(h_base_att, h_base_def, h_form_att, h_form_def, w_seas)
        a_final_att, a_final_def = calcola_forza_squadra(a_base_att, a_base_def, a_form_att, a_form_def, w_seas)
        
        # 3. xG Match
        xg_h_stats = (h_final_att * a_final_def) / avg_goals_league
        xg_a_stats = (a_final_att * h_final_def) / avg_goals_league
        
        # 4. Elo
        elo_diff = (h_elo + (100 if m_type=="Standard" else 0)) - a_elo
        elo_factor_h = 1 + (elo_diff / 1000.0)
        elo_factor_a = 1 - (elo_diff / 1000.0)
        
        # 5. Lambda
        f_xh = (xg_h_stats * elo_factor_h) + home_adv_goals
        f_xa = (xg_a_stats * elo_factor_a)
        
        # 6. Malus
        fatigue_malus = 0.05 
        if h_rest <= 3: f_xh *= (1 - fatigue_malus); f_xa *= (1 + fatigue_malus) 
        if a_rest <= 3: f_xa *= (1 - fatigue_malus); f_xh *= (1 + fatigue_malus)
        f_xh *= (h_str/100.0); f_xa *= (a_str/100.0)
        
        if is_big_match: f_xh *= 0.90; f_xa *= 0.90
        if h_m_a: f_xh *= 0.85
        if h_m_d: f_xa *= 1.20
        if a_m_a: f_xa *= 0.85
        if a_m_d: f_xh *= 1.20

        # 7. Matrix
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
        
        # 8. Stabilità
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
