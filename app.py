import streamlit as st
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date
import pdfplumber
import re
from tableauscraper import TableauScraper

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Mathbet fc", page_icon="⚽", layout="wide")

# --- DATABASE BASE (Fallback) ---
XG_DATABASE = {
    "Inter": {"xg": 40.5, "xga": 17.9},
    "Juventus": {"xg": 37.5, "xga": 21.6},
    "Milan": {"xg": 36.4, "xga": 24.0},
    "Napoli": {"xg": 33.3, "xga": 24.0},
    "Atalanta": {"xg": 34.5, "xga": 24.1},
}

# --- CLUBELO API ---
@st.cache_data(ttl=3600) 
def get_clubelo_database():
    try:
        date_str = date.today().strftime("%Y-%m-%d")
        url = f"http://api.clubelo.com/{date_str}"
        df = pd.read_csv(url)
        return dict(zip(df.Club, df.Elo))
    except: return {}

ELO_DB = get_clubelo_database()

# --- PARAMETRI CAMPIONATI ---
LEAGUES = {
    "🌐 Generico (Media)": { "avg": 1.35, "ha": 0.30, "w_elo_base": 0.40, "rho": -0.13 }, 
    "🇮🇹 Serie A":          { "avg": 1.30, "ha": 0.20, "w_elo_base": 0.50, "rho": -0.14 },
    "🇬🇧 Premier League":   { "avg": 1.55, "ha": 0.30, "w_elo_base": 0.55, "rho": -0.12 },
    "🇩🇪 Bundesliga":       { "avg": 1.65, "ha": 0.35, "w_elo_base": 0.45, "rho": -0.10 },
    "🇪🇸 La Liga":          { "avg": 1.25, "ha": 0.25, "w_elo_base": 0.55, "rho": -0.14 },
}

# --- FUNZIONE SCRAPING TABLEAU ---
@st.cache_data(ttl=3600)
def load_tableau_data(url):
    """Scarica dati xG da un link Tableau specifico"""
    try:
        ts = TableauScraper()
        ts.loads(url)
        workbook = ts.getWorkbook()
        
        target_df = None
        for t in workbook.worksheets:
            df = t.data
            cols_lower = [c.lower() for c in df.columns]
            if any("team" in c or "squad" in c for c in cols_lower) and any("xg" in c for c in cols_lower):
                target_df = df
                break
        
        if target_df is None: target_df = list(workbook.worksheets)[0].data

        tableau_db = {}
        col_team = next((c for c in target_df.columns if "team" in c.lower() or "squad" in c.lower()), None)
        col_xg = next((c for c in target_df.columns if "xg" in c.lower() and "diff" not in c.lower() and "against" not in c.lower() and "xga" not in c.lower()), None)
        col_xga = next((c for c in target_df.columns if "xga" in c.lower() or "against" in c.lower()), None)
        
        if not col_xga:
             candidates = [c for c in target_df.columns if "xg" in c.lower()]
             if len(candidates) >= 2: col_xga = candidates[1]

        if col_team and col_xg:
            for _, row in target_df.iterrows():
                try:
                    team = str(row[col_team])
                    xg_val = float(str(row[col_xg]).replace(',', '.'))
                    xga_val = float(str(row[col_xga]).replace(',', '.')) if col_xga else 1.0
                    if xg_val > 0: tableau_db[team] = {"xg": xg_val, "xga": xga_val}
                except: continue
        return tableau_db
    except: return {}

# --- FUNZIONI DI PARSING PDF ---
def parse_xg_pdf(uploaded_file):
    new_db = {}
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            text = ""
            for page in pdf.pages: text += page.extract_text() + "\n"
        lines = text.split('\n')
        known_teams = list(XG_DATABASE.keys()) + ["Cremonese", "Salernitana", "Spezia", "Sampdoria", "Frosinone", "Sassuolo", "Parma", "Como", "Venezia"]
        for line in lines:
            found_team = None
            for t in known_teams:
                if t.lower() in line.lower():
                    found_team = t; break
            if found_team:
                nums = re.findall(r"(\d+[\.,]\d+|\d+)", line)
                vals = []
                for n in nums:
                    v = float(n.replace(',', '.'))
                    if v > 100: v /= 10
                    vals.append(v)
                valid = [v for v in vals if 5.0 < v < 150.0]
                if len(valid) >= 2: new_db[found_team] = {"xg": valid[0], "xga": valid[1]}
        return new_db
    except: return {}

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

# --- INIZIALIZZAZIONE ---
if 'history' not in st.session_state: st.session_state.history = []
if 'analyzed' not in st.session_state: st.session_state.analyzed = False
if 'db_season' not in st.session_state: st.session_state.db_season = XG_DATABASE
if 'db_form' not in st.session_state: st.session_state.db_form = {}

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Fonti Dati (Dual Mode)")
    
    # SELETTORE FONTE DATI
    data_source = st.radio("Scegli Fonte:", ["Tableau (Live)", "PDF / Manuale"], index=0)
    
    if data_source == "Tableau (Live)":
        st.caption("1. Vai su Tableau, imposta 'All Matches', copia il link.")
        url_season = st.text_input("Link 1: Stagione Intera", "https://public.tableau.com/views/FootballxGLeagueTablesv4/LeagueTablesv2")
        
        st.caption("2. Su Tableau, filtra 'Last 5', copia il NUOVO link.")
        url_form = st.text_input("Link 2: Ultime 5 Partite (Opzionale)")
        
        if st.button("🔄 Scarica Dati Tableau"):
            with st.spinner("Scaricamento dati stagione..."):
                s_data = load_tableau_data(url_season)
                if s_data: 
                    st.session_state.db_season = s_data
                    st.success(f"✅ Stagione: {len(s_data)} squadre")
            
            if url_form:
                with st.spinner("Scaricamento dati forma..."):
                    f_data = load_tableau_data(url_form)
                    if f_data:
                        st.session_state.db_form = f_data
                        st.success(f"✅ Forma: {len(f_data)} squadre")

    elif data_source == "PDF / Manuale":
        uploaded_file = st.file_uploader("Carica PDF Classifica", type="pdf")
        if uploaded_file:
            with st.spinner("Lettura PDF..."):
                pdf_data = parse_xg_pdf(uploaded_file)
                if pdf_data:
                    st.session_state.db_season = pdf_data
                    st.success(f"✅ PDF Letto: {len(pdf_data)} squadre")
        st.info("In questa modalità la forma (L5) sarà stimata matematicamente.")

    st.markdown("---")
    league_name = st.selectbox("Campionato", list(LEAGUES.keys()))
    L_DATA = LEAGUES[league_name]
    
    matchday = st.slider("Giornata Attuale", 1, 38, 20)
    w_elo = (L_DATA["w_elo_base"] + 0.10) if 8 < matchday <= 19 else (max(L_DATA["w_elo_base"], 0.75) if matchday <= 8 else L_DATA["w_elo_base"])
    
    st.markdown("---")
    m_type = st.radio("Contesto", ["Standard", "Derby", "Campo Neutro"])
    is_big_match = st.checkbox("🔥 Big Match")
    use_xg_mode = st.toggle("📊 Usa Dati xG", value=True)
    CURRENT_RHO = L_DATA.get("rho", -0.13)

st.title("Mathbet fc - Tableau Dual Core 🚀")

# --- HELPER FUNCTIONS ---
def get_season_stats(name, default_xg, default_xga):
    # Cerca nel DB Stagionale (db_season)
    for k, v in st.session_state.db_season.items():
        if k.lower() in name.lower() or name.lower() in k.lower():
            matches_played = max(1, matchday)
            # Se il valore è > 5, assumiamo sia un totale e dividiamo per le giornate
            val_xg = v['xg']/matches_played if v['xg'] > 5.0 else v['xg']
            val_xga = v['xga']/matches_played if v['xga'] > 5.0 else v['xga']
            return val_xg, val_xga
    return default_xg, default_xga

def get_form_stats(name, season_xg, season_xga):
    # Cerca nel DB Forma (db_form) se esiste
    if st.session_state.db_form:
        for k, v in st.session_state.db_form.items():
            if k.lower() in name.lower() or name.lower() in k.lower():
                # Qui assumiamo che la vista "Last 5" restituisca il totale delle ultime 5.
                # Se il valore è molto basso (< 3.0), forse è una media, quindi moltiplichiamo per 5?
                # Tableau di solito dà i totali. Prendiamo il valore grezzo.
                return v['xg'], v['xga']
    
    # FALLBACK: Se non abbiamo il link Forma, stimiamo matematicamente (Media x 5)
    return season_xg * 5, season_xga * 5

# --- INPUT SQUADRE ---
sorted_teams = sorted(list(st.session_state.db_season.keys()))
col_h, col_a = st.columns(2)
h_uo_input, a_uo_input = {}, {}
sos_options = ["Media (Standard)", "Difficili (Top Team)", "Facili (Bassa Classifica)"]

# CASA
with col_h:
    st.subheader("🏠 Squadra Casa")
    h_name = st.selectbox("Casa", ["Manuale"] + sorted_teams, index=1 if len(sorted_teams)>0 else 0)
    if h_name == "Manuale": h_name = st.text_input("Nome Casa", "Inter")
    
    auto_elo_h = float(ELO_DB.get(h_name, 1600.0))
    h_elo = st.number_input("Rating Elo", 1000.0, 2500.0, value=auto_elo_h, step=10.0, key="helo")
    
    # Recupero Dati
    def_h_gf, def_h_gs = 1.85, 0.95
    form_h_gf, form_h_gs = 9.25, 4.75 # Default fallback
    
    if h_name in st.session_state.db_season or h_name in str(st.session_state.db_season):
        # 1. Dati Stagionali (Media a partita)
        def_h_gf, def_h_gs = get_season_stats(h_name, 1.85, 0.95)
        # 2. Dati Forma (Totale ultime 5)
        form_h_gf, form_h_gs = get_form_stats(h_name, def_h_gf, def_h_gs)

    with st.expander("📊 Statistiche (Dual Data)", expanded=True):
        c1, c2 = st.columns(2)
        st.caption("Dati Stagionali (Media/Partita)")
        h_gf_s = c1.number_input("xG Season", 0.0, 5.0, float(def_h_gf), 0.01)
        h_gs_s = c2.number_input("xGA Season", 0.0, 5.0, float(def_h_gs), 0.01)
        
        st.caption("Forma Recente (Totale L5)")
        # Qui usiamo i dati dal secondo link Tableau (se presente)
        h_gf_l5 = c1.number_input("xG Last 5", 0.0, 25.0, float(form_h_gf), 0.1)
        h_gs_l5 = c2.number_input("xGA Last 5", 0.0, 25.0, float(form_h_gs), 0.1)
        
        st.markdown("---")
        # Stime Casa
        h_gf_h = c1.number_input("Casa Stima (+15%)", 0.0, 5.0, h_gf_s*1.15, 0.01)
        h_gs_h = c2.number_input("Casa Stima (-15%)", 0.0, 5.0, h_gs_s*0.85, 0.01)
        h_sos = st.selectbox("SoS Avversari", sos_options, key="hsos")

    with st.expander("Trend"):
        for l in [0.5, 1.5, 2.5, 3.5, 4.5]: h_uo_input[l] = st.slider(f"O{l} % H", 0, 100, 50, key=f"ho{l}")

# OSPITE
with col_a:
    st.subheader("✈️ Squadra Ospite")
    a_name = st.selectbox("Ospite", ["Manuale"] + sorted_teams, index=2 if len(sorted_teams)>1 else 0)
    if a_name == "Manuale": a_name = st.text_input("Nome Ospite", "Juventus")

    auto_elo_a = float(ELO_DB.get(a_name, 1550.0))
    a_elo = st.number_input("Rating Elo", 1000.0, 2500.0, value=auto_elo_a, step=10.0, key="aelo")

    # Recupero Dati
    def_a_gf, def_a_gs = 1.45, 0.85
    form_a_gf, form_a_gs = 7.25, 4.25
    
    if a_name in st.session_state.db_season or a_name in str(st.session_state.db_season):
        def_a_gf, def_a_gs = get_season_stats(a_name, 1.45, 0.85)
        form_a_gf, form_a_gs = get_form_stats(a_name, def_a_gf, def_a_gs)

    with st.expander("📊 Statistiche (Dual Data)", expanded=True):
        c3, c4 = st.columns(2)
        st.caption("Dati Stagionali (Media/Partita)")
        a_gf_s = c3.number_input("xG Season", 0.0, 5.0, float(def_a_gf), 0.01)
        a_gs_s = c4.number_input("xGA Season", 0.0, 5.0, float(def_a_gs), 0.01)
        
        st.caption("Forma Recente (Totale L5)")
        a_gf_l5 = c3.number_input("xG Last 5", 0.0, 25.0, float(form_a_gf), 0.1)
        a_gs_l5 = c4.number_input("xGA Last 5", 0.0, 25.0, float(form_a_gs), 0.1)
        
        st.markdown("---")
        # Stime Fuori
        a_gf_a = c3.number_input("Fuori Stima (-15%)", 0.0, 5.0, a_gf_s*0.85, 0.01)
        a_gs_a = c4.number_input("Fuori Stima (+15%)", 0.0, 5.0, a_gs_s*1.15, 0.01)
        a_sos = st.selectbox("SoS Avversari", sos_options, key="asos")

    with st.expander("Trend"):
        for l in [0.5, 1.5, 2.5, 3.5, 4.5]: a_uo_input[l] = st.slider(f"O{l} % A", 0, 100, 50, key=f"ao{l}")

st.subheader("💰 Quote")
qc1, qc2, qc3 = st.columns(3)
b1 = qc1.number_input("Q1", 1.01, 100.0, 2.10)
bX = qc2.number_input("QX", 1.01, 100.0, 3.20)
b2 = qc3.number_input("Q2", 1.01, 100.0, 3.60)

with st.expander("⚙️ Parametri Avanzati"):
    c_str1, c_str2 = st.columns(2)
    h_str = c_str1.slider("Titolari % Casa", 50, 100, 100)
    a_str = c_str2.slider("Titolari % Ospite", 50, 100, 100)
    h_rest = c_str1.slider("Riposo Casa", 2, 10, 7)
    a_rest = c_str2.slider("Riposo Ospite", 2, 10, 7)
    h_m_a = c_str1.checkbox("No Bomber Casa")
    a_m_a = c_str2.checkbox("No Bomber Ospite")
    h_m_d = c_str1.checkbox("No Difensore Casa")
    a_m_d = c_str2.checkbox("No Difensore Ospite")

# --- CALCOLO ---
if st.button("🚀 ANALIZZA PARTITA", type="primary", use_container_width=True):
    ha_val = L_DATA["ha"]
    if m_type == "Campo Neutro": ha_val = 0.0
    elif m_type == "Derby": ha_val *= 0.5
    
    h_gf_l5_c, h_gs_l5_c = h_gf_l5, h_gs_l5
    a_gf_l5_c, a_gs_l5_c = a_gf_l5, a_gs_l5
    if h_sos == "Difficili (Top Team)": h_gf_l5_c *= 1.25; h_gs_l5_c *= 0.85
    elif h_sos == "Facili (Bassa Classifica)": h_gf_l5_c *= 0.85; h_gs_l5_c *= 1.20 
    if a_sos == "Difficili (Top Team)": a_gf_l5_c *= 1.25; a_gs_l5_c *= 0.85
    elif a_sos == "Facili (Bassa Classifica)": a_gf_l5_c *= 0.85; a_gs_l5_c *= 1.20

    w_seas, w_ha, w_l5 = 0.50, 0.35, 0.15 

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
    
    exp_h = 1 / (1 + 10 ** (-((h_elo + ha_val*400) - a_elo)/400.0))
    xg_e_h = L_DATA["avg"] * (exp_h / 0.5) ** 0.85
    xg_e_a = L_DATA["avg"] * ((1 - exp_h) / 0.5) ** 0.85
    
    f_xh = ((xg_e_h * w_elo) + (xg_s_h * (1-w_elo))) * (h_str/100.0)
    f_xa = ((xg_e_a * w_elo) + (xg_s_a * (1-w_elo))) * (a_str/100.0)
    
    fatigue_malus = 0.05 
    if h_rest <= 3: f_xh *= (1 - fatigue_malus); f_xa *= (1 + fatigue_malus) 
    if a_rest <= 3: f_xa *= (1 - fatigue_malus); f_xh *= (1 + fatigue_malus)
    if is_big_match: f_xh *= 0.90; f_xa *= 0.90
    if h_m_a: f_xh *= 0.85
    if a_m_a: f_xa *= 0.85
    if h_m_d: f_xa *= 1.20
    if a_m_d: f_xh *= 1.20

    # Poisson & Stability
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
    if total_prob > 0: matrix /= total_prob; p1 /= total_prob; pX /= total_prob; p2 /= total_prob; pGG /= total_prob

    p1X, pX2, p12 = p1+pX, pX+p2, p1+p2
    
    sim = []
    for _ in range(5000):
        gh = np.random.poisson(max(0.1, np.random.normal(f_xh, 0.15*f_xh)))
        ga = np.random.poisson(max(0.1, np.random.normal(f_xa, 0.15*f_xa)))
        sim.append(1 if gh>ga else (0 if gh==ga else 2))
    s1, sX, s2 = sim.count(1)/5000, sim.count(0)/5000, sim.count(2)/5000
    stability = max(0, 100 - ((abs(p1-s1)+abs(pX-sX)+abs(p2-s2))/3*400))

    st.session_state.analyzed = True
    st.session_state.f_xh = f_xh; st.session_state.f_xa = f_xa
    st.session_state.h_name = h_name; st.session_state.a_name = a_name
    st.session_state.p1 = p1; st.session_state.pX = pX; st.session_state.p2 = p2
    st.session_state.p1X = p1X; st.session_state.pX2 = pX2; st.session_state.p12 = p12
    st.session_state.pGG = pGG; st.session_state.stability = stability
    st.session_state.matrix = matrix; st.session_state.scores = scores
    st.session_state.b1 = b1; st.session_state.bX = bX; st.session_state.b2 = b2

# --- OUTPUT ---
if st.session_state.analyzed:
    st.header(f"📊 {st.session_state.h_name} vs {st.session_state.a_name}")
    st.metric("xG Previsti", f"{st.session_state.f_xh:.2f} - {st.session_state.f_xa:.2f}", delta=f"Stabilità: {st.session_state.stability:.1f}%")

    tab1, tab2, tab3, tab4 = st.tabs(["🏆 Esito & DC", "⚽ Gol & Handicap", "👤 Marcatori", "📝 Storico & Tools"])

    with tab1:
        c_prob, c_chart = st.columns([1, 1])
        with c_prob:
            st.subheader("1X2 & Doppia Chance")
            st.table(pd.DataFrame({
                "Esito": ["1", "X", "2", "1X", "X2", "12"],
                "Prob %": [f"{p1:.1%}", f"{pX:.1%}", f"{p2:.1%}", f"{p1X:.1%}", f"{pX2:.1%}", f"{p12:.1%}"],
                "Fair Odd": [f"{1/p1:.2f}", f"{1/pX:.2f}", f"{1/p2:.2f}", f"{1/p1X:.2f}", f"{1/pX2:.2f}", f"{1/p12:.2f}"],
            }))

        with c_chart:
            st.subheader("Risultati Esatti")
            scores.sort(key=lambda x: x["Prob"], reverse=True)
            st.table(pd.DataFrame([{"Risultato": s["Risultato"], "Prob": f"{s['Prob']:.1%}", "Quota": f"{1/s['Prob']:.2f}"} for s in scores[:6]]))
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.heatmap(st.session_state.matrix[:5,:5], annot=True, fmt=".0%", cmap="Greens", cbar=False)
            plt.xlabel("Gol Ospite"); plt.ylabel("Gol Casa"); st.pyplot(fig)

    with tab2:
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.subheader("📉 Under / Over")
            uo_res = []
            for l in [0.5, 1.5, 2.5, 3.5, 4.5]:
                p_over_pure = np.sum(st.session_state.matrix[np.indices((10,10))[0] + np.indices((10,10))[1] > l])
                trend_avg = (h_uo_input[l] + a_uo_input[l]) / 200.0
                p_over = (p_over_pure * 0.65) + (trend_avg * 0.35)
                uo_res.append({"Linea": l, "Under %": f"{(1-p_over):.1%}", "Fair U": f"{1/(1-p_over):.2f}", "Over %": f"{p_over:.1%}", "Fair O": f"{1/p_over:.2f}"})
            st.table(pd.DataFrame(uo_res))
            st.subheader("Gol / NoGol")
            st.table(pd.DataFrame([{"Esito": "GG", "Prob": f"{pGG:.1%}", "Fair": f"{1/pGG:.2f}"}, {"Esito": "NG", "Prob": f"{(1-pGG):.1%}", "Fair": f"{1/(1-pGG):.2f}"}]))

        with col_g2:
            st.subheader("🔢 Multigol")
            mg_res = []
            for r in [(1,2), (1,3), (2,3), (2,4), (3,5)]:
                mask = (np.indices((10,10))[0] + np.indices((10,10))[1] >= r[0]) & (np.indices((10,10))[0] + np.indices((10,10))[1] <= r[1])
                pm = np.sum(st.session_state.matrix[mask])
                mg_res.append({"Range": f"{r[0]}-{r[1]}", "Prob": f"{pm:.1%}", "Fair": f"{1/pm:.2f}"})
            st.table(pd.DataFrame(mg_res))
            st.subheader("🏁 Handicap")
            h1_minus1 = np.sum(st.session_state.matrix[np.indices((10,10))[0] - 1 > np.indices((10,10))[1]])
            dnb_1 = st.session_state.p1 / (st.session_state.p1 + st.session_state.p2) if (st.session_state.p1 + st.session_state.p2) > 0 else 0
            st.write(f"**Handicap (-1) Casa:** {h1_minus1:.1%} (@{1/h1_minus1:.2f})"); st.write(f"**Draw No Bet (1 DNB):** {dnb_1:.1%} (@{1/dnb_1:.2f})")

    with tab3:
        st.subheader("Calcolatore Marcatore")
        c_p1, c_p2 = st.columns(2)
        pl_name = c_p1.text_input("Nome", "Vlahovic")
        pl_team = c_p2.radio("Team", [st.session_state.h_name, st.session_state.a_name])
        pl_xg90 = c_p1.number_input("xG/90", 0.01, 2.00, 0.45)
        pl_min = c_p2.number_input("Minuti", 10, 100, 85)
        pl_odd = st.number_input("Quota Book", 1.01, 100.0, 2.50)
        team_xg_match = st.session_state.f_xh if pl_team == st.session_state.h_name else st.session_state.f_xa
        prob_score, _ = calculate_player_probability(pl_xg90, pl_min, team_xg_match, 1.35)
        st.metric("Probabilità Goal", f"{prob_score:.1%}", delta=f"Valore: {((pl_odd*prob_score)-1)*100:.1f}%")
        st.write(f"Quota Reale: **{1/prob_score:.2f}**")

    with tab4:
        st.header("Backtesting & Tools")
        if st.button("💾 SALVA IN STORICO"):
            st.session_state.history.append({"Match": f"{st.session_state.h_name} vs {st.session_state.a_name}", "P1": p1, "PX": pX, "P2": p2, "Esito Reale": "In attesa"})
            st.success("Salvato!")
        if st.session_state.history:
            ed_df = st.data_editor(pd.DataFrame(st.session_state.history), num_rows="dynamic", column_config={"Esito Reale": st.column_config.SelectboxColumn("Esito", options=["1", "X", "2", "In attesa"])})
            val = ed_df[ed_df["Esito Reale"] != "In attesa"]
            if not val.empty:
                brier = []
                for _, r in val.iterrows():
                    o = [1 if r["Esito Reale"]=="1" else 0, 1 if r["Esito Reale"]=="X" else 0, 1 if r["Esito Reale"]=="2" else 0]
                    brier.append((r["P1"]-o[0])**2 + (r["PX"]-o[1])**2 + (r["P2"]-o[2])**2)
                st.metric("Brier Score", f"{np.mean(brier):.3f}")
            if st.button("🗑️ Reset"): st.session_state.history = []; st.rerun()
        st.markdown("---")
        cM1, cM2 = st.columns(2)
        with cM1:
            my_prob = st.number_input("Prob %", 0.1, 99.9, 50.0); st.write(f"Quota Reale: **{100/my_prob:.2f}**")
        with cM2:
            mp = st.number_input("Tua Stima %", 1.0, 100.0, 50.0) / 100
            mq = st.number_input("Quota Book", 1.01, 100.0, 2.0)
            if (mp * mq) - 1 > 0: st.success(f"VALORE! Stake: {((((mq-1)*mp)-(1-mp))/(mq-1)*25):.1f}%")
            else: st.error("NO VALUE")
