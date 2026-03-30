import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import json
import engine 
import live_scraper 

# ==============================================================================
# CONFIGURAZIONE API & PAGINA
# ==============================================================================
# 🔴 INSERISCI LE TUE CHIAVI API QUI SOTTO
ODDS_API_KEY = st.secrets["ODDS_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

st.set_page_config(page_title="Mathbet fc", page_icon="⚽", layout="wide")

if 'history' not in st.session_state: st.session_state.history = engine.carica_storico_json()
if 'analyzed' not in st.session_state: st.session_state.analyzed = False
if 'ml_active' not in st.session_state: st.session_state.ml_active = False

# ==============================================================================
# 🎛️ UI SIDEBAR
# ==============================================================================
with st.sidebar:
    st.title("🧠 Configurazione")
    league_name = st.selectbox("Campionato", list(engine.LEAGUES_CONFIG.keys()))
    st.session_state.league_name = league_name
    L_DATA = engine.LEAGUES_CONFIG[league_name]
    
    with st.spinner("Estrazione dati LIVE in corso..."):
        STATS_DB, TEAM_LIST = engine.fetch_understat_data_auto(league_name)
        PLAYERS_DF = engine.fetch_understat_players(league_name)
        ELO_DICT = engine.fetch_clubelo_ratings() 
        ALL_LEAGUE_ODDS = engine.fetch_league_odds(L_DATA["odds_id"], ODDS_API_KEY)
    
    st.markdown("---")
    use_ml_boost = st.toggle("🤖 Attiva Apprendimento ML", value=False)
    data_mode = st.radio("Dati Analisi", ["Solo Gol Reali", "Solo xG (NPxG Mode)", "Ibrido (Consigliato)"], index=2)
    volatility = st.slider("Volatilità", 0.8, 1.4, 1.0, 0.05)
    matchday = st.slider("Giornata Attuale", 1, 38, 22)
    w_seas = min(0.90, 0.30 + (matchday * 0.02)) 
    m_type = st.radio("Contesto", ["Standard", "Derby", "Campo Neutro"])
    is_big_match = st.checkbox("🔥 Big Match")
    
    with st.expander("🔢 Dati Calci d'Angolo", expanded=False):
        avg_lega_corn = st.number_input("Media Angoli Lega (Tot match)", 5.0, 15.0, 10.0, 0.1)
        c1, c2 = st.columns(2)
        corn_f_h = c1.number_input("Fatti Casa", 0.0, 15.0, 5.5, 0.1)
        corn_s_h = c1.number_input("Subiti Casa", 0.0, 15.0, 4.5, 0.1)
        corn_f_a = c2.number_input("Fatti Ospite", 0.0, 15.0, 4.5, 0.1)
        corn_s_a = c2.number_input("Subiti Ospite", 0.0, 15.0, 5.5, 0.1)

    with st.expander("🔢 Altri Dati Extra", expanded=False):
        c_card_h = st.number_input("Cart. Casa", 0.0, 10.0, 2.0, 0.1)
        c_card_a = st.number_input("Cart. Ospite", 0.0, 10.0, 2.5, 0.1)
        c_shot_h = st.number_input("Tiri Casa", 0.0, 50.0, 12.0, 0.5)
        c_shot_a = st.number_input("Tiri Ospite", 0.0, 50.0, 10.0, 0.5)
        c_sot_h = st.number_input("Tiri Porta C", 0.0, 20.0, 4.5, 0.1)
        c_sot_a = st.number_input("Tiri Porta O", 0.0, 20.0, 3.5, 0.1)
        c_foul_h = st.number_input("Falli Casa", 0.0, 30.0, 11.5, 0.5)
        c_foul_a = st.number_input("Falli Ospite", 0.0, 30.0, 12.5, 0.5)

st.title("Mathbet fc ⚽")

ml_models = None
if use_ml_boost:
    ml_models = engine.train_ml_models(st.session_state.history)

# --- SCANNER AUTOMATICO TOP 5 ---
with st.expander("🔥 SCANNER GIORNALIERO: TOP 5 VALUE BETS", expanded=False):
    st.markdown("Cerca le migliori occasioni matematiche sul palinsesto.")
    if st.button("🔍 Cerca Top 5 Value Bets Ora", type="primary"):
        with st.spinner("Elaborazione in corso..."):
            top_5 = engine.find_top_value_bets(ALL_LEAGUE_ODDS, STATS_DB, L_DATA, volatility, m_type, ELO_DICT, w_seas, data_mode, ml_models)
            if top_5:
                st.table(pd.DataFrame(top_5).style.format({"Prob %": "{:.1%}", "Fair Odd": "{:.2f}", "Valore %": "{:.1%}"}).map(lambda x: 'background-color: #d4edda; font-weight: bold;', subset=['Valore %']))
            else: 
                st.warning("Nessuna Value Bet significativa trovata.")
st.divider()

# ==============================================================================
# 🗓️ SELEZIONE MATCH DA PALINSESTO E DATI
# ==============================================================================
st.subheader("🗓️ Selezione Partita")

if ALL_LEAGUE_ODDS:
    match_options = [f"{m['home_team']} - {m['away_team']}" for m in ALL_LEAGUE_ODDS]
    selected_match = st.selectbox("Scegli il Match in programma", match_options)
    bookie_h_name, bookie_a_name = selected_match.split(" - ")
    auto_h_name = engine.fuzzy_match_team(bookie_h_name, TEAM_LIST)
    auto_a_name = engine.fuzzy_match_team(bookie_a_name, TEAM_LIST)
else:
    if ODDS_API_KEY == "INSERISCI_QUI_LA_TUA_CHIAVE":
        st.error("⚠️ La tendina dei Prossimi Match è nascosta. Inserisci la tua vera ODDS_API_KEY nel codice per sbloccarla!")
    else:
        st.warning("⚠️ Nessun match in programma nei prossimi giorni o limite chiamate API esaurito.")
    bookie_h_name, bookie_a_name = "", ""
    auto_h_name = TEAM_LIST[0] if TEAM_LIST else None
    auto_a_name = TEAM_LIST[1] if len(TEAM_LIST)>1 else None

col_h, col_a = st.columns(2)
h_uo_input, a_uo_input = {}, {}

def get_blended_val(stats_dict, metric, mode, w_seas):
    def extract_val(raw_segment):
        matches = raw_segment.get('matches', 0)
        if matches <= 0: return 0.0
        if metric == 'gf': v_goals, v_xg, v_npxg = raw_segment.get('goals_total', 0), raw_segment.get('xg_total', 0), raw_segment.get('npxg_total', 0)
        else: v_goals, v_xg, v_npxg = raw_segment.get('ga_total', 0), raw_segment.get('xga_total', 0), raw_segment.get('npxga_total', 0)
        if mode == "Solo Gol Reali": return v_goals / matches
        elif mode == "Solo xG (NPxG Mode)": return v_npxg / matches
        else: return ((v_npxg * 0.40) + (v_xg * 0.30) + (v_goals * 0.30)) / matches
    return (extract_val(stats_dict.get("total", {})) * w_seas) + (extract_val(stats_dict.get("form", {})) * (1 - w_seas))

with col_h:
    st.subheader("🏠 Squadra Casa")
    h_idx = TEAM_LIST.index(auto_h_name) if auto_h_name in TEAM_LIST else 0
    h_name = st.selectbox("Dati Understat Casa", TEAM_LIST, index=h_idx) if TEAM_LIST else st.text_input("Nome Casa", "Inter")
    h_stats = STATS_DB.get(h_name) if STATS_DB else None
    h_elo = st.number_input("Rating Elo Casa", 1000.0, 2500.0, float(engine.get_elo_for_team(h_name, ELO_DICT, 1600.0) if h_name else 1600.0), step=10.0)
    with st.expander("📊 Dati (Mix Stagione/Trend)", expanded=True):
        def_att_s = get_blended_val(h_stats, 'gf', data_mode, w_seas) if h_stats else 1.85
        def_def_s = get_blended_val(h_stats, 'gs', data_mode, w_seas) if h_stats else 0.95
        c1, c2 = st.columns(2)
        h_att = c1.number_input("Attacco Totale (C)", 0.0, 5.0, float(def_att_s), 0.01)
        h_def = c2.number_input("Difesa Totale (C)", 0.0, 5.0, float(def_def_s), 0.01)
        c3, c4 = st.columns(2)
        h_att_home = c3.number_input("Attacco Casa", 0.0, 5.0, float(get_blended_val({"total": h_stats["home"], "form": h_stats["home"]}, 'gf', data_mode, 1.0) if h_stats else def_att_s*1.15), 0.01)
        h_def_home = c4.number_input("Difesa Casa", 0.0, 5.0, float(get_blended_val({"total": h_stats["home"], "form": h_stats["home"]}, 'gs', data_mode, 1.0) if h_stats else def_def_s*0.85), 0.01)
    with st.expander("Over Trend"):
        for l in [0.5, 1.5, 2.5, 3.5, 4.5]: h_uo_input[l] = st.slider(f"Over {l} % H", 0, 100, 50, key=f"ho{l}")

with col_a:
    st.subheader("✈️ Squadra Ospite")
    a_idx = TEAM_LIST.index(auto_a_name) if auto_a_name in TEAM_LIST else 0
    a_name = st.selectbox("Dati Understat Ospite", TEAM_LIST, index=a_idx) if TEAM_LIST else st.text_input("Nome Ospite", "Juve")
    a_stats = STATS_DB.get(a_name) if STATS_DB else None
    a_elo = st.number_input("Rating Elo Ospite", 1000.0, 2500.0, float(engine.get_elo_for_team(a_name, ELO_DICT, 1550.0) if a_name else 1550.0), step=10.0)
    with st.expander("📊 Dati (Mix Stagione/Trend)", expanded=True):
        def_att_s_a = get_blended_val(a_stats, 'gf', data_mode, w_seas) if a_stats else 1.45
        def_def_s_a = get_blended_val(a_stats, 'gs', data_mode, w_seas) if a_stats else 0.85
        c5, c6 = st.columns(2)
        a_att = c5.number_input("Attacco Totale (O)", 0.0, 5.0, float(def_att_s_a), 0.01)
        a_def = c6.number_input("Difesa Totale (O)", 0.0, 5.0, float(def_def_s_a), 0.01)
        c7, c8 = st.columns(2)
        a_att_away = c7.number_input("Attacco Fuori", 0.0, 5.0, float(get_blended_val({"total": a_stats["away"], "form": a_stats["away"]}, 'gf', data_mode, 1.0) if a_stats else def_att_s_a*0.85), 0.01)
        a_def_away = c8.number_input("Difesa Fuori", 0.0, 5.0, float(get_blended_val({"total": a_stats["away"], "form": a_stats["away"]}, 'gs', data_mode, 1.0) if a_stats else def_def_s_a*1.15), 0.01)
    with st.expander("Over Trend"):
        for l in [0.5, 1.5, 2.5, 3.5, 4.5]: a_uo_input[l] = st.slider(f"Over {l} % A", 0, 100, 50, key=f"ao{l}")

live_match_odds = engine.extract_match_odds(ALL_LEAGUE_ODDS, bookie_h_name, bookie_a_name)

st.subheader("💰 Quote Reali")
q1, qx, q2 = st.columns(3)
val_1 = live_match_odds['h2h'].get(bookie_h_name, 2.10) if live_match_odds and 'h2h' in live_match_odds else 2.10
val_X = live_match_odds['h2h'].get('Draw', 3.20) if live_match_odds and 'h2h' in live_match_odds else 3.20
val_2 = live_match_odds['h2h'].get(bookie_a_name, 3.60) if live_match_odds and 'h2h' in live_match_odds else 3.60
b1 = q1.number_input("Q1", 1.01, 100.0, float(val_1))
bX = qx.number_input("QX", 1.01, 100.0, float(val_X))
b2 = q2.number_input("Q2", 1.01, 100.0, float(val_2))

# ==============================================================================
# 🕵️‍♂️ PRE-ANALISI DEL FIGGHIOZZO (CONSIGLI FINE-TUNING & CHAT)
# ==============================================================================
st.subheader("🕵️‍♂️ Pre-Analisi del Figghiozzo (News & Formazioni)")
st.markdown("Fatti consigliare dal Figghiozzo i parametri. Se sbaglia o manca una news, **puoi scrivergli per correggerlo!**")

if "fig_chat_session" not in st.session_state:
    st.session_state.fig_chat_session = None
if "fig_chat_history" not in st.session_state:
    st.session_state.fig_chat_history = []

if st.button("📰 Cerca News e Avvia Pre-Analisi", type="secondary"):
    if not GEMINI_API_KEY or GEMINI_API_KEY == "INSERISCI_QUI_LA_TUA_CHIAVE_GEMINI":
        st.warning("⚠️ Inserisci la tua vera API Key di Gemini all'inizio del codice (riga 15).")
    else:
        with st.spinner("🔍 Il Figghiozzo sta forzando i sistemi anti-bot per estrarre le formazioni..."):
            try:
                from duckduckgo_search import DDGS
                import google.generativeai as genai
                from datetime import datetime
                import cloudscraper
                from bs4 import BeautifulSoup
                
                oggi_str = datetime.now().strftime("%Y-%m-%d")
                
                # 1. RICERCA MIRATA
                query_news = f"{h_name} {a_name} probabili formazioni infortuni"
                full_articles = []
                
                scraper = cloudscraper.create_scraper(browser={'browser': 'chrome', 'platform': 'windows', 'desktop': True})
                
                try:
                    # Cerca prima nelle news, se vuoto cerca nel testo generale
                    search_results = DDGS().news(query_news, region='it-it', timelimit='w', max_results=4)
                    if not search_results:
                        search_results = DDGS().text(query_news, region='it-it', timelimit='w', max_results=4)
                    
                    piano_c_attivo = False
                    if not search_results:
                        piano_c_attivo = True
                        search_results = DDGS().text(f"{h_name} tabellino formazioni", region='it-it', timelimit='m', max_results=2)
                        search_results += DDGS().text(f"{a_name} tabellino formazioni", region='it-it', timelimit='m', max_results=2)
                        
                    for res in search_results:
                        # FIX FONDAMENTALE: DuckDuckGo a volte usa 'href', a volte 'url'
                        link = res.get('href') or res.get('url') or ''
                        snippet = res.get('body', '')  # Il riassunto è vitale
                        titolo = res.get('title', '')
                        
                        if link:
                            try:
                                page = scraper.get(link, timeout=7)
                                soup = BeautifulSoup(page.text, 'html.parser')
                                article_text = " ".join([p.get_text() for p in soup.find_all('p')])
                                
                                text_lower = article_text.lower()
                                # Se veniamo bloccati dai cookie, passiamo ALMENO il titolo e il riassunto
                                if len(article_text) < 150 or "cookie" in text_lower or "javascript" in text_lower:
                                    full_articles.append(f"--- TITOLO: {titolo} ---\nRIASSUNTO: {snippet}\n")
                                else:
                                    # Se passiamo, diamo all'IA SIA il riassunto SIA l'articolo intero
                                    full_articles.append(f"--- TITOLO: {titolo} ---\nRIASSUNTO: {snippet}\nTESTO ARTICOLO: {article_text[:3500]}\n")
                            except Exception:
                                full_articles.append(f"--- TITOLO: {titolo} ---\nRIASSUNTO: {snippet}\n")
                        else:
                            # Se non c'è link, usiamo almeno lo snippet
                            full_articles.append(f"--- TITOLO: {titolo} ---\nRIASSUNTO: {snippet}\n")
                    
                    if full_articles:
                        news_context = "\n".join(full_articles)
                        if piano_c_attivo:
                            news_context += "\n\n[ATTENZIONE: Queste sono notizie relative all'ULTIMA PARTITA giocata. Usale per dedurre squalificati o infortunati.]"
                    else:
                        news_context = "[Nessuna news trovata. DIVIETO DI INVENTARE NOMI.]"
                        
                except Exception:
                    news_context = "[Ricerca web fallita. DIVIETO DI INVENTARE NOMI.]"

                genai.configure(api_key=GEMINI_API_KEY)
                modello_valido = None
                for m in genai.list_models():
                    if 'generateContent' in m.supported_generation_methods:
                        modello_valido = m.name
                        if 'flash' in m.name: break
                        
                if modello_valido:
                    model = genai.GenerativeModel(modello_valido)
                    
                    chat = model.start_chat(history=[])
                    st.session_state.fig_chat_session = chat
                    st.session_state.fig_chat_history = []
                    
                    prompt_pre = f"""
                    IL TUO NOME È "FIGGHIOZZO". Presentati in modo amichevole e spavaldo.
                    Oggi è il {oggi_str}. Stiamo analizzando il PROSSIMO MATCH IMMINENTE: {h_name} vs {a_name}.
                    
                    [REGOLA TASSATIVA]: Non dire mai "il match di oggi". Usa "il prossimo match" o "la sfida imminente".
                    
                    📰 TESTI, RIASSUNTI E TITOLI DEGLI ARTICOLI ESTTRATTI ORA:
                    {news_context}
                    
                    🔴 IL TUO COMPITO:
                    1. **Formazioni e Assenze:** Cerca in modo MANIACALE i nomi dei giocatori nei testi, nei titoli e nei riassunti qui sopra. 
                       - Elenca con precisione le probabili formazioni in base ai dati che ti ho appena fornito. 
                       - Evidenzia chi è assente, squalificato o infortunato.
                       - SE NON CI SONO NOMI NEI TESTI, HAI IL DIVIETO ASSOLUTO DI INVENTARLI O RICORDARLI DAL PASSATO. Ammetti che non ci sono dati aggiornati per le rose.
                    
                    2. **Scheda Fine-Tuning:** Consiglia all'utente i parametri. Applica la REGOLA D'ORO anti-overfitting:
                    - CASO A (Assenza Mirata): Manca un big? Consiglia "No Bomber" o "No Difensore", ma tieni "Titolari %" al 100%.
                    - CASO B (Turnover/Molte assenze): Consiglia di abbassare "Titolari %" (es. 80-85%), ma lascia FALSO le spunte "No Bomber/Difensore".
                    
                    Formatta in modo chiaro i valori esatti consigliati:
                    - Titolari % Casa: [Valore] | Titolari % Ospite: [Valore]
                    - No Bomber Casa: [VERO/FALSO] | No Bomber Ospite: [VERO/FALSO]
                    - No Difensore Casa: [VERO/FALSO] | No Difensore Ospite: [VERO/FALSO]
                    """
                    
                    response = chat.send_message(prompt_pre)
                    st.session_state.fig_chat_history.append({"role": "model", "text": response.text})
                    
            except Exception as e:
                st.error(f"Errore nella pre-analisi: {e}")

if "fig_chat_history" in st.session_state and st.session_state.fig_chat_history:
    st.divider()
    for msg in st.session_state.fig_chat_history:
        if msg["role"] == "user":
            st.info(f"🗣️ **Tu:** {msg['text']}")
        else:
            st.success(f"🤖 **Figghiozzo:**\n{msg['text']}")
    
    st.markdown("---")
    user_input = st.text_input("💬 Scrivi al Figghiozzo per correggerlo o aggiungere news:", key="fig_input_box")
    
    if st.button("Invia Risposta al Figghiozzo", type="primary"):
        if user_input and st.session_state.fig_chat_session:
            with st.spinner("✍️ Il Figghiozzo sta leggendo la tua nota e ricalcolando la scheda..."):
                try:
                    chat = st.session_state.fig_chat_session
                    st.session_state.fig_chat_history.append({"role": "user", "text": user_input})
                    
                    correction_prompt = f"{user_input}\n\n[ISTRUZIONE: Rispondi all'utente ringraziandolo, NON dire 'nella partita di oggi' e scrivi DI NUOVO la 'SCHEDA FINE-TUNING' aggiornata tenendo conto di questa nuova info.]"
                    
                    new_response = chat.send_message(correction_prompt)
                    st.session_state.fig_chat_history.append({"role": "model", "text": new_response.text})
                    st.rerun()
                except Exception as e:
                    st.error(f"Errore durante la conversazione: {e}")
                
with st.expander("⚙️ Fine Tuning"):
    c1, c2 = st.columns(2)
    h_str = c1.slider("Titolari % Casa", 50, 100, 100); a_str = c2.slider("Titolari % Ospite", 50, 100, 100)
    h_rest = c1.slider("Riposo Casa", 2, 10, 7); a_rest = c2.slider("Riposo Ospite", 2, 10, 7)
    h_m_a = c1.checkbox("No Bomber Casa"); a_m_a = c2.checkbox("No Bomber Ospite")
    h_m_d = c1.checkbox("No Difensore Casa"); a_m_d = c2.checkbox("No Difensore Ospite")

# ==============================================================================
# 🚀 TRIGGER ANALISI
# ==============================================================================
if st.button("🚀 ANALIZZA", type="primary", use_container_width=True):
    with st.spinner("Calcolo Algoritmo ML Multi-Target..."):
        home_adv = L_DATA["ha"] if m_type == "Standard" else (0.0 if m_type == "Campo Neutro" else L_DATA["ha"]*0.5)
        w_split = 0.60
        h_fin_att = (h_att*(1-w_split)) + (h_att_home*w_split); h_fin_def = (h_def*(1-w_split)) + (h_def_home*w_split)
        a_fin_att = (a_att*(1-w_split)) + (a_att_away*w_split); a_fin_def = (a_def*(1-w_split)) + (a_def_away*w_split)
        
        xg_h = (h_fin_att * a_fin_def) / L_DATA["avg"]; xg_a = (a_fin_att * h_fin_def) / L_DATA["avg"]
        elo_diff = (h_elo + (100 if m_type=="Standard" else 0)) - a_elo
        f_xh = (xg_h * (1 + elo_diff/1000.0)) + home_adv
        f_xa = (xg_a * (1 - elo_diff/1000.0))

        h_ppda = a_ppda = h_dc = a_dc = 10.0
        if h_stats and a_stats:
            h_ppda = (h_stats['total']['ppda'] * w_seas) + (h_stats['form']['ppda'] * (1 - w_seas))
            a_ppda = (a_stats['total']['ppda'] * w_seas) + (a_stats['form']['ppda'] * (1 - w_seas))
            h_oppda = (h_stats['total']['oppda'] * w_seas) + (h_stats['form']['oppda'] * (1 - w_seas))
            a_oppda = (a_stats['total']['oppda'] * w_seas) + (a_stats['form']['oppda'] * (1 - w_seas))

            if h_ppda < 10.5 and a_oppda > 10.5: f_xh *= 1.07 
            if a_ppda < 10.5 and h_oppda > 10.5: f_xa *= 1.07

            h_dc = (h_stats['total']['dc'] * w_seas) + (h_stats['form']['dc'] * (1 - w_seas))
            a_dc = (a_stats['total']['dc'] * w_seas) + (a_stats['form']['dc'] * (1 - w_seas))
            if (h_dc + a_dc) > 0:
                h_tilt = h_dc / (h_dc + a_dc)
                if h_tilt > 0.60: f_xh *= 1.05; f_xa *= 0.95 
                elif h_tilt < 0.40: f_xh *= 0.95; f_xa *= 1.05 

            h_pts, h_xpts = h_stats['total']['pts'], h_stats['total']['xpts']
            a_pts, a_xpts = a_stats['total']['pts'], a_stats['total']['xpts']
            if h_pts > (h_xpts * 1.2): f_xh *= 0.97 
            elif h_pts < (h_xpts * 0.8): f_xh *= 1.03 
            if a_pts > (a_xpts * 1.2): f_xa *= 0.97
            elif a_pts < (a_xpts * 0.8): f_xa *= 1.03

        expected_goal_diff = f_xh - f_xa
        if expected_goal_diff > 0.45: f_xh *= 0.96; f_xa *= 1.04
        elif expected_goal_diff < -0.45: f_xa *= 0.96; f_xh *= 1.04

        f_xh *= volatility * (h_str/100.0)
        f_xa *= volatility * (a_str/100.0)
        if h_rest <= 3: f_xh *= 0.95; f_xa *= 1.05
        if a_rest <= 3: f_xa *= 0.95; f_xh *= 1.05
        if is_big_match: f_xh *= 0.9; f_xa *= 0.9
        if h_m_a: f_xh *= 0.85
        if h_m_d: f_xa *= 1.20
        if a_m_a: f_xa *= 0.85
        if a_m_d: f_xh *= 1.20

        matrix = np.zeros((10,10)); scores = []
        p1 = pX = p2 = pGG = pO25 = 0.0
        
        for h in range(10):
            for a in range(10):
                p = engine.dixon_coles_probability(h, a, f_xh, f_xa, L_DATA["rho"])
                matrix[h,a] = p
                if h > a: p1 += p
                elif h == a: pX += p
                else: p2 += p
                if h > 0 and a > 0: pGG += p
                if (h + a) > 2.5: pO25 += p
                if h < 6 and a < 6: scores.append({"Risultato": f"{h}-{a}", "Prob": p})
                
        tot = np.sum(matrix)
        if tot > 0:
            matrix /= tot; p1 /= tot; pX /= tot; p2 /= tot; pGG /= tot; pO25 /= tot
        
        if ml_models and use_ml_boost:
            p1, pX, p2, pO25, pGG = engine.apply_ml_boost(ml_models, f_xh, f_xa, p1, pX, p2, pO25, pGG, h_elo, a_elo, h_ppda, a_ppda, h_dc, a_dc, w_seas, volatility)
            st.session_state.ml_active = True
        else:
            st.session_state.ml_active = False
            
        sim = engine.monte_carlo_simulation(f_xh, f_xa)
        stability = max(0, 100 - ((abs(p1-(sim.count(1)/5000))+abs(pX-(sim.count(0)/5000))+abs(p2-(sim.count(2)/5000)))/3*400))
        
        xCorn_H, xCorn_A = engine.calcola_xCorners_pro(corn_f_h, corn_s_h, corn_f_a, corn_s_a, avg_lega_corn/2, h_dc, a_dc)
        corn_1, corn_X, corn_2, corn_lines = engine.calculate_stats_probs(xCorn_H, xCorn_A)
        card_1, card_X, card_2, card_lines = engine.calculate_stats_probs(c_card_h, c_card_a)
        shot_1, shot_X, shot_2, shot_lines = engine.calculate_stats_probs(c_shot_h, c_shot_a)
        sot_1, sot_X, sot_2, sot_lines = engine.calculate_stats_probs(c_sot_h, c_sot_a)
        foul_1, foul_X, foul_2, foul_lines = engine.calculate_stats_probs(c_foul_h, c_foul_a)
        
        st.session_state.analyzed = True
        st.session_state.update({
            "f_xh": f_xh, "f_xa": f_xa, "h_name": h_name, "a_name": a_name, "league_name": league_name,
            "p1": p1, "pX": pX, "p2": p2, "pGG": pGG, "pO25": pO25, "stability": stability,
            "h_elo": h_elo, "a_elo": a_elo, "h_ppda": h_ppda, "a_ppda": a_ppda, "h_dc": h_dc, "a_dc": a_dc,
            "w_seas": w_seas, "volatility": volatility,
            "matrix": matrix, "scores": scores, "b1": b1, "bX": bX, "b2": b2,
            "stats": {
                "corners": {"1": corn_1, "X": corn_X, "2": corn_2, "lines": corn_lines, "xH": xCorn_H, "xA": xCorn_A},
                "cards": {"1": card_1, "X": card_X, "2": card_2, "lines": card_lines},
                "shots": {"1": shot_1, "X": shot_X, "2": shot_2, "lines": shot_lines},
                "sot": {"1": sot_1, "X": sot_X, "2": sot_2, "lines": sot_lines},
                "fouls": {"1": foul_1, "X": foul_X, "2": foul_2, "lines": foul_lines}
            }
        })

# ==============================================================================
# 📊 VISUALIZZAZIONE RISULTATI (PANNELLO COMPLETO 8 TABS)
# ==============================================================================
if st.session_state.analyzed:
    st.markdown("---")
    if st.session_state.ml_active: 
        st.success("🤖 Multi-Target AI Attiva! Percentuali corrette automaticamente dal Machine Learning.")
        
    st.header(f"📊 {st.session_state.h_name} vs {st.session_state.a_name}")
    c1, c2 = st.columns(2)
    c1.metric("xG Previsti (Adjusted)", f"{st.session_state.f_xh:.2f} - {st.session_state.f_xa:.2f}")
    c2.metric("Affidabilità Previsione", f"{st.session_state.stability:.1f}%")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(["🏆 Esito & Handicap", "⚽ Under / Over", "🎯 Multigol", "👤 Player & Assist", "⛳ Extra & Corners", "📝 Storico & ML", "🧮 Utility & Combo", "✍️ Giornalista AI", "🔴 Trading Live (Sperimentale)"])
    
    # ---------------- TAB 1: ESITO E HANDICAP ----------------
    with tab1:
        c_1, c_2 = st.columns(2)
        with c_1:
            st.subheader("1X2 & Value Bet")
            val_df = pd.DataFrame({
                "Esito": ["1", "X", "2"],
                "Prob": [f"{st.session_state.p1:.1%}", f"{st.session_state.pX:.1%}", f"{st.session_state.p2:.1%}"],
                "Fair": [f"{1/st.session_state.p1:.2f}", f"{1/st.session_state.pX:.2f}", f"{1/st.session_state.p2:.2f}"],
                "Bookie": [st.session_state.b1, st.session_state.bX, st.session_state.b2],
                "Valore": [f"{(st.session_state.b1*st.session_state.p1-1):.1%}", f"{(st.session_state.bX*st.session_state.pX-1):.1%}", f"{(st.session_state.b2*st.session_state.p2-1):.1%}"]
            })
            st.dataframe(val_df.style.map(lambda x: "background-color: #d4edda" if "%" in str(x) and "-" not in str(x) and str(x) != "0.0%" else "", subset=["Valore"]), hide_index=True)
            
            st.subheader("Doppia Chance & Draw No Bet")
            p1X, pX2, p12 = (st.session_state.p1 + st.session_state.pX), (st.session_state.pX + st.session_state.p2), (st.session_state.p1 + st.session_state.p2)
            p1_dnb = st.session_state.p1 / (1 - st.session_state.pX) if st.session_state.pX < 1 else 0
            p2_dnb = st.session_state.p2 / (1 - st.session_state.pX) if st.session_state.pX < 1 else 0
            
            dc_df = pd.DataFrame({
                "Esito": ["1X", "X2", "12", "1 DNB", "2 DNB"],
                "Prob %": [f"{p1X:.1%}", f"{pX2:.1%}", f"{p12:.1%}", f"{p1_dnb:.1%}", f"{p2_dnb:.1%}"],
                "Quota Fair": [f"{1/p1X:.2f}", f"{1/pX2:.2f}", f"{1/p12:.2f}", f"{1/p1_dnb:.2f}" if p1_dnb>0 else "-", f"{1/p2_dnb:.2f}" if p2_dnb>0 else "-"]
            })
            st.dataframe(dc_df, hide_index=True)

        with c_2:
            st.subheader("Risultati Esatti")
            scores = sorted(st.session_state.scores, key=lambda x: x["Prob"], reverse=True)
            st.dataframe(pd.DataFrame([{"Risultato": s["Risultato"], "Prob": f"{s['Prob']:.1%}"} for s in scores[:6]]), hide_index=True)

            st.subheader("Handicap Asiatico (-1.5 / +1.5)")
            matrix = st.session_state.matrix
            ph_minus_15 = np.sum(matrix[np.indices((10,10))[0] - np.indices((10,10))[1] >= 2])
            pa_plus_15 = 1 - ph_minus_15
            pa_minus_15 = np.sum(matrix[np.indices((10,10))[1] - np.indices((10,10))[0] >= 2])
            ph_plus_15 = 1 - pa_minus_15
            
            hc_df = pd.DataFrame({
                "Mercato": ["Casa -1.5", "Ospite +1.5", "Ospite -1.5", "Casa +1.5"],
                "Prob %": [f"{ph_minus_15:.1%}", f"{pa_plus_15:.1%}", f"{pa_minus_15:.1%}", f"{ph_plus_15:.1%}"],
                "Quota": [f"{1/ph_minus_15:.2f}" if ph_minus_15>0 else "-", f"{1/pa_plus_15:.2f}" if pa_plus_15>0 else "-", f"{1/pa_minus_15:.2f}" if pa_minus_15>0 else "-", f"{1/ph_plus_15:.2f}" if ph_plus_15>0 else "-"]
            })
            st.dataframe(hc_df, hide_index=True)

    # ---------------- TAB 2: UNDER / OVER ----------------
    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Mercato Under / Over")
            uo_list = []
            for l in [0.5, 1.5, 2.5, 3.5, 4.5]:
                if l == 2.5: 
                    p_over = st.session_state.pO25
                else:
                    p_pure = np.sum(st.session_state.matrix[np.indices((10,10))[0] + np.indices((10,10))[1] > l])
                    p_over = (p_pure*0.7) + (((h_uo_input.get(l,50) + a_uo_input.get(l,50))/200.0)*0.3)
                p_under = 1.0 - p_over
                
                uo_list.append({
                    "Linea": l,
                    "Over %": f"{p_over:.1%}", "Quota O.": f"{1/p_over:.2f}" if p_over>0 else "-",
                    "Under %": f"{p_under:.1%}", "Quota U.": f"{1/p_under:.2f}" if p_under>0 else "-"
                })
            st.dataframe(pd.DataFrame(uo_list), hide_index=True)
            
        with c2:
            st.subheader("Mercato Goal / No Goal")
            pGG = st.session_state.pGG
            pNG = 1.0 - pGG
            gg_df = pd.DataFrame({
                "Esito": ["Goal (BTTS)", "No Goal"],
                "Prob %": [f"{pGG:.1%}", f"{pNG:.1%}"],
                "Quota": [f"{1/pGG:.2f}" if pGG>0 else "-", f"{1/pNG:.2f}" if pNG>0 else "-"]
            })
            st.dataframe(gg_df, hide_index=True)

    # ---------------- TAB 3: MULTIGOL ----------------
    with tab3:
        cm1, cm2, cm3 = st.columns(3)
        
        with cm1:
            st.subheader("Multigol Totale")
            mg_res = []
            for r in [(1,2), (1,3), (2,3), (2,4), (3,4), (3,5)]:
                pm = np.sum(st.session_state.matrix[(np.indices((10,10))[0] + np.indices((10,10))[1] >= r[0]) & (np.indices((10,10))[0] + np.indices((10,10))[1] <= r[1])])
                mg_res.append({"Range": f"{r[0]}-{r[1]}", "Prob %": f"{pm:.1%}", "Quota": f"{1/pm:.2f}" if pm>0 else "-"})
            st.dataframe(pd.DataFrame(mg_res), hide_index=True)
            
        with cm2:
            st.subheader("Multigol Casa")
            mgh_res = []
            for r in [(1,2), (1,3), (2,3), (2,4)]:
                pm = np.sum(st.session_state.matrix[(np.indices((10,10))[0] >= r[0]) & (np.indices((10,10))[0] <= r[1])])
                mgh_res.append({"Range": f"{r[0]}-{r[1]}", "Prob %": f"{pm:.1%}", "Quota": f"{1/pm:.2f}" if pm>0 else "-"})
            st.dataframe(pd.DataFrame(mgh_res), hide_index=True)
            
        with cm3:
            st.subheader("Multigol Ospite")
            mga_res = []
            for r in [(1,2), (1,3), (2,3), (2,4)]:
                pm = np.sum(st.session_state.matrix[(np.indices((10,10))[1] >= r[0]) & (np.indices((10,10))[1] <= r[1])])
                mga_res.append({"Range": f"{r[0]}-{r[1]}", "Prob %": f"{pm:.1%}", "Quota": f"{1/pm:.2f}" if pm>0 else "-"})
            st.dataframe(pd.DataFrame(mga_res), hide_index=True)

    # ---------------- TAB 4: PLAYER E ASSIST ----------------
    with tab4:
        st.subheader("Analisi Marcatore, Assist e Combo 1X2")
        if PLAYERS_DF is not None and not PLAYERS_DF.empty:
            team_sel = st.radio("Scegli Squadra", [f"Casa: {st.session_state.h_name}", f"Ospite: {st.session_state.a_name}"])
            is_home = "Casa" in team_sel
            t_match = engine.fuzzy_match_team(st.session_state.h_name if is_home else st.session_state.a_name, PLAYERS_DF['team'].unique())
            players_list = PLAYERS_DF[PLAYERS_DF['team'] == t_match]['player'].tolist() if t_match else []
            
            if players_list:
                pl_n = st.selectbox("Seleziona Giocatore", players_list)
                p_data = PLAYERS_DF[PLAYERS_DF['player'] == pl_n].iloc[0]
                p_xg_val = float(p_data.get('xg90', 0))
                p_xa_val = float(p_data.get('xa90', 0))
            else:
                st.warning("Giocatori non trovati. Inserimento manuale.")
                pl_n = st.text_input("Nome", "Player")
                c1, c2 = st.columns(2)
                p_xg_val = c1.number_input("xG/90 (Gol)", 0.0, 2.0, 0.4)
                p_xa_val = c2.number_input("xA/90 (Assist)", 0.0, 2.0, 0.2)
            
            txg = st.session_state.f_xh if is_home else st.session_state.f_xa
            t_type = "Casa" if is_home else "Ospite"
            t_avg_xg_seas = h_stats["total"]["xg_total"]/max(1, h_stats["total"]["matches"]) if is_home else a_stats["total"]["xg_total"]/max(1, a_stats["total"]["matches"])
            pmin = st.number_input("Minuti Previsti", 1, 100, 90)

            st.markdown("---")
            col_prob1, col_prob2 = st.columns(2)

            if p_xg_val > 0:
                pprob_gol = engine.calculate_player_probability(p_xg_val, pmin, txg, t_avg_xg_seas)
                col_prob1.success(f"⚽ **Probabilità GOL {pl_n}:** {pprob_gol:.1%} (@{1/pprob_gol:.2f})")
            
            if p_xa_val > 0:
                pprob_assist = engine.calculate_player_probability(p_xa_val, pmin, txg, t_avg_xg_seas)
                col_prob2.info(f"👟 **Probabilità ASSIST {pl_n}:** {pprob_assist:.1%} (@{1/pprob_assist:.2f})")
                
            st.markdown("### ⚡ Combo Player + 1X2")
            c_c1, c_c2 = st.columns(2)
            sel_res_p = c_c1.selectbox("Scegli Esito Match per la Combo", ["1", "X", "2", "1X", "X2", "12"])
            
            c_btn1, c_btn2 = st.columns(2)
            if c_btn1.button("Calcola Combo GOL + Esito"):
                share_g = min(0.99, (p_xg_val / 90 * pmin) / max(0.1, txg))
                p_combo_g = engine.calculate_combo_player(st.session_state.matrix, sel_res_p, t_type, share_g)
                st.success(f"Combo **{sel_res_p} + Gol {pl_n}**: {p_combo_g:.1%} (@{1/p_combo_g:.2f})")

            if c_btn2.button("Calcola Combo ASSIST + Esito"):
                share_a = min(0.99, (p_xa_val / 90 * pmin) / max(0.1, txg))
                p_combo_a = engine.calculate_combo_player(st.session_state.matrix, sel_res_p, t_type, share_a)
                st.info(f"Combo **{sel_res_p} + Assist {pl_n}**: {p_combo_a:.1%} (@{1/p_combo_a:.2f})")

    # ---------------- TAB 5: EXTRA ----------------
    with tab5:
        c_s1, c_s2 = st.columns(2)
        with c_s1:
            st.markdown("### 🚩 Angoli")
            corn = st.session_state.stats["corners"]
            st.caption(f"xCorners -> Casa: {corn['xH']:.1f} | Ospite: {corn['xA']:.1f}")
            st.dataframe(pd.DataFrame([{"Linea": k, "Over %": f"{v['prob']:.1%}", "Quota": f"{1/v['prob']:.2f}" if v['prob']>0.001 else "-"} for k,v in corn["lines"].items()]), hide_index=True)

            st.markdown("### 🟨 Cartellini")
            card = st.session_state.stats["cards"]
            st.dataframe(pd.DataFrame([{"Linea": k, "Over %": f"{v['prob']:.1%}", "Quota": f"{1/v['prob']:.2f}" if v['prob']>0.001 else "-"} for k,v in card["lines"].items()]), hide_index=True)

        with c_s2:
            st.markdown("### 🥅 Tiri Totali")
            shot = st.session_state.stats["shots"]
            st.dataframe(pd.DataFrame([{"Linea": k, "Over %": f"{v['prob']:.1%}", "Quota": f"{1/v['prob']:.2f}" if v['prob']>0.001 else "-"} for k,v in shot["lines"].items()]), hide_index=True)

            st.markdown("### 🎯 Tiri in Porta")
            sot = st.session_state.stats["sot"]
            st.dataframe(pd.DataFrame([{"Linea": k, "Over %": f"{v['prob']:.1%}", "Quota": f"{1/v['prob']:.2f}" if v['prob']>0.001 else "-"} for k,v in sot["lines"].items()]), hide_index=True)

    # ---------------- TAB 6: ML & STORICO ----------------
    with tab6:
        st.subheader("📝 Storico, Backup & Addestramento ML")
        
        c1, c2 = st.columns(2)
        if c1.button("💾 Salva Match Corrente nel Database"):
            st.session_state.history.append({
                "Match": f"{st.session_state.h_name} - {st.session_state.a_name}",
                "Data": str(datetime.now().strftime("%Y-%m-%d")),
                "f_xh": st.session_state.f_xh, "f_xa": st.session_state.f_xa,
                "P1_Stat": st.session_state.p1, "PX_Stat": st.session_state.pX, "P2_Stat": st.session_state.p2,
                "PO25_Stat": st.session_state.pO25, "PGG_Stat": st.session_state.pGG,
                "h_elo": st.session_state.h_elo, "a_elo": st.session_state.a_elo,
                "h_ppda": st.session_state.h_ppda, "a_ppda": st.session_state.a_ppda,
                "h_dc": st.session_state.h_dc, "a_dc": st.session_state.a_dc,
                "w_seas": st.session_state.w_seas, "volatility": st.session_state.volatility,
                "Real_Score": "-"
            })
            engine.salva_storico_json(st.session_state.history)
            st.success("Dati tattici salvati! Scarica il Backup a fine giornata se chiudi l'app.")
            
        excel_data = engine.generate_excel_report(st.session_state)
        c2.download_button("📊 Scarica Report Excel", excel_data, f"Mathbet_{datetime.now().strftime('%H%M')}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        st.divider()
        st.markdown("#### 🔄 Salvataggio di Sicurezza (Backup JSON)")
        
        col_dl, col_ul = st.columns(2)
        json_string = json.dumps(st.session_state.history, indent=2)
        col_dl.download_button("⬇️ Scarica Backup (JSON)", file_name="mathbet_backup.json", mime="application/json", data=json_string)
        
        uploaded_file = col_ul.file_uploader("⬆️ Ricarica Backup (JSON)", type="json")
        if uploaded_file is not None:
            try:
                loaded_history = json.load(uploaded_file)
                st.session_state.history = loaded_history
                engine.salva_storico_json(loaded_history)
                st.success("✅ Storico ripristinato con successo! Ricarica la pagina.")
            except Exception as e:
                st.error("Errore nel caricamento del file.")

        st.divider()
        st.markdown("#### Inserisci i Risultati Esatti")
        if st.session_state.history:
            for i, match in enumerate(reversed(st.session_state.history[-15:])):
                true_idx = len(st.session_state.history) - 1 - i
                col1, col2 = st.columns([3, 1])
                col1.write(f"**{match['Match']}** (xG: {match.get('f_xh',0):.2f} - {match.get('f_xa',0):.2f})")
                
                curr_score = match.get("Real_Score", "-")
                new_score = col2.text_input("Risultato Esatto", value="" if curr_score=="-" else curr_score, placeholder="es. 2-1", key=f"res_{true_idx}")
                
                if new_score and new_score != curr_score and "-" in new_score:
                    st.session_state.history[true_idx]["Real_Score"] = new_score.replace(" ", "")
                    engine.salva_storico_json(st.session_state.history)
                    st.toast(f"Risultato {new_score} acquisito!")

    # ---------------- TAB 7: UTILITY E COMBO ----------------
    with tab7:
        cu1, cu2 = st.columns(2)
        with cu1:
            st.subheader("⚡ Combo Maker")
            sel_res = st.selectbox("Esito 1X2", ["-", "1", "X", "2", "1X", "X2", "12"])
            sel_uo = st.selectbox("Under/Over", ["-", "Over 1.5", "Under 1.5", "Over 2.5", "Under 2.5", "Over 3.5", "Under 3.5"])
            sel_gg = st.selectbox("Goal/NoGoal", ["-", "Goal", "No Goal"])
            
            if st.button("Calcola Combo", type="primary"):
                prob_combo = 0.0
                for h in range(10):
                    for a in range(10):
                        p = st.session_state.matrix[h, a]
                        if p == 0: continue
                        if sel_res != "-":
                            if sel_res == '1' and not (h > a): continue
                            elif sel_res == 'X' and not (h == a): continue
                            elif sel_res == '2' and not (h < a): continue
                            elif sel_res == '1X' and not (h >= a): continue
                            elif sel_res == 'X2' and not (h <= a): continue
                            elif sel_res == '12' and not (h != a): continue
                        if sel_uo != "-":
                            lim = float(sel_uo.split()[1])
                            if ("Over" in sel_uo and (h+a) <= lim) or ("Under" in sel_uo and (h+a) > lim): continue
                        if sel_gg != "-" and (("Goal" in sel_gg and (h==0 or a==0)) or ("No Goal" in sel_gg and (h>0 and a>0))): continue
                        prob_combo += p
                if prob_combo > 0: 
                    st.success(f"Probabilità Combo: **{prob_combo:.1%}** (@{1/prob_combo:.2f})")
                else: 
                    st.warning("Evento impossibile (0%)")

        with cu2:
            st.subheader("💰 Gestione Bankroll (Kelly)")
            k_bank = st.number_input("Bankroll Totale (€)", 10.0, 100000.0, 1000.0, step=50.0)
            k_prob = st.number_input("Probabilità di Vittoria (%)", 0.1, 100.0, 55.0, step=1.0)
            k_odd = st.number_input("Quota Offerta dal Bookmaker", 1.01, 100.0, 2.00, step=0.05)
            
            val_perc = ((k_prob / 100) * k_odd) - 1
            if val_perc > 0: 
                st.success(f"✅ VALORE MATEMATICO TROVATO: **+{val_perc:.1%}**")
            else:
                st.error(f"❌ NESSUN VALORE. Ritorno atteso: **{val_perc:.1%}**")
                
            if k_odd > 1:
                b = k_odd - 1
                p = k_prob / 100
                q = 1 - p
                f = (b * p - q) / b 
                if f > 0: 
                    st.info(f"Puntata consigliata (1/4 Kelly): **€ {(f * 0.25) * k_bank:.2f}** ({f*0.25*100:.2f}% del bankroll)")
                else:
                    st.warning("Non puntare su questo evento (Valore Negativo).")

# ---------------- TAB 8: GIORNALISTA AI (GEMINI + WEB SCRAPING + COMBO) ----------------
    with tab8:
        st.subheader("✍️ Giornalista AI (Scraping Articoli Completi e Combo)")
        st.markdown("L'IA (il *Figghiozzo*) legge le notizie e incrocia i dati matematici per sfornare i migliori pronostici (incluse Combo e Marcatore/Assist).")
        
        # Inizializza la cronologia della chat se non esiste
        if "figghiozzo_chat" not in st.session_state:
            st.session_state.figghiozzo_chat = []
            
        if st.button("📝 Leggi Articoli e Genera Pronostico", type="primary"):
            if not GEMINI_API_KEY or GEMINI_API_KEY == "INSERISCI_QUI_LA_TUA_CHIAVE_GEMINI":
                st.warning("⚠️ Inserisci la tua vera API Key di Gemini all'inizio del codice (riga 15).")
            else:
                with st.spinner("🔍 Il Figghiozzo sta cercando notizie, estraendo le Combo e studiando i giocatori..."):
                    try:
                        from duckduckgo_search import DDGS
                        import google.generativeai as genai
                        from datetime import datetime
                        import cloudscraper
                        from bs4 import BeautifulSoup
                        import numpy as np # Assicurati di avere numpy importato in alto nel file
                        
                        oggi_str = datetime.now().strftime("%Y-%m-%d")
                        
                        # 1. RICERCA WEB E SCRAPING DEGLI ARTICOLI COMPLETI
                        search_query = f"{st.session_state.h_name} {st.session_state.a_name} probabili formazioni infortuni"
                        full_articles = []
                        scraper = cloudscraper.create_scraper(browser={'browser': 'chrome', 'platform': 'windows', 'desktop': True})
                        
                        try:
                            search_results = DDGS().news(search_query, region='it-it', timelimit='w', max_results=3)
                            
                            if not search_results:
                                search_query_2 = f"{st.session_state.h_name} infortunati squalificati out"
                                search_results = DDGS().text(search_query_2, region='it-it', timelimit='m', max_results=2)
                                search_query_3 = f"{st.session_state.a_name} infortunati squalificati out"
                                search_results += DDGS().text(search_query_3, region='it-it', timelimit='m', max_results=2)
                                
                            for res in search_results:
                                url = res.get('href') or res.get('url') or ''
                                if url:
                                    try:
                                        page = scraper.get(url, timeout=7)
                                        soup = BeautifulSoup(page.text, 'html.parser')
                                        paragraphs = soup.find_all('p')
                                        article_text = " ".join([p.get_text() for p in paragraphs])
                                        if len(article_text) < 150 or "cookie" in article_text.lower() or "javascript" in article_text.lower():
                                            full_articles.append(f"--- ARTICOLO DA: {res.get('source', 'Sito Web')} ---\nTitolo: {res.get('title', '')}\nRiassunto: {res.get('body', '')}\n")
                                        else:
                                            full_articles.append(f"--- ARTICOLO DA: {res.get('source', 'Sito Web')} ---\nTitolo: {res.get('title', '')}\nTesto: {article_text[:5000]}\n")
                                    except Exception:
                                        full_articles.append(f"--- ARTICOLO DA: {res.get('source', 'Sito Web')} ---\nTitolo: {res.get('title', '')}\nRiassunto: {res.get('body', '')}\n")
                            
                            if full_articles:
                                news_context = "\n".join(full_articles)
                            else:
                                news_context = "Nessuna notizia testuale trovata sul web per questa partita. Basati sulle tue conoscenze delle rose."
                                
                        except Exception as e:
                            news_context = "Ricerca web fallita. Basati solo sui dati matematici forniti e sulle tue conoscenze base."

                        # 2. ESTRAZIONE DATI E CALCOLO DELLE COMBO
                        p1, pX, p2 = st.session_state.p1, st.session_state.pX, st.session_state.p2
                        p1X, pX2, p12 = p1+pX, pX+p2, p1+p2
                        b1, bX, b2 = st.session_state.b1, st.session_state.bX, st.session_state.b2
                        val_1, val_X, val_2 = (b1*p1)-1, (bX*pX)-1, (b2*p2)-1
                        
                        mat = st.session_state.matrix
                        idx_h, idx_a = np.indices((10,10))
                        
                        pO15 = np.sum(mat[idx_h + idx_a > 1.5])
                        pU35 = np.sum(mat[idx_h + idx_a <= 3.5])
                        mg_1_3 = np.sum(mat[(idx_h + idx_a >= 1) & (idx_h + idx_a <= 3)])
                        mg_2_4 = np.sum(mat[(idx_h + idx_a >= 2) & (idx_h + idx_a <= 4)])
                        mgh_1_2 = np.sum(mat[(idx_h >= 1) & (idx_h <= 2)])
                        mga_1_2 = np.sum(mat[(idx_a >= 1) & (idx_a <= 2)])
                        ph_minus_15 = np.sum(mat[idx_h - idx_a >= 2])
                        pa_minus_15 = np.sum(mat[idx_a - idx_h >= 2])
                        
                        p1_O15 = np.sum(mat[(idx_h > idx_a) & (idx_h + idx_a > 1.5)])
                        p2_O15 = np.sum(mat[(idx_h < idx_a) & (idx_h + idx_a > 1.5)])
                        p1X_O15 = np.sum(mat[(idx_h >= idx_a) & (idx_h + idx_a > 1.5)])
                        p1X_U35 = np.sum(mat[(idx_h >= idx_a) & (idx_h + idx_a <= 3.5)])
                        pX2_O15 = np.sum(mat[(idx_h <= idx_a) & (idx_h + idx_a > 1.5)])
                        pX2_U35 = np.sum(mat[(idx_h <= idx_a) & (idx_h + idx_a <= 3.5)])
                        p1_GG = np.sum(mat[(idx_h > idx_a) & (idx_h > 0) & (idx_a > 0)])
                        p2_GG = np.sum(mat[(idx_h < idx_a) & (idx_h > 0) & (idx_a > 0)])

                        def qf(prob): return f"{1/prob:.2f}" if prob > 0.01 else "N/A"

                        # 3. CONFIGURA GEMINI
                        genai.configure(api_key=GEMINI_API_KEY)
                        modello_valido = None
                        for m in genai.list_models():
                            if 'generateContent' in m.supported_generation_methods:
                                modello_valido = m.name
                                if 'flash' in m.name: break
                                    
                        if not modello_valido:
                            st.error("❌ Nessun modello compatibile trovato.")
                            st.stop()
                            
                        model = genai.GenerativeModel(modello_valido)
                        
                        # 4. IL SUPER-PROMPT AGGIORNATO
                        prompt = f"""
                        Agisci come un Data Analyst calcistico Senior e Tipster Professionista. 
                        IL TUO NOME È "FIGGHIOZZO". Devi TASSATIVAMENTE iniziare la tua risposta presentandoti in modo amichevole e sicuro di te, dicendo una frase del tipo: "Ciao, sono il Figghiozzo e questa è la mia analisi..." oppure "Benvenuti, qui è il Figghiozzo che vi parla. Ecco i dati per il match...".
                        Dopo esserti presentato, torna ad essere un analista serio, non essere pigro, fornisci un'analisi lunga, dettagliata e precisa.
                        Oggi è il {oggi_str}. Stiamo analizzando la PROSSIMA partita imminente: {st.session_state.h_name} vs {st.session_state.a_name}.
                        
                        📊 1. METRICHE AVANZATE DEL SOFTWARE:
                        - Affidabilità Algoritmo (Stabilità): {st.session_state.stability:.1f}%
                        - xG Aggiustati: {st.session_state.h_name} {st.session_state.f_xh:.2f} | {st.session_state.a_name} {st.session_state.f_xa:.2f}
                        - Pressione (PPDA): Casa {st.session_state.h_ppda:.1f} | Ospite {st.session_state.a_ppda:.1f}
                        
                        💰 2. MERCATO 1X2 E VALUE BET RILEVATA:
                        - Esito 1: Probabilità {p1:.1%} | Quota Bookie: {b1} | Valore Matematico: {val_1:.1%}
                        - Esito X: Probabilità {pX:.1%} | Quota Bookie: {bX} | Valore Matematico: {val_X:.1%}
                        - Esito 2: Probabilità {p2:.1%} | Quota Bookie: {b2} | Valore Matematico: {val_2:.1%}
                        
                        🎯 3. MATRICE PROBABILITÀ E "QUOTE FAIR" (QUOTE REALI):
                        - Doppia Chance: 1X ({p1X:.1%}, Q.Fair: {qf(p1X)}) | X2 ({pX2:.1%}, Q.Fair: {qf(pX2)}) | 12 ({p12:.1%}, Q.Fair: {qf(p12)})
                        - Handicap (-1.5): Casa ({ph_minus_15:.1%}, Q.Fair: {qf(ph_minus_15)}) | Ospite ({pa_minus_15:.1%}, Q.Fair: {qf(pa_minus_15)})
                        - Under/Over: Over 1.5 ({pO15:.1%}, Q.Fair: {qf(pO15)}) | Over 2.5 ({st.session_state.pO25:.1%}, Q.Fair: {qf(st.session_state.pO25)}) | Under 3.5 ({pU35:.1%}, Q.Fair: {qf(pU35)})
                        - Goal/NoGoal: Goal ({st.session_state.pGG:.1%}, Q.Fair: {qf(st.session_state.pGG)})
                        - Multigol Partita: 1-3 ({mg_1_3:.1%}, Q.Fair: {qf(mg_1_3)}) | 2-4 ({mg_2_4:.1%}, Q.Fair: {qf(mg_2_4)})
                        
                        ⚡ 4. MERCATI COMBO (CON QUOTA FAIR CALCOLATA):
                        - 1 + Over 1.5 ({p1_O15:.1%}, Q.Fair: {qf(p1_O15)}) | 1X + Over 1.5 ({p1X_O15:.1%}, Q.Fair: {qf(p1X_O15)}) | 1X + Under 3.5 ({p1X_U35:.1%}, Q.Fair: {qf(p1X_U35)})
                        - 2 + Over 1.5 ({p2_O15:.1%}, Q.Fair: {qf(p2_O15)}) | X2 + Over 1.5 ({pX2_O15:.1%}, Q.Fair: {qf(pX2_O15)}) | X2 + Under 3.5 ({pX2_U35:.1%}, Q.Fair: {qf(pX2_U35)})
                        - 1 + Goal/BTTS ({p1_GG:.1%}, Q.Fair: {qf(p1_GG)}) | 2 + Goal/BTTS ({p2_GG:.1%}, Q.Fair: {qf(p2_GG)})
                        
                        📰 5. TESTO DEGLI ARTICOLI (Per infortuni e giocatori chiave):
                        {news_context}
                        
                        🔴 IL TUO COMPITO (RISPETTA QUESTA STRUTTURA ESATTA E NON ESSERE PIGRO):
                        1. **📋 Formazioni e Assenze:** Scrivi una stima delle formazioni in campo estraendo chi giocherà e chi è assente/infortunato in base agli articoli letti.
                        2. **🧠 Analisi Tattica Quantitativa:** Unisci i dati matematici (xG e PPDA) col contesto reale letto negli articoli. Analizza che tipo di partita ne uscirà. Sii analitico.
                        3. **💎 Pronostici Ufficiali Consigliati dal Figghiozzo:** Fornisci 3 pronostici accurati. Spiega ai lettori la regola d'oro: "Giocate questo pronostico SOLO se il bookmaker offre una quota uguale o superiore alla nostra Quota Fair".
                           - 🟢 **Safe Pick (Basso Rischio):** Scegli una giocata base o Multigol molto coperta. Indica la Quota Fair.
                           - 🟡 **Combo Pick (Valore Tattico):** Usa obbligatoriamente uno dei mercati Combo (sezione 4). Spiega perché l'andamento del match favorisce questa combo. Indica la sua Quota Fair.
                           - 🔴 **Player Pick (Marcatore/Assist + 1X2):** Basandoti sulle formazioni lette e sull'andamento tattico, consiglia una giocata speciale (Es. "Lautaro Segna + 1"). Spiega logicamente la scelta. IMPORTANTE: Concludi questo paragrafo dicendo testualmente: *"Per scoprire la Quota Fair matematica esatta di questa giocata speciale, inserite il nome del giocatore nell'apposita sezione 'Player & Assist' del nostro software (Tab 4)!"*
                        
                        ATTENZIONE: NON menzionare MAI angoli, tiri o cartellini. Scrivi da vero esperto di betting.
                        """
                        
                        # 5. INIZIALIZZA LA CHAT INVECE DELLA SINGOLA GENERAZIONE
                        st.session_state.figghiozzo_chat = [{"role": "user", "parts": [prompt]}]
                        
                        # Ottieni la prima risposta
                        chat_session = model.start_chat(history=[])
                        response = chat_session.send_message(prompt)
                        
                        # Salva la risposta dell'AI
                        st.session_state.figghiozzo_chat.append({"role": "model", "parts": [response.text]})
                        st.success("✅ Analisi del Figghiozzo generata con successo!")
                        
                    except ImportError:
                        st.error("🚨 Mancano le librerie. Aggiungi 'beautifulsoup4' e 'duckduckgo_search' al file requirements.txt e fai un Reboot.")
                    except Exception as e:
                        st.error(f"Errore nella generazione dell'analisi: {e}")

        # --- SEZIONE DI RENDER DELLA CHAT INTERATTIVA ---
        st.markdown("---")
        
        # Se c'è una conversazione attiva, mostrala
        if "figghiozzo_chat" in st.session_state and len(st.session_state.figghiozzo_chat) > 0:
            for message in st.session_state.figghiozzo_chat:
                # Nascondiamo il super-prompt iniziale (il primo messaggio) per non sporcare l'interfaccia
                if message["role"] == "user" and "Agisci come un Data Analyst calcistico" in message["parts"][0]:
                    continue
                
                with st.chat_message("user" if message["role"] == "user" else "assistant"):
                    st.markdown(message["parts"][0])
            
            # Input per rispondere al Figghiozzo
            if user_input := st.chat_input("Fai una domanda al Figghiozzo su questa partita..."):
                # Mostra subito il messaggio dell'utente a schermo
                with st.chat_message("user"):
                    st.markdown(user_input)
                
                # Aggiungi il messaggio alla history
                st.session_state.figghiozzo_chat.append({"role": "user", "parts": [user_input]})
                
                # Chiama Gemini passandogli tutto lo storico
                with st.spinner("Il Figghiozzo sta digitando..."):
                    try:
                        import google.generativeai as genai
                        genai.configure(api_key=GEMINI_API_KEY)
                        # Riconfiguriamo il modello al volo per la chat
                        for m in genai.list_models():
                            if 'generateContent' in m.supported_generation_methods and 'flash' in m.name:
                                modello_valido = m.name
                                break
                        model = genai.GenerativeModel(modello_valido)
                        
                        # Ricrea l'oggetto chat di Gemini con lo storico salvato (escluso l'ultimissimo messaggio utente che invieremo ora)
                        history_for_gemini = st.session_state.figghiozzo_chat[:-1] 
                        chat_session = model.start_chat(history=history_for_gemini)
                        
                        # Invia il nuovo messaggio
                        response = chat_session.send_message(user_input)
                        
                        # Salva e mostra la risposta
                        st.session_state.figghiozzo_chat.append({"role": "model", "parts": [response.text]})
                        with st.chat_message("assistant"):
                            st.markdown(response.text)
                            
                    except Exception as e:
                        st.error(f"Errore di connessione col Figghiozzo: {e}")

    # ---------------- TAB 9: TRADING LIVE ----------------
    with tab9:
        st.subheader("🔴 Trading Live (Sperimentale - Latenza Quasi-Zero)")
        st.markdown("Intercetta i dati JSON puri per ottenere il **Momentum** e le quote **Exchange Live**.")
        
        c_search1, c_search2 = st.columns([3, 1])
        with c_search1:
            st.info("💡 Suggerimento: Assicurati che la partita sia attualmente In-Play ed esatta.")
            
        with c_search2:
            if st.button("🔄 Cerca ID Evento Live", type="primary"):
                with st.spinner("Ricerca su SofaScore..."):
                    ev_id = live_scraper.get_sofascore_search(st.session_state.h_name, st.session_state.a_name)
                    if ev_id:
                        st.session_state.live_event_id = ev_id
                        st.success(f"Trovato! ID: {ev_id}")
                    else:
                        st.error("Partita non trovata live.")
        
        sofa_id_input = st.text_input("SofaScore Event ID", value=st.session_state.get("live_event_id", ""))
        betfair_market_input = st.text_input("Betfair Market ID (es. 1.123456789 - Opzionale)")
        
        if sofa_id_input:
            if st.button("📊 Estrai Dati e Momentum Live"):
                with st.spinner("Scraping Dati Live..."):
                    l_stats = live_scraper.get_sofascore_live_stats(sofa_id_input)
                    if l_stats:
                        minutes_played = st.number_input("Minuto Attuale", 1, 100, 45)
                        idx_h, idx_a = engine.calculate_live_pressure_index(l_stats['home'], l_stats['away'], minutes_played)
                        
                        col_l1, col_l2 = st.columns(2)
                        with col_l1:
                            st.write(f"### {st.session_state.h_name}")
                            st.metric("Pressure Index (Momentum)", f"{idx_h:.2f}")
                            st.write(f"Attacchi Pericolosi: {l_stats['home'].get('Dangerous attacks',0)}")
                            st.write(f"Tiri in Porta: {l_stats['home'].get('Shots on target',0)}")
                            
                        with col_l2:
                            st.write(f"### {st.session_state.a_name}")
                            st.metric("Pressure Index (Momentum)", f"{idx_a:.2f}")
                            st.write(f"Attacchi Pericolosi: {l_stats['away'].get('Dangerous attacks',0)}")
                            st.write(f"Tiri in Porta: {l_stats['away'].get('Shots on target',0)}")
                            
                        try:
                            # Se fornito l'ID mercato, potremo in futuro fare parse completo
                            live_odd_h, live_odd_a = st.session_state.b1, st.session_state.b2
                            if betfair_market_input:
                                bf_data = live_scraper.get_betfair_free_odds(betfair_market_input)
                                st.caption("Risposta Dati Betfair: Collegata")
                                if bf_data:
                                    with st.expander("👀 Dati Grezzi Betfair (Test)"):
                                        st.json(bf_data)
                        except:
                            live_odd_h, live_odd_a = st.session_state.b1, st.session_state.b2
                            
                        signals = engine.calculate_value_in_play(
                            st.session_state.p1, st.session_state.p2, 
                            minutes_played, live_odd_h, live_odd_a, 
                            idx_h, idx_a
                        )
                        
                        if signals:
                            st.warning("🚨 **SEGNALI DI TRADING RILEVATI IN-PLAY!**", icon="📈")
                            for s in signals:
                                st.success(f"**{s['Team']}** | {s['Signal']} (Quota nel db: {s['Odd Live']} / Fair ricalcolata: {s['Fair']})")
                        else:
                            st.info("Nessun segnale statistico evidente pre-bookmaker al momento.")
                        
                        st.session_state.last_momentum_h = idx_h
                        st.session_state.last_momentum_a = idx_a
