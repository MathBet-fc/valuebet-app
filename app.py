import streamlit as st
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import json
from datetime import datetime

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Mathbet fc - ML Ultimate", page_icon="🧠", layout="wide")

# --- PARAMETRI ML (GIÀ AGGIORNATI CON I TUOI DATI DEL TRAINING) ---
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
    """Calcola la probabilità di un risultato esatto con il modello Dixon-Coles"""
    prob = (math.exp(-mu_h) * (mu_h**h_goals) / math.factorial(h_goals)) * \
           (math.exp(-mu_a) * (mu_a**a_goals) / math.factorial(a_goals))
    if h_goals == 0 and a_goals == 0: prob *= (1.0 - (mu_h * mu_a * rho))
    elif h_goals == 0 and a_goals == 1: prob *= (1.0 + (mu_h * rho))
    elif h_goals == 1 and a_goals == 0: prob *= (1.0 + (mu_a * rho))
    elif h_goals == 1 and a_goals == 1: prob *= (1.0 - rho)
    return max(0.0, prob)

def calculate_player_probability(metric_per90, expected_mins, team_match_xg, team_avg_xg):
    """Calcola la probabilità che un giocatore segni"""
    base_lambda = (metric_per90 / 90.0) * expected_mins
    if team_avg_xg <= 0: team_avg_xg = 0.01
    match_factor = team_match_xg / team_avg_xg
    final_lambda = base_lambda * match_factor
    return 1 - math.exp(-final_lambda)

@st.cache_data
def monte_carlo_simulation(f_xh, f_xa, n_sims=5000):
    """Simulazione Monte Carlo con caching per performance migliori"""
    sim = []
    for _ in range(n_sims):
        gh = np.random.poisson(max(0.1, np.random.normal(f_xh, 0.15*f_xh)))
        ga = np.random.poisson(max(0.1, np.random.normal(f_xa, 0.15*f_xa)))
        sim.append(1 if gh>ga else (0 if gh==ga else 2))
    return sim

def calcola_forza_squadra(att_tot, def_tot, att_spec, def_spec, att_form, def_form, w_season):
    """
    Calcola la forza di una squadra bilanciando:
    - Dati stagionali totali
    - Dati specifici (casa/trasferta)
    - Forma recente
    
    Logica: 60% specifico + 40% totale per i dati stagionali, poi mix con forma
    """
    # Mix tra dati specifici (casa/trasferta) e totali stagionali
    # Diamo più peso ai dati specifici (60%) perché sono più rilevanti per il contesto
    att_season = (att_spec * 0.60) + (att_tot * 0.40)
    def_season = (def_spec * 0.60) + (def_tot * 0.40)
    
    # Poi bilancia stagione vs forma recente
    att = (att_season * w_season) + ((att_form/5.0) * (1-w_season))
    def_ = (def_season * w_season) + ((def_form/5.0) * (1-w_season))
    return att, def_

def valida_input(avg_goals, f_xh, f_xa):
    """Validazione input per prevenire errori"""
    errors = []
    if avg_goals <= 0:
        errors.append("⚠️ Media gol del campionato non valida")
    if f_xh < 0:
        errors.append("⚠️ xG Casa negativo - controlla i parametri")
    if f_xa < 0:
        errors.append("⚠️ xG Ospite negativo - controlla i parametri")
    return errors

def salva_storico_json():
    """Salva lo storico su file JSON"""
    try:
        with open('mathbet_history.json', 'w') as f:
            json.dump(st.session_state.history, f, indent=2, default=str)
        return True
    except Exception as e:
        st.error(f"Errore nel salvataggio: {e}")
        return False

def carica_storico_json():
    """Carica lo storico da file JSON"""
    try:
        with open('mathbet_history.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except Exception as e:
        st.error(f"Errore nel caricamento: {e}")
        return []

def export_risultati_csv(data):
    """Esporta i risultati in formato CSV"""
    df = pd.DataFrame([data])
    return df.to_csv(index=False).encode('utf-8')

# --- INIZIALIZZAZIONE ---
if 'history' not in st.session_state: 
    st.session_state.history = carica_storico_json()
if 'analyzed' not in st.session_state: 
    st.session_state.analyzed = False

# --- SIDEBAR ---
with st.sidebar:
    st.title("🧠 Configurazione ML")
    league_name = st.selectbox("Campionato", list(LEAGUES.keys()), 
                               help="Seleziona il campionato per utilizzare parametri ML ottimizzati")
    L_DATA = LEAGUES[league_name]
    
    # Mostriamo i parametri attivi per conferma
    st.info(f"**Parametri Attivi (ML):**\n\n🏠 Home Advantage: {L_DATA['ha']:.3f}\n\n🔗 Rho (Correlazione): {L_DATA['rho']:.3f}\n\n⚽ Media Gol: {L_DATA['avg']:.2f}")
    
    st.markdown("---")
    matchday = st.slider("Giornata (Peso Stagione)", 1, 38, 10, 
                        help="Più alta è la giornata, più contano i dati stagionali rispetto alla forma recente. Giornata 1 = 32% stagione, Giornata 38 = 90% stagione")
    # Calcolo peso dinamico: Giornata 1 = 32% stagione, Giornata 38 = 90% stagione
    w_seas = min(0.90, 0.30 + (matchday * 0.02)) 
    st.caption(f"Peso Stagionale: {w_seas:.0%} | Peso Forma: {(1-w_seas):.0%}")
    
    st.markdown("---")
    m_type = st.radio("Contesto", ["Standard", "Derby", "Campo Neutro"],
                     help="Standard: vantaggio casa normale | Derby: vantaggio casa ridotto | Campo Neutro: nessun vantaggio")
    is_big_match = st.checkbox("🔥 Big Match", 
                              help="Partite importanti tendono ad avere meno gol del previsto")
    
    st.markdown("---")
    st.subheader("📊 Storico")
    if st.session_state.history:
        st.metric("Analisi Salvate", len(st.session_state.history))
        if st.button("💾 Salva su File", use_container_width=True):
            if salva_storico_json():
                st.success("✅ Salvato!")
    
    st.markdown("---")
    st.caption("🧠 Mathbet FC ML v2.0")
    st.caption("Powered by Dixon-Coles Model")

st.title("Mathbet fc - ML Ultimate Edition 🚀")
st.caption("Sistema avanzato di previsione calcistica basato su Machine Learning")

# --- INPUT PRINCIPALE ---
col_h, col_a = st.columns(2)
h_uo_input, a_uo_input = {}, {}

# SQUADRA CASA
with col_h:
    st.subheader("🏠 Squadra Casa")
    h_name = st.text_input("Nome Casa", "Inter")
    h_elo = st.number_input("Rating Elo Casa", 1000.0, 2500.0, 1600.0, step=10.0,
                           help="Rating Elo: 1800+ = Top Team | 1600 = Medio-Alto | 1400 = Media | 1200- = Basso")
    
    with st.expander("📊 Dati Stagionali Totali", expanded=True):
        st.caption("Media Gol (Tutta la Stagione - Casa + Trasferta)")
        c1, c2 = st.columns(2)
        h_att_tot = c1.number_input("Gol Fatti/Partita Totali (C)", 0.0, 5.0, 1.85, 0.01,
                               help="Media gol fatti per partita in tutta la stagione (casa+trasferta)")
        h_def_tot = c2.number_input("Gol Subiti/Partita Totali (C)", 0.0, 5.0, 0.95, 0.01,
                               help="Media gol subiti per partita in tutta la stagione (casa+trasferta)")
    
    with st.expander("🏠 Dati Specifici IN CASA", expanded=True):
        st.caption("Media Gol nelle partite giocate IN CASA")
        c1h, c2h = st.columns(2)
        h_att_home = c1h.number_input("Gol Fatti/Partita in Casa", 0.0, 5.0, 2.10, 0.01,
                               help="Media gol fatti nelle partite IN CASA")
        h_def_home = c2h.number_input("Gol Subiti/Partita in Casa", 0.0, 5.0, 0.75, 0.01,
                               help="Media gol subiti nelle partite IN CASA")
        
    with st.expander("📈 Forma Recente (L5)", expanded=False):
        st.caption("Forma Recente (Totale ultime 5 partite)")
        c3, c4 = st.columns(2)
        h_form_att = c3.number_input("Gol Fatti L5 (C)", 0.0, 25.0, 9.0, 0.5,
                                    help="Totale gol fatti nelle ultime 5 partite")
        h_form_def = c4.number_input("Gol Subiti L5 (C)", 0.0, 25.0, 4.0, 0.5,
                                    help="Totale gol subiti nelle ultime 5 partite")

    with st.expander("📈 Trend Over (Opzionale)", expanded=False):
        st.caption("Percentuale storica di Over nelle partite della squadra")
        for l in [0.5, 1.5, 2.5, 3.5, 4.5]: 
            h_uo_input[l] = st.slider(f"Over {l} % Casa", 0, 100, 50, key=f"ho{l}",
                                     help=f"% di partite con più di {l} gol totali")

# SQUADRA OSPITE
with col_a:
    st.subheader("✈️ Squadra Ospite")
    a_name = st.text_input("Nome Ospite", "Juventus")
    a_elo = st.number_input("Rating Elo Ospite", 1000.0, 2500.0, 1550.0, step=10.0,
                           help="Rating Elo: 1800+ = Top Team | 1600 = Medio-Alto | 1400 = Media | 1200- = Basso")

    with st.expander("📊 Dati Stagionali Totali", expanded=True):
        st.caption("Media Gol (Tutta la Stagione - Casa + Trasferta)")
        c5, c6 = st.columns(2)
        a_att_tot = c5.number_input("Gol Fatti/Partita Totali (O)", 0.0, 5.0, 1.45, 0.01,
                               help="Media gol fatti per partita in tutta la stagione (casa+trasferta)")
        a_def_tot = c6.number_input("Gol Subiti/Partita Totali (O)", 0.0, 5.0, 0.85, 0.01,
                               help="Media gol subiti per partita in tutta la stagione (casa+trasferta)")
    
    with st.expander("✈️ Dati Specifici IN TRASFERTA", expanded=True):
        st.caption("Media Gol nelle partite giocate IN TRASFERTA")
        c5a, c6a = st.columns(2)
        a_att_away = c5a.number_input("Gol Fatti/Partita in Trasferta", 0.0, 5.0, 1.20, 0.01,
                               help="Media gol fatti nelle partite IN TRASFERTA")
        a_def_away = c6a.number_input("Gol Subiti/Partita in Trasferta", 0.0, 5.0, 1.05, 0.01,
                               help="Media gol subiti nelle partite IN TRASFERTA")
        
    with st.expander("📈 Forma Recente (L5)", expanded=False):
        st.caption("Forma Recente (Totale ultime 5 partite)")
        c7, c8 = st.columns(2)
        a_form_att = c7.number_input("Gol Fatti L5 (O)", 0.0, 25.0, 7.0, 0.5,
                                    help="Totale gol fatti nelle ultime 5 partite")
        a_form_def = c8.number_input("Gol Subiti L5 (O)", 0.0, 25.0, 3.0, 0.5,
                                    help="Totale gol subiti nelle ultime 5 partite")

    with st.expander("📈 Trend Over (Opzionale)", expanded=False):
        st.caption("Percentuale storica di Over nelle partite della squadra")
        for l in [0.5, 1.5, 2.5, 3.5, 4.5]: 
            a_uo_input[l] = st.slider(f"Over {l} % Ospite", 0, 100, 50, key=f"ao{l}",
                                     help=f"% di partite con più di {l} gol totali")

st.subheader("💰 Quote Bookmaker")
st.caption("Inserisci le quote per calcolare il value bet")
qc1, qc2, qc3 = st.columns(3)
b1 = qc1.number_input("Quota 1", 1.01, 100.0, 2.10, help="Quota vittoria casa")
bX = qc2.number_input("Quota X", 1.01, 100.0, 3.20, help="Quota pareggio")
b2 = qc3.number_input("Quota 2", 1.01, 100.0, 3.60, help="Quota vittoria ospite")

# --- OPZIONI AVANZATE (FINE TUNING - FUNZIONE ORIGINALE) ---
with st.expander("⚙️ Fine Tuning (Stanchezza & Assenze)"):
    st.caption("Regola gli xG in base a turnover, riposo e assenze chiave")
    c_str1, c_str2 = st.columns(2)
    h_str = c_str1.slider("Titolari % Casa", 50, 100, 100, 
                         help="% di titolari in campo - influenza la qualità complessiva")
    a_str = c_str2.slider("Titolari % Ospite", 50, 100, 100,
                         help="% di titolari in campo - influenza la qualità complessiva")
    h_rest = c_str1.slider("Riposo Casa (gg)", 2, 10, 7,
                          help="Giorni di riposo - meno di 4 giorni causa stanchezza")
    a_rest = c_str2.slider("Riposo Ospite (gg)", 2, 10, 7,
                          help="Giorni di riposo - meno di 4 giorni causa stanchezza")
    h_m_a = c_str1.checkbox("No Bomber Casa", help="Assenza attaccante principale (-15% xG)")
    a_m_a = c_str2.checkbox("No Bomber Ospite", help="Assenza attaccante principale (-15% xG)")
    h_m_d = c_str1.checkbox("No Difensore Casa", help="Assenza difensore chiave (+20% xG avversario)")
    a_m_d = c_str2.checkbox("No Difensore Ospite", help="Assenza difensore chiave (+20% xG avversario)")

# --- CALCOLO CORE ---
if st.button("🚀 ANALIZZA CON ML", type="primary", use_container_width=True):
    
    with st.spinner("🔄 Elaborazione in corso..."):
        progress_bar = st.progress(0)
        
        # 1. Recupero Parametri ML ottimizzati
        progress_bar.progress(10)
        home_adv_goals = L_DATA["ha"]
        rho_val = L_DATA["rho"]
        avg_goals_league = L_DATA["avg"]
        
        # Adattamento contesto
        if m_type == "Campo Neutro": home_adv_goals = 0.0
        elif m_type == "Derby": home_adv_goals *= 0.5

        # 2. Calcolo Forza Squadre (Mix pesato Totale/Specifico/Forma)
        progress_bar.progress(25)
        # Casa: usa dati IN CASA + totali + forma
        h_a_val, h_d_val = calcola_forza_squadra(h_att_tot, h_def_tot, h_att_home, h_def_home, 
                                                  h_form_att, h_form_def, w_seas)
        # Ospite: usa dati IN TRASFERTA + totali + forma
        a_a_val, a_d_val = calcola_forza_squadra(a_att_tot, a_def_tot, a_att_away, a_def_away, 
                                                  a_form_att, a_form_def, w_seas)
        
        # 3. Calcolo xG Base (Attacco x Difesa / Media)
        progress_bar.progress(40)
        xg_h_stats = (h_a_val * a_d_val) / avg_goals_league
        xg_a_stats = (a_a_val * h_d_val) / avg_goals_league
        
        # 4. Elo Adjustment (Correzione basata sulla forza storica)
        elo_diff = (h_elo + (100 if m_type=="Standard" else 0)) - a_elo
        elo_factor_h = 1 + (elo_diff / 1000.0)
        elo_factor_a = 1 - (elo_diff / 1000.0)
        
        # 5. Lambda Definitivi (Con Home Adv ottimizzato da ML)
        progress_bar.progress(50)
        f_xh = (xg_h_stats * elo_factor_h) + home_adv_goals
        f_xa = (xg_a_stats * elo_factor_a)
        
        # VALIDAZIONE INPUT
        errors = valida_input(avg_goals_league, f_xh, f_xa)
        if errors:
            progress_bar.empty()
            for error in errors:
                st.error(error)
            st.stop()
        
        # 6. Applicazione Malus Fine Tuning (FUNZIONE ORIGINALE)
        progress_bar.progress(60)
        fatigue_malus = 0.05 
        if h_rest <= 3: f_xh *= (1 - fatigue_malus); f_xa *= (1 + fatigue_malus) 
        if a_rest <= 3: f_xa *= (1 - fatigue_malus); f_xh *= (1 + fatigue_malus)
        f_xh *= (h_str/100.0)
        f_xa *= (a_str/100.0)
        
        if is_big_match: f_xh *= 0.90; f_xa *= 0.90
        if h_m_a: f_xh *= 0.85
        if h_m_d: f_xa *= 1.20
        if a_m_a: f_xa *= 0.85
        if a_m_d: f_xh *= 1.20

        # 7. Generazione Matrice Dixon-Coles
        progress_bar.progress(75)
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
                
        # Normalizzazione
        tot = np.sum(matrix)
        if tot > 0: matrix /= tot; p1 /= tot; pX /= tot; p2 /= tot; pGG /= tot
        p1X, pX2, p12 = p1+pX, pX+p2, p1+p2

        # 8. Calcolo Stabilità (Simulazione Monte Carlo con CACHE)
        progress_bar.progress(90)
        sim = monte_carlo_simulation(f_xh, f_xa, 5000)
        s1, sX, s2 = sim.count(1)/5000, sim.count(0)/5000, sim.count(2)/5000
        stability = max(0, 100 - ((abs(p1-s1)+abs(pX-sX)+abs(p2-s2))/3*400))

        # Salvataggio Stato
        progress_bar.progress(100)
        st.session_state.analyzed = True
        st.session_state.f_xh = f_xh; st.session_state.f_xa = f_xa
        st.session_state.h_name = h_name; st.session_state.a_name = a_name
        st.session_state.p1 = p1; st.session_state.pX = pX; st.session_state.p2 = p2
        st.session_state.p1X = p1X; st.session_state.pX2 = pX2; st.session_state.p12 = p12
        st.session_state.pGG = pGG; st.session_state.stability = stability
        st.session_state.matrix = matrix; st.session_state.scores = scores
        st.session_state.b1 = b1; st.session_state.bX = bX; st.session_state.b2 = b2
        st.session_state.league = league_name
        st.session_state.avg_goals_league = avg_goals_league
        
        progress_bar.empty()
        st.success("✅ Analisi completata con successo!")

# --- OUTPUT VISIVO ---
if st.session_state.analyzed:
    st.markdown("---")
    st.header(f"📊 {st.session_state.h_name} vs {st.session_state.a_name}")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Expected Goals (xG)", 
                  f"{st.session_state.f_xh:.2f} - {st.session_state.f_xa:.2f}")
    col_m2.metric("Stabilità Modello", 
                  f"{st.session_state.stability:.1f}%",
                  delta="Eccellente" if st.session_state.stability > 85 else "Buona" if st.session_state.stability > 70 else "Media")
    col_m3.metric("Campionato",
                  st.session_state.league.split(" ")[-1] if " " in st.session_state.league else "Generico")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏆 Esito", "⚽ Gol", "👤 Marcatori", "📊 Grafici", "📝 Storico"])

    with tab1:
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Probabilità 1X2")
            df_1x2 = pd.DataFrame({
                "Esito": ["1 (Casa)", "X (Pareggio)", "2 (Ospite)", "1X", "X2", "12"],
                "Prob %": [f"{st.session_state.p1:.1%}", f"{st.session_state.pX:.1%}", f"{st.session_state.p2:.1%}", 
                          f"{st.session_state.p1X:.1%}", f"{st.session_state.pX2:.1%}", f"{st.session_state.p12:.1%}"],
                "Fair Odd": [f"{1/st.session_state.p1:.2f}", f"{1/st.session_state.pX:.2f}", f"{1/st.session_state.p2:.2f}", 
                            f"{1/st.session_state.p1X:.2f}", f"{1/st.session_state.pX2:.2f}", f"{1/st.session_state.p12:.2f}"],
                "Bookie": [st.session_state.b1, st.session_state.bX, st.session_state.b2, "-", "-", "-"],
                "Value %": [f"{(st.session_state.b1*st.session_state.p1-1)*100:.1f}%", 
                           f"{(st.session_state.bX*st.session_state.pX-1)*100:.1f}%", 
                           f"{(st.session_state.b2*st.session_state.p2-1)*100:.1f}%", "-", "-", "-"]
            })
            
            # Evidenzia value bet
            def highlight_value(row):
                if row['Esito'] in ['1 (Casa)', 'X (Pareggio)', '2 (Ospite)']:
                    val = float(row['Value %'].strip('%'))
                    if val > 5:
                        return ['background-color: #90EE90'] * len(row)
                    elif val > 0:
                        return ['background-color: #FFFFE0'] * len(row)
                return [''] * len(row)
            
            st.dataframe(df_1x2.style.apply(highlight_value, axis=1), hide_index=True, use_container_width=True)
            
            st.caption("🟢 Verde = Value Bet forte (>5%) | 🟡 Giallo = Value Bet (>0%)")
            
        with c2:
            st.subheader("Risultati Esatti Top")
            scores = st.session_state.scores.copy()
            scores.sort(key=lambda x: x["Prob"], reverse=True)
            df_scores = pd.DataFrame([{
                "Score": s["Risultato"], 
                "Probabilità": f"{s['Prob']:.1%}", 
                "Quota Fair": f"{1/s['Prob']:.2f}"
            } for s in scores[:8]])
            st.dataframe(df_scores, hide_index=True, use_container_width=True)
            
            st.caption("Heatmap Probabilità Risultati")
            fig, ax = plt.subplots(figsize=(6,4))
            sns.heatmap(st.session_state.matrix[:6,:6], annot=True, fmt=".1%", 
                       cmap="RdYlGn", cbar=True, xticklabels=range(6), yticklabels=range(6))
            plt.xlabel("Gol Ospite"); plt.ylabel("Gol Casa")
            plt.title("Matrice Dixon-Coles")
            st.pyplot(fig)

    with tab2:
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Under / Over")
            uo_res = []
            for l in [0.5, 1.5, 2.5, 3.5, 4.5]:
                p_pure = np.sum(st.session_state.matrix[np.indices((10,10))[0] + np.indices((10,10))[1] > l])
                # Mix tra calcolo puro e trend inserito dall'utente
                trend = (h_uo_input.get(l,50) + a_uo_input.get(l,50))/200.0
                p_final = (p_pure * 0.7) + (trend * 0.3) 
                uo_res.append({
                    "Linea": f"O/U {l}", 
                    "Under %": f"{(1-p_final):.1%}", 
                    "Quota U": f"{1/(1-p_final):.2f}", 
                    "Over %": f"{p_final:.1%}", 
                    "Quota O": f"{1/p_final:.2f}"
                })
            st.dataframe(pd.DataFrame(uo_res), hide_index=True, use_container_width=True)
            
        with c2:
            st.subheader("Gol & Handicap")
            df_gg = pd.DataFrame([
                {"Esito": "Goal (GG)", "Probabilità": f"{st.session_state.pGG:.1%}", "Quota Fair": f"{1/st.session_state.pGG:.2f}"}, 
                {"Esito": "No Goal (NG)", "Probabilità": f"{(1-st.session_state.pGG):.1%}", "Quota Fair": f"{1/(1-st.session_state.pGG):.2f}"}
            ])
            st.dataframe(df_gg, hide_index=True, use_container_width=True)
            
            st.markdown("**Handicap Asiatico**")
            h_hand = np.sum(st.session_state.matrix[np.indices((10,10))[0] - 1 > np.indices((10,10))[1]])
            a_hand = np.sum(st.session_state.matrix[np.indices((10,10))[0] + 1 < np.indices((10,10))[1]])
            
            st.write(f"🏠 **Casa (-1):** {h_hand:.1%} (@{1/h_hand:.2f})")
            st.write(f"✈️ **Ospite (-1):** {a_hand:.1%} (@{1/a_hand:.2f})")
            
            # Multigol
            st.markdown("**Multigol**")
            mg_1_2 = np.sum(st.session_state.matrix[(np.indices((10,10))[0] + np.indices((10,10))[1] >= 1) & 
                                                    (np.indices((10,10))[0] + np.indices((10,10))[1] <= 2)])
            mg_2_3 = np.sum(st.session_state.matrix[(np.indices((10,10))[0] + np.indices((10,10))[1] >= 2) & 
                                                    (np.indices((10,10))[0] + np.indices((10,10))[1] <= 3)])
            mg_3_4 = np.sum(st.session_state.matrix[(np.indices((10,10))[0] + np.indices((10,10))[1] >= 3) & 
                                                    (np.indices((10,10))[0] + np.indices((10,10))[1] <= 4)])
            st.write(f"1-2 Gol: {mg_1_2:.1%} (@{1/mg_1_2:.2f})")
            st.write(f"2-3 Gol: {mg_2_3:.1%} (@{1/mg_2_3:.2f})")
            st.write(f"3-4 Gol: {mg_3_4:.1%} (@{1/mg_3_4:.2f})")

    with tab3:
        st.subheader("🎯 Calcolatore Marcatore")
        st.caption("Stima la probabilità che un giocatore segni in base ai suoi xG e ai minuti previsti")
        
        c1, c2 = st.columns(2)
        pl_n = c1.text_input("Nome Giocatore", "Vlahovic")
        pl_xg = c1.number_input("xG per 90 minuti", 0.01, 2.0, 0.45, 0.01,
                               help="Expected Goals medi ogni 90 minuti giocati")
        pl_min = c2.number_input("Minuti Previsti", 1, 100, 85, 1,
                                help="Minuti che il giocatore dovrebbe giocare")
        is_home_team = c1.checkbox("È della squadra di Casa?", value=True)
        
        # Usa gli xG totali calcolati per la squadra e la media del campionato
        team_xg = st.session_state.f_xh if is_home_team else st.session_state.f_xa
        team_avg_xg = st.session_state.avg_goals_league
        
        p_goal = calculate_player_probability(pl_xg, pl_min, team_xg, team_avg_xg)
        p_2plus = calculate_player_probability(pl_xg * 1.8, pl_min, team_xg, team_avg_xg)
        
        col1, col2, col3 = st.columns(3)
        col1.metric(f"Prob. Gol {pl_n}", f"{p_goal:.1%}")
        col2.metric("Quota Fair", f"{1/p_goal:.2f}")
        col3.metric("Prob. 2+ Gol", f"{p_2plus:.1%}")
        
        st.info(f"💡 Con xG di squadra a {team_xg:.2f} e media campionato {team_avg_xg:.2f}, "
               f"il giocatore ha un fattore moltiplicativo di {team_xg/team_avg_xg:.2f}x")

    with tab4:
        st.subheader("📊 Visualizzazioni Avanzate")
        
        # Grafico a barre probabilità 1X2
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = go.Figure(data=[
                go.Bar(name='Probabilità', 
                      x=['1 (Casa)', 'X', '2 (Ospite)'], 
                      y=[st.session_state.p1*100, st.session_state.pX*100, st.session_state.p2*100],
                      marker_color=['#2E7D32', '#FBC02D', '#C62828'],
                      text=[f"{st.session_state.p1:.1%}", f"{st.session_state.pX:.1%}", f"{st.session_state.p2:.1%}"],
                      textposition='auto')
            ])
            fig1.update_layout(title="Distribuzione Probabilità 1X2",
                             yaxis_title="Probabilità %",
                             showlegend=False,
                             height=400)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Confronto Fair Odds vs Bookie
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(name='Quota Fair',
                                 x=['1', 'X', '2'],
                                 y=[1/st.session_state.p1, 1/st.session_state.pX, 1/st.session_state.p2],
                                 marker_color='lightblue'))
            fig2.add_trace(go.Bar(name='Quota Bookmaker',
                                 x=['1', 'X', '2'],
                                 y=[st.session_state.b1, st.session_state.bX, st.session_state.b2],
                                 marker_color='coral'))
            fig2.update_layout(title="Confronto Quote Fair vs Bookmaker",
                             yaxis_title="Quota",
                             barmode='group',
                             height=400)
            st.plotly_chart(fig2, use_container_width=True)
        
        # Distribuzione gol totali
        gol_dist = []
        for tot_gol in range(11):
            prob = np.sum(st.session_state.matrix[np.indices((10,10))[0] + np.indices((10,10))[1] == tot_gol])
            gol_dist.append({"Gol": tot_gol, "Probabilità": prob*100})
        
        df_gol = pd.DataFrame(gol_dist)
        fig3 = px.bar(df_gol, x='Gol', y='Probabilità', 
                     title='Distribuzione Probabilità Gol Totali',
                     labels={'Probabilità': 'Probabilità %'},
                     color='Probabilità',
                     color_continuous_scale='Greens')
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)

    with tab5:
        st.subheader("📝 Storico Analisi & Backtesting")
        
        col_save1, col_save2 = st.columns([3, 1])
        
        with col_save1:
            if st.button("💾 Salva Risultato Corrente", use_container_width=True):
                risultato_corrente = {
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "Match": f"{st.session_state.h_name} - {st.session_state.a_name}",
                    "xG": f"{st.session_state.f_xh:.2f} - {st.session_state.f_xa:.2f}",
                    "P1": round(st.session_state.p1, 3),
                    "PX": round(st.session_state.pX, 3),
                    "P2": round(st.session_state.p2, 3),
                    "Stabilità": round(st.session_state.stability, 1),
                    "Campionato": st.session_state.league,
                    "Esito Reale": "?"
                }
                st.session_state.history.append(risultato_corrente)
                st.success("✅ Risultato salvato nello storico!")
        
        with col_save2:
            if st.session_state.history:
                # Export CSV
                csv_data = export_risultati_csv(st.session_state.history[-1])
                st.download_button(
                    label="📥 Export CSV",
                    data=csv_data,
                    file_name=f"mathbet_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        if st.session_state.history:
            st.markdown("---")
            st.write(f"**Totale Analisi Salvate:** {len(st.session_state.history)}")
            
            # Editor con risultati reali
            df_history = pd.DataFrame(st.session_state.history)
            
            edited_df = st.data_editor(
                df_history,
                column_config={
                    "Esito Reale": st.column_config.SelectboxColumn(
                        "Esito Reale",
                        options=["1", "X", "2", "?"],
                        help="Inserisci il risultato reale della partita",
                        required=True
                    ),
                    "Timestamp": st.column_config.TextColumn("Data/Ora", disabled=True),
                    "Match": st.column_config.TextColumn("Partita", disabled=True),
                    "P1": st.column_config.NumberColumn("Prob 1", format="%.3f", disabled=True),
                    "PX": st.column_config.NumberColumn("Prob X", format="%.3f", disabled=True),
                    "P2": st.column_config.NumberColumn("Prob 2", format="%.3f", disabled=True),
                },
                hide_index=True,
                use_container_width=True,
                num_rows="dynamic"
            )
            
            # Aggiorna storico con modifiche
            st.session_state.history = edited_df.to_dict('records')
            
            # Calcolo Brier Score e Accuracy
            validated = edited_df[edited_df["Esito Reale"] != "?"]
            
            if not validated.empty:
                st.markdown("---")
                st.subheader("📈 Performance del Modello")
                
                col_perf1, col_perf2, col_perf3, col_perf4 = st.columns(4)
                
                # Brier Score
                brier_scores = []
                correct_predictions = 0
                
                for _, row in validated.iterrows():
                    actual = [1 if row["Esito Reale"]=="1" else 0, 
                             1 if row["Esito Reale"]=="X" else 0, 
                             1 if row["Esito Reale"]=="2" else 0]
                    predicted = [row["P1"], row["PX"], row["P2"]]
                    
                    brier = sum((predicted[i] - actual[i])**2 for i in range(3))
                    brier_scores.append(brier)
                    
                    # Check se la previsione con prob più alta è corretta
                    max_prob_idx = predicted.index(max(predicted))
                    if actual[max_prob_idx] == 1:
                        correct_predictions += 1
                
                avg_brier = np.mean(brier_scores)
                accuracy = (correct_predictions / len(validated)) * 100
                
                col_perf1.metric("Brier Score", f"{avg_brier:.3f}",
                               help="0 = Perfetto | <0.25 = Ottimo | <0.50 = Buono | >0.50 = Da migliorare")
                col_perf2.metric("Accuracy", f"{accuracy:.1f}%",
                               help="% di previsioni corrette (esito con prob maggiore)")
                col_perf3.metric("Partite Validate", len(validated))
                col_perf4.metric("Stabilità Media", f"{validated['Stabilità'].mean():.1f}%")
                
                # Grafico performance nel tempo
                if len(validated) > 1:
                    fig_perf = go.Figure()
                    fig_perf.add_trace(go.Scatter(
                        y=brier_scores,
                        mode='lines+markers',
                        name='Brier Score',
                        line=dict(color='royalblue', width=2),
                        marker=dict(size=8)
                    ))
                    fig_perf.add_hline(y=0.25, line_dash="dash", line_color="green", 
                                      annotation_text="Soglia Ottimo")
                    fig_perf.update_layout(
                        title="Evoluzione Brier Score nel Tempo",
                        xaxis_title="Partita #",
                        yaxis_title="Brier Score",
                        height=300
                    )
                    st.plotly_chart(fig_perf, use_container_width=True)
            
            # Pulsanti gestione storico
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("🗑️ Reset Storico", use_container_width=True):
                    st.session_state.history = []
                    st.rerun()
            with col_btn2:
                if st.button("💾 Salva su File JSON", use_container_width=True):
                    if salva_storico_json():
                        st.success("✅ Storico salvato su mathbet_history.json!")
        else:
            st.info("📭 Nessuna analisi salvata. Clicca su 'Salva Risultato Corrente' per iniziare il tracking!")

# --- FOOTER ---
st.markdown("---")
st.caption("🧠 Mathbet FC - ML Ultimate v2.0 | Modello Dixon-Coles con parametri ottimizzati per campionato")
st.caption("⚠️ Questo tool è solo a scopo educativo. Gioca responsabilmente.")
