import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import engine 

# ==============================================================================
# CONFIGURAZIONE API & PAGINA
# ==============================================================================
ODDS_API_KEY = "1b31f14ebbc80b505c8412a5dc4d6791"

st.set_page_config(page_title="Mathbet fc - ML Ultimate Pro", page_icon="🧠", layout="wide")

if 'history' not in st.session_state: st.session_state.history = engine.carica_storico_json()
if 'analyzed' not in st.session_state: st.session_state.analyzed = False

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
        ELO_DICT = engine.fetch_clubelo_ratings() # 🟢 NUOVO: Carica Elo Rating
        ALL_LEAGUE_ODDS = engine.fetch_league_odds(L_DATA["odds_id"], ODDS_API_KEY)
    
    if STATS_DB: st.success("✅ Dati Squadre Sincronizzati")
    if ELO_DICT: st.success("✅ ClubElo Rating Sincronizzato")
    
    st.markdown("---")
    data_mode = st.radio("Dati Analisi", ["Solo Gol Reali", "Solo xG (NPxG Mode)", "Ibrido (Consigliato)"], index=2)
    volatility = st.slider("Volatilità", 0.8, 1.4, 1.0, 0.05)
    matchday = st.slider("Giornata", 1, 38, 22)
    m_type = st.radio("Contesto", ["Standard", "Derby", "Campo Neutro"])
    is_big_match = st.checkbox("🔥 Big Match")
    
    with st.expander("🔢 Dati Calci d'Angolo (Modello xCorners)", expanded=False):
        avg_lega_corn = st.number_input("Media Angoli Lega (Tot match)", 5.0, 15.0, 10.0, 0.1)
        c1, c2 = st.columns(2)
        corn_f_h = c1.number_input("Fatti Casa", 0.0, 15.0, 5.5, 0.1)
        corn_s_h = c1.number_input("Subiti Casa", 0.0, 15.0, 4.5, 0.1)
        corn_f_a = c2.number_input("Fatti Ospite", 0.0, 15.0, 4.5, 0.1)
        corn_s_a = c2.number_input("Subiti Ospite", 0.0, 15.0, 5.5, 0.1)

    with st.expander("🔢 Altri Dati Extra (Tiri, Cartellini)", expanded=False):
        c_card_h = st.number_input("Cartellini Casa (Avg)", 0.0, 10.0, 2.0, 0.1)
        c_card_a = st.number_input("Cartellini Ospite (Avg)", 0.0, 10.0, 2.5, 0.1)
        c_shot_h = st.number_input("Tiri Totali Casa", 0.0, 50.0, 12.0, 0.5)
        c_shot_a = st.number_input("Tiri Totali Ospite", 0.0, 50.0, 10.0, 0.5)
        c_sot_h = st.number_input("Tiri in Porta Casa", 0.0, 20.0, 4.5, 0.1)
        c_sot_a = st.number_input("Tiri in Porta Ospite", 0.0, 20.0, 3.5, 0.1)
        c_foul_h = st.number_input("Falli Casa", 0.0, 30.0, 11.5, 0.5)
        c_foul_a = st.number_input("Falli Ospite", 0.0, 30.0, 12.5, 0.5)

    with st.expander("🧮 Calcolatori Utility", expanded=False):
        my_prob = st.number_input("Tua Probabilità (%)", 0.1, 100.0, 50.0, step=1.0)
        book_odd = st.number_input("Quota Bookmaker", 1.01, 100.0, 2.00, step=0.05)
        value_calc = ((my_prob / 100) * book_odd) - 1
        if value_calc > 0: st.success(f"✅ VALUE BET! (+{value_calc:.1%})")
        st.divider()
        k_bank = st.number_input("Bankroll Totale (€)", 0.0, 100000.0, 1000.0, step=10.0)
        k_prob = st.number_input("Prob Vittoria (%) (K)", 0.1, 100.0, 55.0, step=1.0)
        k_odd = st.number_input("Quota Evento (K)", 1.01, 100.0, 2.00, step=0.05)
        if k_odd > 1:
            b = k_odd - 1; p = k_prob / 100; q = 1 - p
            f = (b * p - q) / b 
            if f > 0: st.success(f"💰 Punta: **€ {(f * 0.25) * k_bank:.2f}** ({f*0.25*100:.2f}%)")

st.title("Mathbet fc - ML Ultimate Pro 🚀")

# --- SCANNER AUTOMATICO TOP 5 ---
with st.expander("🔥 SCANNER GIORNALIERO: TOP 5 VALUE BETS", expanded=False):
    st.markdown("Cerca le migliori occasioni matematiche sul palinsesto (Scraping Integrato ClubElo).")
    if st.button("🔍 Cerca Top 5 Value Bets Ora", type="primary"):
        with st.spinner(f"Elaborazione in corso..."):
            # 🟢 INIETTA ELO_DICT NELLO SCANNER
            top_5 = engine.find_top_value_bets(ALL_LEAGUE_ODDS, STATS_DB, L_DATA, volatility, m_type, ELO_DICT)
            if top_5:
                df_top5 = pd.DataFrame(top_5)
                st.table(df_top5.style.format({"Prob %": "{:.1%}", "Fair Odd": "{:.2f}", "Valore %": "{:.1%}"}).applymap(lambda x: 'background-color: #d4edda; font-weight: bold;', subset=['Valore %']))
            else:
                st.warning("Nessuna Value Bet significativa (>3%) trovata.")
st.divider()

col_h, col_a = st.columns(2)
h_uo_input, a_uo_input = {}, {}

def get_val_ui(stats_dict, metric, mode):
    raw = stats_dict["total"]
    matches = raw.get('matches', 0)
    if matches <= 0: return 0.0
    if metric == 'gf': 
        v_goals, v_xg, v_npxg = raw.get('goals_total', 0), raw.get('xg_total', 0), raw.get('npxg_total', 0)
    else: 
        v_goals, v_xg, v_npxg = raw.get('ga_total', 0), raw.get('xga_total', 0), raw.get('npxga_total', 0)
    if mode == "Solo Gol Reali": return v_goals / matches
    elif mode == "Solo xG (NPxG Mode)": return v_npxg / matches
    else: return ((v_npxg * 0.40) + (v_xg * 0.30) + (v_goals * 0.30)) / matches

with col_h:
    st.subheader("🏠 Squadra Casa")
    h_name = st.selectbox("Seleziona Casa", TEAM_LIST, index=0) if TEAM_LIST else st.text_input("Nome Casa", "Inter")
    h_stats = STATS_DB.get(h_name) if STATS_DB else None
    
    # 🟢 CARICAMENTO AUTOMATICO ELO
    default_elo_h = engine.get_elo_for_team(h_name, ELO_DICT, 1600.0) if h_name else 1600.0
    h_elo = st.number_input("Rating Elo Casa (Auto)", 1000.0, 2500.0, float(default_elo_h), step=10.0)
    
    with st.expander("📊 Dati", expanded=True):
        def_att_s = get_val_ui(h_stats, 'gf', data_mode) if h_stats else 1.85
        def_def_s = get_val_ui(h_stats, 'gs', data_mode) if h_stats else 0.95
        c1, c2 = st.columns(2)
        h_att = c1.number_input("Attacco Totale (C)", 0.0, 5.0, float(def_att_s), 0.01)
        h_def = c2.number_input("Difesa Totale (C)", 0.0, 5.0, float(def_def_s), 0.01)
        c3, c4 = st.columns(2)
        h_att_home = c3.number_input("Attacco Casa", 0.0, 5.0, float(def_att_s*1.15), 0.01)
        h_def_home = c4.number_input("Difesa Casa", 0.0, 5.0, float(def_def_s*0.85), 0.01)
    with st.expander("Over Trend"):
        for l in [0.5, 1.5, 2.5, 3.5, 4.5]: h_uo_input[l] = st.slider(f"Over {l} % H", 0, 100, 50, key=f"ho{l}")

with col_a:
    st.subheader("✈️ Squadra Ospite")
    a_name = st.selectbox("Seleziona Ospite", TEAM_LIST, index=1 if len(TEAM_LIST)>1 else 0) if TEAM_LIST else st.text_input("Nome Ospite", "Juve")
    a_stats = STATS_DB.get(a_name) if STATS_DB else None
    
    # 🟢 CARICAMENTO AUTOMATICO ELO
    default_elo_a = engine.get_elo_for_team(a_name, ELO_DICT, 1550.0) if a_name else 1550.0
    a_elo = st.number_input("Rating Elo Ospite (Auto)", 1000.0, 2500.0, float(default_elo_a), step=10.0)
    
    with st.expander("📊 Dati", expanded=True):
        def_att_s_a = get_val_ui(a_stats, 'gf', data_mode) if a_stats else 1.45
        def_def_s_a = get_val_ui(a_stats, 'gs', data_mode) if a_stats else 0.85
        c5, c6 = st.columns(2)
        a_att = c5.number_input("Attacco Totale (O)", 0.0, 5.0, float(def_att_s_a), 0.01)
        a_def = c6.number_input("Difesa Totale (O)", 0.0, 5.0, float(def_def_s_a), 0.01)
        c7, c8 = st.columns(2)
        a_att_away = c7.number_input("Attacco Fuori", 0.0, 5.0, float(def_att_s_a*0.85), 0.01)
        a_def_away = c8.number_input("Difesa Fuori", 0.0, 5.0, float(def_def_s_a*1.15), 0.01)
    with st.expander("Over Trend"):
        for l in [0.5, 1.5, 2.5, 3.5, 4.5]: a_uo_input[l] = st.slider(f"Over {l} % A", 0, 100, 50, key=f"ao{l}")

live_match_odds = engine.extract_match_odds(ALL_LEAGUE_ODDS, h_name, a_name)

st.subheader("💰 Quote Reali")
if not live_match_odds:
    st.warning(f"⚠️ Quote non trovate nel palinsesto per {h_name} - {a_name}. Il match potrebbe essere passato, troppo lontano nel tempo, o con nomi incompatibili. Inserimento manuale attivo:")
else:
    st.success("✅ Quote sincronizzate con successo dai Bookmaker")

q1, qx, q2 = st.columns(3)
val_1 = live_match_odds['h2h'].get(h_name, 2.10) if live_match_odds and 'h2h' in live_match_odds else 2.10
val_X = live_match_odds['h2h'].get('Draw', 3.20) if live_match_odds and 'h2h' in live_match_odds else 3.20
val_2 = live_match_odds['h2h'].get(a_name, 3.60) if live_match_odds and 'h2h' in live_match_odds else 3.60

b1 = q1.number_input("Q1", 1.01, 100.0, float(val_1))
bX = qx.number_input("QX", 1.01, 100.0, float(val_X))
b2 = q2.number_input("Q2", 1.01, 100.0, float(val_2))

with st.expander("⚙️ Fine Tuning"):
    c1, c2 = st.columns(2)
    h_str = c1.slider("Titolari % Casa", 50, 100, 100); a_str = c2.slider("Titolari % Ospite", 50, 100, 100)
    h_rest = c1.slider("Riposo Casa", 2, 10, 7); a_rest = c2.slider("Riposo Ospite", 2, 10, 7)
    h_m_a = c1.checkbox("No Bomber Casa"); a_m_a = c2.checkbox("No Bomber Ospite")
    h_m_d = c1.checkbox("No Difensore Casa"); a_m_d = c2.checkbox("No Difensore Ospite")

# ==============================================================================
# 🚀 TRIGGER ANALISI E INVIO A ENGINE
# ==============================================================================
if st.button("🚀 ANALIZZA", type="primary", use_container_width=True):
    with st.spinner("Calcolo Algoritmo ML (xG + xCorners + Deep Completions)..."):
        home_adv = L_DATA["ha"] if m_type == "Standard" else (0.0 if m_type == "Campo Neutro" else L_DATA["ha"]*0.5)
        w_split = 0.60
        h_fin_att = (h_att*(1-w_split)) + (h_att_home*w_split); h_fin_def = (h_def*(1-w_split)) + (h_def_home*w_split)
        a_fin_att = (a_att*(1-w_split)) + (a_att_away*w_split); a_fin_def = (a_def*(1-w_split)) + (a_def_away*w_split)
        
        xg_h = (h_fin_att * a_fin_def) / L_DATA["avg"]; xg_a = (a_fin_att * h_fin_def) / L_DATA["avg"]
        elo_diff = (h_elo + (100 if m_type=="Standard" else 0)) - a_elo
        f_xh = (xg_h * (1 + elo_diff/1000.0)) + home_adv
        f_xa = (xg_a * (1 - elo_diff/1000.0))

        h_dc, a_dc = 5.0, 5.0 
        if h_stats and a_stats:
            h_ppda, a_ppda = h_stats['total']['ppda'], a_stats['total']['ppda']
            h_oppda, a_oppda = h_stats['total']['oppda'], a_stats['total']['oppda']

            ppda_factor_h, ppda_factor_a = 1.0, 1.0
            if h_ppda < 10.5 and a_oppda > 10.5: ppda_factor_h += 0.07 
            if a_ppda < 10.5 and h_oppda > 10.5: ppda_factor_a += 0.07
            f_xh *= ppda_factor_h; f_xa *= ppda_factor_a

            h_dc, a_dc = h_stats['total']['dc'], a_stats['total']['dc']
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

        f_xh *= volatility * (h_str/100.0); f_xa *= volatility * (a_str/100.0)
        if h_rest <= 3: f_xh*=0.95; f_xa*=1.05
        if a_rest <= 3: f_xa*=0.95; f_xh*=1.05
        if is_big_match: f_xh*=0.9; f_xa*=0.9
        if h_m_a: f_xh *= 0.85 
        if h_m_d: f_xa *= 1.20
        if a_m_a: f_xa *= 0.85 
        if a_m_d: f_xh *= 1.20

        matrix = np.zeros((10,10)); scores = []
        p1, pX, p2, pGG = 0,0,0,0
        for h in range(10):
            for a in range(10):
                p = engine.dixon_coles_probability(h, a, f_xh, f_xa, L_DATA["rho"])
                matrix[h,a] = p
                if h>a: p1+=p
                elif h==a: pX+=p
                else: p2+=p
                if h>0 and a>0: pGG+=p
                if h<6 and a<6: scores.append({"Risultato": f"{h}-{a}", "Prob": p})
        tot = np.sum(matrix); matrix/=tot; p1/=tot; pX/=tot; p2/=tot; pGG/=tot
        
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
            "p1": p1, "pX": pX, "p2": p2, "pGG": pGG, "stability": stability,
            "matrix": matrix, "scores": scores, "b1": b1, "bX": bX, "b2": b2, "history": st.session_state.history,
            "stats": {
                "corners": {"1": corn_1, "X": corn_X, "2": corn_2, "lines": corn_lines, "xH": xCorn_H, "xA": xCorn_A},
                "cards": {"1": card_1, "X": card_X, "2": card_2, "lines": card_lines},
                "shots": {"1": shot_1, "X": shot_X, "2": shot_2, "lines": shot_lines},
                "sot": {"1": sot_1, "X": sot_X, "2": sot_2, "lines": sot_lines},
                "fouls": {"1": foul_1, "X": foul_X, "2": foul_2, "lines": foul_lines}
            }
        })

# ==============================================================================
# 📊 VISUALIZZAZIONE RISULTATI
# ==============================================================================
if st.session_state.analyzed:
    st.markdown("---")
    st.header(f"📊 {st.session_state.h_name} vs {st.session_state.a_name}")
    c1, c2 = st.columns(2)
    c1.metric("xG Previsti (Adjusted)", f"{st.session_state.f_xh:.2f} - {st.session_state.f_xa:.2f}")
    c2.metric("Affidabilità", f"{st.session_state.stability:.1f}%")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🏆 Esito", "⚽ Goal", "👤 Player & Assist", "⛳ Stats Extra & Corners", "📝 Storico", "⚡ Combo"])
    
    with tab1:
        c_1, c_2 = st.columns(2)
        with c_1:
            st.subheader("1X2 & Doppia Chance")
            res_df = pd.DataFrame({
                "Esito": ["1", "X", "2", "1X", "X2", "12"],
                "Prob %": [f"{st.session_state.p1:.1%}", f"{st.session_state.pX:.1%}", f"{st.session_state.p2:.1%}", f"{(st.session_state.p1+st.session_state.pX):.1%}", f"{(st.session_state.pX+st.session_state.p2):.1%}", f"{(st.session_state.p1+st.session_state.p2):.1%}"],
                "Quota Fair": [f"{1/st.session_state.p1:.2f}", f"{1/st.session_state.pX:.2f}", f"{1/st.session_state.p2:.2f}", f"{1/(st.session_state.p1+st.session_state.pX):.2f}", f"{1/(st.session_state.pX+st.session_state.p2):.2f}", f"{1/(st.session_state.p1+st.session_state.p2):.2f}"]
            })
            st.dataframe(res_df, hide_index=True)
            
            st.subheader("Value Bet")
            val_df = pd.DataFrame({
                "Esito": ["1", "X", "2"],
                "Fair Odd": [f"{1/st.session_state.p1:.2f}", f"{1/st.session_state.pX:.2f}", f"{1/st.session_state.p2:.2f}"],
                "Bookie": [st.session_state.b1, st.session_state.bX, st.session_state.b2],
                "Valore": [f"{(st.session_state.b1*st.session_state.p1-1):.1%}", f"{(st.session_state.bX*st.session_state.pX-1):.1%}", f"{(st.session_state.b2*st.session_state.p2-1):.1%}"]
            })
            st.dataframe(val_df.style.applymap(lambda x: "background-color: #d4edda" if "%" in str(x) and "-" not in str(x) and str(x) != "0.0%" else "", subset=["Valore"]), hide_index=True)

        with c_2:
            st.subheader("Risultati Esatti")
            scores = sorted(st.session_state.scores, key=lambda x: x["Prob"], reverse=True)
            st.dataframe(pd.DataFrame([{"Risultato": s["Risultato"], "Prob": f"{s['Prob']:.1%}"} for s in scores[:6]]), hide_index=True)
            
            fig, ax = plt.subplots(figsize=(5,4))
            sns.heatmap(st.session_state.matrix[:6,:6], annot=True, fmt=".0%", cmap="Greens", cbar=False, xticklabels=range(6), yticklabels=range(6))
            plt.xlabel(st.session_state.a_name); plt.ylabel(st.session_state.h_name)
            st.pyplot(fig)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Under / Over")
            uo_list = []
            for l in [0.5, 1.5, 2.5, 3.5, 4.5]:
                p_pure = np.sum(st.session_state.matrix[np.indices((10,10))[0] + np.indices((10,10))[1] > l])
                pf = (p_pure*0.7) + (((h_uo_input.get(l,50) + a_uo_input.get(l,50))/200.0)*0.3)
                uo_list.append({"Linea": f"Over {l}", "Prob %": f"{pf:.1%}", "Quota": f"{1/pf:.2f}"})
            st.dataframe(pd.DataFrame(uo_list), hide_index=True)
            st.write(f"**Goal / Goal:** {st.session_state.pGG:.1%} (@{1/st.session_state.pGG:.2f})")

        with c2:
            st.subheader("Multigol")
            mg_res = []
            for r in [(1,2), (1,3), (2,3), (2,4), (3,4), (3,5)]:
                pm = np.sum(st.session_state.matrix[(np.indices((10,10))[0] + np.indices((10,10))[1] >= r[0]) & (np.indices((10,10))[0] + np.indices((10,10))[1] <= r[1])])
                mg_res.append({"Range": f"{r[0]}-{r[1]}", "Prob %": f"{pm:.1%}", "Quota": f"{1/pm:.2f}"})
            st.dataframe(pd.DataFrame(mg_res), hide_index=True)

    with tab3:
        st.subheader("Analisi Marcatore, Assist e Combo")
        if PLAYERS_DF is not None and not PLAYERS_DF.empty:
            team_sel = st.radio("Scegli Squadra", [f"Casa: {st.session_state.h_name}", f"Ospite: {st.session_state.a_name}"])
            is_home = "Casa" in team_sel
            t_match = engine.fuzzy_match_team(st.session_state.h_name if is_home else st.session_state.a_name, PLAYERS_DF['team'].unique())
            players_list = PLAYERS_DF[PLAYERS_DF['team'] == t_match]['player'].tolist() if t_match else []
            
            if players_list:
                pl_n = st.selectbox("Seleziona Giocatore", players_list)
                p_data = PLAYERS_DF[PLAYERS_DF['player'] == pl_n].iloc[0]
                p_xg_val, p_xa_val = float(p_data.get('xg90', 0)), float(p_data.get('xa90', 0))
            else:
                st.warning("Giocatori non trovati. Inserimento manuale.")
                pl_n = st.text_input("Nome", "Player")
                c1, c2 = st.columns(2)
                p_xg_val = c1.number_input("xG/90 (Gol)", 0.0, 2.0, 0.4)
                p_xa_val = c2.number_input("xA/90 (Assist)", 0.0, 2.0, 0.2)
            
            txg = st.session_state.f_xh if is_home else st.session_state.f_xa
            t_type = "Casa" if is_home else "Ospite"
            t_avg_xg_seas = h_stats["total"]["xg_total"]/h_stats["total"]["matches"] if is_home else a_stats["total"]["xg_total"]/a_stats["total"]["matches"]
            pmin = st.number_input("Minuti Previsti", 1, 100, 90)

            st.markdown("---")
            col_prob1, col_prob2 = st.columns(2)

            if p_xg_val > 0:
                pprob_gol = engine.calculate_player_probability(p_xg_val, pmin, txg, t_avg_xg_seas)
                col_prob1.success(f"⚽ **Probabilità GOL {pl_n}:** {pprob_gol:.1%} (@{1/pprob_gol:.2f})")
            
            if p_xa_val > 0:
                pprob_assist = engine.calculate_player_probability(p_xa_val, pmin, txg, t_avg_xg_seas)
                col_prob2.info(f"👟 **Probabilità ASSIST {pl_n}:** {pprob_assist:.1%} (@{1/pprob_assist:.2f})")
                
            st.markdown("### ⚡ Combo Player")
            c_c1, c_c2 = st.columns(2)
            sel_res_p = c_c1.selectbox("Scegli Esito Match per la Combo", ["1", "X", "2", "1X", "X2", "12"])
            
            c_btn1, c_btn2 = st.columns(2)
            if c_btn1.button("Calcola Combo GOL"):
                share_g = min(0.99, (p_xg_val / 90 * pmin) / max(0.1, txg))
                p_combo_g = engine.calculate_combo_player(st.session_state.matrix, sel_res_p, t_type, share_g)
                st.success(f"Combo **{sel_res_p} + Gol {pl_n}**: {p_combo_g:.1%} (@{1/p_combo_g:.2f})")

            if c_btn2.button("Calcola Combo ASSIST"):
                share_a = min(0.99, (p_xa_val / 90 * pmin) / max(0.1, txg))
                p_combo_a = engine.calculate_combo_player(st.session_state.matrix, sel_res_p, t_type, share_a)
                st.info(f"Combo **{sel_res_p} + Assist {pl_n}**: {p_combo_a:.1%} (@{1/p_combo_a:.2f})")

    with tab4:
        st.subheader("⛳ Modello xCorners & Stats Extra")
        def safe_odd(prob): return f"{1/prob:.2f}" if prob > 0.001 else "N/A"

        c_s1, c_s2 = st.columns(2)
        with c_s1:
            st.markdown("### 🚩 Angoli")
            corn = st.session_state.stats["corners"]
            st.caption(f"xCorners -> Casa: {corn['xH']:.1f} | Ospite: {corn['xA']:.1f}")
            st.dataframe(pd.DataFrame([{"Linea": k, "Over %": f"{v['prob']:.1%}", "Quota": safe_odd(v['prob'])} for k,v in corn["lines"].items()]), hide_index=True)

            st.markdown("### 🟨 Cartellini")
            card = st.session_state.stats["cards"]
            st.dataframe(pd.DataFrame([{"Linea": k, "Over %": f"{v['prob']:.1%}", "Quota": safe_odd(v['prob'])} for k,v in card["lines"].items()]), hide_index=True)

        with c_s2:
            st.markdown("### 🥅 Tiri Totali")
            shot = st.session_state.stats["shots"]
            st.dataframe(pd.DataFrame([{"Linea": k, "Over %": f"{v['prob']:.1%}", "Quota": safe_odd(v['prob'])} for k,v in shot["lines"].items()]), hide_index=True)

            st.markdown("### 🎯 Tiri in Porta")
            sot = st.session_state.stats["sot"]
            st.dataframe(pd.DataFrame([{"Linea": k, "Over %": f"{v['prob']:.1%}", "Quota": safe_odd(v['prob'])} for k,v in sot["lines"].items()]), hide_index=True)

    with tab5:
        c1, c2 = st.columns(2)
        if c1.button("💾 Salva in Storico"):
            st.session_state.history.append({"Match": f"{st.session_state.h_name}-{st.session_state.a_name}", "xG": f"{st.session_state.f_xh:.2f}-{st.session_state.f_xa:.2f}", "P1": st.session_state.p1})
            engine.salva_storico_json(st.session_state.history)
            st.success("Salvato!")
        
        excel_data = engine.generate_excel_report(st.session_state)
        c2.download_button("📥 Scarica Report Excel", excel_data, f"Mathbet_{datetime.now().strftime('%H%M')}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with tab6:
        st.subheader("⚡ Combo Maker")
        c1, c2, c3 = st.columns(3)
        sel_res = c1.selectbox("Esito 1X2", ["-", "1", "X", "2", "1X", "X2", "12"])
        sel_uo = c2.selectbox("Under/Over", ["-", "Over 1.5", "Under 1.5", "Over 2.5", "Under 2.5", "Over 3.5", "Under 3.5", "Over 4.5", "Under 4.5"])
        sel_gg = c3.selectbox("Goal/NoGoal", ["-", "Goal", "No Goal"])
        
        if st.button("Calcola Combo", type="primary"):
            prob_combo = 0.0
            for h in range(10):
                for a in range(10):
                    p = st.session_state.matrix[h, a]
                    if p == 0: continue
                    if sel_res != "-" and not eval(f"h {'>' if sel_res=='1' else '==' if sel_res=='X' else '<' if sel_res=='2' else '>=' if sel_res=='1X' else '<=' if sel_res=='X2' else '!='} a"): continue
                    if sel_uo != "-":
                        lim = float(sel_uo.split()[1])
                        if ("Over" in sel_uo and (h+a) <= lim) or ("Under" in sel_uo and (h+a) > lim): continue
                    if sel_gg != "-" and (("Goal" in sel_gg and (h==0 or a==0)) or ("No Goal" in sel_gg and (h>0 and a>0))): continue
                    prob_combo += p
            if prob_combo > 0: st.success(f"Probabilità Combo: **{prob_combo:.1%}** (@{1/prob_combo:.2f})")
            else: st.warning("Evento impossibile (0%)")
