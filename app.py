import streamlit as st
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# --- CONFIGURAZIONE PAGINA ---
# MODIFICA: Rimosso "Pro Ultra" dal titolo della pagina
st.set_page_config(page_title="Mathbet fc",
                   page_icon="⚽",
                   layout="wide")

# --- DATABASE CAMPIONATI CON PESI DINAMICI ---
LEAGUES = {
    "🌐 Generico (Media)": {
        "avg": 1.35,
        "ha": 100,
        "w_elo": 0.35,
        "w_seas": 0.35,
        "w_form": 0.30
    },
    "🇮🇹 Serie A": {
        "avg": 1.30,
        "ha": 90,
        "w_elo": 0.40,
        "w_seas": 0.35,
        "w_form": 0.25
    },
    "🇮🇹 Serie B": {
        "avg": 1.15,
        "ha": 85,
        "w_elo": 0.20,
        "w_seas": 0.30,
        "w_form": 0.50
    },
    "🇬🇧 Premier League": {
        "avg": 1.48,
        "ha": 105,
        "w_elo": 0.45,
        "w_seas": 0.35,
        "w_form": 0.20
    },
    "🇩🇪 Bundesliga": {
        "avg": 1.58,
        "ha": 105,
        "w_elo": 0.35,
        "w_seas": 0.40,
        "w_form": 0.25
    },
    "🇪🇸 La Liga": {
        "avg": 1.25,
        "ha": 100,
        "w_elo": 0.50,
        "w_seas": 0.30,
        "w_form": 0.20
    },
    "🇫🇷 Ligue 1": {
        "avg": 1.28,
        "ha": 95,
        "w_elo": 0.35,
        "w_seas": 0.35,
        "w_form": 0.30
    },
}

# Parametri Globali Fissi
SCALING_FACTOR = 1000
W_VENUE = 0.20
WEIGHT_HISTORICAL_UO = 0.40
KELLY_FRACTION = 0.25
DRAW_BOOST = 1.10


# --- FUNZIONI MATEMATICHE ---
def normalize_elo(val):
    if val <= 100: return 1100 + (val * 9.5)
    if val > 2200: return val / 1.75
    return val


def poisson(k, lmbda):
    return (lmbda**k * math.exp(-lmbda)) / math.factorial(k)


def calculate_kelly(prob_true, odds_book):
    if odds_book <= 1.01 or prob_true <= 0: return 0.0
    b = odds_book - 1
    p = prob_true
    q = 1 - p
    kelly = ((b * p) - q) / b
    return max(0.0, (kelly * KELLY_FRACTION) * 100)


# Calcolo xG Dinamico (accetta pesi variabili)
def calculate_complex_xg_dynamic(elo_h, elo_a, att_h, def_h, att_a, def_a,
                                 current_avg, current_ha, w_elo_l):
    # Peso Stats è il rimanente dopo aver tolto il peso Elo
    w_stats_l = 1.0 - w_elo_l

    diff_h = (elo_h + current_ha) - elo_a
    diff_a = elo_a - (elo_h + current_ha)
    xg_elo_h = current_avg * (1 + (diff_h / SCALING_FACTOR))
    xg_elo_a = current_avg * (1 + (diff_a / SCALING_FACTOR))
    xg_stats_h = (att_h * def_a) / current_avg
    xg_stats_a = (att_a * def_h) / current_avg

    final_xg_h = (xg_elo_h * w_elo_l) + (xg_stats_h * w_stats_l)
    final_xg_a = (xg_elo_a * w_elo_l) + (xg_stats_a * w_stats_l)
    return final_xg_h, final_xg_a


# --- FUNZIONI GIOCATORI ---
def calculate_player_prop_kelly(prob_true, odds_book):
    KELLY_F = 0.25
    if odds_book <= 1.01 or prob_true <= 0: return 0.0
    b = odds_book - 1
    p = prob_true
    q = 1 - p
    kelly = ((b * p) - q) / b
    return max(0.0, (kelly * KELLY_F) * 100)


def calculate_player_probability(metric_per90, expected_mins, team_match_xg,
                                 team_avg_xg):
    base_lambda = (metric_per90 / 90.0) * expected_mins
    match_factor = 1.0
    if team_avg_xg > 0:
        match_factor = team_match_xg / team_avg_xg
    final_lambda = base_lambda * match_factor
    prob_at_least_one = 1 - math.exp(-final_lambda)
    prob_at_least_two = 1 - (math.exp(-final_lambda) +
                             (final_lambda * math.exp(-final_lambda)))
    return prob_at_least_one, prob_at_least_two, final_lambda


# --- INTERFACCIA UTENTE ---

# Sidebar
with st.sidebar:
    # MODIFICA: Rimosso "Pro"
    st.title("⚙️ Configurazione")
    st.write("I parametri si adattano automaticamente al campionato scelto.")
    league_name = st.selectbox("Campionato", list(LEAGUES.keys()))

    # Estrazione Parametri Dinamici
    L_DATA = LEAGUES[league_name]
    CURRENT_AVG_GOALS = L_DATA["avg"]
    CURRENT_HOME_ADV = L_DATA["ha"]
    W_ELO = L_DATA["w_elo"]
    W_SEASON = L_DATA["w_seas"]
    W_FORM = L_DATA["w_form"]

    st.info(f"""
    📊 **Parametri Attivi:**
    Avg Gol: {CURRENT_AVG_GOALS}
    Home Adv: {CURRENT_HOME_ADV}

    ⚖️ **Pesi Algoritmo:**
    Elo (Blasone): {W_ELO*100:.0f}%
    Stagione: {W_SEASON*100:.0f}%
    Forma (Last 5): {W_FORM*100:.0f}%
    """)

# MODIFICA: Rimosso "Pro Ultra" dal titolo principale
st.title("Mathbet fc ⚽")
st.markdown(
    "ℹ️ *Algoritmo Ibrido: Monte Carlo (10k Sim) + Poisson + Pesi Dinamici*")

# Link Dati
with st.expander("🔗 Link Utili (Clicca per aprire)", expanded=False):
    lc1, lc2, lc3 = st.columns(3)
    with lc1:
        st.caption("Rating")
        st.link_button("ClubElo", "http://clubelo.com")
        st.link_button(
            "StatsSulCalcio",
            "https://www.statistichesulcalcio.com/classifiche_elo.php")
    with lc2:
        st.caption("Stats")
        st.link_button("FootyStats",
                       "https://footystats.org/it/italy/serie-a/form-table")
        st.link_button("Diretta.it", "https://www.diretta.it")
    with lc3:
        st.caption("Giocatori & Quote")
        st.link_button("FBref (Players)",
                       "https://fbref.com/en/comps/11/Serie-A-Stats")
        st.link_button("OddsPortal", "https://www.oddsportal.com")

st.markdown("---")

# Input Squadre
col_h, col_a = st.columns(2)

with col_h:
    st.subheader("🏠 Casa")
    h_name = st.text_input("Nome Casa", "Home Team")
    raw_h_elo = st.number_input("Rating (Elo/Opta)",
                                value=1500.0,
                                step=10.0,
                                key="h_elo")
    h_elo = normalize_elo(raw_h_elo)
    h_str = st.slider("Formazione Casa %", 50, 100, 100, key="h_str")

    with st.expander("📊 Stats Casa", expanded=True):
        st.caption("Stagione")
        h_gf_seas = st.number_input("GF Media Stagione",
                                    0.0,
                                    5.0,
                                    1.5,
                                    key="h1")
        h_gs_seas = st.number_input("GS Media Stagione",
                                    0.0,
                                    5.0,
                                    1.2,
                                    key="h2")
        st.caption(f"Ultime 5 (Peso: {W_FORM*100:.0f}%)")
        h_gf_l5 = st.number_input("GF Totali (Last 5)", 0, 20, 7, key="h3")
        h_gs_l5 = st.number_input("GS Totali (Last 5)", 0, 20, 5, key="h4")
        st.caption("Casa")
        h_gf_venue = st.number_input("GF Media Casa", 0.0, 5.0, 1.8, key="h5")
        h_gs_venue = st.number_input("GS Media Casa", 0.0, 5.0, 0.9, key="h6")

    with st.expander("📈 % Over Casa"):
        h_uo = {
            0.5: 95,
            1.5: st.slider("% Over 1.5 Casa", 0, 100, 75, key="h_o15"),
            2.5: st.slider("% Over 2.5 Casa", 0, 100, 50, key="h_o25"),
            3.5: st.slider("% Over 3.5 Casa", 0, 100, 30, key="h_o35"),
            4.5: st.slider("% Over 4.5 Casa", 0, 100, 15, key="h_o45"),
        }

with col_a:
    st.subheader("✈️ Ospite")
    a_name = st.text_input("Nome Ospite", "Away Team")
    raw_a_elo = st.number_input("Rating (Elo/Opta)",
                                value=1450.0,
                                step=10.0,
                                key="a_elo")
    a_elo = normalize_elo(raw_a_elo)
    a_str = st.slider("Formazione Ospite %", 50, 100, 100, key="a_str")

    with st.expander("📊 Stats Ospite", expanded=True):
        st.caption("Stagione")
        a_gf_seas = st.number_input("GF Media Stagione",
                                    0.0,
                                    5.0,
                                    1.2,
                                    key="a1")
        a_gs_seas = st.number_input("GS Media Stagione",
                                    0.0,
                                    5.0,
                                    1.4,
                                    key="a2")
        st.caption(f"Ultime 5 (Peso: {W_FORM*100:.0f}%)")
        a_gf_l5 = st.number_input("GF Totali (Last 5)", 0, 20, 5, key="a3")
        a_gs_l5 = st.number_input("GS Totali (Last 5)", 0, 20, 8, key="a4")
        st.caption("Trasferta")
        a_gf_venue = st.number_input("GF Media Fuori", 0.0, 5.0, 1.0, key="a5")
        a_gs_venue = st.number_input("GS Media Fuori", 0.0, 5.0, 1.6, key="a6")

    with st.expander("📈 % Over Ospite"):
        a_uo = {
            0.5: 95,
            1.5: st.slider("% Over 1.5 Ospite", 0, 100, 70, key="a_o15"),
            2.5: st.slider("% Over 2.5 Ospite", 0, 100, 45, key="a_o25"),
            3.5: st.slider("% Over 3.5 Ospite", 0, 100, 25, key="a_o35"),
            4.5: st.slider("% Over 4.5 Ospite", 0, 100, 10, key="a_o45"),
        }

st.markdown("---")
st.subheader("💰 Quote")
qc1, qc2, qc3, qc4, qc5 = st.columns(5)
b1 = qc1.number_input("1", 1.01, 100.0, 2.50)
bX = qc2.number_input("X", 1.01, 100.0, 3.20)
b2 = qc3.number_input("2", 1.01, 100.0, 2.90)
bGG = qc4.number_input("GG", 1.01, 100.0, 1.75)
bNG = qc5.number_input("NG", 1.01, 100.0, 2.05)

# Variabili globali
xg_h, xg_a = 0, 0
all_scores_list = []

# --- CORE LOGIC ---
if st.button("🚀 ANALIZZA PARTITA (Simulazione Avanzata)",
             type="primary",
             use_container_width=True):

    # 1. Calcolo Pesi Dinamici
    h_avg_gf_form = h_gf_l5 / 5.0
    h_avg_gs_form = h_gs_l5 / 5.0
    # Formula con Pesi Dinamici
    att_h = (h_gf_seas * W_SEASON) + (h_avg_gf_form * W_FORM) + (h_gf_venue *
                                                                 W_VENUE)
    def_h = (h_gs_seas * W_SEASON) + (h_avg_gs_form * W_FORM) + (h_gs_venue *
                                                                 W_VENUE)

    a_avg_gf_form = a_gf_l5 / 5.0
    a_avg_gs_form = a_gs_l5 / 5.0
    att_a = (a_gf_seas * W_SEASON) + (a_avg_gf_form * W_FORM) + (a_gf_venue *
                                                                 W_VENUE)
    def_a = (a_gs_seas * W_SEASON) + (a_avg_gs_form * W_FORM) + (a_gs_venue *
                                                                 W_VENUE)

    xg_h, xg_a = calculate_complex_xg_dynamic(h_elo, a_elo, att_h, def_h,
                                              att_a, def_a, CURRENT_AVG_GOALS,
                                              CURRENT_HOME_ADV, W_ELO)
    xg_h = max(0.1, xg_h * (h_str / 100.0))
    xg_a = max(0.1, xg_a * (a_str / 100.0))

    # 2. Modello Matematico (Poisson + Dixon Coles)
    prob_1, prob_X, prob_2 = 0, 0, 0
    prob_GG, prob_NG = 0, 0
    total_goals_probs = {k: 0 for k in range(20)}
    
    uo_probs_math = {0.5: [0, 0], 1.5: [0, 0], 2.5: [0, 0], 3.5: [0, 0], 4.5: [0, 0]}
    RHO = -0.13

    # Matrice per Heatmap
    score_matrix = np.zeros((6, 6))

    for h in range(10):
        for a in range(10):
            p = poisson(h, xg_h) * poisson(a, xg_a)
            correction = 1.0
            if h == 0 and a == 0: correction = 1.0 - (xg_h * xg_a * RHO)
            elif h == 0 and a == 1: correction = 1.0 + (xg_h * RHO)
            elif h == 1 and a == 0: correction = 1.0 + (xg_a * RHO)
            elif h == 1 and a == 1: correction = 1.0 - RHO
            p = max(0, p * correction)
            if h == a: p *= DRAW_BOOST

            if h > a: prob_1 += p
            elif h == a: prob_X += p
            else: prob_2 += p

            if h > 0 and a > 0: prob_GG += p
            else: prob_NG += p

            tot = h + a
            if tot in total_goals_probs: total_goals_probs[tot] += p
            # Il ciclo qui funziona automaticamente per tutte le chiavi in uo_probs_math
            for line in uo_probs_math:
                if tot < line: uo_probs_math[line][0] += p
                else: uo_probs_math[line][1] += p

            all_scores_list.append({"score": f"{h}-{a}", "prob": p})

            # Popola Heatmap (solo fino a 5-5)
            if h < 6 and a < 6:
                score_matrix[h, a] = p

    # 3. Monte Carlo Simulation (10,000 Match)
    n_sims = 10000
    sim_h = np.random.poisson(xg_h, n_sims)
    sim_a = np.random.poisson(xg_a, n_sims)
    sim_1 = np.sum(sim_h > sim_a) / n_sims
    sim_X = np.sum(sim_h == sim_a) / n_sims
    sim_2 = np.sum(sim_h < sim_a) / n_sims

    # Normalizzazione Matematica
    total_prob = prob_1 + prob_X + prob_2
    factor = 1.0 / total_prob if total_prob > 0 else 1.0
    p1, pX, p2 = prob_1 * factor, prob_X * factor, prob_2 * factor
    p_gg, p_ng = prob_GG * factor, prob_NG * factor

    # Doppia Chance
    p1X, pX2, p12 = p1 + pX, pX + p2, p1 + p2

    # Multigol
    def get_multigol_prob(min_g, max_g):
        prob = 0
        for g in range(min_g, max_g + 1):
            prob += total_goals_probs.get(g, 0)
        return prob * factor

    mg_1_2, mg_1_3, mg_2_3, mg_2_4, mg_3_5 = get_multigol_prob(
        1, 2), get_multigol_prob(1, 3), get_multigol_prob(
            2, 3), get_multigol_prob(2, 4), get_multigol_prob(3, 5)

    # Ordinamento Risultati Esatti
    for item in all_scores_list:
        item["prob"] *= factor
    all_scores_list.sort(key=lambda x: x["prob"], reverse=True)
    top_3_scores = all_scores_list[:3]

    # --- OUTPUT ---
    st.balloons()
    st.header(f"📊 {h_name} vs {a_name}")

    # xG e Confronto Monte Carlo
    col_xg1, col_xg2 = st.columns(2)
    col_xg1.metric("xG Stimati", f"{xg_h:.2f} - {xg_a:.2f}")
    col_xg2.info(
        f"🎲 **Check Monte Carlo (10k Sim):**\n1: {sim_1:.1%} | X: {sim_X:.1%} | 2: {sim_2:.1%}"
    )

    # 1. TABELLA PRINCIPALE 1X2
    col_main, col_dc = st.columns(2)
    with col_main:
        st.subheader("🏆 Esito Finale")
        f1, fX, f2 = (1 / p1 if p1 > 0 else 0), (1 / pX if pX > 0 else
                                                 0), (1 / p2 if p2 > 0 else 0)
        e1, eX, e2 = ((b1 / f1) - 1) * 100, ((bX / fX) - 1) * 100, (
            (b2 / f2) - 1) * 100
        k1, kX, k2 = calculate_kelly(p1, b1), calculate_kelly(
            pX, bX), calculate_kelly(p2, b2)

        data_1x2 = {
            "Esito": ["1", "X", "2"],
            "Prob %": [f"{p1*100:.1f}%", f"{pX*100:.1f}%", f"{p2*100:.1f}%"],
            "Fair": [f"{f1:.2f}", f"{fX:.2f}", f"{f2:.2f}"],
            "Book": [b1, bX, b2],
            "Val": [f"{e1:+.1f}%", f"{eX:+.1f}%", f"{e2:+.1f}%"],
            "Stake": [f"{k1:.1f}%", f"{kX:.1f}%", f"{k2:.1f}%"]
        }
        st.table(pd.DataFrame(data_1x2))

    with col_dc:
        st.subheader("🛡️ Doppia Chance")
        f1X, fX2, f12 = (1 / p1X if p1X > 0 else 0), (
            1 / pX2 if pX2 > 0 else 0), (1 / p12 if p12 > 0 else 0)
        data_dc = {
            "Esito": ["1X", "X2", "12"],
            "Prob %":
            [f"{p1X*100:.1f}%", f"{pX2*100:.1f}%", f"{p12*100:.1f}%"],
            "Fair": [f"{f1X:.2f}", f"{fX2:.2f}", f"{f12:.2f}"]
        }
        st.table(pd.DataFrame(data_dc))

    # 2. RISULTATI ESATTI + HEATMAP
    st.subheader("🎯 Risultati Esatti")

    # Top 3 Cards
    cols_cs = st.columns(3)
    for i, item in enumerate(top_3_scores):
        fair = 1 / item["prob"] if item["prob"] > 0 else 0
        with cols_cs[i]:
            st.metric(f"Top {i+1}: {item['score']}",
                      f"{item['prob']*100:.1f}%", f"Fair: {fair:.2f}")

    # Heatmap
    with st.expander("🗺️ Apri Mappa Probabilità (Heatmap)"):
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(score_matrix,
                    annot=True,
                    fmt=".1%",
                    cmap="YlGnBu",
                    xticklabels=range(6),
                    yticklabels=range(6),
                    cbar=False)
        plt.xlabel(f"Gol {a_name}")
        plt.ylabel(f"Gol {h_name}")
        plt.title("Probabilità Punteggio")
        st.pyplot(fig)

    # 3. UNDER/OVER E GG/NG
    col_uo, col_gg = st.columns(2)
    with col_uo:
        st.subheader("📉 Under / Over")
        uo_data = []
        # AGGIORNATO: Loop esteso per includere 3.5 e 4.5
        for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
            u_math = uo_probs_math[line][0] * factor
            o_math = uo_probs_math[line][1] * factor
            
            # Qui prende i valori dagli slider. Se non esistono, usa 50 come default
            h_stat_o = h_uo.get(line, 50) / 100.0 
            a_stat_o = a_uo.get(line, 50) / 100.0
            
            avg_hist_over = (h_stat_o + a_stat_o) / 2.0
            o_final = (o_math *
                       (1 - WEIGHT_HISTORICAL_UO)) + (avg_hist_over *
                                                      WEIGHT_HISTORICAL_UO)
            u_final = 1.0 - o_final
            f_u = 1 / u_final if u_final > 0 else 0
            f_o = 1 / o_final if o_final > 0 else 0
            uo_data.append({
                "Linea": line,
                "U %": f"{u_final*100:.1f}%",
                "Fair U": f"{f_u:.2f}",
                "O %": f"{o_final*100:.1f}%",
                "Fair O": f"{f_o:.2f}"
            })
        st.dataframe(pd.DataFrame(uo_data), hide_index=True)

    with col_gg:
        st.subheader("⚽ Gol / No Gol")
        f_gg, f_ng = (1 / p_gg if p_gg > 0 else 0), (1 /
                                                     p_ng if p_ng > 0 else 0)
        e_gg, e_ng = ((bGG / f_gg) - 1) * 100 if f_gg > 0 else -100, (
            (bNG / f_ng) - 1) * 100 if f_ng > 0 else -100
        data_gg = {
            "Esito": ["GOL", "NO GOL"],
            "Prob %": [f"{p_gg*100:.1f}%", f"{p_ng*100:.1f}%"],
            "Fair": [f"{f_gg:.2f}", f"{f_ng:.2f}"],
            "Book": [bGG, bNG],
            "Val": [f"{e_gg:+.1f}%", f"{e_ng:+.1f}%"]
        }
        st.dataframe(pd.DataFrame(data_gg), hide_index=True)

    # 4. MULTIGOL
    st.subheader("🔢 Multigol Probabili")
    mg_data = [
        {
            "Range": "1-2",
            "Prob": mg_1_2,
            "Fair": 1 / mg_1_2
        },
        {
            "Range": "1-3",
            "Prob": mg_1_3,
            "Fair": 1 / mg_1_3
        },
        {
            "Range": "2-3",
            "Prob": mg_2_3,
            "Fair": 1 / mg_2_3
        },
        {
            "Range": "2-4",
            "Prob": mg_2_4,
            "Fair": 1 / mg_2_4
        },
        {
            "Range": "3-5",
            "Prob": mg_3_5,
            "Fair": 1 / mg_3_5
        },
    ]
    # Ordina per probabilità
    mg_data.sort(key=lambda x: x["Prob"], reverse=True)
    # Formattazione per tabella
    mg_display = [{
        "Range": m["Range"],
        "Prob %": f"{m['Prob']*100:.1f}%",
        "Fair Odd": f"{m['Fair']:.2f}"
    } for m in mg_data]
    st.dataframe(pd.DataFrame(mg_display).head(3),
                 hide_index=True,
                 use_container_width=True)

    # --- PLAYER PROP ---
    st.markdown("---")
    st.header("👤 Marcatore / Assist")
    with st.expander("Apri Calcolatore", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            player_name = st.text_input("Giocatore", "Nome")
            team_sel = st.radio("Squadra", [h_name, a_name], horizontal=True)
            mins = st.slider("Minuti", 15, 95, 85)
        with c2:
            type_s = st.radio("Tipo", ["GOL", "ASSIST"], horizontal=True)
            val_90 = st.number_input(f"x{'G' if 'GOL' in type_s else 'A'}/90",
                                     0.0, 2.0, 0.40)
            b_odd = st.number_input("Quota", 1.0, 50.0, 2.50)

        ctx_xg = xg_h if team_sel == h_name else xg_a
        ctx_avg = h_gf_seas if team_sel == h_name else a_gf_seas
        prob_p1, prob_p2, _ = calculate_player_probability(
            val_90, mins, ctx_xg, ctx_avg)
        fair_p = 1 / prob_p1 if prob_p1 > 0 else 0
        edge_p = ((b_odd / fair_p) - 1) * 100 if fair_p > 0 else -100

        c_res1, c_res2, c_res3 = st.columns(3)
        c_res1.metric(f"Prob {type_s}", f"{prob_p1*100:.1f}%")
        c_res2.metric("Fair Odd", f"{fair_p:.2f}")
        c_res3.metric("Valore",
                      f"{edge_p:+.1f}%",
                      delta_color="normal" if edge_p < 0 else "inverse")

    # CSV
    csv_row = {
        "Date": datetime.now().strftime("%Y-%m-%d"),
        "Home": h_name,
        "Away": a_name,
        "xG_H": f"{xg_h:.2f}",
        "xG_A": f"{xg_a:.2f}",
        "Fair_1": f"{f1:.2f}",
        "Fair_X": f"{fX:.2f}",
        "Fair_2": f"{f2:.2f}",
        "Top_Score": f"{top_3_scores[0]['score']}"
    }
    st.download_button(
        "💾 CSV",
        pd.DataFrame([csv_row]).to_csv(index=False).encode('utf-8'),
        f"bet_{h_name}_{a_name}.csv", "text/csv")

# --- NUOVA FUNZIONE: REVERSE ENGINEERING ---
st.markdown("---")
st.header("🕵️ Reverse Engineering Quote")
st.write("Scopri cosa sta 'pensando' il Bookmaker.")

with st.expander("Apri Analizzatore Quote Inverse", expanded=False):
    col_rev1, col_rev2 = st.columns(2)
    with col_rev1:
        rev_quota = st.number_input("Quota Bookmaker (es. 1.90)", 1.01, 100.0,
                                    1.90)

    # Implied Probability
    implied_prob = 1 / rev_quota

    # Stima xG dominance necessaria (approssimazione euristica basata su Poisson)
    # Esempio: 50% prob vittoria richiede circa +0.4/0.5 xG dominance
    # Questa è una stima visuale per aiutare l'utente

    st.metric("Probabilità Implicita (senza aggio)",
              f"{implied_prob*100:.1f}%")

    if implied_prob > 0.60:
        st.warning(
            "Il bookmaker vede una vittoria NETTA. Serve una differenza xG > 1.0"
        )
    elif implied_prob > 0.45:
        st.info(
            "Il bookmaker vede un vantaggio SOLIDO. Serve una differenza xG tra 0.4 e 0.8"
        )
    elif implied_prob > 0.35:
        st.write("Quota equilibrata/incerta.")
    else:
        st.write(
            "Il bookmaker vede questo esito come sfavorito/poco probabile.")

# MANUALE
st.markdown("---")
st.header("🧮 Calcolatore Manuale")
with st.expander("Apri", expanded=False):
    k1, k2, k3 = st.columns(3)
    mp = k1.number_input("Prob (%)", 0.1, 100.0, 50.0) / 100
    mq = k2.number_input("Quota", 1.01, 100.0, 2.0)
    mb = k3.number_input("Bankroll", 0.0, 10000.0, 1000.0)
    mev = (mp * mq) - 1
    if mev > 0:
        mk = (((mq - 1) * mp) - (1 - mp)) / (mq - 1) * 0.25
        st.success(f"✅ Value! Edge: {mev*100:.1f}% | Stake: € {mb*mk:.2f}")
    else:
        st.error("No Value")
