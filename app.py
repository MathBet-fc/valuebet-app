import streamlit as st
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Mathbet fc",
                   page_icon="⚽",
                   layout="wide")

# --- INIZIALIZZAZIONE SESSION STATE ---
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
    st.session_state.xg_h = 0
    st.session_state.xg_a = 0
    st.session_state.team_avg_h = 0 
    st.session_state.team_avg_a = 0 
    st.session_state.prob_1 = 0
    st.session_state.prob_X = 0
    st.session_state.prob_2 = 0
    st.session_state.score_matrix = None
    st.session_state.all_scores = []
    # Dati storici U/O salvati per calcoli successivi
    st.session_state.hist_uo_h = {}
    st.session_state.hist_uo_a = {}

# --- DATABASE CAMPIONATI (HOME ADVANTAGE RIDOTTO) ---
LEAGUES = {
    "🌐 Generico (Media)": { "avg": 1.35, "ha": 0.30, "w_elo_base": 0.40 }, 
    "🇮🇹 Serie A":          { "avg": 1.30, "ha": 0.20, "w_elo_base": 0.50 },
    "🇮🇹 Serie B":          { "avg": 1.15, "ha": 0.25, "w_elo_base": 0.30 },
    "🇬🇧 Premier League":   { "avg": 1.55, "ha": 0.30, "w_elo_base": 0.55 },
    "🇩🇪 Bundesliga":       { "avg": 1.65, "ha": 0.35, "w_elo_base": 0.45 },
    "🇪🇸 La Liga":          { "avg": 1.25, "ha": 0.25, "w_elo_base": 0.55 },
    "🇫🇷 Ligue 1":          { "avg": 1.30, "ha": 0.24, "w_elo_base": 0.45 },
}

# Parametri Globali
SCALING_FACTOR = 400.0  
KELLY_FRACTION = 0.25
RHO = -0.13 # Correlazione Dixon-Coles
WEIGHT_HIST_UO = 0.40 # Quanto pesano i dati storici inseriti dall'utente (40%)

# --- FUNZIONI MATEMATICHE ---

def calculate_dynamic_weights(matchday, base_w_elo):
    if matchday <= 8:
        w_elo = max(base_w_elo, 0.75)
    elif matchday <= 19:
        w_elo = base_w_elo + 0.10
    else:
        w_elo = base_w_elo
    w_stats = 1.0 - w_elo
    return w_elo, w_stats

def dixon_coles_probability(h_goals, a_goals, mu_h, mu_a, rho):
    prob = (math.exp(-mu_h) * (mu_h**h_goals) / math.factorial(h_goals)) * \
           (math.exp(-mu_a) * (mu_a**a_goals) / math.factorial(a_goals))
    correction = 1.0
    if h_goals == 0 and a_goals == 0: correction = 1.0 - (mu_h * mu_a * rho)
    elif h_goals == 0 and a_goals == 1: correction = 1.0 + (mu_h * rho)
    elif h_goals == 1 and a_goals == 0: correction = 1.0 + (mu_a * rho)
    elif h_goals == 1 and a_goals == 1: correction = 1.0 - rho
    return max(0.0, prob * correction)

def calculate_kelly(prob_true, odds_book):
    if odds
