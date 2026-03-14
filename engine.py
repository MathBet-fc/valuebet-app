import math
import pandas as pd
import numpy as np
import json
import io
import difflib
import requests
from datetime import datetime
from understatapi import UnderstatClient
import streamlit as st

# ==============================================================================
# CONFIGURAZIONE COSTANTI
# ==============================================================================
LEAGUES_CONFIG = {
    "🇮🇹 Serie A": {"avg": 1.28, "ha": 0.059, "rho": -0.025, "understat": "Serie_A", "odds_id": "soccer_italy_serie_a"},
    "🇬🇧 Premier League": {"avg": 1.47, "ha": 0.046, "rho": 0.005, "understat": "EPL", "odds_id": "soccer_epl"},
    "🇪🇸 La Liga": {"avg": 1.31, "ha": 0.143, "rho": 0.050, "understat": "La_Liga", "odds_id": "soccer_spain_la_liga"},
    "🇩🇪 Bundesliga": {"avg": 1.57, "ha": 0.066, "rho": -0.080, "understat": "Bundesliga", "odds_id": "soccer_germany_bundesliga"},
    "🇫🇷 Ligue 1": {"avg": 1.49, "ha": 0.120, "rho": -0.020, "understat": "Ligue_1", "odds_id": "soccer_france_ligue_one"}
}

# ==============================================================================
# FUNZIONI DI DATA EXTRACTION (CON CACHE "ZERO SPRECHI")
# ==============================================================================
@st.cache_data(ttl=3600)
def fetch_understat_data_auto(league_name):
    import streamlit as st
    u_league = LEAGUES_CONFIG[league_name]["understat"]
    try:
        with UnderstatClient() as client:
            # Uso '2025' per la stagione corrente 2025/2026
            data = client.league(u_league).get_team_data('2025')
            stats_db, team_list = {}, []
            for team_id, details in data.items():
                t_name = details['title']
                team_list.append(t_name)
                hist = details['history']
                m = max(1, len(hist))
                
                ppda_val = sum([h['ppda']['att']/h['ppda']['def'] if h['ppda']['def'] > 0 else 12 for h in hist]) / m
                oppda_val = sum([h['oppda']['att']/h['oppda']['def'] if h['oppda']['def'] > 0 else 12 for h in hist]) / m
                
                stats_db[t_name] = {
                    "total": {
                        "goals_total": sum([h['scored'] for h in hist]), "ga_total": sum([h['missed'] for h in hist]),
                        "xg_total": sum([h['xG'] for h in hist]), "xga_total": sum([h['xGA'] for h in hist]),
                        "npxg_total": sum([h['npxG'] for h in hist]), "npxga_total": sum([h['npxGA'] for h in hist]),
                        "ppda": ppda_val, "oppda": oppda_val, "dc": sum([h['deep'] for h in hist]) / m, 
                        "odc": 5.0, "pts": sum([h['pts'] for h in hist]), "xpts": sum([h['xpts'] for h in hist]), "matches": m
                    }
                }
            return stats_db, sorted(team_list)
    except Exception as e:
        # QUESTO È IL PUNTO CHIAVE: ora stamperà l'errore sullo schermo
        st.error(f"🚨 BLOCCO UNDERSTAT ({league_name}): {str(e)}")
        return {}, []

@st.cache_data(ttl=3600)
def fetch_understat_players(league_name):
    u_league = LEAGUES_CONFIG[league_name]["understat"]
    try:
        with UnderstatClient() as client:
            players_data = client.league(u_league).get_player_data('2025')
            df = pd.DataFrame(players_data)
            df['time'] = pd.to_numeric(df['time'])
            df['xG'], df['xA'] = pd.to_numeric(df['xG']), pd.to_numeric(df['xA'])
            df['xg90'] = (df['xG'] / df['time']) * 90
            df['xa90'] = (df['xA'] / df['time']) * 90
            return df[['player_name', 'team_title', 'xg90', 'xa90']].rename(columns={'player_name': 'player', 'team_title': 'team'})
    except Exception:
        return None

@st.cache_data(ttl=3600)
def fetch_league_odds(league_id, api_key):
    """Scarica tutte le quote del campionato una sola volta all'ora."""
    if not api_key or api_key == "INSERISCI_QUI_LA_TUA_CHIAVE": return []
    url = f"https://api.the-odds-api.com/v4/sports/{league_id}/odds/"
    params = {"apiKey": api_key, "regions": "eu", "markets": "h2h,totals,btts", "oddsFormat": "decimal"}
    try:
        res = requests.get(url, params=params).json()
        return res if isinstance(res, list) else []
    except Exception:
        return []

def extract_match_odds(all_odds, home_team, away_team):
    """Filtra le quote di un singolo match dalla memoria cache."""
    for match in all_odds:
        if difflib.get_close_matches(home_team, [match['home_team']], cutoff=0.55):
            bookie = match['bookmakers'][0] 
            odds_data = {"h2h": {}, "totals": {}, "btts": {}}
            for m in bookie['markets']:
                if m['key'] == 'h2h':
                    odds_data['h2h'] = {o['name']: o['price'] for o in m['outcomes']}
                elif m['key'] == 'totals':
                    o25 = [o['price'] for o in m['outcomes'] if o['name'] == 'Over' and o['point'] == 2.5]
                    u25 = [o['price'] for o in m['outcomes'] if o['name'] == 'Under' and o['point'] == 2.5]
                    if o25 and u25: odds_data['totals'] = {"Over 2.5": o25[0], "Under 2.5": u25[0]}
                elif m['key'] == 'btts':
                    odds_data['btts'] = {o['name']: o['price'] for o in m['outcomes']}
            return odds_data
    return None

def fuzzy_match_team(api_name, csv_team_list):
    matches = difflib.get_close_matches(api_name, csv_team_list, n=1, cutoff=0.55)
    return matches[0] if matches else None

# ==============================================================================
# FUNZIONI MATEMATICHE & ML
# ==============================================================================
def dixon_coles_probability(h_goals, a_goals, mu_h, mu_a, rho):
    prob = (math.exp(-mu_h) * (mu_h**h_goals) / math.factorial(h_goals)) * (math.exp(-mu_a) * (mu_a**a_goals) / math.factorial(a_goals))
    if h_goals == 0 and a_goals == 0: prob *= (1.0 - (mu_h * mu_a * rho))
    elif h_goals == 0 and a_goals == 1: prob *= (1.0 + (mu_h * rho))
    elif h_goals == 1 and a_goals == 0: prob *= (1.0 + (mu_a * rho))
    elif h_goals == 1 and a_goals == 1: prob *= (1.0 - rho)
    return max(0.0, prob)

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
        lines[f"Over {line}"] = {"prob": p_over, "odd": 1/p_over if p_over > 0.001 else 999}
    return p1, pX, p2, lines

def calcola_xCorners_pro(corn_f_H, corn_s_H, corn_f_A, corn_s_A, media_lega_team, dc_H, dc_A):
    m_lega = max(media_lega_team, 1.0)
    att_H, def_H = corn_f_H / m_lega, corn_s_H / m_lega
    att_A, def_A = corn_f_A / m_lega, corn_s_A / m_lega
    base_H, base_A = att_H * def_A * m_lega, att_A * def_H * m_lega
    tot_dc = dc_H + dc_A
    if tot_dc > 0:
        tilt_H = dc_H / tot_dc
        if tilt_H > 0.60: base_H *= 1.15; base_A *= 0.85
        elif tilt_H < 0.40: base_H *= 0.85; base_A *= 1.15
    return max(0.1, base_H), max(0.1, base_A)

def calculate_player_probability(metric_per90, expected_mins, team_match_xg, team_avg_xg):
    base_lambda = (metric_per90 / 90.0) * expected_mins
    if team_avg_xg <= 0.1: team_avg_xg = max(0.1, team_match_xg)
    raw_factor = team_match_xg / team_avg_xg
    final_factor = min(2.5, pow(raw_factor, 0.75))
    return 1 - math.exp(-(base_lambda * final_factor))

def calculate_combo_player(matrix, outcome_type, team_type, player_share):
    prob_combo = 0.0
    for h in range(10):
        for a in range(10):
            p_score = matrix[h, a]
            if p_score == 0: continue
            cond_outcome = False
            if outcome_type == "1": cond_outcome = (h > a)
            elif outcome_type == "X": cond_outcome = (h == a)
            elif outcome_type == "2": cond_outcome = (h < a)
            elif outcome_type == "1X": cond_outcome = (h >= a)
            elif outcome_type == "X2": cond_outcome = (h <= a)
            elif outcome_type == "12": cond_outcome = (h != a)
            else: cond_outcome = True
            if not cond_outcome: continue
            team_goals = h if team_type == "Casa" else a
            p_player_given_score = 0.0 if team_goals == 0 else (1 - (1 - player_share)**team_goals)
            prob_combo += (p_score * p_player_given_score)
    return prob_combo

def monte_carlo_simulation(f_xh, f_xa, n_sims=5000):
    sim = []
    for _ in range(n_sims):
        gh = np.random.poisson(max(0.1, np.random.normal(f_xh, 0.20*f_xh)))
        ga = np.random.poisson(max(0.1, np.random.normal(f_xa, 0.20*f_xa)))
        sim.append(1 if gh>ga else (0 if gh==ga else 2))
    return sim

# ==============================================================================
# MOTORE TOP 5 VALUE BETS
# ==============================================================================
def find_top_value_bets(all_odds, stats_db, L_DATA, volatility):
    if not all_odds: return []
    team_list = list(stats_db.keys())
    value_bets = []
    
    for match in all_odds:
        h_match = difflib.get_close_matches(match['home_team'], team_list, cutoff=0.55)
        a_match = difflib.get_close_matches(match['away_team'], team_list, cutoff=0.55)
        if not h_match or not a_match: continue
        
        h_name, a_name = h_match[0], a_match[0]
        h_stats, a_stats = stats_db[h_name], stats_db[a_name]
        
        try:
            bookie = match['bookmakers'][0]
            b1 = bX = b2 = bO25 = bGG = 0.0
            for m in bookie['markets']:
                if m['key'] == 'h2h':
                    b1 = next((o['price'] for o in m['outcomes'] if o['name'] == match['home_team']), 0)
                    bX = next((o['price'] for o in m['outcomes'] if o['name'] == 'Draw'), 0)
                    b2 = next((o['price'] for o in m['outcomes'] if o['name'] == match['away_team']), 0)
                elif m['key'] == 'totals':
                    bO25 = next((o['price'] for o in m['outcomes'] if o['name'] == 'Over' and o['point'] == 2.5), 0)
                elif m['key'] == 'btts':
                    bGG = next((o['price'] for o in m['outcomes'] if o['name'] == 'Yes'), 0)
        except: continue
        
        h_att = h_stats['total']['xg_total'] / h_stats['total']['matches']
        h_def = h_stats['total']['xga_total'] / h_stats['total']['matches']
        a_att = a_stats['total']['xg_total'] / a_stats['total']['matches']
        a_def = a_stats['total']['xga_total'] / a_stats['total']['matches']
        
        xg_h = ((h_att * a_def) / L_DATA["avg"]) + L_DATA["ha"]
        xg_a = (a_att * h_def) / L_DATA["avg"]
        if h_stats['total']['ppda'] < 10.5: xg_h *= 1.05
        if a_stats['total']['ppda'] < 10.5: xg_a *= 1.05
        
        f_xh, f_xa = xg_h * volatility, xg_a * volatility
        p1, pX, p2, pGG, pO25 = 0,0,0,0,0
        for h in range(10):
            for a in range(10):
                p = dixon_coles_probability(h, a, f_xh, f_xa, L_DATA["rho"])
                if h>a: p1+=p; 
                elif h==a: pX+=p; 
                else: p2+=p
                if h>0 and a>0: pGG+=p
                if (h+a) > 2.5: pO25+=p
                
        evals = [("1", p1, b1), ("X", pX, bX), ("2", p2, b2), ("Over 2.5", pO25, bO25), ("Goal (GG)", pGG, bGG)]
        for bet_type, prob, odd in evals:
            if odd > 1.0 and prob > 0:
                val = (prob * odd) - 1
                if val > 0.03: 
                    value_bets.append({"Match": f"{h_name} - {a_name}", "Mercato": bet_type, "Prob %": prob, "Fair Odd": 1/prob, "Quota Bookie": odd, "Valore %": val})
                    
    value_bets.sort(key=lambda x: x["Valore %"], reverse=True)
    return value_bets[:5]

# ==============================================================================
# GESTIONE STORICO ED EXPORT
# ==============================================================================
def carica_storico_json():
    try:
        with open('mathbet_history.json', 'r') as f: return json.load(f)
    except FileNotFoundError: return []

def salva_storico_json(history_list):
    try:
        with open('mathbet_history.json', 'w') as f: json.dump(history_list, f, indent=2, default=str)
        return True
    except Exception: return False

def generate_excel_report(session_data):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        data_main = {
            "Parametro": ["Partita", "Data", "Campionato", "xG Casa", "xG Ospite", "Stabilità", "Prob 1", "Prob X", "Prob 2"],
            "Valore": [
                f"{session_data['h_name']} - {session_data['a_name']}", datetime.now().strftime("%Y-%m-%d"),
                session_data.get('league_name', 'N/A'), round(session_data['f_xh'], 2), round(session_data['f_xa'], 2),
                f"{session_data['stability']:.1f}%", f"{session_data['p1']:.1%}", f"{session_data['pX']:.1%}", f"{session_data['p2']:.1%}"
            ]
        }
        pd.DataFrame(data_main).to_excel(writer, sheet_name='Analisi Match', index=False)
        data_odds = {
            "Esito": ["1", "X", "2"],
            "Prob %": [session_data['p1'], session_data['pX'], session_data['p2']],
            "Quota Fair": [1/session_data['p1'], 1/session_data['pX'], 1/session_data['p2']],
            "Quota Bookie": [session_data['b1'], session_data['bX'], session_data['b2']],
            "Value Bet %": [(session_data['b1'] * session_data['p1']) - 1, (session_data['bX'] * session_data['pX']) - 1, (session_data['b2'] * session_data['p2']) - 1]
        }
        pd.DataFrame(data_odds).to_excel(writer, sheet_name='Analisi Match', startrow=12, index=False)
        scores = session_data['scores'].copy()
        scores.sort(key=lambda x: x["Prob"], reverse=True)
        pd.DataFrame(scores[:10]).to_excel(writer, sheet_name='Analisi Match', startrow=18, index=False)
        
        if session_data.get('history'):
            pd.DataFrame(session_data['history']).to_excel(writer, sheet_name='Storico Completo', index=False)
    return output.getvalue()
