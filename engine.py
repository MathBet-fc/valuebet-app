import math
import pandas as pd
import numpy as np
import json
import io
import difflib
import requests
from datetime import datetime, timedelta
import cloudscraper
import re
import streamlit as st

def _get_understat_json(url, var_name):
    scraper = cloudscraper.create_scraper(browser={'browser': 'chrome', 'platform': 'windows', 'desktop': True})
    res = scraper.get(url, timeout=15)
    match = re.search(r"var\s+" + var_name + r"\s*=\s*JSON\.parse\('([^']+)'\)", res.text)
    if match:
        return json.loads(match.group(1).encode('utf-8').decode('unicode_escape'))
    return {}
from sklearn.ensemble import RandomForestClassifier

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
# FUNZIONI DI DATA EXTRACTION
# ==============================================================================
@st.cache_data(ttl=43200) 
def fetch_clubelo_ratings():
    try:
        date_str = datetime.now().strftime('%Y-%m-%d')
        df = pd.read_csv(f"http://api.clubelo.com/{date_str}")
    except Exception:
        try:
            date_str = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            df = pd.read_csv(f"http://api.clubelo.com/{date_str}")
        except Exception:
            return {}
    if 'Club' in df.columns and 'Elo' in df.columns:
        return dict(zip(df['Club'], df['Elo']))
    return {}

def get_elo_for_team(team_name, elo_dict, default_elo=1500.0):
    if not elo_dict or not team_name: 
        return default_elo
    if team_name in elo_dict: 
        return float(elo_dict[team_name])
    matches = difflib.get_close_matches(team_name, elo_dict.keys(), n=1, cutoff=0.55)
    if matches: 
        return float(elo_dict[matches[0]])
    return default_elo

@st.cache_data(ttl=3600)
def fetch_understat_data_auto(league_name):
    u_league = LEAGUES_CONFIG[league_name]["understat"]
    try:
        url = f"https://understat.com/league/{u_league}/2024"
        data = _get_understat_json(url, "teamsData")
        if not data: return {}, []
        stats_db = {}
        team_list = []
        
        def calculate_segment(hist_list):
            m = len(hist_list)
            if m == 0:
                return {"goals_total": 0, "ga_total": 0, "xg_total": 0, "xga_total": 0, "npxg_total": 0, "npxga_total": 0, "ppda": 12.0, "oppda": 12.0, "dc": 5.0, "odc": 5.0, "pts": 0, "xpts": 0, "matches": 0}
            
            goals = ga = xg = xga = npxg = npxga = dc = odc = pts = xpts = 0
            ppda_sum = oppda_sum = 0
            
            for h in hist_list:
                goals += h.get('scored', 0)
                ga += h.get('missed', 0)
                xg += h.get('xG', 0)
                xga += h.get('xGA', 0)
                npxg += h.get('npxG', h.get('xG', 0))
                npxga += h.get('npxGA', h.get('xGA', 0))
                dc += h.get('deep', 5)
                odc += h.get('deep_allowed', 5)
                pts += h.get('pts', 0)
                xpts += h.get('xpts', 0)
                
                p_def = h.get('ppda', {}).get('def', 0)
                if p_def > 0:
                    ppda_sum += h.get('ppda', {}).get('att', 0) / p_def
                else:
                    ppda_sum += 12.0
                    
                op_def = h.get('ppda_allowed', {}).get('def', 0)
                if op_def > 0:
                    oppda_sum += h.get('ppda_allowed', {}).get('att', 0) / op_def
                else:
                    oppda_sum += 12.0
                    
            return {
                "goals_total": goals, "ga_total": ga, "xg_total": xg, "xga_total": xga, 
                "npxg_total": npxg, "npxga_total": npxga, "ppda": ppda_sum / m, "oppda": oppda_sum / m, 
                "dc": dc / m, "odc": odc / m, "pts": pts, "xpts": xpts, "matches": m
            }

        for team_id, details in data.items():
            t_name = details['title']
            team_list.append(t_name)
            hist = details['history']
            stats_db[t_name] = {
                "total": calculate_segment(hist),
                "form": calculate_segment(hist[-5:]),
                "home": calculate_segment([m for m in hist if m.get('h_a') == 'h']),
                "away": calculate_segment([m for m in hist if m.get('h_a') == 'a'])
            }
        return stats_db, sorted(team_list)
    except Exception as e:
        st.error(f"🚨 ERRORE SORGENTE DATI ({league_name}): {str(e)}")
        st.stop()
        return {}, []

@st.cache_data(ttl=3600)
def fetch_understat_players(league_name):
    u_league = LEAGUES_CONFIG[league_name]["understat"]
    try:
        url = f"https://understat.com/league/{u_league}/2024"
        players_data = _get_understat_json(url, "playersData")
        if not players_data: return None
        df = pd.DataFrame(players_data)
        df['time'] = pd.to_numeric(df['time'])
        df['xG'] = pd.to_numeric(df['xG'])
        df['xA'] = pd.to_numeric(df['xA'])
        df['xg90'] = (df['xG'] / df['time']) * 90
        df['xa90'] = (df['xA'] / df['time']) * 90
        return df[['player_name', 'team_title', 'xg90', 'xa90']].rename(columns={'player_name': 'player', 'team_title': 'team'})
    except Exception:
        return None

@st.cache_data(ttl=3600)
def fetch_league_odds(league_id, api_key):
    if not api_key or api_key == "INSERISCI_QUI_LA_TUA_CHIAVE": 
        return []
    url = f"https://api.the-odds-api.com/v4/sports/{league_id}/odds/"
    params = {"apiKey": api_key, "regions": "eu", "markets": "h2h,totals", "oddsFormat": "decimal"}
    try:
        res = requests.get(url, params=params)
        res.raise_for_status() 
        data = res.json()
        if not isinstance(data, list): 
            return []
        return data
    except Exception:
        return []

def extract_match_odds(all_odds, home_team, away_team):
    if not all_odds: 
        return None
    feed_home_teams = [m['home_team'] for m in all_odds]
    h_match = difflib.get_close_matches(home_team, feed_home_teams, n=1, cutoff=0.45)
    if h_match:
        matched_home = h_match[0]
        for match in all_odds:
            if match['home_team'] == matched_home:
                if not match.get('bookmakers'): 
                    return None
                bookie = match['bookmakers'][0] 
                odds_data = {"h2h": {}, "totals": {}, "btts": {}}
                for m in bookie['markets']:
                    if m['key'] == 'h2h':
                        odds_data['h2h'] = {o['name']: o['price'] for o in m['outcomes']}
                    elif m['key'] == 'totals':
                        o25 = [o['price'] for o in m['outcomes'] if o['name'] == 'Over' and o['point'] == 2.5]
                        u25 = [o['price'] for o in m['outcomes'] if o['name'] == 'Under' and o['point'] == 2.5]
                        if o25 and u25: 
                            odds_data['totals'] = {"Over 2.5": o25[0], "Under 2.5": u25[0]}
                return odds_data
    return None

def fuzzy_match_team(api_name, csv_team_list):
    matches = difflib.get_close_matches(api_name, csv_team_list, n=1, cutoff=0.55)
    if matches:
        return matches[0]
    return None

# ==============================================================================
# 🤖 MACHINE LEARNING (MULTI-TARGET)
# ==============================================================================
@st.cache_resource
def train_ml_models(history_data):
    if not history_data:
        return None
    X = []
    y_1x2 = []
    y_uo25 = []
    y_btts = []
    
    for match in history_data:
        score = match.get("Real_Score", "-")
        if score != "-" and "-" in score and "f_xh" in match:
            try:
                hg, ag = map(int, score.replace(" ", "").split("-"))
                features = [
                    float(match.get("f_xh", 1.0)), float(match.get("f_xa", 1.0)),
                    float(match.get("P1_Stat", 0.33)), float(match.get("PX_Stat", 0.33)), float(match.get("P2_Stat", 0.33)),
                    float(match.get("PO25_Stat", 0.50)), float(match.get("PGG_Stat", 0.50)),
                    float(match.get("h_elo", 1500.0)), float(match.get("a_elo", 1500.0)),
                    float(match.get("h_ppda", 10.0)), float(match.get("a_ppda", 10.0)),
                    float(match.get("h_dc", 5.0)), float(match.get("a_dc", 5.0)),
                    float(match.get("w_seas", 0.70)), float(match.get("volatility", 1.0))
                ]
                X.append(features)
                y_1x2.append(1 if hg > ag else (0 if hg == ag else 2))
                y_uo25.append(1 if (hg + ag) > 2.5 else 0)
                y_btts.append(1 if (hg > 0 and ag > 0) else 0)
            except Exception: 
                continue
                
    if len(X) < 15: 
        return None
    
    model_1x2 = RandomForestClassifier(n_estimators=200, max_depth=7, random_state=42).fit(X, y_1x2)
    model_uo25 = RandomForestClassifier(n_estimators=200, max_depth=7, random_state=42).fit(X, y_uo25)
    model_btts = RandomForestClassifier(n_estimators=200, max_depth=7, random_state=42).fit(X, y_btts)
    return {"1x2": model_1x2, "uo25": model_uo25, "btts": model_btts}

def apply_ml_boost(models_dict, f_xh, f_xa, p1_stat, px_stat, p2_stat, po25_stat, pgg_stat, h_elo, a_elo, h_ppda, a_ppda, h_dc, a_dc, w_seas, volatility):
    if not models_dict: 
        return p1_stat, px_stat, p2_stat, po25_stat, pgg_stat
        
    features = np.array([[f_xh, f_xa, p1_stat, px_stat, p2_stat, po25_stat, pgg_stat, h_elo, a_elo, h_ppda, a_ppda, h_dc, a_dc, w_seas, volatility]])
    
    m_1x2 = models_dict["1x2"]
    probs_1x2 = m_1x2.predict_proba(features)[0]
    classes_1x2 = list(m_1x2.classes_)
    ml_p1 = probs_1x2[classes_1x2.index(1)] if 1 in classes_1x2 else 0.0
    ml_px = probs_1x2[classes_1x2.index(0)] if 0 in classes_1x2 else 0.0
    ml_p2 = probs_1x2[classes_1x2.index(2)] if 2 in classes_1x2 else 0.0
    
    m_uo25 = models_dict["uo25"]
    probs_uo25 = m_uo25.predict_proba(features)[0]
    classes_uo25 = list(m_uo25.classes_)
    ml_po25 = probs_uo25[classes_uo25.index(1)] if 1 in classes_uo25 else 0.0
    
    m_btts = models_dict["btts"]
    probs_btts = m_btts.predict_proba(features)[0]
    classes_btts = list(m_btts.classes_)
    ml_pgg = probs_btts[classes_btts.index(1)] if 1 in classes_btts else 0.0
    
    f_p1 = (p1_stat * 0.75) + (ml_p1 * 0.25)
    f_px = (px_stat * 0.75) + (ml_px * 0.25)
    f_p2 = (p2_stat * 0.75) + (ml_p2 * 0.25)
    f_po25 = (po25_stat * 0.75) + (ml_po25 * 0.25)
    f_pgg = (pgg_stat * 0.75) + (ml_pgg * 0.25)
    
    tot_1x2 = f_p1 + f_px + f_p2
    return f_p1/tot_1x2, f_px/tot_1x2, f_p2/tot_1x2, f_po25, f_pgg

# ==============================================================================
# FUNZIONI MATEMATICHE & ML STATISTICO
# ==============================================================================
def dixon_coles_probability(h_goals, a_goals, mu_h, mu_a, rho):
    prob = (math.exp(-mu_h) * (mu_h**h_goals) / math.factorial(h_goals)) * (math.exp(-mu_a) * (mu_a**a_goals) / math.factorial(a_goals))
    if h_goals == 0 and a_goals == 0: 
        prob *= (1.0 - (mu_h * mu_a * rho))
    elif h_goals == 0 and a_goals == 1: 
        prob *= (1.0 + (mu_h * rho))
    elif h_goals == 1 and a_goals == 0: 
        prob *= (1.0 + (mu_a * rho))
    elif h_goals == 1 and a_goals == 1: 
        prob *= (1.0 - rho)
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
        if tilt_H > 0.60: 
            base_H *= 1.15
            base_A *= 0.85
        elif tilt_H < 0.40: 
            base_H *= 0.85
            base_A *= 1.15
    return max(0.1, base_H), max(0.1, base_A)

def calculate_player_probability(metric_per90, expected_mins, team_match_xg, team_avg_xg):
    base_lambda = (metric_per90 / 90.0) * expected_mins
    if team_avg_xg <= 0.1: 
        team_avg_xg = max(0.1, team_match_xg)
    raw_factor = team_match_xg / team_avg_xg
    final_factor = min(2.5, pow(raw_factor, 0.75))
    return 1 - math.exp(-(base_lambda * final_factor))

def calculate_combo_player(matrix, outcome_type, team_type, player_share):
    prob_combo = 0.0
    for h in range(10):
        for a in range(10):
            p_score = matrix[h, a]
            if p_score == 0: continue
            
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
# MOTORE TOP 5 VALUE BETS (ALLINEAMENTO TOTALE DATI IBRIDI E MATRICE)
# ==============================================================================
def find_top_value_bets(all_odds, stats_db, L_DATA, volatility, m_type, elo_dict, w_seas, data_mode, ml_models=None):
    if not all_odds: 
        return []
    team_list = list(stats_db.keys())
    value_bets = []
    
    # 🟢 Funzione interna per estrarre Gol, xG o Ibrido (identica a quella in app.py)
    def get_team_metric(team_stats, segment, metric):
        raw = team_stats.get(segment, {})
        matches = raw.get('matches', 0)
        if matches <= 0: return 0.0
        
        if metric == 'gf':
            v_goals, v_xg, v_npxg = raw.get('goals_total', 0), raw.get('xg_total', 0), raw.get('npxg_total', 0)
        else:
            v_goals, v_xg, v_npxg = raw.get('ga_total', 0), raw.get('xga_total', 0), raw.get('npxga_total', 0)
            
        if data_mode == "Solo Gol Reali": return v_goals / matches
        elif data_mode == "Solo xG (NPxG Mode)": return v_npxg / matches
        else: return ((v_npxg * 0.40) + (v_xg * 0.30) + (v_goals * 0.30)) / matches

    # 🟢 Mix tra Stagione e Trend Forma
    def get_blended(team_stats, metric):
        val_tot = get_team_metric(team_stats, 'total', metric)
        val_form = get_team_metric(team_stats, 'form', metric)
        return (val_tot * w_seas) + (val_form * (1 - w_seas))
    
    for match in all_odds:
        h_match = difflib.get_close_matches(match['home_team'], team_list, cutoff=0.55)
        a_match = difflib.get_close_matches(match['away_team'], team_list, cutoff=0.55)
        if not h_match or not a_match: 
            continue
        
        h_name, a_name = h_match[0], a_match[0]
        h_stats, a_stats = stats_db[h_name], stats_db[a_name]
        
        try:
            bookie = match['bookmakers'][0]
            b1 = bX = b2 = bO25 = 0.0
            for m in bookie['markets']:
                if m['key'] == 'h2h':
                    b1 = next((o['price'] for o in m['outcomes'] if o['name'] == match['home_team']), 0)
                    bX = next((o['price'] for o in m['outcomes'] if o['name'] == 'Draw'), 0)
                    b2 = next((o['price'] for o in m['outcomes'] if o['name'] == match['away_team']), 0)
                elif m['key'] == 'totals':
                    bO25 = next((o['price'] for o in m['outcomes'] if o['name'] == 'Over' and o['point'] == 2.5), 0)
        except: 
            continue
        
        h_elo = get_elo_for_team(h_name, elo_dict, 1500.0)
        a_elo = get_elo_for_team(a_name, elo_dict, 1500.0)
        
        # 🟢 Estrazione dati allineata alla UI principale
        h_att_base = get_blended(h_stats, 'gf')
        h_def_base = get_blended(h_stats, 'gs')
        a_att_base = get_blended(a_stats, 'gf')
        a_def_base = get_blended(a_stats, 'gs')
        
        h_att_home = get_team_metric(h_stats, 'home', 'gf') if h_stats['home']['matches'] > 0 else h_att_base
        h_def_home = get_team_metric(h_stats, 'home', 'gs') if h_stats['home']['matches'] > 0 else h_def_base
        a_att_away = get_team_metric(a_stats, 'away', 'gf') if a_stats['away']['matches'] > 0 else a_att_base
        a_def_away = get_team_metric(a_stats, 'away', 'gs') if a_stats['away']['matches'] > 0 else a_def_base
        
        w_split = 0.60
        h_fin_att = (h_att_base * (1-w_split)) + (h_att_home * w_split)
        h_fin_def = (h_def_base * (1-w_split)) + (h_def_home * w_split)
        a_fin_att = (a_att_base * (1-w_split)) + (a_att_away * w_split)
        a_fin_def = (a_def_base * (1-w_split)) + (a_def_away * w_split)
        
        xg_h_base = (h_fin_att * a_fin_def) / L_DATA["avg"]
        xg_a_base = (a_fin_att * h_fin_def) / L_DATA["avg"]
        
        home_adv = L_DATA["ha"] if m_type == "Standard" else (0.0 if m_type == "Campo Neutro" else L_DATA["ha"]*0.5)
        elo_diff = (h_elo + (100 if m_type=="Standard" else 0)) - a_elo
        f_xh = (xg_h_base * (1 + elo_diff/1000.0)) + home_adv
        f_xa = (xg_a_base * (1 - elo_diff/1000.0))
        
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
            
        f_xh *= volatility
        f_xa *= volatility
        
        matrix = np.zeros((10,10))
        p1 = pX = p2 = pGG = pO25 = 0.0
        
        for h in range(10):
            for a in range(10):
                p = dixon_coles_probability(h, a, f_xh, f_xa, L_DATA["rho"])
                matrix[h,a] = p
                if h > a: p1 += p
                elif h == a: pX += p
                else: p2 += p
                if h > 0 and a > 0: pGG += p
                if (h + a) > 2.5: pO25 += p
                
        tot = np.sum(matrix)
        if tot > 0:
            p1 /= tot; pX /= tot; p2 /= tot; pGG /= tot; pO25 /= tot
                
        if ml_models:
            p1, pX, p2, pO25, pGG = apply_ml_boost(ml_models, f_xh, f_xa, p1, pX, p2, pO25, pGG, h_elo, a_elo, h_ppda, a_ppda, h_dc, a_dc, w_seas, volatility)
            
        evals = [("1", p1, b1), ("X", pX, bX), ("2", p2, b2), ("Over 2.5", pO25, bO25)]
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
        with open('mathbet_history.json', 'r') as f: 
            return json.load(f)
    except FileNotFoundError: 
        return []

def salva_storico_json(history_list):
    try:
        with open('mathbet_history.json', 'w') as f: 
            json.dump(history_list, f, indent=2, default=str)
        return True
    except Exception: 
        return False

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

# ==============================================================================
# 🔴 TRADING SPORTIVO LIVE (MOMENTUM & VALUE IN-PLAY)
# ==============================================================================
def calculate_live_pressure_index(stats_home, stats_away, current_minute):
    """
    Calcola l'indice di pressione (Momentum) basato su Attacchi Pericolosi e Tiri in Porta.
    """
    if not stats_home or not stats_away or current_minute <= 0:
        return 0.0, 0.0
        
    da_h = stats_home.get('Dangerous attacks', 0)
    da_a = stats_away.get('Dangerous attacks', 0)
    sot_h = stats_home.get('Shots on target', 0)
    sot_a = stats_away.get('Shots on target', 0)
    c_h = stats_home.get('Corner kicks', 0)
    c_a = stats_away.get('Corner kicks', 0)
    
    # Ponderazione Pressione: Tiri in porta * 5, Attacchi Pericolosi * 1.5, Corner * 3
    pressure_h = (sot_h * 5) + (da_h * 1.5) + (c_h * 3)
    pressure_a = (sot_a * 5) + (da_a * 1.5) + (c_a * 3)
    
    idx_h = pressure_h / current_minute
    idx_a = pressure_a / current_minute
    
    return idx_h, idx_a

def adjust_probs_for_time(p1, px, p2, po25, current_minute, goals_h, goals_a):
    """
    Ricalcola le probabilità in base ai minuti residui sfruttando il decadimento.
    """
    rem_time_ratio = max(0.01, (90 - current_minute) / 90.0)
    
    if goals_h == goals_a:
        new_px = px + ((1 - px) * (1 - rem_time_ratio) * 0.7)
        new_p1 = p1 * (1 - new_px) / max(0.01, (p1 + p2))
        new_p2 = p2 * (1 - new_px) / max(0.01, (p1 + p2))
    elif goals_h > goals_a:
        new_p1 = p1 + ((1 - p1) * (1 - rem_time_ratio) * 0.8)
        new_px = px * (1 - new_p1) / max(0.01, (px + p2))
        new_p2 = p2 * (1 - new_p1) / max(0.01, (px + p2))
    else:
        new_p2 = p2 + ((1 - p2) * (1 - rem_time_ratio) * 0.8)
        new_px = px * (1 - new_p2) / max(0.01, (px + p1))
        new_p1 = p1 * (1 - new_p2) / max(0.01, (px + p1))
        
    return new_p1, new_px, new_p2

def calculate_value_in_play(prematch_prob_h, prematch_prob_a, current_minute, live_odd_h, live_odd_a, idx_h, idx_a):
    """
    Incrocia le quote live con la probabilità e la pressione per generare Segnali Trading.
    """
    signals = []
    if current_minute > 20: 
        if idx_h > 1.8 and idx_h > (idx_a * 2): # Dominio netto Casa
            fair_odd = 1 / max(0.01, prematch_prob_h)
            if live_odd_h > fair_odd * 1.15: # Quota live più alta del fair pre-match di un margine
                signals.append({"Team": "Casa", "Signal": "Alta Pressione + Valore (Punta)", "Pressione": round(idx_h, 2), "Odd Live": live_odd_h, "Fair": round(fair_odd,2)})
                
        if idx_a > 1.8 and idx_a > (idx_h * 2): # Dominio netto Ospite
            fair_odd = 1 / max(0.01, prematch_prob_a)
            if live_odd_a > fair_odd * 1.15:
                signals.append({"Team": "Ospite", "Signal": "Alta Pressione + Valore (Punta)", "Pressione": round(idx_a, 2), "Odd Live": live_odd_a, "Fair": round(fair_odd,2)})
                
    return signals
