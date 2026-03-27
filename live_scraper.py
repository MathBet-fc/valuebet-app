import requests
import json
import time

def get_sofascore_search(home_team, away_team):
    """
    Cerca una partita su SofaScore per ottenere il suo event_id.
    """
    query = f"{home_team} {away_team}"
    url = f"https://api.sofascore.com/api/v1/search/events?q={query}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json"
    }
    try:
        res = requests.get(url, headers=headers, timeout=7)
        if res.status_code == 200:
            data = res.json()
            if 'results' in data and len(data['results']) > 0:
                # Prende il primo ID evento corrispondente
                return data['results'][0]['id']
    except Exception as e:
        print(f"Errore ricerca SofaScore: {e}")
    
    return None

def get_sofascore_live_stats(event_id):
    """
    Recupera le statistiche Live (Tiri, Angoli, Attacchi Pericolosi) di un event_id SofaScore.
    """
    url = f"https://api.sofascore.com/api/v1/event/{event_id}/statistics"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json"
    }
    try:
        res = requests.get(url, headers=headers, timeout=7)
        if res.status_code == 200:
            data = res.json()
            
            # Parsea le statistiche in un dictionary strutturato
            stats_dict = {"home": {}, "away": {}}
            if 'statistics' in data and len(data['statistics']) > 0:
                period_stats = data['statistics'][0]['groups']
                for group in period_stats:
                    for item in group['statisticsItems']:
                        name = item['name']
                        stats_dict["home"][name] = item.get('home', 0)
                        stats_dict["away"][name] = item.get('away', 0)
            return stats_dict
    except Exception as e:
        print(f"Errore download stats SofaScore: {e}")
    return None

def get_sofascore_live_momentum(event_id):
    """
    Recupera il grafico live della Pressione (Momentum Index).
    """
    url = f"https://api.sofascore.com/api/v1/event/{event_id}/graph"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json"
    }
    try:
        res = requests.get(url, headers=headers, timeout=7)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return None

def get_betfair_free_odds(market_id, app_key="it.betfair.desktop.web"):
    """
    Recupera quote e volumi di scambio live intercettando le API di sola lettura di Betfair.
    Il market_id è usualmente nel formato '1.123456789'
    """
    url = f"https://ero.betfair.it/www/sports/exchange/readonly/v1/bymarket?_ak={app_key}&alt=json&currencyCode=EUR&locale=it&marketIds={market_id}&types=MARKET_STATE,RUNNER_STATE,RUNNER_EXCHANGE_PRICES_BEST,RUNNER_DESCRIPTION"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json"
    }
    try:
        res = requests.get(url, headers=headers, timeout=7)
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        print(f"Errore download quote Betfair Frontend: {e}")
    return None
