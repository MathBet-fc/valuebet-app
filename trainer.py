import pandas as pd
import numpy as np
from scipy.optimize import minimize
import math

# --- CONFIGURAZIONE ---
FILE_DATASET = "I1.csv" # Il nome del file scaricato (I1 = Serie A)

# 1. CARICAMENTO DATI
print(f"📂 Caricamento dataset: {FILE_DATASET}...")
try:
    df = pd.read_csv(FILE_DATASET)
    # Filtriamo solo le colonne necessarie
    df = df[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']].dropna()
    print(f"✅ Trovate {len(df)} partite valide.")
except:
    print("❌ Errore: File non trovato. Scarica il CSV da football-data.co.uk")
    exit()

# 2. DEFINIZIONE MODELLO (DIXON-COLES)
# Deve essere IDENTICO a quello usato nell'App per funzionare
def dixon_coles_prob(h_g, a_g, exp_h, exp_a, rho):
    if exp_h <= 0: exp_h = 0.01
    if exp_a <= 0: exp_a = 0.01
    
    # Poisson
    prob = (math.exp(-exp_h) * (exp_h**h_g) / math.factorial(h_g)) * \
           (math.exp(-exp_a) * (exp_a**a_g) / math.factorial(a_g))
    
    # Correzione Rho
    if h_g == 0 and a_g == 0: prob *= (1.0 - (exp_h * exp_a * rho))
    elif h_g == 0 and a_g == 1: prob *= (1.0 + (exp_h * rho))
    elif h_g == 1 and a_g == 0: prob *= (1.0 + (exp_a * rho))
    elif h_g == 1 and a_g == 1: prob *= (1.0 - rho)
    
    return max(0.00001, prob)

# 3. FUNZIONE DI COSTO (LOG LOSS)
# L'obiettivo è minimizzare questo numero
def negative_log_likelihood(params):
    home_adv, rho = params
    loss = 0
    
    # Media gol campionato (Attacco medio)
    avg_goals = (df['FTHG'].mean() + df['FTAG'].mean()) / 2
    
    for _, row in df.iterrows():
        real_h = int(row['FTHG'])
        real_a = int(row['FTAG'])
        
        # Nel training semplificato assumiamo che la forza prevista fosse la media campionato
        # + il vantaggio casa che stiamo cercando di scoprire
        lambda_h = avg_goals + home_adv
        lambda_a = avg_goals
        
        # Calcoliamo la probabilità che il modello avrebbe dato al risultato REALE
        prob = dixon_coles_prob(real_h, real_a, lambda_h, lambda_a, rho)
        
        # Sommiamo l'errore (Log Loss negativo)
        loss -= np.log(prob)
        
    return loss

# 4. OTTIMIZZAZIONE
print("\n🤖 Allenamento in corso... (Ricerca parametri ottimali)")
# Punto di partenza [VantaggioCasa, Rho]
initial_guess = [0.25, -0.10] 

# Minimizzazione
result = minimize(negative_log_likelihood, initial_guess, method='Nelder-Mead')

# 5. OUTPUT PER L'UTENTE
best_ha = result.x[0]
best_rho = result.x[1]

print("\n" + "="*50)
print("🏆 TRAINING COMPLETATO CON SUCCESSO")
print("="*50)
print(f"📊 Parametri Ottimizzati per {FILE_DATASET}:")
print(f"   ► Home Advantage (ha): {best_ha:.4f}")
print(f"   ► Rho (Fattore Pareggi): {best_rho:.4f}")
print("-" * 50)
print("📝 COPIA E INCOLLA QUESTO NEL TUO SITO (Sezione LEAGUES):")
print(f'"ha": {best_ha:.2f}, "rho": {best_rho:.2f}')
print("="*50)
