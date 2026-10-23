import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    
    ece = 0.0
    for i in range(n_bins):
        bin_mask = (binids == i)
        if np.sum(bin_mask) > 0:
            bin_conf = np.mean(y_prob[bin_mask])
            bin_acc = np.mean(y_true[bin_mask])
            ece += (np.sum(bin_mask) / len(y_prob)) * np.abs(bin_conf - bin_acc)
    return ece

def evaluate_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Log Loss': log_loss(y_true, y_prob, labels=[0,1]),
        'Brier Score': brier_score_loss(y_true, y_prob),
        'ECE': expected_calibration_error(y_true, y_prob),
        'AUC': roc_auc_score(y_true, y_prob)
    }

def print_metrics(title, metrics):
    print(f"\n--- {title} ---")
    print(f"Accuracy:    {metrics['Accuracy']:.4f}")
    print(f"Log Loss:    {metrics['Log Loss']:.4f}")
    print(f"Brier Score: {metrics['Brier Score']:.4f}")
    print(f"ECE:         {metrics['ECE']:.4f}")
    print(f"AUC:         {metrics['AUC']:.4f}")

print("Loading dataset...")
df = pd.read_csv('data/processed/live_dataset.csv')

# Features expected by preprocessor (excluding identifiers and target)
exclude_cols = ['match_id', 'season', 'date', 'target_team_a_win']
features_inn1 = [c for c in df.columns if c not in exclude_cols]
features_inn2 = [c for c in df.columns if c not in exclude_cols]

print("Loading models and preprocessors...")
prep_inn1 = joblib.load('models/live/primary_live_preprocessor_inn1.joblib')
prep_inn2 = joblib.load('models/live/primary_live_preprocessor_inn2.joblib')
model_inn1 = joblib.load('models/live/primary_live_model_inn1.joblib')
model_inn2 = joblib.load('models/live/primary_live_model_inn2.joblib')

def predict_probabilities(df_split):
    if len(df_split) == 0:
        return np.array([])
    
    inn1_mask = df_split['innings'] == 1
    inn2_mask = df_split['innings'] == 2
    
    probs = np.zeros(len(df_split))
    
    if inn1_mask.sum() > 0:
        X1 = df_split.loc[inn1_mask, features_inn1]
        X1_transformed = prep_inn1.transform(X1)
        probs[inn1_mask] = model_inn1.predict_proba(X1_transformed)[:, 1]
        
    if inn2_mask.sum() > 0:
        X2 = df_split.loc[inn2_mask, features_inn2]
        X2_transformed = prep_inn2.transform(X2)
        probs[inn2_mask] = model_inn2.predict_proba(X2_transformed)[:, 1]
        
    return probs

# Create temporal splits
val_df = df[df['season'].astype(str).str.startswith(('2022', '2023'))].copy()
test_df = df[df['season'].astype(str).str.startswith(('2024', '2025', '2026'))].copy()

print(f"\nEvaluating Validation Set (2022-2023) N={len(val_df)}")
val_prob = predict_probabilities(val_df)
val_metrics = evaluate_metrics(val_df['target_team_a_win'], val_prob)
print_metrics("Validation Overall", val_metrics)

print(f"\nEvaluating Final Test Set (2024-2026) N={len(test_df)}")
test_prob = predict_probabilities(test_df)
test_metrics = evaluate_metrics(test_df['target_team_a_win'], test_prob)
print_metrics("Final Test Overall", test_metrics)

# Error analysis on Validation
print("\n=== VALIDATION ERROR ANALYSIS ===")
val_df['predicted_prob'] = val_prob

# By Innings
print("\nBy Innings:")
for inn in [1, 2]:
    subset = val_df[val_df['innings'] == inn]
    metrics = evaluate_metrics(subset['target_team_a_win'], subset['predicted_prob'])
    print(f" Innings {inn}: Acc={metrics['Accuracy']:.4f}, AUC={metrics['AUC']:.4f}, LogLoss={metrics['Log Loss']:.4f}")

# By Phase
def get_phase(overs):
    if overs < 6: return 'Powerplay'
    elif overs < 15: return 'Middle'
    else: return 'Death'

val_df['phase'] = val_df['overs_completed'].apply(get_phase)
print("\nBy Phase:")
for phase in ['Powerplay', 'Middle', 'Death']:
    subset = val_df[val_df['phase'] == phase]
    metrics = evaluate_metrics(subset['target_team_a_win'], subset['predicted_prob'])
    print(f" {phase}: Acc={metrics['Accuracy']:.4f}, AUC={metrics['AUC']:.4f}, LogLoss={metrics['Log Loss']:.4f}")

# Wickets remaining
print("\nBy Wickets Remaining (Innings 2):")
for w in [10, 8, 5, 2]:
    subset = val_df[(val_df['innings'] == 2) & (val_df['wickets_remaining'] <= w) & (val_df['wickets_remaining'] > w-3)]
    if len(subset) > 0:
        metrics = evaluate_metrics(subset['target_team_a_win'], subset['predicted_prob'])
        print(f" Wickets {w-2}-{w}: Acc={metrics['Accuracy']:.4f}, LogLoss={metrics['Log Loss']:.4f}")

