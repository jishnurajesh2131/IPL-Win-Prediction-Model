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

def evaluate_metrics(y_true, y_prob, threshold=0.480):
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

# Apply the same interaction features
df['interaction_wickets_balls'] = df['wickets_remaining'] * df['legal_balls_remaining']
df['interaction_crr_rrr'] = df['current_run_rate'] / (df['required_run_rate'].replace(0, 0.1) + 0.1)
df['interaction_phase_wickets'] = (df['overs_completed'] > 15).astype(int) * df['wickets_remaining']

exclude_cols = ['match_id', 'season', 'date', 'target_team_a_win']

test_df = df[df['season'].astype(str).str.startswith(('2024', '2025', '2026'))].copy()
features = [c for c in df.columns if c not in exclude_cols]

print("Loading candidate models and preprocessors...")
prep_inn1 = joblib.load('models/live/candidates/candidate_v1_preprocessor_inn1.joblib')
prep_inn2 = joblib.load('models/live/candidates/candidate_v1_preprocessor_inn2.joblib')
model_inn1 = joblib.load('models/live/candidates/candidate_v1_model_inn1.joblib')
model_inn2 = joblib.load('models/live/candidates/candidate_v1_model_inn2.joblib')

def predict_probabilities(df_split):
    if len(df_split) == 0:
        return np.array([])
    
    inn1_mask = df_split['innings'] == 1
    inn2_mask = df_split['innings'] == 2
    
    probs = np.zeros(len(df_split))
    
    if inn1_mask.sum() > 0:
        X1 = df_split.loc[inn1_mask, features]
        X1_transformed = prep_inn1.transform(X1)
        probs[inn1_mask] = model_inn1.predict_proba(X1_transformed)[:, 1]
        
    if inn2_mask.sum() > 0:
        X2 = df_split.loc[inn2_mask, features]
        X2_transformed = prep_inn2.transform(X2)
        probs[inn2_mask] = model_inn2.predict_proba(X2_transformed)[:, 1]
        
    return probs

print(f"\nEvaluating Final Test Set (2024-2026) N={len(test_df)}")
test_prob = predict_probabilities(test_df)
# using the optimal threshold found on Validation (0.480)
test_metrics = evaluate_metrics(test_df['target_team_a_win'], test_prob, threshold=0.480)
print_metrics("Final Test Overall (Candidate V1, Threshold 0.480)", test_metrics)

# Also test with 0.50 for comparison
test_metrics_50 = evaluate_metrics(test_df['target_team_a_win'], test_prob, threshold=0.50)
print_metrics("Final Test Overall (Candidate V1, Threshold 0.50)", test_metrics_50)
