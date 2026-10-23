import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import scipy.stats as stats

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

print("Loading dataset...")
df = pd.read_csv('data/processed/live_dataset.csv')

# --- FEATURE ENGINEERING ---
# Adding interaction features safely
df['interaction_wickets_balls'] = df['wickets_remaining'] * df['legal_balls_remaining']
df['interaction_crr_rrr'] = df['current_run_rate'] / (df['required_run_rate'].replace(0, 0.1) + 0.1)
df['interaction_phase_wickets'] = (df['overs_completed'] > 15).astype(int) * df['wickets_remaining']

exclude_cols = ['match_id', 'season', 'date', 'target_team_a_win']
cat_cols = ['venue', 'city', 'team_a', 'team_b', 'batting_team', 'bowling_team', 'striker', 'non_striker', 'bowler']
# Num cols are everything else
num_cols = [c for c in df.columns if c not in exclude_cols + cat_cols]

# Temporal splits
train_df = df[~df['season'].astype(str).str.startswith(('2022', '2023', '2024', '2025', '2026'))].copy()
val_df = df[df['season'].astype(str).str.startswith(('2022', '2023'))].copy()
# TEST IS EXCLUDED COMPLETELY

print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

# Create Candidates directory
os.makedirs('models/live/candidates', exist_ok=True)

def train_and_tune(innings_num, n_iter=15):
    print(f"\n--- Tuning Innings {innings_num} ---")
    tr = train_df[train_df['innings'] == innings_num]
    vl = val_df[val_df['innings'] == innings_num]
    
    y_tr = tr['target_team_a_win']
    y_vl = vl['target_team_a_win']
    X_tr = tr.drop(columns=exclude_cols)
    X_vl = vl.drop(columns=exclude_cols)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), cat_cols)
        ]
    )
    
    print("Fitting preprocessor...")
    X_tr_prep = preprocessor.fit_transform(X_tr)
    X_vl_prep = preprocessor.transform(X_vl)
    
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'min_child_weight': [1, 5, 10],
        'gamma': [0, 1, 5]
    }
    
    xgb = XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1)
    
    search = RandomizedSearchCV(
        xgb, param_distributions=param_dist,
        n_iter=n_iter, scoring='neg_log_loss', cv=3, verbose=1, random_state=42, n_jobs=-1
    )
    
    print("Searching hyperparameters...")
    search.fit(X_tr_prep, y_tr)
    
    best_model = search.best_estimator_
    print(f"Best Params for Innings {innings_num}: {search.best_params_}")
    
    val_probs = best_model.predict_proba(X_vl_prep)[:, 1]
    metrics = evaluate_metrics(y_vl, val_probs)
    
    print(f"Validation Metrics Innings {innings_num}:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
        
    return preprocessor, best_model, val_probs, y_vl

prep1, mod1, val_prob1, y_val1 = train_and_tune(1, n_iter=10)
prep2, mod2, val_prob2, y_val2 = train_and_tune(2, n_iter=10)

print("\n--- Overall Validation Metrics (New Candidate) ---")
all_val_prob = np.concatenate([val_prob1, val_prob2])
all_y_val = np.concatenate([y_val1, y_val2])
overall_metrics = evaluate_metrics(all_y_val, all_val_prob)
for k, v in overall_metrics.items():
    print(f"  {k}: {v:.4f}")

# Find optimal threshold on validation set for Accuracy
best_thresh = 0.5
best_acc = 0
for t in np.linspace(0.4, 0.6, 21):
    acc = accuracy_score(all_y_val, (all_val_prob >= t).astype(int))
    if acc > best_acc:
        best_acc = acc
        best_thresh = t
print(f"\nOptimal threshold for Accuracy on Val: {best_thresh:.3f} (Acc: {best_acc:.4f})")

joblib.dump(prep1, 'models/live/candidates/candidate_v1_preprocessor_inn1.joblib')
joblib.dump(mod1, 'models/live/candidates/candidate_v1_model_inn1.joblib')
joblib.dump(prep2, 'models/live/candidates/candidate_v1_preprocessor_inn2.joblib')
joblib.dump(mod2, 'models/live/candidates/candidate_v1_model_inn2.joblib')
print("\nCandidate models saved to models/live/candidates/")
