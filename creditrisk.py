import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, classification_report, precision_recall_curve
import lightgbm as lgb
import shap
import joblib
import textwrap
import json
import warnings
warnings.filterwarnings('ignore')


DATA_PATH = "Credit_risk_project/credit_risk_dataset.csv"
TARGET_COL = "default"
ID_COL = "loan_id"
DATE_COL = "month"
RANDOM_SEED = 42
OUTPUT_DIR = "shap_credit_out"

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(RANDOM_SEED)


if DATA_PATH and os.path.exists(DATA_PATH):
    print(f"Loading dataset from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found")
else:
    print("No dataset found — simulating dataset")

    n_samples = 20000
    n_borrower_feats = 12
    n_macro = 6

    dates = pd.date_range(end=pd.Timestamp("2024-12-31"), periods=n_samples // 50, freq="W")
    date_values = np.random.choice(dates, size=n_samples)

    df = pd.DataFrame({DATE_COL: date_values})
    df[ID_COL] = np.arange(1, n_samples + 1)

    for i in range(n_borrower_feats):
        df[f"borrower_feat_{i+1}"] = np.random.normal(0, 1, n_samples) + (i % 3) * 0.5

    # Macro features
    macro_base = np.sin(np.linspace(0, 12 * np.pi, n_samples))
    for j in range(n_macro):
        df[f"macro_{j+1}"] = macro_base + np.random.normal(0, 0.5, n_samples) + j * 0.1

    # Engineered features
    df["credit_util"] = np.clip(np.random.beta(2, 5, n_samples) + 0.2 * (df["borrower_feat_1"] > 0), 0, 1)
    df["num_past_defaults"] = np.random.poisson(0.3, n_samples)

    # Logistic target generation
    linear_term = (
        0.8 * df["credit_util"]
        + 0.6 * df["num_past_defaults"]
        + 0.4 * df["borrower_feat_2"]
        - 0.3 * df["borrower_feat_5"]
        + 0.5 * df["macro_1"]
    )

    prob = 1 / (1 + np.exp(-(-2 + 0.9 * linear_term)))
    df[TARGET_COL] = (np.random.rand(n_samples) < prob).astype(int)

    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

print(f"Dataset shape: {df.shape}")


df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

try:
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
except:
    df[DATE_COL] = pd.date_range(end=pd.Timestamp("2024-12-31"), periods=len(df))

df = df.sort_values(DATE_COL).reset_index(drop=True)

feature_cols = [c for c in df.columns if c not in [TARGET_COL, ID_COL, DATE_COL]]
X = df[feature_cols].copy()
y = df[TARGET_COL].copy()


n = len(df)
train_end = int(0.6 * n)
val_end = int(0.8 * n)

X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

print("Train / Val / Test:", X_train.shape, X_val.shape, X_test.shape)


base_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "verbosity": -1,
    "seed": RANDOM_SEED,
    "is_unbalance": True,
}

param_dist = {
    "num_leaves": [31, 50, 80, 120],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "n_estimators": [100, 300, 600],
    "min_child_samples": [5, 10, 20, 50],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}

from lightgbm import early_stopping, log_evaluation

best_auc = -1
best_params = None

for params in ParameterSampler(param_dist, n_iter=12, random_state=RANDOM_SEED):
    p = base_params.copy()
    p.update(params)

    model = lgb.LGBMClassifier(**p)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            early_stopping(stopping_rounds=50),
            log_evaluation(period=0)
        ]
    )

    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)

    if auc > best_auc:
        best_auc = auc
        best_params = p.copy()

print("Best Parameters:", best_params)


final_model = lgb.LGBMClassifier(**best_params)
final_model.fit(
    pd.concat([X_train, X_val]),
    pd.concat([y_train, y_val]),
    callbacks=[log_evaluation(period=0)]
)


joblib.dump(final_model, os.path.join(OUTPUT_DIR, "final_lgbm.pkl"))


probs_test = final_model.predict_proba(X_test)[:, 1]
auc_test = roc_auc_score(y_test, probs_test)

precision, recall, thresholds = precision_recall_curve(y_test, probs_test)
f1_scores = 2 * precision * recall / (precision + recall + 1e-9)

best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
preds_test = (probs_test >= best_threshold).astype(int)

print("Test AUC:", auc_test)
print("Best Threshold:", best_threshold)
print(classification_report(y_test, preds_test))

metrics = {
    "auc_test": auc_test,
    "threshold": float(best_threshold),
    "f1": f1_score(y_test, preds_test),
    "precision": precision_score(y_test, preds_test),
    "recall": recall_score(y_test, preds_test),
}
pd.Series(metrics).to_csv(os.path.join(OUTPUT_DIR, "metrics.csv"))


explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_test)

if isinstance(shap_values, list):
    shap_vals = shap_values[1]
else:
    shap_vals = shap_values

mean_abs = np.abs(shap_vals).mean(axis=0)
shap_importance = pd.DataFrame({"feature": feature_cols, "importance": mean_abs})
shap_importance = shap_importance.sort_values("importance", ascending=False)

shap_importance.to_csv(os.path.join(OUTPUT_DIR, "shap_feature_importance.csv"), index=False)

plt.figure()
shap.summary_plot(shap_vals, X_test, show=False)
plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary_plot.png"))
plt.close()

top3 = shap_importance["feature"].head(3).tolist()
for feat in top3:
    plt.figure()
    shap.dependence_plot(feat, shap_vals, X_test, show=False)
    plt.savefig(os.path.join(OUTPUT_DIR, f"shap_dependence_{feat}.png"))
    plt.close()


high_idx = np.where(probs_test >= 0.8)[0][:2].tolist()
low_idx = np.where(probs_test <= 0.2)[0][:3].tolist()

if len(high_idx) < 2:
    high_idx = np.argsort(-probs_test)[:2].tolist()
if len(low_idx) < 3:
    low_idx = np.argsort(probs_test)[:3].tolist()

selected_idx = high_idx + low_idx
local_list = []

for idx in selected_idx:
    instance = X_test.iloc[idx:idx+1]
    shap_vec = shap_vals[idx]

    contrib = pd.DataFrame({
        "feature": feature_cols,
        "shap": shap_vec,
        "value": instance.values.flatten()
    })

    pos = contrib.sort_values("shap", ascending=False).head(5)
    neg = contrib.sort_values("shap").head(5)

    local_list.append({
        "index": int(idx),
        "prob": float(probs_test[idx]),
        "true": int(y_test.iloc[idx]),
        "positive": pos.to_dict("records"),
        "negative": neg.to_dict("records")
    })

with open(os.path.join(OUTPUT_DIR, "local_explanations.json"), "w") as f:
    json.dump(local_list, f, indent=2)


report = []
report.append("Model Performance:")
report.append(f"AUC: {auc_test:.4f}")
report.append(f"Best Threshold: {best_threshold:.4f}")
report.append("")

report.append("Top SHAP Features:")
for i, row in shap_importance.head(10).iterrows():
    report.append(f"{i+1}. {row['feature']} (|SHAP|={row['importance']:.5f})")

with open(os.path.join(OUTPUT_DIR, "technical_report.txt"), "w") as f:
    f.write("\n".join(report))

# Executive Summary
summary = f"""
Executive Summary

This credit-risk model uses LightGBM and SHAP to provide interpretable default predictions.
The model achieved a test AUC of {auc_test:.3f}. SHAP enables transparency by identifying
which features contribute most strongly to default risk. The top drivers were:
{", ".join(top3)}. Dependence plots reveal nonlinear risk effects and interaction patterns.
Local SHAP explanations show why specific loans were classified as high- or low-risk.
"""

with open(os.path.join(OUTPUT_DIR, "executive_summary.txt"), "w") as f:
    f.write(summary.strip())

print("\nAll outputs saved in:", OUTPUT_DIR)
