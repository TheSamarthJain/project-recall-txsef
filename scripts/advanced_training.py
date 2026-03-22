import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
import lightgbm as lgb
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load features
print("=" * 80)
print("ADVANCED ALZHEIMER'S DETECTION MODEL TRAINING")
print("=" * 80)
print("\nLoading features...")
df = pd.read_csv('results/all_features_advanced.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nLabel distribution:\n{df['label'].value_counts()}")
print(f"\nCorpus distribution:\n{df['corpus'].value_counts()}")
print(f"Unique participants: {df['participant_id'].nunique()}")

# Prepare data
X = df.drop(['filename', 'label', 'participant_id', 'corpus'], axis=1).values
y = (df['label'] == 'AD').astype(int).values
groups = df['participant_id'].values

# Handle any NaN or inf values
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

print(f"\nFeatures: {X.shape[1]}")
print(f"Samples: {X.shape[0]}")
print(f"AD samples: {np.sum(y)}")
print(f"HC samples: {len(y) - np.sum(y)}")

# Define models
models = {
    'Logistic Regression': LogisticRegression(
        max_iter=2000,
        class_weight='balanced',
        C=0.1,
        solver='saga',
        random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        random_state=42
    ),
    'SVM': SVC(
        kernel='rbf',
        C=1.5,
        gamma='scale',
        class_weight='balanced',
        probability=True,
        random_state=42
    ),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=len(y[y == 0]) / len(y[y == 1]),
        random_state=42,
        eval_metric='logloss'
    ),
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight='balanced',
        random_state=42,
        verbose=-1
    )
}

# Cross-validation
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

best_model = None
best_score = 0
best_model_name = ""
best_scaler = None
best_auc = 0

results = {}

print("\n" + "=" * 80)
print("TRAINING MODELS WITH 5-FOLD STRATIFIED GROUP CROSS-VALIDATION")
print("=" * 80)

for model_name, model in models.items():
    print(f"\n{model_name}:")
    print("-" * 70)

    scores = []
    aucs = []
    sensitivities = []
    specificities = []

    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train
        model.fit(X_train_scaled, y_train)

        # Predict
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        scores.append(acc)
        aucs.append(auc)
        sensitivities.append(sensitivity)
        specificities.append(specificity)

        print(f"  Fold {fold}: Acc={acc:.4f}, AUC={auc:.4f}, Sens={sensitivity:.4f}, Spec={specificity:.4f}")

    mean_score = np.mean(scores)
    std_score = np.std(scores)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    mean_sens = np.mean(sensitivities)
    mean_spec = np.mean(specificities)

    results[model_name] = {
        'accuracy': mean_score,
        'accuracy_std': std_score,
        'auc': mean_auc,
        'auc_std': std_auc,
        'sensitivity': mean_sens,
        'specificity': mean_spec
    }

    print(f"\n  MEAN Accuracy: {mean_score:.4f} ± {std_score:.4f}")
    print(f"  MEAN AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"  MEAN Sensitivity: {mean_sens:.4f}")
    print(f"  MEAN Specificity: {mean_spec:.4f}")

    if mean_score > best_score:
        best_score = mean_score
        best_model = model
        best_model_name = model_name
        best_scaler = scaler
        best_auc = mean_auc

print("\n" + "=" * 80)
print("INDIVIDUAL MODEL RESULTS")
print("=" * 80)
for name, res in results.items():
    print(f"{name:20s} | Acc: {res['accuracy']:.4f} ± {res['accuracy_std']:.4f} | AUC: {res['auc']:.4f}")

print("\n" + "=" * 80)
print(f"BEST INDIVIDUAL MODEL: {best_model_name}")
print(f"Accuracy: {best_score:.4f}")
print(f"AUC: {best_auc:.4f}")
print("=" * 80)

# ============ ENSEMBLE MODEL (STACKING) ============
print("\n" + "=" * 80)
print("TRAINING ENSEMBLE MODEL (VOTING CLASSIFIER)")
print("=" * 80)

# Select top 3 models for ensemble
top_models = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)[:3]
print(f"\nTop 3 models for ensemble:")
for i, (name, res) in enumerate(top_models, 1):
    print(f"  {i}. {name}: {res['accuracy']:.4f}")

# Create voting classifier with top models
ensemble_estimators = [
    (name, models[name]) for name, _ in top_models
]

ensemble = VotingClassifier(
    estimators=ensemble_estimators,
    voting='soft',
    n_jobs=-1
)

# Evaluate ensemble
print("\nEvaluating ensemble model...")
ensemble_scores = []
ensemble_aucs = []
ensemble_sens = []
ensemble_spec = []

for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ensemble.fit(X_train_scaled, y_train)

    y_pred = ensemble.predict(X_test_scaled)
    y_prob = ensemble.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    ensemble_scores.append(acc)
    ensemble_aucs.append(auc)
    ensemble_sens.append(sensitivity)
    ensemble_spec.append(specificity)

    print(f"  Fold {fold}: Acc={acc:.4f}, AUC={auc:.4f}")

ensemble_mean_acc = np.mean(ensemble_scores)
ensemble_mean_auc = np.mean(ensemble_aucs)
ensemble_mean_sens = np.mean(ensemble_sens)
ensemble_mean_spec = np.mean(ensemble_spec)

print(f"\nENSEMBLE RESULTS:")
print(f"  Mean Accuracy: {ensemble_mean_acc:.4f} ± {np.std(ensemble_scores):.4f}")
print(f"  Mean AUC: {ensemble_mean_auc:.4f} ± {np.std(ensemble_aucs):.4f}")
print(f"  Mean Sensitivity: {ensemble_mean_sens:.4f}")
print(f"  Mean Specificity: {ensemble_mean_spec:.4f}")

# Choose final model
if ensemble_mean_acc > best_score:
    print(f"\n🎯 ENSEMBLE WINS! ({ensemble_mean_acc:.4f} vs {best_score:.4f})")
    final_model = ensemble
    final_model_name = "Ensemble (Voting)"
    final_accuracy = ensemble_mean_acc
    final_auc = ensemble_mean_auc
else:
    print(f"\n🎯 {best_model_name} WINS! ({best_score:.4f} vs {ensemble_mean_acc:.4f})")
    final_model = best_model
    final_model_name = best_model_name
    final_accuracy = best_score
    final_auc = best_auc

# Train final model on full dataset
print(f"\n{'=' * 80}")
print(f"TRAINING FINAL MODEL ON COMPLETE DATASET")
print(f"{'=' * 80}")
print(f"Model: {final_model_name}")

scaler_final = StandardScaler()
X_scaled = scaler_final.fit_transform(X)
final_model.fit(X_scaled, y)

# Save models
os.makedirs('models', exist_ok=True)
print("\nSaving models...")

with open('models/final_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)

with open('models/final_scaler.pkl', 'wb') as f:
    pickle.dump(scaler_final, f)

model_info = {
    'model_name': final_model_name,
    'accuracy': final_accuracy,
    'auc': final_auc,
    'n_features': X.shape[1],
    'n_samples': X.shape[0],
    'n_participants': len(np.unique(groups))
}

with open('models/model_info.pkl', 'wb') as f:
    pickle.dump(model_info, f)

print("✅ Models saved!")

# Generate predictions
y_pred_final = final_model.predict(X_scaled)
y_prob_final = final_model.predict_proba(X_scaled)[:, 1]

# Visualizations
print("\nGenerating visualizations...")

# Confusion Matrix
cm = confusion_matrix(y, y_pred_final)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Healthy', 'Alzheimer'],
            yticklabels=['Healthy', 'Alzheimer'])
plt.title(f'Confusion Matrix - {final_model_name}\nAccuracy: {final_accuracy:.2%}', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('results/confusion_matrix_final.png', dpi=300)
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y, y_prob_final)
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, label=f'{final_model_name} (AUC = {final_auc:.3f})', linewidth=3)
plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve - {final_model_name}', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('results/roc_curve_final.png', dpi=300)
plt.close()

# Model Comparison
plt.figure(figsize=(12, 6))
model_names = list(results.keys()) + ['Ensemble']
accuracies = [results[name]['accuracy'] for name in results.keys()] + [ensemble_mean_acc]
colors = ['#A8DADC' if acc < max(accuracies) else '#E63946' for acc in accuracies]

bars = plt.bar(model_names, accuracies, color=colors, edgecolor='black')
plt.axhline(y=0.85, color='green', linestyle='--', linewidth=2, label='85% Target')
plt.xlabel('Model', fontweight='bold')
plt.ylabel('Accuracy', fontweight='bold')
plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.ylim(0.5, 1.0)
plt.legend()
plt.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.3f}',
             ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('results/model_comparison.png', dpi=300)
plt.close()

print("✅ Visualizations saved!")

# Final Report
print(f"\n{'=' * 80}")
print("FINAL CLASSIFICATION REPORT")
print(f"{'=' * 80}")
print(classification_report(y, y_pred_final, target_names=['Healthy', 'Alzheimer']))

print(f"\n{'=' * 80}")
print("🎉 TRAINING COMPLETE!")
print(f"{'=' * 80}")
print(f"Final Model: {final_model_name}")
print(f"Accuracy: {final_accuracy:.4f} ({final_accuracy * 100:.2f}%)")
print(f"AUC: {final_auc:.4f}")
print(f"Total Samples: {len(y)}")
print(f"Features: {X.shape[1]}")
print(f"{'=' * 80}")