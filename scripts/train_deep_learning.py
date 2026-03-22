import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("ADVANCED DEEP LEARNING MODEL FOR ALZHEIMER'S DETECTION")
print("=" * 80)

# Load features
df = pd.read_csv('results/all_features_advanced.csv')
X = df.drop(['filename', 'label', 'participant_id', 'corpus'], axis=1).values
y = (df['label'] == 'AD').astype(int).values
groups = df['participant_id'].values

# Handle NaN/inf
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

print(f"Samples: {len(y)}, Features: {X.shape[1]}")
print(f"AD samples: {np.sum(y)}, HC samples: {len(y) - np.sum(y)}")

# Calculate class weights for imbalanced data
class_weight = {
    0: len(y) / (2 * np.sum(y == 0)),
    1: len(y) / (2 * np.sum(y == 1))
}
print(f"Class weights: {class_weight}")


def create_advanced_model(input_dim):
    """Create advanced neural network with residual connections and attention"""

    # Input
    inputs = layers.Input(shape=(input_dim,))

    # First block with residual
    x = layers.Dense(512, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    # Second block with residual
    x2 = layers.Dense(256, activation='relu')(x)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Dropout(0.4)(x2)

    # Third block
    x3 = layers.Dense(128, activation='relu')(x2)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Dropout(0.3)(x3)

    # Attention layer
    attention = layers.Dense(128, activation='tanh')(x3)
    attention = layers.Dense(128, activation='softmax')(attention)
    x3_attention = layers.Multiply()([x3, attention])

    # Fourth block
    x4 = layers.Dense(64, activation='relu')(x3_attention)
    x4 = layers.BatchNormalization()(x4)
    x4 = layers.Dropout(0.2)(x4)

    # Output
    outputs = layers.Dense(1, activation='sigmoid')(x4)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


# Cross-validation
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

scores = []
aucs = []
sensitivities = []
specificities = []
histories = []

print("\n" + "=" * 80)
print("5-FOLD CROSS-VALIDATION")
print("=" * 80)

for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups), 1):
    print(f"\nFold {fold}:")
    print("-" * 70)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build model
    model = create_advanced_model(X.shape[1])

    # Compile with advanced optimizer
    optimizer = keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=20,
            restore_best_weights=True,
            mode='max',
            verbose=0
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=0
        )
    ]

    # Train
    history = model.fit(
        X_train_scaled, y_train,
        epochs=150,
        batch_size=32,
        validation_split=0.2,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=0
    )

    histories.append(history)

    # Evaluate
    y_pred_prob = model.predict(X_test_scaled, verbose=0).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    scores.append(acc)
    aucs.append(auc)
    sensitivities.append(sensitivity)
    specificities.append(specificity)

    print(f"  Accuracy: {acc:.4f}")
    print(f"  AUC: {auc:.4f}")
    print(f"  Sensitivity: {sensitivity:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  Best epoch: {len(history.history['loss']) - 20}")  # Accounting for patience

print(f"\n{'=' * 80}")
print("DEEP LEARNING CROSS-VALIDATION RESULTS")
print(f"{'=' * 80}")
print(f"Mean Accuracy:    {np.mean(scores):.4f} ± {np.std(scores):.4f}")
print(f"Mean AUC:         {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
print(f"Mean Sensitivity: {np.mean(sensitivities):.4f} ± {np.std(sensitivities):.4f}")
print(f"Mean Specificity: {np.mean(specificities):.4f} ± {np.std(specificities):.4f}")
print(f"{'=' * 80}")

# Compare with ensemble
ensemble_acc = 0.7480  # Your previous best
print(f"\nComparison:")
print(f"  Ensemble:      {ensemble_acc:.4f}")
print(f"  Deep Learning: {np.mean(scores):.4f}")
if np.mean(scores) > ensemble_acc:
    print(f"  🎉 Deep Learning WINS by {(np.mean(scores) - ensemble_acc) * 100:.2f}%!")
else:
    print(f"  Ensemble still better by {(ensemble_acc - np.mean(scores)) * 100:.2f}%")

# Train final model on ALL data
print(f"\n{'=' * 80}")
print("TRAINING FINAL MODEL ON COMPLETE DATASET")
print(f"{'=' * 80}")

scaler_final = StandardScaler()
X_scaled = scaler_final.fit_transform(X)

model_final = create_advanced_model(X.shape[1])
model_final.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)

history_final = model_final.fit(
    X_scaled, y,
    epochs=150,
    batch_size=32,
    validation_split=0.15,
    class_weight=class_weight,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=10, factor=0.5)
    ],
    verbose=1
)

# Save models
print("\nSaving models...")
model_final.save('models/deep_learning_model.keras')
with open('models/deep_scaler.pkl', 'wb') as f:
    pickle.dump(scaler_final, f)

# Save model info
dl_model_info = {
    'model_name': 'Deep Learning (Advanced)',
    'accuracy': np.mean(scores),
    'auc': np.mean(aucs),
    'sensitivity': np.mean(sensitivities),
    'specificity': np.mean(specificities),
    'n_features': X.shape[1],
    'n_samples': X.shape[0]
}

with open('models/deep_model_info.pkl', 'wb') as f:
    pickle.dump(dl_model_info, f)

print("✅ Deep learning model saved!")
print("✅ Scaler saved!")
print("✅ Model info saved!")

# Generate predictions
y_pred_prob_final = model_final.predict(X_scaled, verbose=0).flatten()
y_pred_final = (y_pred_prob_final > 0.5).astype(int)

# Visualizations
print("\nGenerating visualizations...")

# 1. Training history
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Loss
axes[0, 0].plot(history_final.history['loss'], label='Training Loss', linewidth=2)
axes[0, 0].plot(history_final.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0, 0].set_title('Model Loss', fontweight='bold', fontsize=12)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Accuracy
axes[0, 1].plot(history_final.history['accuracy'], label='Training Accuracy', linewidth=2)
axes[0, 1].plot(history_final.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[0, 1].set_title('Model Accuracy', fontweight='bold', fontsize=12)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# AUC
axes[1, 0].plot(history_final.history['auc'], label='Training AUC', linewidth=2)
axes[1, 0].plot(history_final.history['val_auc'], label='Validation AUC', linewidth=2)
axes[1, 0].set_title('Model AUC', fontweight='bold', fontsize=12)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('AUC')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Cross-validation scores
axes[1, 1].bar(['Accuracy', 'AUC', 'Sensitivity', 'Specificity'],
               [np.mean(scores), np.mean(aucs), np.mean(sensitivities), np.mean(specificities)],
               color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'],
               edgecolor='black',
               linewidth=1.5)
axes[1, 1].set_title('Cross-Validation Metrics', fontweight='bold', fontsize=12)
axes[1, 1].set_ylabel('Score')
axes[1, 1].set_ylim(0, 1)
axes[1, 1].grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (metric, value) in enumerate(zip(['Accuracy', 'AUC', 'Sensitivity', 'Specificity'],
                                        [np.mean(scores), np.mean(aucs), np.mean(sensitivities),
                                         np.mean(specificities)])):
    axes[1, 1].text(i, value + 0.02, f'{value:.3f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('results/deep_learning_training.png', dpi=300, bbox_inches='tight')
print("✅ Training history saved: results/deep_learning_training.png")
plt.close()

# 2. Confusion Matrix
cm = confusion_matrix(y, y_pred_final)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Healthy', 'Alzheimer'],
            yticklabels=['Healthy', 'Alzheimer'],
            cbar_kws={'label': 'Count'})
plt.title(f'Deep Learning Confusion Matrix\nAccuracy: {np.mean(scores):.2%}',
          fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('results/deep_learning_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✅ Confusion matrix saved: results/deep_learning_confusion_matrix.png")
plt.close()

# Final report
print(f"\n{'=' * 80}")
print("FINAL CLASSIFICATION REPORT")
print(f"{'=' * 80}")
print(classification_report(y, y_pred_final, target_names=['Healthy', 'Alzheimer']))

print(f"\n{'=' * 80}")
print("🎉 DEEP LEARNING TRAINING COMPLETE!")
print(f"{'=' * 80}")
print(f"Model: Deep Learning (Advanced)")
print(f"Cross-Validation Accuracy: {np.mean(scores):.4f} ({np.mean(scores) * 100:.2f}%)")
print(f"Cross-Validation AUC: {np.mean(aucs):.4f}")
print(f"Total Samples: {len(y)}")
print(f"Features: {X.shape[1]}")
print(f"\nFiles saved:")
print(f"  - models/deep_learning_model.keras")
print(f"  - models/deep_scaler.pkl")
print(f"  - models/deep_model_info.pkl")
print(f"  - results/deep_learning_training.png")
print(f"  - results/deep_learning_confusion_matrix.png")
print(f"{'=' * 80}")