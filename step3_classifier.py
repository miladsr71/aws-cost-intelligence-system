import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("\n── Step 1: Loading data ─────────────────────────────────────────────────")
df = pd.read_csv(os.path.join(BASE_DIR, "detected_anomalies.csv"))
df["date"] = pd.to_datetime(df["date"])
print(f"✅ Loaded {len(df)} rows")
print(f"   Anomaly type distribution:")
print(df["anomaly_type"].value_counts().to_string())

# ── 2. Prepare features ───────────────────────────────────────────────────────
print("\n── Step 2: Preparing classification dataset ─────────────────────────────")

FEATURES = [
    "daily_cost", "usage_amount", "rolling_mean_7d", "rolling_std_7d",
    "z_score", "cost_ratio", "cost_per_usage",
    "cw_cpu_utilization", "cw_network_out_mb",
    "cw_lambda_invocations", "cw_lambda_duration_ms",
]

df[FEATURES] = df[FEATURES].fillna(0)

le = LabelEncoder()
df["anomaly_type_encoded"] = le.fit_transform(df["anomaly_type"])

print(f"✅ Classes: {list(le.classes_)}")

# ── 3. Handle class imbalance ─────────────────────────────────────────────────
print("\n── Step 3: Handling class imbalance ─────────────────────────────────────")

normal_df  = df[df["anomaly_type"] == "none"]
anomaly_df = df[df["anomaly_type"] != "none"]

upsampled = []
for atype in anomaly_df["anomaly_type"].unique():
    subset = anomaly_df[anomaly_df["anomaly_type"] == atype]
    subset_up = subset.sample(60, replace=True, random_state=42)
    upsampled.append(subset_up)

balanced_df = pd.concat([normal_df] + upsampled, ignore_index=True)
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
balanced_df["anomaly_type_encoded"] = le.transform(balanced_df["anomaly_type"])

print(f"✅ Balanced dataset: {len(balanced_df)} rows")
print(balanced_df["anomaly_type"].value_counts().to_string())

# ── 4. Train / Test split ─────────────────────────────────────────────────────
print("\n── Step 4: Train/Test split ─────────────────────────────────────────────")

X = balanced_df[FEATURES]
y = balanced_df["anomaly_type_encoded"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"✅ Training set : {len(X_train)} rows")
print(f"✅ Test set     : {len(X_test)} rows")

# ── 5. Train Random Forest ────────────────────────────────────────────────────
print("\n── Step 5: Training Random Forest Classifier ───────────────────────────")

rf = RandomForestClassifier(
    n_estimators=200, max_depth=10,
    min_samples_split=4, class_weight="balanced",
    random_state=42
)
rf.fit(X_train, y_train)

cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring="accuracy")
print(f"✅ Random Forest trained")
print(f"✅ Cross-validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ── 6. Evaluate ───────────────────────────────────────────────────────────────
print("\n── Step 6: Evaluation ───────────────────────────────────────────────────")

y_pred = rf.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(pd.DataFrame(cm, index=le.classes_, columns=le.classes_).to_string())

# ── 7. Feature Importance ─────────────────────────────────────────────────────
print("\n── Step 7: Feature Importance ───────────────────────────────────────────")
importances = pd.Series(rf.feature_importances_, index=FEATURES)
importances = importances.sort_values(ascending=False)
print("✅ Top features:")
print(importances.round(4).to_string())

# ── 8. Classify full dataset ──────────────────────────────────────────────────
print("\n── Step 8: Classifying full dataset ─────────────────────────────────────")

X_full = scaler.transform(df[FEATURES].fillna(0))
df["predicted_type"]        = le.inverse_transform(rf.predict(X_full))
df["prediction_confidence"] = rf.predict_proba(X_full).max(axis=1).round(4)

detected = df[df["detected_anomaly"] == 1][
    ["date","service","daily_cost","z_score",
     "anomaly_type","predicted_type","prediction_confidence"]
].sort_values("date")

print(f"\n✅ Anomalies classified ({len(detected)} rows):")
print(detected.to_string(index=False))

# ── 9. Visualizations ─────────────────────────────────────────────────────────
print("\n── Step 9: Generating visualizations ───────────────────────────────────")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("AWS Cost Intelligence — Random Forest Classification Results",
             fontsize=14, fontweight="bold")

# Plot 1: Feature importance
importances.plot(kind="barh", ax=axes[0][0], color="steelblue")
axes[0][0].set_title("Feature Importance")
axes[0][0].set_xlabel("Importance Score")
axes[0][0].invert_yaxis()
axes[0][0].grid(True, alpha=0.3)

# Plot 2: Confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[0][1])
axes[0][1].set_title("Confusion Matrix — Random Forest")
axes[0][1].set_xlabel("Predicted")
axes[0][1].set_ylabel("Actual")
plt.setp(axes[0][1].get_xticklabels(), rotation=30, ha="right", fontsize=8)
plt.setp(axes[0][1].get_yticklabels(), rotation=0, fontsize=8)

# Plot 3: Daily cost coloured by anomaly type
color_map = {
    "none"                  : "steelblue",
    "ec2_spike"             : "red",
    "s3_gradual_drift"      : "orange",
    "lambda_burst"          : "green",
    "unexplained_cost_spike": "purple",
}
daily = df.groupby(["date","predicted_type"])["daily_cost"].sum().reset_index()
for atype, color in color_map.items():
    subset = daily[daily["predicted_type"] == atype]
    if len(subset) > 0:
        axes[1][0].scatter(subset["date"], subset["daily_cost"],
                           color=color, label=atype, s=30, alpha=0.8)
axes[1][0].axvline(x=pd.Timestamp("2026-04-14"), color="green",
                   linestyle="--", linewidth=1.5, label="Real → Synthetic")
axes[1][0].set_title("Daily Cost by Predicted Anomaly Type")
axes[1][0].set_ylabel("Cost (USD)")
axes[1][0].legend(fontsize=7)
axes[1][0].grid(True, alpha=0.3)

# Plot 4: Confidence distribution
conf_data = df[df["detected_anomaly"] == 1]["prediction_confidence"]
if len(conf_data) > 0:
    axes[1][1].hist(conf_data, bins=15, color="steelblue", edgecolor="white")
    axes[1][1].set_title("Prediction Confidence Distribution")
    axes[1][1].set_xlabel("Confidence Score")
    axes[1][1].set_ylabel("Count")
    axes[1][1].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(BASE_DIR, "classification_results.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"✅ Plot saved → {plot_path}")
plt.show()

# ── 10. Save ──────────────────────────────────────────────────────────────────
out_path = os.path.join(BASE_DIR, "classified_anomalies.csv")
df.to_csv(out_path, index=False)
print(f"✅ Results saved → {out_path}")

print("\n── Final Summary ───────────────────────────────────────────────────────")
print(f"Total rows classified  : {len(df)}")
print(f"Cross-val accuracy     : {cv_scores.mean():.1%}")
print(f"Test set accuracy      : {(y_pred == y_test).mean():.1%}")
print(f"\nPredicted anomaly types in full dataset:")
print(df[df["detected_anomaly"]==1]["predicted_type"].value_counts().to_string())