import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 1. Load augmented data ────────────────────────────────────────────────────
print("\n── Step 1: Loading augmented data ──────────────────────────────────────")
df = pd.read_csv(os.path.join(BASE_DIR, "augmented_cost_data.csv"))
df["date"] = pd.to_datetime(df["date"])
df = df[df["service"].str.strip() != ""].copy()
df = df.dropna(subset=["service"])
print(f"✅ Loaded {len(df)} rows")
print(f"   Real rows     : {(df['data_source'] == 'real').sum()}")
print(f"   Synthetic rows: {(df['data_source'] == 'synthetic').sum()}")


# ── 2. Feature Engineering ────────────────────────────────────────────────────
print("\n── Step 2: Feature Engineering ─────────────────────────────────────────")

df = df.sort_values(["service", "date"]).reset_index(drop=True)

df["rolling_mean_7d"] = df.groupby("service")["daily_cost"].transform(
                         lambda x: x.rolling(7, min_periods=1).mean())
df["rolling_std_7d"]  = df.groupby("service")["daily_cost"].transform(
                         lambda x: x.rolling(7, min_periods=1).std().fillna(0))

df["z_score"] = np.where(
    df["rolling_std_7d"] > 0,
    (df["daily_cost"] - df["rolling_mean_7d"]) / df["rolling_std_7d"],
    0
)

df["cost_ratio"] = df.groupby("service")["daily_cost"].transform(
                    lambda x: x / x.shift(1).replace(0, np.nan)).fillna(1)

df["cost_per_usage"] = np.where(
    df["usage_amount"] > 0,
    df["daily_cost"] / df["usage_amount"],
    0
)

df["cw_cpu_utilization"]    = df["cw_cpu_utilization"].fillna(0)
df["cw_network_out_mb"]     = df["cw_network_out_mb"].fillna(0)
df["cw_lambda_invocations"] = df["cw_lambda_invocations"].fillna(0)
df["cw_lambda_duration_ms"] = df["cw_lambda_duration_ms"].fillna(0)

print("✅ Features engineered:")
print("   • rolling_mean_7d   — 7-day rolling average cost per service")
print("   • rolling_std_7d    — 7-day rolling std deviation")
print("   • z_score           — deviation from normal behavior")
print("   • cost_ratio        — today vs yesterday cost")
print("   • cost_per_usage    — cost efficiency ratio")
print("   • cw_*              — CloudWatch metrics")


# ── 3. Z-Score detection ──────────────────────────────────────────────────────
print("\n── Step 3: Z-Score Anomaly Detection ───────────────────────────────────")
ZSCORE_THRESHOLD = 2.0
df["zscore_flag"] = (df["z_score"].abs() > ZSCORE_THRESHOLD).astype(int)
zscore_anomalies  = df[df["zscore_flag"] == 1]
print(f"✅ Z-Score detected {len(zscore_anomalies)} anomalies (threshold: ±{ZSCORE_THRESHOLD})")
if len(zscore_anomalies) > 0:
    print(zscore_anomalies[["date","service","daily_cost","z_score","anomaly_type"]].to_string(index=False))


# ── 4. Isolation Forest ───────────────────────────────────────────────────────
print("\n── Step 4: Isolation Forest Detection ──────────────────────────────────")

FEATURES = [
    "daily_cost", "usage_amount", "rolling_mean_7d", "rolling_std_7d",
    "z_score", "cost_ratio", "cost_per_usage",
    "cw_cpu_utilization", "cw_network_out_mb",
    "cw_lambda_invocations", "cw_lambda_duration_ms",
]

X        = df[FEATURES].copy()
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

iso_forest = IsolationForest(n_estimators=200, contamination=0.04, random_state=42)
df["if_score"] = iso_forest.fit_predict(X_scaled)
df["if_flag"]  = (df["if_score"] == -1).astype(int)

if_anomalies = df[df["if_flag"] == 1]
print(f"✅ Isolation Forest detected {len(if_anomalies)} anomalies")
print(if_anomalies[["date","service","daily_cost","z_score","anomaly_type"]].to_string(index=False))


# ── 5. Combined flag ──────────────────────────────────────────────────────────
df["detected_anomaly"] = ((df["zscore_flag"] == 1) | (df["if_flag"] == 1)).astype(int)
print(f"\n✅ Combined detection: {df['detected_anomaly'].sum()} total anomalies flagged")


# ── 6. Validation ─────────────────────────────────────────────────────────────
print("\n── Step 5: Validation ──────────────────────────────────────────────────")
df["actual_anomaly"] = (df["anomaly_label"] == "anomaly").astype(int)

print("\nClassification Report (Isolation Forest):")
print(classification_report(
    df["actual_anomaly"], df["if_flag"],
    target_names=["Normal", "Anomaly"], zero_division=0
))

cm = confusion_matrix(df["actual_anomaly"], df["if_flag"])
print("Confusion Matrix:")
print(f"  True  Normal : {cm[0][0]}  |  False Alarm  : {cm[0][1]}")
print(f"  Missed Anom  : {cm[1][0]}  |  True Anomaly : {cm[1][1]}")


# ── 7. Visualizations ─────────────────────────────────────────────────────────
print("\n── Step 6: Generating visualizations ───────────────────────────────────")

fig, axes = plt.subplots(3, 1, figsize=(14, 14))
fig.suptitle("AWS Cost Intelligence — Anomaly Detection Results", fontsize=15, fontweight="bold")

# Plot 1: Daily total cost with anomaly markers
daily_total  = df.groupby("date")["daily_cost"].sum().reset_index()
anomaly_days = df[df["detected_anomaly"] == 1]["date"].unique()

axes[0].plot(daily_total["date"], daily_total["daily_cost"],
             color="steelblue", linewidth=1.5, label="Daily Total Cost")
for aday in anomaly_days:
    axes[0].axvline(x=aday, color="red", alpha=0.3, linewidth=1)
axes[0].scatter(
    df[df["detected_anomaly"] == 1]["date"],
    df[df["detected_anomaly"] == 1]["daily_cost"],
    color="red", zorder=5, s=60, label="Detected Anomaly"
)
# Mark where real data ends and synthetic begins
axes[0].axvline(x=pd.Timestamp("2026-04-14"), color="green",
                linestyle="--", linewidth=1.5, label="Real → Synthetic")
axes[0].set_title("Daily Cost Over Time (Green line = start of synthetic data)")
axes[0].set_ylabel("Cost (USD)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Z-scores per service
top_services = ["AmazonEC2", "Amazon Simple Storage Service", "AWSLambda"]
colors       = ["steelblue", "darkorange", "green"]
for svc, color in zip(top_services, colors):
    sdf = df[df["service"] == svc]
    if len(sdf) > 0:
        axes[1].plot(sdf["date"], sdf["z_score"], label=svc, color=color, linewidth=1.2)
axes[1].axhline(y= ZSCORE_THRESHOLD, color="red", linestyle="--", alpha=0.7, label=f"Threshold ±{ZSCORE_THRESHOLD}")
axes[1].axhline(y=-ZSCORE_THRESHOLD, color="red", linestyle="--", alpha=0.7)
axes[1].axvline(x=pd.Timestamp("2026-04-14"), color="green", linestyle="--", linewidth=1.5)
axes[1].set_title("Z-Score per Service")
axes[1].set_ylabel("Z-Score")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

# Plot 3: Confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Predicted Normal", "Predicted Anomaly"],
            yticklabels=["Actual Normal", "Actual Anomaly"],
            ax=axes[2])
axes[2].set_title("Confusion Matrix — Isolation Forest")

plt.tight_layout()
plot_path = os.path.join(BASE_DIR, "anomaly_detection_results.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"✅ Plot saved → {plot_path}")
plt.show()

# ── 8. Save ───────────────────────────────────────────────────────────────────
out_path = os.path.join(BASE_DIR, "detected_anomalies.csv")
df.to_csv(out_path, index=False)
print(f"✅ Results saved → {out_path}")

print("\n── Summary ─────────────────────────────────────────────────────────────")
print(f"Total rows analysed  : {len(df)}")
print(f"Actual anomalies     : {df['actual_anomaly'].sum()}")
print(f"Z-Score detections   : {df['zscore_flag'].sum()}")
print(f"Isolation Forest     : {df['if_flag'].sum()}")
print(f"Combined detections  : {df['detected_anomaly'].sum()}")