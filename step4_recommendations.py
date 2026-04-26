import pandas as pd
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# load classified anomalies from step 3
print("\nloading classified anomalies...")
df = pd.read_csv(os.path.join(BASE_DIR, "classified_anomalies.csv"))
df["date"] = pd.to_datetime(df["date"])

# only work with rows that were actually flagged as anomalies
detected = df[df["detected_anomaly"] == 1].copy()
print(f"found {len(detected)} flagged rows to process")


def generate_recommendation(row):
    atype      = row["predicted_type"]
    service    = row["service"]
    cost       = round(row["daily_cost"], 2)
    confidence = round(row["prediction_confidence"] * 100, 1)
    date       = row["date"].strftime("%Y-%m-%d")

    # calculate extra spend vs normal
    normal_cost  = round(row["rolling_mean_7d"], 2)
    extra_cost   = round(cost - normal_cost, 2)
    pct_increase = round(((cost - normal_cost) / normal_cost * 100), 1) if normal_cost > 0 else 0

    base = {
        "date"          : date,
        "service"       : service,
        "anomaly_type"  : atype,
        "confidence_pct": confidence,
        "actual_cost"   : cost,
        "normal_cost"   : normal_cost,
        "extra_cost"    : extra_cost,
        "pct_increase"  : pct_increase,
    }

    # ec2 spike - cost and cpu both went up
    if atype == "ec2_spike":
        base["what_happened"]   = f"EC2 cost jumped to ${cost} on {date} - that is {pct_increase}% above your normal ${normal_cost}/day"
        base["why_it_happened"] = f"EC2 usage went up {pct_increase}% and CPU spiked at the same time - something started using a lot of compute, probably an unexpected instance or a runaway process"
        base["actions"] = [
            "check EC2 console for any instances you did not intentionally launch",
            "review Auto Scaling group activity logs around this date",
            "look at CloudWatch CPU metrics to identify which instance was responsible",
            "consider Reserved Instances if this workload runs regularly - can save up to 72%",
            "set up a daily cost budget alert at 2x your normal EC2 spend",
        ]
        base["severity"] = "HIGH"

    # s3 gradual drift - cost slowly creeping up over multiple days
    elif atype == "s3_gradual_drift":
        base["what_happened"]   = f"S3 cost has been slowly increasing and reached ${cost} on {date} - started at roughly ${normal_cost}/day"
        base["why_it_happened"] = f"S3 costs have been slowly growing and reached {pct_increase}% above normal - your app is probably uploading more data than it should or old files are not being deleted"
        base["actions"] = [
            "check S3 access logs to find which bucket is growing",
            "look for any recent app changes that might have changed upload behavior",
            "review S3 lifecycle policies - make sure old objects are being deleted",
            "enable S3 Storage Lens for a detailed breakdown of storage usage",
            "consider S3 Intelligent-Tiering for objects that are not accessed often",
        ]
        base["severity"] = "MEDIUM"

    # lambda burst - invocations and duration both exploded
    elif atype == "lambda_burst":
        base["what_happened"]   = f"Lambda cost spiked to ${cost} on {date} - normally around ${normal_cost}/day"
        base["why_it_happened"] = f"Lambda ran {pct_increase}% more than normal - your function was probably stuck in a retry loop and kept calling itself over and over"
        base["actions"] = [
            "check Lambda CloudWatch logs for the function that was running on this date",
            "look for error patterns that might have triggered automatic retries",
            "review Lambda event source mappings - make sure SQS or Kinesis triggers are not looping",
            "add a dead letter queue to catch failed invocations before they retry forever",
            "set a Lambda concurrency limit as a safety cap to prevent runaway costs",
            "review function timeout settings - a short timeout can cause many retries",
        ]
        base["severity"] = "HIGH"

    # unexplained spike - cost up but usage stayed normal
    elif atype == "unexplained_cost_spike":
        base["what_happened"]   = f"EC2 cost jumped to ${cost} on {date} but CloudWatch shows CPU and usage were completely normal"
        base["why_it_happened"] = f"Cost jumped {pct_increase}% but nothing changed in actual usage - this is a pricing issue not a usage issue, most likely a Reserved Instance or Savings Plan expired"
        base["actions"] = [
            "check AWS Cost Explorer for any pricing changes around this date",
            "review your Reserved Instances and Savings Plans - check if any expired",
            "compare cost per usage hour before and after the spike",
            "check if any instance types were changed to a higher price tier",
            "contact AWS support if no pricing change is found - could be a billing error",
            "note: this type of anomaly cannot be detected by usage monitoring alone",
        ]
        base["severity"] = "HIGH"

    # false positive - classifier said it is normal
    else:
        base["what_happened"]   = f"flagged by detection but classifier says this is normal behavior"
        base["why_it_happened"] = "probably a false positive - common in days right after a real spike when rolling averages are still elevated"
        base["actions"]         = ["no action needed - keep an eye on it over the next few days"]
        base["severity"]        = "LOW"

    return pd.Series(base)


# apply recommendations to all detected rows
print("generating recommendations...")
rec_df = detected.apply(generate_recommendation, axis=1)
print(f"recommendations generated for {len(rec_df)} rows")


# print readable summary for each real anomaly
print("\n" + "="*70)
print("COST ANOMALY RECOMMENDATIONS REPORT")
print(f"generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("="*70)

real_anomalies = rec_df[rec_df["anomaly_type"] != "none"].drop_duplicates(
    subset=["anomaly_type", "service"]
)

for _, row in real_anomalies.iterrows():
    print(f"\n{'='*70}")
    print(f"[{row['severity']}] {row['anomaly_type'].upper().replace('_', ' ')}")
    print(f"date      : {row['date']}")
    print(f"service   : {row['service']}")
    print(f"cost      : ${row['actual_cost']} (normal: ${row['normal_cost']}, extra: +${row['extra_cost']})")
    print(f"increase  : +{row['pct_increase']}%")
    print(f"confidence: {row['confidence_pct']}%")
    print(f"\nwhat happened:")
    print(f"  {row['what_happened']}")
    print(f"\nwhy it happened:")
    print(f"  {row['why_it_happened']}")
    print(f"\nrecommended actions:")
    for i, action in enumerate(row["actions"], 1):
        print(f"  {i}. {action}")

print(f"\n{'='*70}")

# save to csv
out_path = os.path.join(BASE_DIR, "recommendations.csv")
rec_df.to_csv(out_path, index=False)
print(f"\nrecommendations saved to {out_path}")

print("\n--- summary ---")
print(f"total flagged rows processed : {len(rec_df)}")
severity_counts = rec_df["severity"].value_counts()
for sev in ["HIGH", "MEDIUM", "LOW"]:
    count = severity_counts.get(sev, 0)
    print(f"{sev} severity : {count}")
print(f"\nanomaly types found:")
print(rec_df[rec_df["anomaly_type"] != "none"]["anomaly_type"].value_counts().to_string())