import boto3
import pandas as pd
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# sns config - topic was created manually in aws console
SNS_TOPIC_ARN = "arn:aws:sns:us-east-1:YOUR_ACCOUNT_ID:cost-anomaly-alerts"
REGION        = "us-east-1"

sns = boto3.client("sns", region_name=REGION)
print("\nconnected to SNS")


def send_alert(anomaly_type, service, date, actual_cost,
               normal_cost, extra_cost, pct_increase,
               confidence, why_it_happened, actions):

    # build a clean readable email subject
    severity_label = {
        "ec2_spike"             : "HIGH",
        "lambda_burst"          : "HIGH",
        "unexplained_cost_spike": "HIGH",
        "s3_gradual_drift"      : "MEDIUM",
    }.get(anomaly_type, "LOW")

    subject = f"[{severity_label}] AWS Cost Anomaly Detected - {anomaly_type.replace('_',' ').title()} on {date}"

    # format the actions list as numbered text
    actions_text = "\n".join([f"  {i+1}. {a}" for i, a in enumerate(actions)])

    # build the full email body
    # keeping it clean and readable - no html just plain text
    body = f"""
AWS COST ANOMALY ALERT
{'='*60}

Severity    : {severity_label}
Date        : {date}
Service     : {service}
Anomaly Type: {anomaly_type.replace('_', ' ').upper()}
Confidence  : {confidence}%

COST IMPACT
{'─'*40}
Normal Daily Cost : ${normal_cost}
Actual Cost Today : ${actual_cost}
Extra Spend       : +${extra_cost}
Increase          : +{pct_increase}%

WHAT HAPPENED
{'─'*40}
{why_it_happened}

RECOMMENDED ACTIONS
{'─'*40}
{actions_text}

{'='*60}
This alert was generated automatically by the AWS Cost
Intelligence System - CS596 MS Project
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}
{'='*60}
    """.strip()

    # send the email via sns
    try:
        response = sns.publish(
            TopicArn = SNS_TOPIC_ARN,
            Subject  = subject,
            Message  = body,
        )
        msg_id = response["MessageId"]
        print(f"   alert sent - message id: {msg_id}")
        return True
    except Exception as e:
        print(f"   failed to send alert: {e}")
        return False


# load recommendations from step 4
print("loading recommendations...")
rec_df = pd.read_csv(os.path.join(BASE_DIR, "recommendations.csv"))

# only send alerts for real anomalies not false positives
# also skip rows labeled as none - those are the false alarm recovery days
real_anomalies = rec_df[
    (rec_df["anomaly_type"] != "none") &
    (rec_df["severity"].isin(["HIGH", "MEDIUM"]))
].drop_duplicates(subset=["anomaly_type", "service"])

print(f"found {len(real_anomalies)} real anomalies to alert on")


# send one alert per unique anomaly event
print("\nsending alerts...")
sent_count   = 0
failed_count = 0

for _, row in real_anomalies.iterrows():
    print(f"\n sending alert for {row['anomaly_type']} on {row['date']}...")

    # parse the actions back from the csv
    # they were stored as a python list string so need to clean them up
    try:
        actions_raw = row["actions"]
        # handle both list format and plain string
        if isinstance(actions_raw, str):
            actions_raw = actions_raw.strip("[]").replace("'", "")
            actions = [a.strip() for a in actions_raw.split(",") if a.strip()]
        else:
            actions = list(actions_raw)
    except Exception:
        actions = ["review aws cost explorer for details"]

    success = send_alert(
        anomaly_type    = row["anomaly_type"],
        service         = row["service"],
        date            = row["date"],
        actual_cost     = row["actual_cost"],
        normal_cost     = row["normal_cost"],
        extra_cost      = row["extra_cost"],
        pct_increase    = row["pct_increase"],
        confidence      = row["confidence_pct"],
        why_it_happened = row["why_it_happened"],
        actions         = actions,
    )

    if success:
        sent_count += 1
    else:
        failed_count += 1

print(f"\n--- alert summary ---")
print(f"alerts sent     : {sent_count}")
print(f"alerts failed   : {failed_count}")

if sent_count > 0:
    print(f"\ncheck your email - {sent_count} alert(s) should arrive shortly")
    print("subject lines will start with [HIGH] or [MEDIUM]")

# save a log of what was sent
log_df = real_anomalies[["date","service","anomaly_type","severity",
                          "actual_cost","extra_cost","pct_increase"]].copy()
log_df["alert_sent"] = True
log_df["sent_at"]    = datetime.now().strftime("%Y-%m-%d %H:%M")
log_path = os.path.join(BASE_DIR, "alerts_log.csv")
log_df.to_csv(log_path, index=False)
print(f"\nalert log saved to {log_path}")