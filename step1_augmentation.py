import pandas as pd
import numpy as np
import boto3
import time
import os
from datetime import datetime, timedelta

# set this to False if your account has real EC2/Lambda usage
# and you don't need synthetic data or injected anomalies
USE_SYNTHETIC_DATA = True

# athena config
REGION    = "us-east-1"
DATABASE  = "your_cur_database"
TABLE     = "your_cur_table"
S3_OUTPUT = "s3://YOUR-BUCKET-NAME/athena-results/"
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))


def run_athena_query(query):
    client   = boto3.client("athena", region_name=REGION)
    response = client.start_query_execution(
        QueryString=query,
        QueryExecutionContext={"Database": DATABASE},
        ResultConfiguration={"OutputLocation": S3_OUTPUT},
    )
    query_id = response["QueryExecutionId"]
    print(f"   query id: {query_id}")

    while True:
        result = client.get_query_execution(QueryExecutionId=query_id)
        state  = result["QueryExecution"]["Status"]["State"]
        if state == "SUCCEEDED":
            print("   done")
            break
        elif state in ("FAILED", "CANCELLED"):
            reason = result["QueryExecution"]["Status"].get("StateChangeReason", "")
            raise Exception(f"query failed: {state} - {reason}")
        time.sleep(2)

    paginator = client.get_paginator("get_query_results")
    pages     = paginator.paginate(QueryExecutionId=query_id)
    rows      = []
    columns   = None
    for page in pages:
        result_rows = page["ResultSet"]["Rows"]
        if columns is None:
            columns     = [col["VarCharValue"] for col in result_rows[0]["Data"]]
            result_rows = result_rows[1:]
        for row in result_rows:
            rows.append([col.get("VarCharValue", "") for col in row["Data"]])

    return pd.DataFrame(rows, columns=columns)


# pull real CUR data from athena
print("\npulling real CUR data from Athena...")

query = f"""
    SELECT
        DATE(line_item_usage_start_date)   AS date,
        product_servicename                AS service,
        product_region                     AS region,
        SUM(line_item_unblended_cost)      AS daily_cost,
        SUM(line_item_usage_amount)        AS usage_amount
    FROM {DATABASE}.{TABLE}
    WHERE line_item_unblended_cost IS NOT NULL
    GROUP BY
        DATE(line_item_usage_start_date),
        product_servicename,
        product_region
    ORDER BY date, service
"""

real_df = run_athena_query(query)

real_df["daily_cost"]   = pd.to_numeric(real_df["daily_cost"],   errors="coerce").fillna(0)
real_df["usage_amount"] = pd.to_numeric(real_df["usage_amount"], errors="coerce").fillna(0)
real_df["anomaly_label"]         = "normal"
real_df["anomaly_type"]          = "none"
real_df["cw_cpu_utilization"]    = np.nan
real_df["cw_network_out_mb"]     = np.nan
real_df["cw_lambda_invocations"] = np.nan
real_df["cw_lambda_duration_ms"] = np.nan
real_df["data_source"]           = "real"

real_df = real_df[real_df["service"].str.strip() != ""].copy()
real_df = real_df.dropna(subset=["service"])

print(f"real rows pulled: {len(real_df)}")
print(f"date range: {real_df['date'].min()} to {real_df['date'].max()}")
print("services in real data:")
print(real_df["service"].value_counts().to_string())

real_df["date"] = pd.to_datetime(real_df["date"])


if USE_SYNTHETIC_DATA:
    # synthetic mode - adds EC2, Lambda, boosts real services, injects anomalies
    # use this for personal accounts with low usage
    print("\nUSE_SYNTHETIC_DATA is True - generating synthetic data...")

    real_start = real_df["date"].min()
    real_end   = real_df["date"].max()
    dates      = pd.date_range(start=real_start, end=real_end, freq="D")
    print(f"generating synthetic data for {len(dates)} days")
    print(f"same date range as real data: {real_start.date()} to {real_end.date()}")

    np.random.seed(42)

    base_costs = {
        "AmazonEC2" : 4.50,
        "AWSLambda" : 0.30,
    }
    base_usage = {
        "AmazonEC2" : 24.0,
        "AWSLambda" : 1000.0,
    }

    boost_costs = {
        "AWS Glue"                          : 0.50,
        "AmazonCloudWatch"                  : 0.20,
        "Amazon Simple Storage Service"     : 0.80,
        "AWS Data Transfer"                 : 0.10,
    }
    boost_usage = {
        "AWS Glue"                          : 2.0,
        "AmazonCloudWatch"                  : 50.0,
        "Amazon Simple Storage Service"     : 500.0,
        "AWS Data Transfer"                 : 1.0,
    }

    rows = []
    for day_idx, date in enumerate(dates):
        for service, base_cost in base_costs.items():
            noise = np.random.normal(1.0, 0.12)
            trend = 1.0 + (day_idx / len(dates)) * 0.06
            daily_cost  = round(max(base_cost * noise * trend, 0), 6)
            daily_usage = round(max(base_usage[service] * noise * trend, 0), 4)
            rows.append({
                "date"                  : date.strftime("%Y-%m-%d"),
                "service"               : service,
                "region"                : "us-east-1",
                "daily_cost"            : daily_cost,
                "usage_amount"          : daily_usage,
                "anomaly_label"         : "normal",
                "anomaly_type"          : "none",
                "cw_cpu_utilization"    : np.nan,
                "cw_network_out_mb"     : np.nan,
                "cw_lambda_invocations" : np.nan,
                "cw_lambda_duration_ms" : np.nan,
                "data_source"           : "synthetic",
            })
        for service, base_cost in boost_costs.items():
            noise = np.random.normal(1.0, 0.12)
            trend = 1.0 + (day_idx / len(dates)) * 0.06
            daily_cost  = round(max(base_cost * noise * trend, 0), 6)
            daily_usage = round(max(boost_usage[service] * noise * trend, 0), 4)
            rows.append({
                "date"                  : date.strftime("%Y-%m-%d"),
                "service"               : service,
                "region"                : "us-east-1",
                "daily_cost"            : daily_cost,
                "usage_amount"          : daily_usage,
                "anomaly_label"         : "normal",
                "anomaly_type"          : "none",
                "cw_cpu_utilization"    : np.nan,
                "cw_network_out_mb"     : np.nan,
                "cw_lambda_invocations" : np.nan,
                "cw_lambda_duration_ms" : np.nan,
                "data_source"           : "synthetic",
            })

    synth_df = pd.DataFrame(rows)
    synth_df["date"] = pd.to_datetime(synth_df["date"])
    print(f"synthetic baseline: {len(synth_df)} rows")

    # inject anomalies
    print("\ninjecting anomalies...")
    date_list = sorted(synth_df["date"].dt.strftime("%Y-%m-%d").unique())
    total_days = len(date_list)

    # anomaly 1: ec2 spike
    spike_date = date_list[int(total_days * 0.20)]
    mask = (synth_df["date"].dt.strftime("%Y-%m-%d") == spike_date) & \
           (synth_df["service"] == "AmazonEC2")
    synth_df.loc[mask, "daily_cost"]   *= 8.5
    synth_df.loc[mask, "usage_amount"] *= 8.5
    synth_df.loc[mask, "anomaly_label"] = "anomaly"
    synth_df.loc[mask, "anomaly_type"]  = "ec2_spike"
    print(f"   ec2 spike on {spike_date}")

    # anomaly 2: s3 gradual drift over 16 days
    drift_start_idx = int(total_days * 0.35)
    drift_dates = date_list[drift_start_idx : drift_start_idx + 16]
    for i, d in enumerate(drift_dates):
        mask = (synth_df["date"].dt.strftime("%Y-%m-%d") == d) & \
               (synth_df["service"] == "AmazonEC2")
        multiplier = 1.0 + (i * 0.35)
        synth_df.loc[mask, "daily_cost"]   *= multiplier
        synth_df.loc[mask, "usage_amount"] *= multiplier
        synth_df.loc[mask, "anomaly_label"] = "anomaly"
        synth_df.loc[mask, "anomaly_type"]  = "s3_gradual_drift"
    print(f"   s3 gradual drift from {drift_dates[0]} to {drift_dates[-1]}")

    # anomaly 3: lambda burst
    burst_date = date_list[int(total_days * 0.60)]
    mask = (synth_df["date"].dt.strftime("%Y-%m-%d") == burst_date) & \
           (synth_df["service"] == "AWSLambda")
    synth_df.loc[mask, "daily_cost"]   *= 12.0
    synth_df.loc[mask, "usage_amount"] *= 12.0
    synth_df.loc[mask, "anomaly_label"] = "anomaly"
    synth_df.loc[mask, "anomaly_type"]  = "lambda_burst"
    print(f"   lambda burst on {burst_date}")

    # anomaly 4: unexplained cost spike - cost up but usage normal
    unexplained_date = date_list[int(total_days * 0.80)]
    mask = (synth_df["date"].dt.strftime("%Y-%m-%d") == unexplained_date) & \
           (synth_df["service"] == "AmazonEC2")
    synth_df.loc[mask, "daily_cost"]   *= 5.0
    synth_df.loc[mask, "anomaly_label"] = "anomaly"
    synth_df.loc[mask, "anomaly_type"]  = "unexplained_cost_spike"
    print(f"   unexplained spike on {unexplained_date} - cost up but usage stays normal")

    # add cloudwatch metrics
    print("\nadding cloudwatch metrics...")

    def add_cloudwatch(row):
        atype = row["anomaly_type"]
        if row["service"] == "AmazonEC2":
            base_cpu = 35.0 + np.random.normal(0, 5)
            base_net = 500.0 + np.random.normal(0, 50)
            if atype == "ec2_spike":
                row["cw_cpu_utilization"] = round(min(base_cpu * 8.0, 99.0), 2)
                row["cw_network_out_mb"]  = round(base_net * 7.0, 2)
            elif atype == "unexplained_cost_spike":
                row["cw_cpu_utilization"] = round(base_cpu, 2)
                row["cw_network_out_mb"]  = round(base_net, 2)
            else:
                row["cw_cpu_utilization"] = round(base_cpu, 2)
                row["cw_network_out_mb"]  = round(base_net, 2)
        if row["service"] == "AWSLambda":
            base_inv = 1000.0 + np.random.normal(0, 100)
            base_dur = 250.0  + np.random.normal(0, 30)
            if atype == "lambda_burst":
                row["cw_lambda_invocations"] = round(base_inv * 12.0, 2)
                row["cw_lambda_duration_ms"] = round(base_dur * 3.5, 2)
            else:
                row["cw_lambda_invocations"] = round(base_inv, 2)
                row["cw_lambda_duration_ms"] = round(base_dur, 2)
        return row

    synth_df = synth_df.apply(add_cloudwatch, axis=1)
    print("cloudwatch metrics added")

    # combine real + synthetic
    print("\ncombining datasets...")
    combined_df = pd.concat([real_df, synth_df], ignore_index=True)

else:
    # real data only mode - no synthetic data, no injected anomalies
    # use this for accounts with real EC2/Lambda usage
    print("\nUSE_SYNTHETIC_DATA is False - using real data only")
    combined_df = real_df.copy()

combined_df["date"] = pd.to_datetime(combined_df["date"])
combined_df = combined_df.sort_values(["date", "service"]).reset_index(drop=True)
print(f"total rows: {len(combined_df)}")

out_path = os.path.join(BASE_DIR, "augmented_cost_data.csv")
combined_df.to_csv(out_path, index=False)
print(f"saved to {out_path}")

print("\n--- summary ---")
print(f"synthetic data : {'enabled' if USE_SYNTHETIC_DATA else 'disabled'}")
print(f"total rows     : {len(combined_df)}")
print(f"date range     : {combined_df['date'].min().date()} to {combined_df['date'].max().date()}")
print(f"normal rows    : {(combined_df['anomaly_label'] == 'normal').sum()}")
print(f"anomalies      : {(combined_df['anomaly_label'] == 'anomaly').sum()}")
print("\ncost by service:")
print(combined_df.groupby("service")["daily_cost"].sum().sort_values(ascending=False).round(4).to_string())