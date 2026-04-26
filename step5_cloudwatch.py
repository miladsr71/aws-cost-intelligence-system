import boto3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# connect to cloudwatch
# using same region and credentials as athena
cw = boto3.client("cloudwatch", region_name="us-east-1")
ec2 = boto3.client("ec2", region_name="us-east-1")
lamb = boto3.client("lambda", region_name="us-east-1")

print("\nconnecting to AWS CloudWatch...")


def get_ec2_instances():
    # grab all ec2 instances in the account - running or stopped
    # need the instance ids to query cloudwatch metrics
    try:
        response = ec2.describe_instances()
        instances = []
        for reservation in response["Reservations"]:
            for instance in reservation["Instances"]:
                instance_id = instance["InstanceId"]
                state       = instance["State"]["Name"]
                itype       = instance["InstanceType"]
                # try to get the name tag if it exists
                name = "unnamed"
                for tag in instance.get("Tags", []):
                    if tag["Key"] == "Name":
                        name = tag["Value"]
                instances.append({
                    "instance_id"   : instance_id,
                    "instance_type" : itype,
                    "state"         : state,
                    "name"          : name,
                })
        return instances
    except Exception as e:
        print(f"   could not get ec2 instances: {e}")
        return []


def get_lambda_functions():
    # list all lambda functions in the account
    try:
        response  = lamb.list_functions()
        functions = []
        for fn in response["Functions"]:
            functions.append({
                "name"    : fn["FunctionName"],
                "runtime" : fn.get("Runtime", "unknown"),
                "memory"  : fn.get("MemorySize", 0),
            })
        return functions
    except Exception as e:
        print(f"   could not get lambda functions: {e}")
        return []


def get_metric(namespace, metric_name, dimensions, start_time, end_time, stat="Average"):
    # helper to pull a single cloudwatch metric for a date range
    # returns daily averages - period=86400 means one datapoint per day
    try:
        response = cw.get_metric_statistics(
            Namespace=namespace,
            MetricName=metric_name,
            Dimensions=dimensions,
            StartTime=start_time,
            EndTime=end_time,
            Period=86400,  # one day in seconds
            Statistics=[stat],
        )
        # sort by timestamp and return as a simple list of (date, value) tuples
        datapoints = sorted(response["Datapoints"], key=lambda x: x["Timestamp"])
        return [(dp["Timestamp"].strftime("%Y-%m-%d"), dp[stat]) for dp in datapoints]
    except Exception as e:
        print(f"   metric pull failed for {metric_name}: {e}")
        return []


# set the date range for cloudwatch queries
# going back 75 days to cover the real data period we have in athena
end_time   = datetime.utcnow()
start_time = end_time - timedelta(days=75)
print(f"pulling metrics from {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")


# get ec2 instances and pull their cpu metrics
print("\nchecking for EC2 instances...")
instances = get_ec2_instances()

ec2_metrics = []
if instances:
    print(f"found {len(instances)} ec2 instance(s)")
    for inst in instances:
        print(f"   {inst['instance_id']} ({inst['instance_type']}) - {inst['state']} - {inst['name']}")

        # cpu utilization - most important metric for detecting ec2 spikes
        cpu_data = get_metric(
            namespace   = "AWS/EC2",
            metric_name = "CPUUtilization",
            dimensions  = [{"Name": "InstanceId", "Value": inst["instance_id"]}],
            start_time  = start_time,
            end_time    = end_time,
        )

        # network out - helps confirm if traffic increased alongside cost
        net_data = get_metric(
            namespace   = "AWS/EC2",
            metric_name = "NetworkOut",
            dimensions  = [{"Name": "InstanceId", "Value": inst["instance_id"]}],
            start_time  = start_time,
            end_time    = end_time,
            stat        = "Sum",
        )

        # combine cpu and network into daily rows
        cpu_dict = dict(cpu_data)
        net_dict = dict(net_data)
        all_dates = set(list(cpu_dict.keys()) + list(net_dict.keys()))

        for date in all_dates:
            ec2_metrics.append({
                "date"                : date,
                "service"             : "AmazonEC2",
                "resource_id"         : inst["instance_id"],
                "resource_name"       : inst["name"],
                "cw_cpu_utilization"  : round(cpu_dict.get(date, np.nan), 2),
                "cw_network_out_mb"   : round(net_dict.get(date, 0) / 1024 / 1024, 2),  # bytes to MB
            })

    if ec2_metrics:
        print(f"   pulled {len(ec2_metrics)} ec2 metric datapoints")
    else:
        print("   no metric data returned - instances may not have been running long enough")
else:
    print("no ec2 instances found in this account")


# get lambda functions and pull their metrics
print("\nchecking for Lambda functions...")
functions = get_lambda_functions()

lambda_metrics = []
if functions:
    print(f"found {len(functions)} lambda function(s)")
    for fn in functions:
        print(f"   {fn['name']} ({fn['runtime']}, {fn['memory']}MB)")

        # invocation count - if this spikes it means something is calling the function too much
        inv_data = get_metric(
            namespace   = "AWS/Lambda",
            metric_name = "Invocations",
            dimensions  = [{"Name": "FunctionName", "Value": fn["name"]}],
            start_time  = start_time,
            end_time    = end_time,
            stat        = "Sum",
        )

        # duration - if functions are running longer than expected cost goes up
        dur_data = get_metric(
            namespace   = "AWS/Lambda",
            metric_name = "Duration",
            dimensions  = [{"Name": "FunctionName", "Value": fn["name"]}],
            start_time  = start_time,
            end_time    = end_time,
        )

        inv_dict = dict(inv_data)
        dur_dict = dict(dur_data)
        all_dates = set(list(inv_dict.keys()) + list(dur_dict.keys()))

        for date in all_dates:
            lambda_metrics.append({
                "date"                   : date,
                "service"                : "AWSLambda",
                "resource_id"            : fn["name"],
                "resource_name"          : fn["name"],
                "cw_lambda_invocations"  : round(inv_dict.get(date, np.nan), 2),
                "cw_lambda_duration_ms"  : round(dur_dict.get(date, np.nan), 2),
            })

    if lambda_metrics:
        print(f"   pulled {len(lambda_metrics)} lambda metric datapoints")
    else:
        print("   no lambda metric data found - functions may not have been invoked recently")
else:
    print("no lambda functions found in this account")


# save raw cloudwatch data
print("\nsaving raw cloudwatch data...")
all_metrics = []

if ec2_metrics:
    ec2_df = pd.DataFrame(ec2_metrics)
    all_metrics.append(ec2_df)
    print(f"ec2 metrics: {len(ec2_df)} rows")
    print(ec2_df.head(5).to_string(index=False))

if lambda_metrics:
    lambda_df = pd.DataFrame(lambda_metrics)
    all_metrics.append(lambda_df)
    print(f"lambda metrics: {len(lambda_df)} rows")
    print(lambda_df.head(5).to_string(index=False))

if all_metrics:
    cw_df = pd.concat(all_metrics, ignore_index=True)
    cw_path = os.path.join(BASE_DIR, "cloudwatch_metrics.csv")
    cw_df.to_csv(cw_path, index=False)
    print(f"\ncloudwatch data saved to {cw_path}")
else:
    print("\nno live cloudwatch data found - this is expected for a personal account")
    print("the system will use the correlated synthetic cloudwatch values from step 1")
    # create empty file so later steps don't break
    cw_df = pd.DataFrame(columns=["date","service","resource_id",
                                   "resource_name","cw_cpu_utilization",
                                   "cw_network_out_mb","cw_lambda_invocations",
                                   "cw_lambda_duration_ms"])
    cw_path = os.path.join(BASE_DIR, "cloudwatch_metrics.csv")
    cw_df.to_csv(cw_path, index=False)


# now merge live cloudwatch data back into the main dataset
# if we got real metrics they replace the synthetic ones
# if not the synthetic ones stay as-is
print("\nmerging cloudwatch data into main dataset...")
main_df = pd.read_csv(os.path.join(BASE_DIR, "classified_anomalies.csv"))
main_df["date"] = pd.to_datetime(main_df["date"])

if len(cw_df) > 0:
    cw_df["date"] = pd.to_datetime(cw_df["date"])

    # merge on date and service
    # only update rows where we have real cloudwatch data
    for _, cw_row in cw_df.iterrows():
        mask = (
            (main_df["date"] == cw_row["date"]) &
            (main_df["service"] == cw_row["service"])
        )
        if mask.sum() > 0:
            if "cw_cpu_utilization" in cw_row and not pd.isna(cw_row.get("cw_cpu_utilization", np.nan)):
                main_df.loc[mask, "cw_cpu_utilization"] = cw_row["cw_cpu_utilization"]
            if "cw_network_out_mb" in cw_row and not pd.isna(cw_row.get("cw_network_out_mb", np.nan)):
                main_df.loc[mask, "cw_network_out_mb"] = cw_row["cw_network_out_mb"]
            if "cw_lambda_invocations" in cw_row and not pd.isna(cw_row.get("cw_lambda_invocations", np.nan)):
                main_df.loc[mask, "cw_lambda_invocations"] = cw_row["cw_lambda_invocations"]
            if "cw_lambda_duration_ms" in cw_row and not pd.isna(cw_row.get("cw_lambda_duration_ms", np.nan)):
                main_df.loc[mask, "cw_lambda_duration_ms"] = cw_row["cw_lambda_duration_ms"]

    print(f"merged live cloudwatch data into {mask.sum()} rows")
else:
    print("no live data to merge - keeping synthetic cloudwatch values")

# save final enriched dataset
out_path = os.path.join(BASE_DIR, "final_enriched_data.csv")
main_df.to_csv(out_path, index=False)
print(f"final enriched dataset saved to {out_path}")


# print a summary of what cloudwatch data we have
print("\n--- cloudwatch summary ---")
print(f"ec2 instances found    : {len(instances)}")
print(f"lambda functions found : {len(functions)}")
print(f"ec2 metric rows        : {len(ec2_metrics)}")
print(f"lambda metric rows     : {len(lambda_metrics)}")

cw_filled = main_df["cw_cpu_utilization"].notna().sum()
print(f"\nrows with cloudwatch data in final dataset: {cw_filled}")
print(f"rows without cloudwatch data              : {main_df['cw_cpu_utilization'].isna().sum()}")

print("\ndone - cloudwatch integration complete")
print("if no live metrics were found that is ok for a personal account")
print("the synthetic cloudwatch values are still valid for the ml model")