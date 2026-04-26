# AWS Cost Intelligence System

A custom cloud cost monitoring and anomaly detection system built on AWS infrastructure using machine learning. The system connects to AWS billing data, detects unusual cost behavior, classifies the type of anomaly, explains the root cause, and sends email alerts automatically.

## What It Does

- Pulls real billing data from AWS Athena every run
- Detects cost anomalies using Z-Score analysis and Isolation Forest
- Classifies each anomaly into one of 4 types using Random Forest
- Generates plain English recommendations with specific action steps
- Sends formatted email alerts via AWS SNS
- Produces an interactive HTML dashboard with charts and analysis
- Runs fully automated on EC2 with daily scheduling

## Why It Matters

AWS provides a native Cost Anomaly Detection service, but it only tells you that costs went up. It cannot explain why or what to do about it.

| Capability | AWS Native | This System |
|---|---|---|
| Detects cost increase | Yes | Yes |
| Identifies which service | Yes | Yes |
| Classifies anomaly type | No | Yes |
| Explains root cause | No | Yes |
| Correlates with CPU/usage | No | Yes |
| Gives specific recommendations | No | Yes |
| Distinguishes usage vs pricing anomaly | No | Yes |

## Anomaly Types Detected

| Type | What It Means |
|---|---|
| EC2 Spike | Cost and CPU both jumped — unexpected instance or runaway process |
| S3 Gradual Drift | Cost slowly creeping up over days — app bug or missing cleanup |
| Lambda Burst | Invocations exploded — infinite retry loop or runaway function |
| Unexplained Cost Spike | Cost jumped but CPU is normal — pricing change or Reserved Instance expiry |

## Architecture

```
AWS CUR → S3 → Glue Crawler → Athena
                                  ↓
                          Python Pipeline
                                  ↓
              ┌─────────────────────────────────────┐
              │  Step 1: Data Ingestion             │
              │  Step 2: Anomaly Detection          │
              │  Step 3: ML Classification          │
              │  Step 4: Recommendations            │
              │  Step 5: CloudWatch Integration     │
              │  Step 6: SNS Email Alerts           │
              │  Step 7: HTML Dashboard             │
              └─────────────────────────────────────┘
                                  ↓
                    Email Alert + Dashboard Report
```

## How to Run

### Prerequisites

- AWS account with CUR enabled
- Python 3.9+
- AWS CLI configured with credentials

### Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn boto3
```

### Run the full pipeline

```bash
python3 run_pipeline.py
```

This single command runs all 7 steps and takes about 60 seconds.

### Open the dashboard

```bash
open dashboard.html
```

## Configuration

If your AWS account has real EC2 and Lambda usage, set `USE_SYNTHETIC_DATA = False` in `step1_augmentation.py`. The system will use only your real billing data and detect actual anomalies without any synthetic data.

If your account has low usage (personal or dev account), keep `USE_SYNTHETIC_DATA = True` to see the system in action with simulated anomalies.

### AWS Settings

Update these values in `step1_augmentation.py` to match your account:

```python
REGION    = "us-east-1"
DATABASE  = "your_cur_database"
TABLE     = "your_cur_table"
S3_OUTPUT = "s3://your-bucket/athena-results/"
```

Update the SNS topic ARN in `step6_alerts.py`:

```python
SNS_TOPIC_ARN = "arn:aws:sns:us-east-1:YOUR_ACCOUNT_ID:your-topic-name"
```

## Pipeline Steps

| Step | Script | What It Does |
|---|---|---|
| 1 | step1_augmentation.py | Pulls billing data from Athena, adds synthetic data if enabled |
| 2 | step2_anomaly_detection.py | Engineers features, runs Z-Score + Isolation Forest detection |
| 3 | step3_classifier.py | Trains Random Forest to classify anomaly types |
| 4 | step4_recommendations.py | Generates root cause explanations and action steps |
| 5 | step5_cloudwatch.py | Pulls real CloudWatch metrics for EC2 and Lambda |
| 6 | step6_alerts.py | Sends email alerts via AWS SNS |
| 7 | step7_dashboard.py | Generates HTML dashboard with charts and analysis |

## Algorithms Used

| Algorithm | Type | Purpose |
|---|---|---|
| Z-Score | Statistical | Flags cost deviations beyond 2.0 standard deviations |
| Isolation Forest | Unsupervised ML | Detects multivariate anomalies across 11 features |
| Random Forest | Supervised ML | Classifies anomalies into 4 types with confidence scores |

## Results

- Detection accuracy: 98%
- Classification cross-validation: 99.8%
- Classification test set: ~100%
- All 4 anomaly types detected and classified correctly
- Email alerts delivered successfully via SNS

## Tech Stack

- Python 3.9+
- pandas, numpy, scikit-learn, matplotlib, seaborn
- boto3 (AWS SDK)
- AWS: S3, CUR, Glue, Athena, CloudWatch, SNS, EC2

## Automated Deployment

The system can be deployed to an EC2 instance with a cron job for fully automated daily execution. No manual commands needed — the pipeline runs every morning, checks for anomalies, and sends alerts if anything unusual is found.

## About the Data

This system pulls real billing data from AWS Athena. For accounts with low usage, synthetic EC2 and Lambda data is added for the same billing period to demonstrate the anomaly detection capabilities. The synthetic data includes 4 labeled anomaly types used for model training and validation.

On accounts with real EC2 and Lambda workloads, the system detects actual cost anomalies without any synthetic data needed.

## Created By

Milad Zahmatkesh
