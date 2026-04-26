import subprocess
import sys
import os
from datetime import datetime

# change to the directory where all the scripts are
# this makes sure relative paths work correctly
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

# all steps in order - each one depends on the previous
steps = [
    ("step1_augmentation.py",      "Data Ingestion & Augmentation"),
    ("step2_anomaly_detection.py", "Anomaly Detection"),
    ("step3_classifier.py",        "ML Classification"),
    ("step4_recommendations.py",   "Recommendations Engine"),
    ("step5_cloudwatch.py",        "CloudWatch Integration"),
    ("step6_alerts.py",            "SNS Alerts"),
    ("step7_dashboard.py",         "Dashboard Generation"),
]

print("="*60)
print("AWS Cost Intelligence Pipeline")
print(f"started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("="*60)

failed = False
for script, name in steps:
    print(f"\nrunning {name}...")
    print(f"   script: {script}")

    # run each script and wait for it to finish
    result = subprocess.run(
        [sys.executable, os.path.join(BASE_DIR, script)],
        capture_output=False  # show output in terminal
    )

    if result.returncode != 0:
        print(f"\nsomething went wrong in {name} - stopping pipeline")
        failed = True
        break
    else:
        print(f"   {name} done ✓")

print("\n" + "="*60)
if not failed:
    print("pipeline complete!")
    print(f"finished: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("\noutput files:")
    print("   augmented_cost_data.csv  - full dataset")
    print("   detected_anomalies.csv   - step 2 results")
    print("   classified_anomalies.csv - step 3 results")
    print("   recommendations.csv      - step 4 results")
    print("   final_enriched_data.csv  - step 5 results")
    print("   alerts_log.csv           - step 6 log")
    print("   dashboard.html           - open in browser")
    print("\ncheck your email for any anomaly alerts!")
else:
    print("pipeline stopped due to an error")
    print("check the output above to see what went wrong")
print("="*60)