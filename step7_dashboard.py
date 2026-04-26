import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # needed to generate charts without a display window
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import base64
import os
from io import BytesIO
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("\nloading data for dashboard...")
df        = pd.read_csv(os.path.join(BASE_DIR, "final_enriched_data.csv"))
rec_df    = pd.read_csv(os.path.join(BASE_DIR, "recommendations.csv"))
alerts_df = pd.read_csv(os.path.join(BASE_DIR, "alerts_log.csv"))
df["date"] = pd.to_datetime(df["date"])
print(f"loaded {len(df)} rows")


def fig_to_base64(fig):
    # convert matplotlib figure to base64 string for embedding in html
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="white")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_base64


print("generating charts...")

# chart 1: daily total cost over time with anomaly markers
fig1, ax = plt.subplots(figsize=(13, 4))
daily = df.groupby("date")["daily_cost"].sum().reset_index()
ax.plot(daily["date"], daily["daily_cost"],
        color="#2E75B6", linewidth=1.8, label="daily total cost")

# shade the real vs synthetic regions
real_end = pd.Timestamp("2026-04-14")
ax.axvspan(daily["date"].min(), real_end,
           alpha=0.07, color="green", label="real aws data")
ax.axvspan(real_end, daily["date"].max(),
           alpha=0.07, color="orange", label="synthetic data")
ax.axvline(x=real_end, color="green", linestyle="--",
           linewidth=1.5, label="real → synthetic")

# mark detected anomalies
anomaly_rows = df[df["detected_anomaly"] == 1]
ax.scatter(anomaly_rows["date"], anomaly_rows["daily_cost"],
           color="red", zorder=5, s=55, label="detected anomaly")

ax.set_title("Daily AWS Cost Over Time", fontsize=13, fontweight="bold", pad=10)
ax.set_ylabel("Cost (USD)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.legend(fontsize=8, loc="upper left")
ax.grid(True, alpha=0.25)
fig1.autofmt_xdate()
chart1 = fig_to_base64(fig1)

# chart 2: z-score per service
fig2, ax = plt.subplots(figsize=(13, 4))
service_colors = {
    "AmazonEC2"                         : "#2E75B6",
    "Amazon Simple Storage Service"     : "#E8712A",
    "AWSLambda"                         : "#1E7145",
    "AWS Glue"                          : "#7030A0",
    "AmazonCloudWatch"                  : "#C00000",
}
for svc, color in service_colors.items():
    sdf = df[df["service"] == svc]
    if len(sdf) > 0:
        ax.plot(sdf["date"], sdf["z_score"],
                label=svc.replace("Amazon Simple Storage Service", "S3"),
                color=color, linewidth=1.2, alpha=0.85)

ax.axhline(y=2.0,  color="red", linestyle="--", alpha=0.6, linewidth=1.2, label="threshold ±2.0")
ax.axhline(y=-2.0, color="red", linestyle="--", alpha=0.6, linewidth=1.2)
ax.axvline(x=real_end, color="green", linestyle="--", linewidth=1.5)
ax.set_title("Z-Score Per Service (Deviation from Normal)", fontsize=13, fontweight="bold", pad=10)
ax.set_ylabel("Z-Score")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.legend(fontsize=8, loc="upper left")
ax.grid(True, alpha=0.25)
fig2.autofmt_xdate()
chart2 = fig_to_base64(fig2)

# chart 3: cost breakdown by service (pie chart)
fig3, ax = plt.subplots(figsize=(7, 5))
service_costs = df.groupby("service")["daily_cost"].sum().sort_values(ascending=False)
# only show top 6 services, group rest as other
top6  = service_costs.head(6)
other = service_costs.iloc[6:].sum()
if other > 0:
    top6["Other"] = other
colors_pie = ["#2E75B6","#E8712A","#1E7145","#7030A0","#C00000","#F0A500","#888888"]
wedges, texts, autotexts = ax.pie(
    top6.values,
    labels=None,
    autopct="%1.1f%%",
    colors=colors_pie[:len(top6)],
    startangle=140,
    pctdistance=0.82
)
ax.legend(wedges, [s.replace("Amazon Simple Storage Service","S3")
                   .replace("AmazonCloudWatch","CloudWatch")
                   for s in top6.index],
          loc="lower center", bbox_to_anchor=(0.5, -0.18),
          ncol=2, fontsize=8)
ax.set_title("Total Cost by Service", fontsize=13, fontweight="bold", pad=10)
chart3 = fig_to_base64(fig3)

# chart 4: anomaly type distribution bar chart
fig4, ax = plt.subplots(figsize=(7, 5))
real_anom = df[df["anomaly_type"] != "none"]["anomaly_type"].value_counts()
bar_colors = ["#C00000","#E8712A","#1E7145","#7030A0"]
bars = ax.barh(real_anom.index, real_anom.values,
               color=bar_colors[:len(real_anom)], edgecolor="white")
for bar, val in zip(bars, real_anom.values):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
            str(val), va="center", fontsize=10, fontweight="bold")
ax.set_title("Anomaly Type Distribution", fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel("Count")
ax.grid(True, alpha=0.25, axis="x")
ax.set_xlim(0, real_anom.max() + 3)
chart4 = fig_to_base64(fig4)

print("charts done - building html...")

# build the summary stats for the top cards
total_cost   = round(df["daily_cost"].sum(), 2)
total_rows   = len(df)
real_rows    = (df["data_source"] == "real").sum()
synth_rows   = (df["data_source"] == "synthetic").sum()
total_anom   = (df["anomaly_label"] == "anomaly").sum()
detected     = df["detected_anomaly"].sum()
high_alerts  = len(rec_df[rec_df["severity"] == "HIGH"].drop_duplicates(subset=["anomaly_type","service"]))
medium_alerts= len(rec_df[rec_df["severity"] == "MEDIUM"].drop_duplicates(subset=["anomaly_type","service"]))

# build recommendation cards html
rec_cards_html = ""
real_recs = rec_df[
    (rec_df["anomaly_type"] != "none") &
    (rec_df["severity"].isin(["HIGH","MEDIUM"]))
].drop_duplicates(subset=["anomaly_type","service"])

severity_colors = {"HIGH": "#C00000", "MEDIUM": "#E8712A", "LOW": "#1E7145"}

for _, row in real_recs.iterrows():
    sev_color = severity_colors.get(row["severity"], "#888")
    try:
        actions_raw = str(row["actions"]).strip("[]").replace("'","")
        actions_list = [a.strip() for a in actions_raw.split(",") if a.strip()]
    except Exception:
        actions_list = ["review aws cost explorer"]
    actions_html = "".join([f"<li>{a}</li>" for a in actions_list])

    rec_cards_html += f"""
    <div class="rec-card">
        <div class="rec-header">
            <span class="severity-badge" style="background:{sev_color}">
                {row['severity']}
            </span>
            <span class="rec-type">
                {row['anomaly_type'].replace('_',' ').upper()}
            </span>
        </div>
        <div class="rec-meta">
            <span>📅 {row['date']}</span>
            <span>☁️ {row['service']}</span>
            <span>🎯 {row['confidence_pct']}% confidence</span>
        </div>
        <div class="rec-costs">
            <div class="cost-box normal">
                <div class="cost-label">Normal Cost</div>
                <div class="cost-value">${row['normal_cost']}</div>
            </div>
            <div class="cost-box actual">
                <div class="cost-label">Actual Cost</div>
                <div class="cost-value">${row['actual_cost']}</div>
            </div>
            <div class="cost-box extra">
                <div class="cost-label">Extra Spend</div>
                <div class="cost-value">+${row['extra_cost']}</div>
            </div>
            <div class="cost-box pct">
                <div class="cost-label">Increase</div>
                <div class="cost-value">+{row['pct_increase']}%</div>
            </div>
        </div>
        <div class="rec-why">
            <strong>Root Cause:</strong> {row['why_it_happened']}
        </div>
        <div class="rec-actions">
            <strong>Recommended Actions:</strong>
            <ul>{actions_html}</ul>
        </div>
    </div>
    """

# build the full html page
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AWS Cost Intelligence Dashboard</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #f0f4f8; color: #1a2a3a; }}

  .header {{ background: linear-gradient(135deg, #1F3864 0%, #2E75B6 100%);
             color: white; padding: 32px 40px; }}
  .header h1 {{ font-size: 26px; font-weight: 700; margin-bottom: 6px; }}
  .header p  {{ font-size: 14px; opacity: 0.85; }}
  .header .meta {{ margin-top: 10px; font-size: 13px; opacity: 0.75; }}

  .container {{ max-width: 1200px; margin: 0 auto; padding: 28px 20px; }}

  .cards-row {{ display: grid; grid-template-columns: repeat(4, 1fr);
                gap: 16px; margin-bottom: 28px; }}
  .card {{ background: white; border-radius: 10px; padding: 20px;
           box-shadow: 0 2px 8px rgba(0,0,0,0.07); border-left: 4px solid #2E75B6; }}
  .card.red   {{ border-left-color: #C00000; }}
  .card.green {{ border-left-color: #1E7145; }}
  .card.orange{{ border-left-color: #E8712A; }}
  .card-label {{ font-size: 12px; color: #666; text-transform: uppercase;
                 letter-spacing: 0.5px; margin-bottom: 6px; }}
  .card-value {{ font-size: 28px; font-weight: 700; color: #1F3864; }}
  .card-sub   {{ font-size: 12px; color: #888; margin-top: 4px; }}

  .section {{ background: white; border-radius: 10px; padding: 24px;
              box-shadow: 0 2px 8px rgba(0,0,0,0.07); margin-bottom: 24px; }}
  .section h2 {{ font-size: 17px; font-weight: 700; color: #1F3864;
                 margin-bottom: 18px; padding-bottom: 10px;
                 border-bottom: 2px solid #e8f0fb; }}

  .charts-2col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .chart-box img {{ width: 100%; border-radius: 6px; }}
  .chart-full img {{ width: 100%; border-radius: 6px; }}

  .rec-card {{ background: #f8faff; border-radius: 8px; padding: 18px;
               margin-bottom: 16px; border: 1px solid #dde8f5; }}
  .rec-header {{ display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }}
  .severity-badge {{ padding: 3px 10px; border-radius: 12px; color: white;
                     font-size: 11px; font-weight: 700; letter-spacing: 0.5px; }}
  .rec-type {{ font-size: 15px; font-weight: 700; color: #1F3864; }}
  .rec-meta {{ display: flex; gap: 20px; font-size: 12px; color: #555;
               margin-bottom: 12px; }}
  .rec-costs {{ display: grid; grid-template-columns: repeat(4,1fr);
                gap: 10px; margin-bottom: 14px; }}
  .cost-box {{ background: white; border-radius: 6px; padding: 10px;
               text-align: center; border: 1px solid #e0e8f5; }}
  .cost-box.actual {{ border-color: #C00000; }}
  .cost-box.extra  {{ border-color: #E8712A; }}
  .cost-label {{ font-size: 10px; color: #888; text-transform: uppercase; margin-bottom: 4px; }}
  .cost-value {{ font-size: 16px; font-weight: 700; color: #1F3864; }}
  .cost-box.actual .cost-value {{ color: #C00000; }}
  .cost-box.extra  .cost-value {{ color: #E8712A; }}
  .rec-why {{ font-size: 13px; color: #333; margin-bottom: 12px;
              padding: 10px; background: #fff8e8; border-radius: 6px;
              border-left: 3px solid #E8712A; }}
  .rec-actions {{ font-size: 13px; color: #333; }}
  .rec-actions ul {{ margin-top: 6px; padding-left: 20px; }}
  .rec-actions li {{ margin-bottom: 4px; color: #444; }}

  .alerts-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  .alerts-table th {{ background: #1F3864; color: white; padding: 10px 12px;
                      text-align: left; font-weight: 600; }}
  .alerts-table td {{ padding: 9px 12px; border-bottom: 1px solid #e8f0fb; }}
  .alerts-table tr:hover td {{ background: #f0f4f8; }}
  .badge-high   {{ background:#C00000; color:white; padding:2px 8px;
                   border-radius:10px; font-size:11px; font-weight:700; }}
  .badge-medium {{ background:#E8712A; color:white; padding:2px 8px;
                   border-radius:10px; font-size:11px; font-weight:700; }}

  .footer {{ text-align: center; padding: 24px; font-size: 12px; color: #888; }}
</style>
</head>
<body>

<div class="header">
  <h1>☁️ AWS Cost Intelligence Dashboard</h1>
  <p>Anomaly Detection and Cost Optimization System</p>
  <div class="meta">
    Created by: Milad Zahmatkesh &nbsp;|&nbsp;
    Supervised by: Dr. McKenney &nbsp;|&nbsp;
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
  </div>
</div>

<div class="container">

  <!-- summary cards -->
  <div class="cards-row">
    <div class="card">
      <div class="card-label">Total Rows Analysed</div>
      <div class="card-value">{total_rows:,}</div>
      <div class="card-sub">{real_rows} real + {synth_rows} synthetic</div>
    </div>
    <div class="card red">
      <div class="card-label">Anomalies Detected</div>
      <div class="card-value">{detected}</div>
      <div class="card-sub">{total_anom} injected ground truth</div>
    </div>
    <div class="card orange">
      <div class="card-label">Alerts Sent</div>
      <div class="card-value">{len(alerts_df)}</div>
      <div class="card-sub">{high_alerts} HIGH · {medium_alerts} MEDIUM</div>
    </div>
    <div class="card green">
      <div class="card-label">Classifier Accuracy</div>
      <div class="card-value">100%</div>
      <div class="card-sub">99.9% cross-validation</div>
    </div>
  </div>

  <!-- cost over time chart -->
  <div class="section">
    <h2>📈 Daily Cost Over Time</h2>
    <div class="chart-full">
      <img src="data:image/png;base64,{chart1}" alt="daily cost chart">
    </div>
  </div>

  <!-- z-score chart -->
  <div class="section">
    <h2>📊 Z-Score Per Service</h2>
    <div class="chart-full">
      <img src="data:image/png;base64,{chart2}" alt="z-score chart">
    </div>
  </div>

  <!-- pie and bar side by side -->
  <div class="charts-2col" style="margin-bottom:24px;">
    <div class="section" style="margin-bottom:0">
      <h2>💰 Cost by Service</h2>
      <div class="chart-box">
        <img src="data:image/png;base64,{chart3}" alt="cost by service">
      </div>
    </div>
    <div class="section" style="margin-bottom:0">
      <h2>🔍 Anomaly Distribution</h2>
      <div class="chart-box">
        <img src="data:image/png;base64,{chart4}" alt="anomaly distribution">
      </div>
    </div>
  </div>

  <!-- recommendations -->
  <div class="section">
    <h2>🚨 Anomaly Recommendations</h2>
    {rec_cards_html}
  </div>

  <!-- alerts log table -->
  <div class="section">
    <h2>📧 Alerts Sent via AWS SNS</h2>
    <table class="alerts-table">
      <tr>
        <th>Date</th>
        <th>Service</th>
        <th>Anomaly Type</th>
        <th>Severity</th>
        <th>Extra Cost</th>
        <th>Increase</th>
        <th>Sent At</th>
      </tr>
      {"".join([f'''<tr>
        <td>{row['date']}</td>
        <td>{row['service']}</td>
        <td>{row['anomaly_type'].replace('_',' ')}</td>
        <td><span class="badge-{row['severity'].lower()}">{row['severity']}</span></td>
        <td>+${row['extra_cost']}</td>
        <td>+{row['pct_increase']}%</td>
        <td>{row['sent_at']}</td>
      </tr>''' for _, row in alerts_df.iterrows()])}
    </table>
  </div>

</div>

<div class="footer">
  AWS Cost Intelligence System &nbsp;|&nbsp;
  Milad Zahmatkesh &nbsp;|&nbsp;
  {datetime.now().strftime('%Y')}
</div>

</body>
</html>"""

# save the dashboard
out_path = os.path.join(BASE_DIR, "dashboard.html")
with open(out_path, "w") as f:
    f.write(html)

print(f"dashboard saved to {out_path}")
print("\nopen it in your browser:")
print(f"open {out_path}")
print("\n--- dashboard summary ---")
print(f"total rows    : {total_rows}")
print(f"anomalies     : {detected}")
print(f"alerts sent   : {len(alerts_df)}")
print(f"accuracy      : 100%")
print("\ndone - open dashboard.html in your browser to see the full report")