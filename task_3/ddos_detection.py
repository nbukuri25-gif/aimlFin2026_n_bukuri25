# ==========================================
# Task 3: Web server log DDoS interval detection using regression
# Colab-ready single block
#
# Expected input file in notebook root:
#   n_bukuri25_73625_server.log
#
# Outputs:
#   - Prints detected DDoS time interval(s)
#   - Saves plots: traffic_baseline.png, residuals.png
# ==========================================

import os
import re
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import HuberRegressor

# -----------------------------
# 0) Configuration
# -----------------------------
LOG_FILENAME = "n_bukuri25_73625_server.log"  # must exist in Colab root (as you requested)

# OPTIONAL: download if missing (kept off by default)
DOWNLOAD_IF_MISSING = False
LOG_URL = "http://max.ge/aiml_final/n_bukuri25_73625_server.log"

# Aggregation bin size (regression is clearer on minute bins; you can set "10S" or "1S" if needed)
BIN = "1min"  # pandas resample frequency: "1min", "10S", "1S", "5min" etc.

# DDoS detection parameters
Z_THRESHOLD = 4.0          # residual threshold in robust z-score units
MIN_CONSECUTIVE_BINS = 3   # how many consecutive bins to call an "attack interval"

# -----------------------------
# 1) Ensure log file exists
# -----------------------------
if not os.path.exists(LOG_FILENAME):
    if DOWNLOAD_IF_MISSING:
        try:
            import subprocess
            print(f"[i] {LOG_FILENAME} not found. Downloading from: {LOG_URL}")
            subprocess.check_call(["bash", "-lc", f"wget -O {LOG_FILENAME} --timeout=25 --tries=2 {LOG_URL}"])
        except Exception as e:
            raise FileNotFoundError(
                f"Could not find or download {LOG_FILENAME}. "
                f"Upload it to Colab root or set DOWNLOAD_IF_MISSING=False and upload manually.\n{e}"
            )
    else:
        raise FileNotFoundError(
            f"{LOG_FILENAME} not found in the current directory.\n"
            f"Please upload it to Colab (Files panel) so it appears at ./ {LOG_FILENAME}"
        )

# -----------------------------
# 2) Parse timestamps from logs (UPDATED for your format)
# -----------------------------
import re
import pandas as pd
from datetime import datetime

# 1) Your observed format: [2024-03-22 18:00:54+04:00] or [2024-03-22T18:00:54+04:00]
BRACKET_ISO_RE = re.compile(
    r"\[(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[+\-]\d{2}:\d{2})?)\]"
)

# 2) Classic Apache format: [10/Oct/2000:13:55:36 -0700]
APACHE_TS_RE = re.compile(
    r"\[(\d{2}/[A-Za-z]{3}/\d{4}:\d{2}:\d{2}:\d{2}) ([+\-]\d{4})\]"
)

MONTHS = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
}

def parse_apache_ts(ts_str: str) -> datetime:
    # ts_str: "12/Feb/2026:15:04:05"
    d, mon, rest = ts_str.split("/", 2)            # "12", "Feb", "2026:15:04:05"
    year, hms = rest.split(":", 1)                 # "2026", "15:04:05"
    hh, mm, ss = hms.split(":")
    return datetime(int(year), MONTHS[mon], int(d), int(hh), int(mm), int(ss))

timestamps = []
bad_lines = 0

with open(LOG_FILENAME, "r", errors="ignore") as f:
    for line in f:
        m = BRACKET_ISO_RE.search(line)
        if m:
            # keep timezone if present, then convert to naive local time (no tz) for resampling
            try:
                ts = pd.to_datetime(m.group(1), utc=False)  # keeps +04:00 offset if present
                ts = ts.tz_localize(None) if getattr(ts, "tzinfo", None) else ts
                timestamps.append(ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts)
            except Exception:
                bad_lines += 1
            continue

        m2 = APACHE_TS_RE.search(line)
        if m2:
            try:
                timestamps.append(parse_apache_ts(m2.group(1)))
            except Exception:
                bad_lines += 1
            continue

        bad_lines += 1

if len(timestamps) == 0:
    raise ValueError(
        "No timestamps parsed from the log file. "
        "Share 3–5 raw lines (including the timestamp part) and we’ll adjust parsing."
    )

print(f"[i] Parsed timestamps: {len(timestamps):,}")
print(f"[i] Unparsed lines (ignored): {bad_lines:,}")
print(f"[i] Time range: {min(timestamps)}  ->  {max(timestamps)}")


# -----------------------------
# 3) Build time series: requests per BIN
# -----------------------------
df = pd.DataFrame({"ts": pd.to_datetime(timestamps)})
df = df.set_index("ts").sort_index()

# Count requests per bin
series = df.resample(BIN).size().rename("req_count").to_frame()
series["t"] = (series.index - series.index[0]).total_seconds()

# If extremely sparse, fill missing bins
series["req_count"] = series["req_count"].fillna(0)

# -----------------------------
# 4) Regression baseline (robust)
# -----------------------------
# Use robust regression on polynomial time features to model smooth trend.
X = series[["t"]].values
y = series["req_count"].values

model = Pipeline([
    ("poly", PolynomialFeatures(degree=3, include_bias=False)),
    ("huber", HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=2000))
])

model.fit(X, y)
y_hat = model.predict(X)

series["baseline"] = y_hat
series["residual"] = series["req_count"] - series["baseline"]

# Robust z-score using MAD (median absolute deviation)
med = np.median(series["residual"])
mad = np.median(np.abs(series["residual"] - med)) + 1e-9
series["rz"] = (series["residual"] - med) / (1.4826 * mad)

# Flag suspicious bins
series["is_ddos_bin"] = series["rz"] >= Z_THRESHOLD

# -----------------------------
# 5) Convert flagged bins to contiguous time intervals
# -----------------------------
ddos_intervals = []
in_run = False
run_start = None
run_len = 0

idx = series.index.to_list()
flags = series["is_ddos_bin"].to_list()

for i, flag in enumerate(flags):
    if flag and not in_run:
        in_run = True
        run_start = idx[i]
        run_len = 1
    elif flag and in_run:
        run_len += 1
    elif (not flag) and in_run:
        # run ended at previous bin end
        if run_len >= MIN_CONSECUTIVE_BINS:
            run_end = idx[i]  # end boundary (current bin start)
            ddos_intervals.append((run_start, run_end))
        in_run = False
        run_start = None
        run_len = 0

# If ends with run
if in_run and run_len >= MIN_CONSECUTIVE_BINS:
    ddos_intervals.append((run_start, idx[-1] + pd.Timedelta(BIN)))

# Print intervals
print("\n=== Detected DDoS interval(s) ===")
if not ddos_intervals:
    print("No DDoS intervals detected with current thresholds.")
else:
    for a, b in ddos_intervals:
        print(f"- {a}  ->  {b}   (bin={BIN}, z>={Z_THRESHOLD}, min_bins={MIN_CONSECUTIVE_BINS})")

# -----------------------------
# 6) Visualizations (saved as PNGs for GitHub)
# -----------------------------
# Plot 1: traffic + baseline + highlight detected bins
plt.figure(figsize=(12, 5))
plt.plot(series.index, series["req_count"], label="Requests per bin")
plt.plot(series.index, series["baseline"], label="Regression baseline")
# Highlight ddos bins
ddos_points = series[series["is_ddos_bin"]]
plt.scatter(ddos_points.index, ddos_points["req_count"], label="Flagged bins", s=12)
plt.title(f"Traffic Volume with Regression Baseline (bin={BIN})")
plt.xlabel("Time")
plt.ylabel("Requests per bin")
plt.legend()
plt.tight_layout()
plt.savefig("traffic_baseline.png", dpi=200)
plt.show()

# Plot 2: residual robust z-score
plt.figure(figsize=(12, 4))
plt.plot(series.index, series["rz"], label="Robust z-score of residuals")
plt.axhline(Z_THRESHOLD, linestyle="--", label=f"Threshold z={Z_THRESHOLD}")
plt.title("Residuals (Robust z-score) for DDoS Detection")
plt.xlabel("Time")
plt.ylabel("Robust z-score")
plt.legend()
plt.tight_layout()
plt.savefig("residuals.png", dpi=200)
plt.show()

print("\nSaved plots: traffic_baseline.png, residuals.png")
