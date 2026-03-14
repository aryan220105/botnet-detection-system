# AI-Powered Botnet Detection System

> **Network Security Mini Project** — Final-Year Capstone  
> Real-time botnet traffic detection using Machine Learning and Network Traffic Analysis

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Architecture](#architecture)  
3. [Features](#features)  
4. [Tech Stack](#tech-stack)  
5. [Project Structure](#project-structure)  
6. [Installation](#installation)  
7. [How to Run](#how-to-run)  
8. [Module Details](#module-details)  
9. [Dashboard Screenshots](#dashboard-screenshots)  
10. [Future Improvements](#future-improvements)  
11. [References](#references)

---

## Project Overview

This system simulates a **Security Operations Center (SOC) monitoring tool** that:

- **Captures** live network packets (or simulates them for demo purposes)
- **Extracts** traffic features such as packet size, flow duration, DNS entropy, and connection frequency
- **Detects** botnet behaviour using two complementary ML approaches:
  - **Random Forest Classifier** (supervised, label-based)
  - **Isolation Forest** (unsupervised anomaly detection)
- **Classifies** each packet/flow as `NORMAL`, `SUSPICIOUS`, or `BOTNET`
- **Alerts** security analysts via a real-time web dashboard with interactive charts
- **Logs** all threats to persistent log files for forensic analysis

### Dataset

The models are trained on the **real-world CICIDS-2017 dataset** from the Canadian Institute for Cybersecurity (University of New Brunswick). The Friday capture file is used, which contains:

| Traffic Type | Samples |
|---|---|
| BENIGN | 288,544 |
| Portscan | 159,066 |
| DDoS | 95,144 |
| Botnet - Attempted | 4,067 |
| Botnet | 736 |

A balanced subset of 15,000 flows (7,500 normal + 7,500 attack) is extracted and used for training, achieving **99.89% accuracy** with Random Forest.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        NETWORK TRAFFIC                          │
│              (Live Capture via Scapy / Simulated)               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   Packet Capture       │
              │   Module               │
              │   (packet_capture/)    │
              └───────────┬────────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │   Feature Extraction   │
              │   Engine               │
              │   (feature_engineering/│)
              └───────────┬────────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │   Detection Engine     │◄──── Trained Models (.pkl)
              │   (detection_engine/)  │      ├─ Random Forest
              │                        │      └─ Isolation Forest
              │   Classification:      │
              │   NORMAL / SUSPICIOUS  │
              │   / BOTNET             │
              └──┬──────────┬──────────┘
                 │          │
        ┌────────┘          └────────┐
        ▼                            ▼
┌──────────────┐          ┌──────────────────┐
│  Alert &     │          │  Flask Dashboard  │
│  Log System  │          │  (dashboard/)     │
│  (utils/)    │          │                   │
│              │          │  • Live stats     │
│  alerts.log  │          │  • Threat table   │
│  alerts.json │          │  • Charts         │
└──────────────┘          │  • IP blacklist   │
                          └──────────────────┘
```

---

## Features

| Feature | Description |
|---|---|
| **Live Packet Capture** | Sniff real traffic with Scapy (requires admin) |
| **Simulated Traffic** | Built-in simulator for demos — no privileges needed |
| **ML Detection** | Random Forest (supervised) + Isolation Forest (anomaly) |
| **9 Traffic Features** | Packet size, flow duration, protocol, DNS entropy, etc. |
| **SOC Dashboard** | Dark-themed, real-time Flask web UI with Chart.js |
| **Alert System** | Structured alerts with threat level, attack type, timestamps |
| **IP Blacklist** | Auto-blacklists IPs that trigger CRITICAL alerts |
| **Geo-location** | IP geo-lookup via ip-api.com |
| **Persistent Logs** | Rotating log files + JSON alert archive |
| **Heuristic Fallback** | Rule-based detection layer active even without models |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Packet Capture | Scapy |
| Data Processing | pandas, NumPy |
| Machine Learning | scikit-learn (RandomForest, IsolationForest) |
| Model Persistence | joblib |
| Web Dashboard | Flask, Jinja2, Bootstrap 5, Chart.js |
| Visualization | matplotlib, seaborn (training plots) |
| Logging | Python `logging` (rotating file handlers) |

---

## Project Structure

```
botnet-detection-system/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── data/
│   ├── dataset_sample.csv             # Small sample dataset for quick testing
│   ├── cicids2017_processed.csv       # Real CICIDS-2017 dataset (15,000 flows)
│   ├── download_dataset.py            # Script to download & preprocess CICIDS-2017
│   └── ip_blacklist.json              # Auto-generated blacklist
│
├── models/
│   ├── train_model.py                 # Training pipeline
│   ├── random_forest_model.pkl        # Trained RF model (generated)
│   ├── isolation_forest_model.pkl     # Trained IF model (generated)
│   └── training_metrics.json          # Accuracy & F1 scores
│
├── packet_capture/
│   └── capture.py                     # Live + simulated packet capture
│
├── feature_engineering/
│   └── feature_extractor.py           # Feature extraction & flow aggregation
│
├── detection_engine/
│   └── detector.py                    # Core classification engine
│
├── dashboard/
│   ├── app.py                         # Flask application
│   └── templates/
│       └── index.html                 # SOC dashboard template
│
├── utils/
│   ├── logger.py                      # Centralized logging
│   └── ip_blacklist.py                # Blacklist & geo-location utilities
│
├── scripts/
│   └── run_system.py                  # One-click system launcher
│
└── logs/                              # Runtime logs (auto-created)
    ├── alerts.log
    ├── alerts.json
    ├── traffic.log
    └── system.log
```

---

## Installation

### Prerequisites

- **Python 3.10+** installed and on PATH
- **pip** package manager
- *(Optional)* Administrator/root privileges for live packet capture

### Steps

```bash
# 1. Navigate to the project directory
cd botnet-detection-system

# 2. Create a virtual environment (recommended)
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## How to Run

### Option A — One-Click Launcher (recommended)

```bash
python scripts/run_system.py
```

This will:
1. Train both ML models on the sample dataset (if not already trained)
2. Start the background traffic simulator
3. Launch the Flask dashboard at **http://127.0.0.1:5000**

Open your browser and navigate to `http://127.0.0.1:5000`.

### Option B — Step-by-Step

```bash
# Step 1: Download & preprocess the real CICIDS-2017 dataset (~200 MB download)
python data/download_dataset.py

# Step 2: Train the models on real data
python models/train_model.py --dataset data/cicids2017_processed.csv

# Step 3: (Optional) Test the detection engine standalone
python detection_engine/detector.py

# Step 4: Launch the dashboard
python dashboard/app.py
```

### Option C — Custom Port / Skip Training

```bash
# Skip training if models already exist
python scripts/run_system.py --skip-train

# Use a custom port
python scripts/run_system.py --port 8080

# Bind to all interfaces (e.g. for LAN access)
python scripts/run_system.py --host 0.0.0.0 --port 5000
```

### Testing the Packet Capture Module Standalone

```bash
# Simulated capture (no admin needed)
python packet_capture/capture.py --mode sim --count 100

# Live capture (requires admin/root)
python packet_capture/capture.py --mode live --count 50
```

---

## Module Details

### 1. Packet Capture (`packet_capture/capture.py`)

- **Live mode**: Uses Scapy's `sniff()` to capture real packets from a network interface
- **Simulated mode**: Generates realistic synthetic packets with configurable malicious-to-normal ratio
- Extracts per-packet fields: source/destination IP, ports, protocol, packet size, TTL, DNS queries

### 2. Feature Engineering (`feature_engineering/feature_extractor.py`)

Converts raw packets into 9 ML features:

| # | Feature | Description |
|---|---|---|
| 1 | `packet_size` | Size of the packet in bytes |
| 2 | `flow_duration` | Duration of the network flow |
| 3 | `protocol_type` | IP protocol number (6=TCP, 17=UDP) |
| 4 | `packets_per_flow` | Total packets in the flow |
| 5 | `bytes_per_flow` | Total bytes transferred |
| 6 | `connection_frequency` | Connections per second |
| 7 | `dns_entropy` | Shannon entropy of DNS query names |
| 8 | `src_port` | Source port number |
| 9 | `dst_port` | Destination port number |

### 3. Dataset Download (`data/download_dataset.py`)

- Downloads the **CICIDS-2017 Friday** capture (~200 MB) from HuggingFace
- Maps 80+ raw CICFlowMeter columns to our 9-feature schema
- Creates a balanced 15,000-row training dataset (7,500 normal + 7,500 attack)
- Supports custom sample sizes via `--samples` flag

### 4. Model Training (`models/train_model.py`)

- Loads the labelled CSV dataset (real CICIDS-2017 or sample)
- Normalizes features with `StandardScaler`
- Trains a **Random Forest** (100 estimators, max depth 20) — **99.89% accuracy**
- Trains an **Isolation Forest** (150 estimators, 15% contamination)
- Saves confusion matrix and feature importance plots
- Persists models to `.pkl` files via joblib

### 5. Detection Engine (`detection_engine/detector.py`)

Multi-layered classification:

1. **Blacklist check** — instant CRITICAL alert for known bad IPs
2. **Random Forest** — supervised prediction with probability scores
3. **Isolation Forest** — anomaly detection for zero-day patterns
4. **Heuristic rules** — DGA entropy checks, suspicious port detection

### 6. Dashboard (`dashboard/app.py`)

Flask web application with:
- **Stats cards**: total packets, normal, suspicious, botnet counts
- **Charts**: protocol distribution (doughnut), classification breakdown, threat timeline
- **Alert table**: real-time feed of detected threats
- **Suspicious IPs**: ranked by alert count
- **IP Blacklist**: visual display of banned IPs
- **Auto-refresh**: polls the API every 2 seconds

### 7. Alert & Logging (`utils/logger.py`)

- Rotating file handlers (5 MB per file, 5 backups)
- Separate log streams: `system.log`, `traffic.log`, `alerts.log`
- JSON alert archive (`alerts.json`) for structured querying

---

## Dashboard Screenshots

After running `python scripts/run_system.py`, open **http://127.0.0.1:5000** in your browser.

The dashboard features:

- **Dark-themed SOC interface** with real-time monitoring indicators
- **Four stat cards** showing live packet classification counts
- **Three interactive charts**: protocol distribution, classification breakdown, threat timeline
- **Scrollable alert table** with threat levels colour-coded (CRITICAL=red, HIGH=orange, MEDIUM=yellow)
- **Suspicious IP panel** ranking the most frequently flagged source addresses
- **Live traffic feed** showing every packet with classification and confidence score
- **IP Blacklist panel** displaying currently banned addresses

---

## Future Improvements

- [ ] **Deep Learning**: Replace Random Forest with LSTM or Transformer for sequence-based detection
- [ ] **Real PCAP ingestion**: Load `.pcap` files captured from tools like Wireshark or tcpdump
- [ ] **Email alerts**: SMTP integration for sending alert digests to SOC analysts
- [ ] **Attack heatmap**: Geographic visualization of threat sources on a world map
- [ ] **MITRE ATT&CK mapping**: Tag each detection with the relevant MITRE technique ID
- [ ] **Database backend**: Move from JSON logs to SQLite/PostgreSQL for production use
- [ ] **Docker deployment**: Containerize the full stack for one-command deployment
- [ ] **REST API authentication**: JWT-based auth for the dashboard API endpoints
- [ ] **Threat intelligence feeds**: Integrate with AbuseIPDB or VirusTotal APIs
- [ ] **Streamlit alternative UI**: Optional Streamlit-based dashboard for rapid prototyping

---

## References

1. **CICIDS 2017 Dataset** — Canadian Institute for Cybersecurity  
   https://www.unb.ca/cic/datasets/ids-2017.html

2. **CTU-13 Botnet Dataset** — Czech Technical University  
   https://www.stratosphereips.org/datasets-ctu13

3. **Scapy Documentation**  
   https://scapy.readthedocs.io/

4. **scikit-learn Documentation**  
   https://scikit-learn.org/stable/

5. **Flask Documentation**  
   https://flask.palletsprojects.com/

---

*Developed as a Final-Year Network Security Mini Project — 2026*
