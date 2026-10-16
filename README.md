# 🛡️ CampusNet Guardian
**Intelligent Network Monitoring & Anomaly Detection System**

> **Academic prototype.** This project uses synthetic/demo network traffic. It is intended for
> educational purposes and does not represent real-time monitoring of a production campus network.

## 1. Description
CampusNet Guardian is a Streamlit dashboard that analyzes campus network traffic records, uses an
**Isolation Forest** (scikit-learn) to label each record as `NORMAL` or `ANOMALOUS`, explains why
a record was flagged, and shows alerts, statistics and charts.

## 2. Features
- Automatic synthetic dataset generation (8,000 records, about 5% anomalous patterns)
- Data cleaning and validation (missing values, invalid ports, bad numbers, missing columns)
- Isolation Forest with anomaly score and NORMAL / ANOMALOUS status
- "Why was this flagged?" rule-based hints
- Alert levels: NORMAL, WARNING, HIGH (worded as *potential anomalies*)
- Dashboard with KPI cards and 6 Plotly charts
- Traffic Analysis page with filters and search
- **Live Monitoring simulation** (synthetic records, never real traffic)
- CSV upload with validation and downloadable analysis results
- Automatic model training if `models/anomaly_model.pkl` is missing

## 3. Technology stack
Python 3, Streamlit, Pandas, NumPy, scikit-learn, Plotly, Joblib

## 4. Architecture
```
Synthetic Network Data  (generate_data.py)
        ↓
Data Collection / Upload   (src/data_processing.py: load_dataset, read_csv_bytes)
        ↓
Data Cleaning & Processing (src/data_processing.py: clean_data)
        ↓
Feature Extraction         (src/data_processing.py: add_derived_features, build_feature_matrix)
        ↓
Isolation Forest           (src/anomaly_detection.py)
        ↓
Normal / Anomalous         (src/anomaly_detection.py: predict)
        ↓
Alert Generation           (src/alerts.py)
        ↓
Streamlit Dashboard        (app.py + src/charts.py)
```

## 5. Folder structure
```
CampusNet-Guardian/
├── app.py                  # Streamlit dashboard (all pages)
├── generate_data.py        # creates synthetic traffic CSV files
├── train_model.py          # trains and saves the model
├── requirements.txt
├── README.md
├── .streamlit/config.toml  # dark theme
├── data/                   # network_traffic.csv, sample_upload.csv
├── models/                 # anomaly_model.pkl
├── src/
│   ├── config.py           # paths and settings
│   ├── data_processing.py  # load, validate, clean, features
│   ├── anomaly_detection.py# train, predict, explain
│   ├── alerts.py           # alert levels and messages
│   └── charts.py           # Plotly charts
├── tests/test_project.py   # automated tests
└── docs/VIVA_GUIDE.md      # viva preparation
```

## 6. Installation (Windows)
```bat
mkdir CampusNet-Guardian
cd CampusNet-Guardian
:: copy all project files into this folder, then:
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
macOS/Linux: `source venv/bin/activate` instead of `venv\Scripts\activate`.

## 7. Generate data
```bash
python generate_data.py
```
Creates `data/network_traffic.csv` and `data/sample_upload.csv`. (Also done automatically if missing.)

## 8. Train the model
```bash
python train_model.py
```
Prints precision and recall on a held-out test set and saves `models/anomaly_model.pkl`.
(Also done automatically by the app if the model file is missing.)

## 9. Run the dashboard
```bash
streamlit run app.py
```
Open the link shown in the terminal (usually http://localhost:8501).

**VS Code:** open the project folder → `Ctrl+Shift+P` → *Python: Select Interpreter* → choose
the `venv` one → open a terminal (`Ctrl+``) → run the commands above.

## 10. How to use the dashboard
| Page | What to do |
|---|---|
| Dashboard | See KPI cards and 6 charts |
| Traffic Analysis | Filter by protocol, status, port, IP, date; view statistics |
| Anomaly Detection | See highest-risk records; pick one to see *Why was this flagged?* |
| Live Monitoring | Click **Start Monitoring Simulation**; adjust the unusual-traffic chance; click Stop |
| Upload Dataset | Upload a CSV (or download the sample first), view results, click **Download Analysis Results** |
| About Project | Problem statement, method, disclaimer |

## 11. How anomaly detection works
1. Each record gets 7 features: `packet_count`, `bytes_transferred`, `duration`, `source_port`,
   `destination_port`, plus derived `bytes_per_packet` and `packets_per_second`.
2. Skewed features are log-transformed, then standardized (`StandardScaler`).
3. **Isolation Forest** builds 200 random trees. Records that can be separated from the rest in
   very few splits get a high anomaly score.
4. `anomaly_score > 0` → `ANOMALOUS`, otherwise `NORMAL`. Flagged records with a score above the
   median flagged score get a HIGH alert, the others WARNING.
5. "Why flagged" hints compare the record with normal ranges learned from the training data
   (99th percentile of normal traffic, common ports, short connections). These are hints, not proof.

**No data leakage:** the model trains on 80% of the data only; the `label` column of the synthetic
data is never used as a feature and is used only to measure accuracy on the other 20%. The same
preprocessing code and the same saved scaler are used for training and prediction.

## 12. Testing
```bash
python -m unittest discover -s tests -v
```
Expected: `Ran 10 tests ... OK`. Manual checklist:

| Test | Expected result |
|---|---|
| `python generate_data.py` | Two CSV files created in `data/` |
| `python train_model.py` | Precision/recall printed (about 0.8–0.9), model file created |
| `streamlit run app.py` | Dashboard opens, 4 KPI cards, 6 charts, about 5% anomalies |
| Upload `data/sample_upload.csv` | Success message, results table, charts |
| Upload CSV missing a column | Red error naming the missing column |
| Upload an empty file | Friendly error, no crash |
| Download Analysis Results | CSV with `anomaly_score`, `status`, `why_flagged`, `alert_level` |
| Live Monitoring → Start | New record every 1.5 s, alerts change, Stop works |

## 13. Screenshots
_Add screenshots here: Dashboard, Anomaly Detection, Live Monitoring, Upload._

## 14. Limitations
- Uses synthetic data; real networks are much messier.
- Isolation Forest finds *unusual* traffic, not necessarily *malicious* traffic.
- "Why flagged" reasons are simple rules, not model internals.
- No real packet capture, no user/device identity, no long-term storage.
- Live Monitoring is a simulation.

## 15. Future scope
Use a public dataset (e.g. CIC-IDS), compare more models (One-Class SVM, Autoencoder), use SHAP
for explanations, add email alerts, store history in a database, import authorized flow logs.

## 16. Academic disclaimer
This project is an academic prototype that uses synthetic/demo network traffic. It is intended for
educational purposes and does not represent real-time monitoring of a production campus network.
