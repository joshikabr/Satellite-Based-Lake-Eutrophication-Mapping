#  TNPCB Water Quality Monitor
### Satellite-Based River Pollution Detection for the Cauvery Basin, Tamil Nadu

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Google%20Earth%20Engine-4285F4?style=for-the-badge&logo=google&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Sentinel--2-ESA-003247?style=for-the-badge"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Accuracy-87.7%25-00c896?style=flat-square"/>
  <img src="https://img.shields.io/badge/Cohen's%20κ-0.720-00aaff?style=flat-square"/>
  <img src="https://img.shields.io/badge/Trees-600-a855f7?style=flat-square"/>
  <img src="https://img.shields.io/badge/Features-35-ffb347?style=flat-square"/>
  <img src="https://img.shields.io/badge/Stations-20-ff3b5c?style=flat-square"/>
</p>

---

## Overview

A production-grade **water quality monitoring dashboard** that uses **Google Earth Engine Sentinel-2 satellite imagery** and a trained **Random Forest classifier** to detect river pollution across 20 monitoring stations in the Cauvery basin.

The system classifies each station as **Clean** or **Polluted** using 35 engineered spectral features derived from multispectral satellite bands, providing actionable insights for regulators, researchers, and industries operating along Tamil Nadu's most critical river system.

### Why Satellite-Based Monitoring?

| Traditional Method | This System |
|---|---|
| Manual field sampling — weekly/monthly | Sentinel-2 imagery — every 5 days |
| Covers 1–2 points per visit | Covers all 20 stations in one run |
| Delayed lab results (days) | Predictions in < 30 seconds |
| High operational cost | Near-zero cost (GEE free tier) |
| Limited spatial coverage | 10–60m spatial resolution |

---

## Features

### Single Station Prediction
- Select any of 20 Cauvery basin stations and a date
- Automatically fetches the nearest cloud-free Sentinel-2 image via GEE
- Returns Clean/Polluted classification with probability score
- Displays **Water Quality Score (0–100)** and **Grade (A–F)**
- **Water use safety flags** — Drinking / Irrigation / Aquaculture / Bathing
- **Pollution source inference** — identifies likely cause (algal bloom, industrial discharge, sediment)
- Spectral radar chart with clean baseline reference
- Station industrial context (industry type, risk level, population)

### Monthly Trend Analysis
- Run all 12 months for any station in a given year
- Dual-axis chart — P(Polluted) % and WQ Score on same timeline
- Seasonal index trends (NDCI, Turbidity, CDOM, BOD Proxy)
- Identifies best and worst months automatically
- Color-coded monthly detail table

### Multi-Station Comparison
- Compare 2–10 stations on the same date simultaneously
- Side-by-side pollution probability + WQ Score bar chart
- Index heatmap across all selected stations
- Interactive Leaflet map with color-coded station markers
- Grouped index comparison bar chart

### Data Dashboard
- Training data balance donut chart
- Historical pollution rate heatmap (Station × Month)
- Monthly and per-station pollution rate charts
- Seasonal NDCI box plots
- **Top 15 Model Feature Importances** from the Random Forest
- Model performance summary

### Prediction History
- Full session history with timeline scatter plot
- Map view of all predicted stations in the session
- Export-ready data table

---

## Satellite Data

| Property | Value |
|---|---|
| Satellite | Sentinel-2A / 2B (ESA Copernicus) |
| Product | `COPERNICUS/S2_SR_HARMONIZED` (Surface Reflectance) |
| Bands Used | B2 (Blue), B3 (Green), B4 (Red), B5 (Red Edge), B8 (NIR) |
| Spatial Resolution | 10–60 m |
| Temporal Resolution | 5-day revisit cycle |
| Data Access | Google Earth Engine API |
| Cloud Filter | < 20% cloud pixel percentage |
| Window | ±15 days around selected date |

---

## Model Architecture

### Random Forest Classifier
```
Estimators  : 600 trees
Max Features : sqrt
CV Strategy  : 5-fold stratified cross-validation
Threshold    : 0.49 (tuned for precision-recall balance)
```

### Performance
| Metric | Value |
|---|---|
| Overall Accuracy (OA) | **87.7%** |
| Cohen's Kappa (κ) | **0.720** |
| Classes | Binary (1 = Clean, 2 = Polluted) |

### Feature Engineering (35 Features)

**Raw Bands (5)**
```
B2, B3, B4, B5, B8
```

**Spectral Indices (15)**
```
NDWI    — Normalized Difference Water Index       (water clarity)
NDCI    — Normalized Difference Chlorophyll Index (algae/eutrophication)
SABI    — Surface Algal Bloom Index
NDRE    — Normalized Difference Red Edge          (vegetation stress)
FAI     — Floating Algae Index
WDRVI   — Wide Dynamic Range Vegetation Index
NDRB    — Normalized Difference Red-Blue
TI      — Turbidity Index
CDOM    — Coloured Dissolved Organic Matter proxy
BOD     — Biochemical Oxygen Demand proxy
turbidity, B3_B4, B5_B4, B8_B4, B2_B3
```

**Temporal Features (2)**
```
sin_month, cos_month   — Cyclic encoding of seasonality
```

**Station Z-Score Features (8)**
```
B3_z, B4_z, B8_z, NDWI_z, NDCI_z, turbidity_z, CDOM_z, BOD_proxy_z
```

**Interaction Features (5)**
```
FAI_norm, high_FAI, NDCI_sq, NDWI_NDCI, CDOM_NDWI
```

---

## Monitoring Stations (20)

| Station | Lat | Lon | Industrial Context |
|---|---|---|---|
| Mettur | 11.7858 | 77.8003 | Chemical & Fertilizer Plants |
| Erode | 11.3400 | 77.7200 | Textile & Dyeing Units |
| Bhavani | 11.4466 | 77.6825 | Textile Mills |
| Trichy | 10.8594 | 78.7264 | Mixed Industrial |
| Kumbakonam | 10.9250 | 79.3800 | Agro-processing |
| Thanjavur | 10.8620 | 79.0980 | Agricultural Runoff |
| Komarapalayam | 11.4400 | 77.8450 | Dyeing & Bleaching |
| Pitchavaram | 11.4484 | 79.7222 | Aquaculture / Mangroves |
| Tirunelveli | 8.5595 | 77.5734 | Paper & Pulp Mills |
| Napier Bridge | 13.0658 | 80.2868 | Urban Sewage (Chennai) |
| Musiri | 10.9576 | 78.5592 | — |
| Mayiladuthurai | 11.1137 | 79.6104 | — |
| Mohanur | 11.0763 | 78.1411 | — |
| Pugalur | 11.0925 | 77.9827 | — |
| Bhavani Sagar | 11.4145 | 77.7587 | — |
| Grand Anaicut | 10.8594 | 78.7264 | — |
| Coleroon | 10.8686 | 78.7033 | — |
| Pettaivaithalai | 10.9197 | 78.8499 | — |
| Saidapet | 13.0200 | 80.2200 | — |
| Kotturpuram | 13.0263 | 80.2427 | — |

---

## Installation & Setup

### Prerequisites
- Python 3.9+
- A Google Earth Engine account ([sign up free](https://earthengine.google.com/))
- The trained model file (`tnpcb_wq_model_v2.pkl` or `tnpcb_wq_model.pkl`)

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/tnpcb-water-quality.git
cd tnpcb-water-quality
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install earthengine-api streamlit joblib scikit-learn==1.6.1 pandas numpy plotly scipy
```

Or using the requirements file:
```bash
pip install -r requirements.txt
```

### 4. Authenticate Google Earth Engine
```bash
earthengine authenticate
```
This opens a browser window. Sign in with your Google account that has GEE access approved.

### 5. Place Model File
Place your trained model file in the project root:
```
tnpcb-water-quality/
 app.py
 tnpcb_wq_model_v2.pkl     ← your model here
 tnpcb_extracted_features.csv   ← optional, enables Dashboard
 README.md
```

### 6. Run the App
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## Project Structure

```
tnpcb-water-quality/

 app.py                          # Main Streamlit application
 requirements.txt                # Python dependencies
 README.md                       # This file

 tnpcb_wq_model_v2.pkl           # Trained Random Forest model (required)
 tnpcb_extracted_features.csv    # Reference dataset (optional)

 assets/                         # Screenshots / demo images (optional)
     screenshot.png
```

---

## Requirements

```txt
earthengine-api>=0.1.370
streamlit>=1.28.0
joblib>=1.3.0
scikit-learn==1.6.1
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.17.0
scipy>=1.11.0
```

>  **Important:** `scikit-learn==1.6.1` is pinned. Loading a `.pkl` model requires the **exact same version** used during training.

---

## Configuration

### Model Files
The app looks for model files in this priority order:
1. `tnpcb_wq_model_v2.pkl`
2. `tnpcb_wq_model.pkl`

The model file must be a `joblib`-serialized dict with keys:
```python
{
    'model'       : sklearn_estimator,   # trained RandomForestClassifier
    'feature_cols': list_of_35_features,
    'threshold'   : 0.49,                # classification threshold
    'oa'          : 87.7,                # overall accuracy %
    'kappa'       : 0.720                # Cohen's kappa
}
```

### Reference CSV (Optional)
`tnpcb_extracted_features.csv` enables the **Dashboard** and **historical context** features. Expected columns:
```
Station (or Name), Month, B2, B3, B4, B5, B8, Binary_True (or True_Class)
```
Where `Binary_True`: `1` = Clean, `2` = Polluted.

---

## Water Quality Score Explained

Beyond the binary Clean/Polluted label, the app computes a **composite WQ Score (0–100)**:

```
Base Score    = 100 - P(Polluted)%
NDCI Penalty  = max(0, (NDCI  - 0.10) × 80)
Turb Penalty  = max(0, (Turb  - 0.80) × 30)
CDOM Penalty  = max(0, (CDOM  - 1.50) × 20)
NDWI Bonus    = max(0,  NDWI          × 10)

WQ Score = Base - Penalties + Bonus   [clamped 0–100]
```

| Grade | Score | Meaning |
|---|---|---|
| **A** | 80–100 | Excellent |
| **B** | 60–79 | Good |
| **C** | 40–59 | Moderate |
| **D** | 20–39 | Poor |
| **F** | 0–19 | Critical |

---

## Limitations

- **Cloud cover** — Sentinel-2 is optical; cloudy images return no data (monsoon months affected)
- **Revisit cycle** — Data is 2–5 days old at best; not truly real-time
- **Spatial resolution** — 60m buffer per station; may miss narrow pollution plumes
- **Binary classification** — Model predicts Clean vs Polluted; not continuous WQI values
- **Training data scope** — Model trained on Cauvery basin stations; may need retraining for other river systems

---

## Roadmap

- [ ] **Live Monitor page** — Auto-fetch latest Sentinel-2 image for all stations
- [ ] **Email/SMS alerts** — Notify when stations exceed pollution threshold
- [ ] **Multi-class WQI** — Extend beyond binary to 4–5 quality classes
- [ ] **Streamlit Cloud deployment** — One-click public deployment
- [ ] **IoT sensor fusion** — Combine satellite with TNPCB ground sensor APIs
- [ ] **Cauvery Delta extension** — Add downstream delta stations
- [ ] **PDF report export** — Auto-generate compliance reports

---

## Spectral Indices Reference

| Index | Formula | Pollution Indicator |
|---|---|---|
| NDWI | (B3−B8)/(B3+B8) | Water clarity — lower = murkier |
| NDCI | (B5−B4)/(B5+B4) | Algae / chlorophyll concentration |
| SABI | (B8−B3)/(B2+B4) | Surface algal bloom |
| Turbidity | B4/B3 | Suspended sediment load |
| CDOM | B3/B4 | Dissolved organic matter (dyes, sewage) |
| BOD proxy | B3/B8 | Biochemical oxygen demand indicator |
| FAI | B8−(B4+(B5−B4)×0.5) | Floating algae / foam detection |
| NDRE | (B5−B4)/(B5+B4) | Red-edge vegetation stress |

---

## Contributing

Contributions are welcome! Please open an issue first to discuss what you'd like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/live-monitor`)
3. Commit your changes (`git commit -m 'Add live monitor page'`)
4. Push to the branch (`git push origin feature/live-monitor`)
5. Open a Pull Request

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- **European Space Agency (ESA)** — Sentinel-2 satellite constellation
- **Google Earth Engine** — Cloud-based geospatial analysis platform
- **Tamil Nadu Pollution Control Board (TNPCB)** — Ground truth water quality data
- **Copernicus Programme** — Open and free Earth observation data policy

---

## Contact

For questions, collaborations, or deployment support:

- **GitHub Issues** — Bug reports and feature requests
- **Pull Requests** — Code contributions

---

<p align="center">
  Made for cleaner rivers in Tamil Nadu
</p>

<p align="center">
  <sub>Sentinel-2 · Random Forest · Google Earth Engine · Streamlit · Cauvery Basin</sub>
</p>
