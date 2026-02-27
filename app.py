# ── REQUIREMENTS ──────────────────────────────────────────────────────────────
# pip install earthengine-api streamlit joblib scikit-learn==1.6.1
#             pandas numpy plotly scipy

import ee
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import json

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
FEATURE_COLS_SAFE = [
    'B2','B3','B4','B5','B8',
    'NDWI','NDCI','SABI','turbidity','B3_B4',
    'FAI','NDRE','WDRVI','TI','NDRB',
    'CDOM','BOD_proxy','B5_B4','B8_B4','B2_B3',
    'sin_month','cos_month',
    'FAI_norm','high_FAI','NDCI_sq','NDWI_NDCI','CDOM_NDWI',
    'B3_z','B4_z','B8_z','NDWI_z','NDCI_z',
    'turbidity_z','CDOM_z','BOD_proxy_z'
]

RIVER_STATIONS = {
    'Mettur'          : (11.7858, 77.8003),
    'Musiri'          : (10.9576, 78.5592),
    'Trichy'          : (10.8594, 78.7264),
    'Erode'           : (11.3400, 77.7200),
    'Bhavani'         : (11.4466, 77.6825),
    'Kumbakonam'      : (10.9250, 79.3800),
    'Thanjavur'       : (10.8620, 79.0980),
    'Mayiladuthurai'  : (11.1137, 79.6104),
    'Mohanur'         : (11.0763, 78.1411),
    'Komarapalayam'   : (11.4400, 77.8450),
    'Pettaivaithalai' : (10.9197, 78.8499),
    'Pugalur'         : (11.0925, 77.9827),
    'Bhavani Sagar'   : (11.4145, 77.7587),
    'Grand Anaicut'   : (10.8594, 78.7264),
    'Coleroon'        : (10.8686, 78.7033),
    'Pitchavaram'     : (11.4484, 79.7222),
    'Tirunelveli'     : (8.5595,  77.5734),
    'Napier Bridge'   : (13.0658, 80.2868),
    'Saidapet'        : (13.0200, 80.2200),
    'Kotturpuram'     : (13.0263, 80.2427),
}

# Industrial context for each station
STATION_CONTEXT = {
    'Mettur'         : {'industry': 'Chemical & Fertilizer Plants', 'risk': 'High', 'pop': '52,000'},
    'Erode'          : {'industry': 'Textile & Dyeing Units',       'risk': 'High', 'pop': '2,10,000'},
    'Bhavani'        : {'industry': 'Textile Mills',                'risk': 'High', 'pop': '80,000'},
    'Trichy'         : {'industry': 'Mixed Industrial',             'risk': 'Medium','pop': '9,20,000'},
    'Kumbakonam'     : {'industry': 'Agro-processing',              'risk': 'Low',  'pop': '1,40,000'},
    'Thanjavur'      : {'industry': 'Agricultural Runoff',          'risk': 'Low',  'pop': '2,20,000'},
    'Komarapalayam'  : {'industry': 'Dyeing & Bleaching',           'risk': 'High', 'pop': '75,000'},
    'Pitchavaram'    : {'industry': 'Aquaculture / Mangroves',      'risk': 'Low',  'pop': '5,000'},
    'Tirunelveli'    : {'industry': 'Paper & Pulp Mills',           'risk': 'Medium','pop': '4,74,000'},
    'Napier Bridge'  : {'industry': 'Urban Sewage (Chennai)',        'risk': 'High', 'pop': '70,00,000'},
}

MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']

# Pollution thresholds with WQI labels
WQI_THRESHOLDS = {
    'NDCI'      : {'clean': (-0.1, 0.1),  'unit': '',     'label': 'Chlorophyll/Algae Index'},
    'turbidity' : {'clean': (0.0,  0.8),  'unit': '',     'label': 'Turbidity Ratio'},
    'CDOM'      : {'clean': (0.0,  1.5),  'unit': '',     'label': 'Dissolved Organics (CDOM)'},
    'NDWI'      : {'clean': (0.0,  1.0),  'unit': '',     'label': 'Water Clarity Index'},
}

CLR_CLEAN    = '#00e5a0'
CLR_POLLUTED = '#ff3b5c'
CLR_WARN     = '#ffb347'
CLR_ACCENT   = '#00aaff'
CLR_ACCENT2  = '#a855f7'
CLR_BG       = '#060d17'
CLR_CARD     = '#0d1825'
CLR_CARD2    = '#111f30'
CLR_BORDER   = '#1e3048'
CLR_TEXT     = '#e2f0ff'
CLR_MUTED    = '#5a7a9a'

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
def add_gee_indices(df):
    df = df.copy()
    df['NDWI']      = (df['B3']-df['B8'])  / (df['B3']+df['B8']  +1e-6)
    df['NDCI']      = (df['B5']-df['B4'])  / (df['B5']+df['B4']  +1e-6)
    df['SABI']      = (df['B8']-df['B3'])  / (df['B2']+df['B4']  +1e-6)
    df['turbidity'] = df['B4'] / (df['B3']+1e-6)
    df['B3_B4']     = df['B3'] / (df['B4']+1e-6)
    return df

def build_spectral_features(df):
    df = df.copy()
    df['FAI']       = df['B8']-(df['B4']+(df['B5']-df['B4'])*0.5)
    df['NDRE']      = (df['B5']-df['B4'])  / (df['B5']+df['B4']  +1e-6)
    df['WDRVI']     = (0.1*df['B8']-df['B4']) / (0.1*df['B8']+df['B4']+1e-6)
    df['TI']        = df['B4'] / (df['B2']+1e-6)
    df['NDRB']      = (df['B4']-df['B2'])  / (df['B4']+df['B2']  +1e-6)
    df['CDOM']      = df['B3'] / (df['B4']+1e-6)
    df['BOD_proxy'] = df['B3'] / (df['B8']+1e-6)
    df['B5_B4']     = df['B5'] / (df['B4']+1e-6)
    df['B8_B4']     = df['B8'] / (df['B4']+1e-6)
    df['B2_B3']     = df['B2'] / (df['B3']+1e-6)
    df['sin_month'] = np.sin(2*np.pi*df['Month']/12)
    df['cos_month'] = np.cos(2*np.pi*df['Month']/12)
    return df

def build_station_zscores(df, reference_df=None):
    df = df.copy()
    station = df['Station'].iloc[0]
    if reference_df is not None:
        fai_mean = reference_df['FAI'].mean()
        fai_std  = reference_df['FAI'].std()+1e-6
        fai_q75  = reference_df['FAI'].quantile(0.75)
    else:
        fai_mean = df['FAI'].mean()
        fai_std  = df['FAI'].std()+1e-6
        fai_q75  = df['FAI'].quantile(0.75)
    df['FAI_norm']  = (df['FAI']-fai_mean)/fai_std
    df['high_FAI']  = (df['FAI']>fai_q75).astype(int)
    df['NDCI_sq']   = df['NDCI']**2
    df['NDWI_NDCI'] = df['NDWI']*df['NDCI']
    df['CDOM_NDWI'] = df['CDOM']*df['NDWI']
    for col in ['B3','B4','B8','NDWI','NDCI','turbidity','CDOM','BOD_proxy']:
        mu, sig = 0.0, 1.0
        if reference_df is not None and station in reference_df['Station'].values:
            s = reference_df[reference_df['Station']==station][col].dropna()
            if len(s)>1:
                mu  = s.mean()
                sig = s.std() if s.std()>0 else 1.0
        df[f'{col}_z'] = (df[col]-mu)/(sig+1e-6)
    return df

def prepare_features(raw_dict, reference_df=None):
    df = pd.DataFrame([raw_dict])
    df = add_gee_indices(df)
    df = build_spectral_features(df)
    df = build_station_zscores(df, reference_df)
    return df

# ══════════════════════════════════════════════════════════════════════════════
# WATER QUALITY SCORE ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def compute_wq_score(prob_polluted, indices):
    """Convert model probability + indices into a 0-100 water quality score."""
    base = 100 - prob_polluted

    # Penalty from individual indices
    ndci_penalty  = max(0, (indices['NDCI'] - 0.1) * 80)
    turb_penalty  = max(0, (indices['turbidity'] - 0.8) * 30)
    cdom_penalty  = max(0, (indices['CDOM'] - 1.5) * 20)
    ndwi_bonus    = max(0, indices['NDWI'] * 10)

    score = base - ndci_penalty - turb_penalty - cdom_penalty + ndwi_bonus
    return max(0, min(100, score))

def wq_grade(score):
    if score >= 80: return 'A', 'Excellent', CLR_CLEAN
    if score >= 60: return 'B', 'Good',      '#7fff7f'
    if score >= 40: return 'C', 'Moderate',  CLR_WARN
    if score >= 20: return 'D', 'Poor',      '#ff8c42'
    return 'F', 'Critical', CLR_POLLUTED

def pollution_source_analysis(indices, prob):
    """Infer likely pollution sources from spectral indices."""
    sources = []
    if indices['NDCI'] > 0.15:
        sources.append({'source': '🌿 Algal Bloom / Eutrophication',
                        'confidence': min(99, int(indices['NDCI']*300)),
                        'cause': 'Excess nutrients (N, P) from agricultural runoff or sewage'})
    if indices['turbidity'] > 1.2:
        sources.append({'source': '🟤 High Suspended Sediments',
                        'confidence': min(99, int((indices['turbidity']-0.8)*80)),
                        'cause': 'Industrial discharge, construction, or heavy rainfall runoff'})
    if indices['CDOM'] > 2.0:
        sources.append({'source': '🏭 Industrial Organic Discharge',
                        'confidence': min(99, int((indices['CDOM']-1.5)*60)),
                        'cause': 'Textile dye effluents, paper mills, or tanneries'})
    if prob > 70 and not sources:
        sources.append({'source': '⚠️ Mixed Pollution',
                        'confidence': int(prob),
                        'cause': 'Multiple overlapping pollution sources detected'})
    return sources

def safe_water_assessment(score, station):
    """Generate drinking/irrigation/aquaculture safety flags."""
    ctx = STATION_CONTEXT.get(station, {})
    flags = {
        'drinking'    : score >= 75,
        'irrigation'  : score >= 45,
        'aquaculture' : score >= 55,
        'bathing'     : score >= 60,
    }
    return flags

# ══════════════════════════════════════════════════════════════════════════════
# CACHED RESOURCES
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def init_gee():
    try:
        ee.Initialize()
    except Exception:
        try:
            ee.Authenticate()
            ee.Initialize()
        except Exception as e:
            st.error(f"❌ GEE init failed: {e}")
            st.stop()

@st.cache_resource
def load_model():
    for fname in ['tnpcb_wq_model_v2.pkl','tnpcb_wq_model.pkl']:
        if os.path.exists(fname):
            raw = joblib.load(fname)
            if not isinstance(raw, dict):
                return {'model':raw,'feature_cols':FEATURE_COLS_SAFE,
                        'threshold':0.49,'oa':87.7,'kappa':0.720}
            return raw
    st.error("❌ Model file not found.")
    st.stop()

@st.cache_data
def load_reference_data():
    if os.path.exists('tnpcb_extracted_features.csv'):
        df = pd.read_csv('tnpcb_extracted_features.csv')
        if 'Name' in df.columns and 'Station' not in df.columns:
            df['Station'] = df['Name']
        if 'True_Class' in df.columns and 'Binary_True' not in df.columns:
            df['Binary_True'] = df['True_Class']
        df = add_gee_indices(df)
        df = build_spectral_features(df)
        return df
    return None

# ══════════════════════════════════════════════════════════════════════════════
# GEE FETCH
# ══════════════════════════════════════════════════════════════════════════════
def fetch_sentinel2(lat, lon, date_str, station_name, cloud_pct=20, window=15):
    date  = datetime.strptime(date_str,'%Y-%m-%d')
    start = (date-timedelta(days=window)).strftime('%Y-%m-%d')
    end   = (date+timedelta(days=window)).strftime('%Y-%m-%d')
    point  = ee.Geometry.Point([lon,lat])
    buffer = point.buffer(100)
    col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
             .filterBounds(point).filterDate(start,end)
             .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',cloud_pct))
             .sort('CLOUDY_PIXEL_PERCENTAGE'))
    if col.size().getInfo()==0:
        col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                 .filterBounds(point).filterDate(start,end)
                 .sort('CLOUDY_PIXEL_PERCENTAGE'))
        if col.size().getInfo()==0:
            return None, f"No image for {date_str} ±{window}d"
    img      = col.first()
    img_date = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd').getInfo()
    vals = img.select(['B2','B3','B4','B5','B8']).reduceRegion(
        reducer=ee.Reducer.mean(), geometry=buffer, scale=60, maxPixels=1e6
    ).getInfo()
    if not vals or not vals.get('B3'):
        return None, "Could not extract band values"
    return {'Station':station_name,'Month':date.month,
            'B2':vals.get('B2',np.nan),'B3':vals.get('B3',np.nan),
            'B4':vals.get('B4',np.nan),'B5':vals.get('B5',np.nan),
            'B8':vals.get('B8',np.nan),'image_date':img_date}, None

# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
def predict_single(station, date_str, bundle, ref_df):
    lat, lon = RIVER_STATIONS[station]
    raw, err = fetch_sentinel2(lat, lon, date_str, station)
    if err:
        return {'error': err}
    feat_df   = prepare_features(raw, ref_df)
    model     = bundle['model']
    threshold = bundle.get('threshold', 0.49)
    feat_cols = bundle.get('feature_cols', FEATURE_COLS_SAFE)
    missing   = [c for c in feat_cols if c not in feat_df.columns]
    if missing:
        return {'error': f"Missing features: {missing}"}
    proba      = model.predict_proba(feat_df[feat_cols].values)[0,1]
    pred       = int(proba>=threshold)
    label      = 'Polluted' if pred==1 else 'Clean'
    confidence = proba*100 if pred==1 else (1-proba)*100
    indices = {
        'NDWI'     : round(float(feat_df['NDWI'].iloc[0]),4),
        'NDCI'     : round(float(feat_df['NDCI'].iloc[0]),4),
        'turbidity': round(float(feat_df['turbidity'].iloc[0]),4),
        'CDOM'     : round(float(feat_df['CDOM'].iloc[0]),4),
        'FAI'      : round(float(feat_df['FAI'].iloc[0]),4),
        'NDRE'     : round(float(feat_df['NDRE'].iloc[0]),4),
        'BOD_proxy': round(float(feat_df['BOD_proxy'].iloc[0]),4),
        'WDRVI'    : round(float(feat_df['WDRVI'].iloc[0]),4),
    }
    wq_score = compute_wq_score(proba*100, indices)
    grade, grade_label, grade_color = wq_grade(wq_score)
    return {
        'station'      : station,
        'query_date'   : date_str,
        'image_date'   : raw['image_date'],
        'result'       : label,
        'pred'         : pred,
        'confidence'   : round(confidence,1),
        'prob_polluted': round(proba*100,1),
        'wq_score'     : round(wq_score,1),
        'grade'        : grade,
        'grade_label'  : grade_label,
        'grade_color'  : grade_color,
        'bands'        : {k:round(float(raw[k]),2) for k in ['B2','B3','B4','B5','B8']},
        'indices'      : indices,
        'sources'      : pollution_source_analysis(indices, proba*100),
        'safety'       : safe_water_assessment(wq_score, station),
        'lat'          : lat,
        'lon'          : lon,
    }

# ══════════════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def dark_layout(fig, title='', height=400):
    fig.update_layout(
        title         = dict(text=title, font=dict(color=CLR_TEXT, size=15, family='Sora')),
        paper_bgcolor = CLR_BG,
        plot_bgcolor  = CLR_CARD,
        font          = dict(color=CLR_MUTED, family='Sora'),
        height        = height,
        margin        = dict(l=50,r=30,t=55,b=45),
        legend        = dict(bgcolor=CLR_CARD2, bordercolor=CLR_BORDER, borderwidth=1,
                             font=dict(size=12)),
        xaxis         = dict(gridcolor=CLR_BORDER, zerolinecolor=CLR_BORDER,
                             tickfont=dict(size=12)),
        yaxis         = dict(gridcolor=CLR_BORDER, zerolinecolor=CLR_BORDER,
                             tickfont=dict(size=12)),
    )
    return fig

def wq_score_gauge(score, grade, grade_label, grade_color):
    """Large WQ Score gauge — 0 to 100."""
    fig = go.Figure(go.Indicator(
        mode   = "gauge+number",
        value  = score,
        domain = {'x': [0,1], 'y': [0.15, 1]},
        title  = {'text': f"Water Quality Score<br><span style='font-size:13px;color:{grade_color}'>"
                          f"Grade {grade} — {grade_label}</span>",
                  'font': {'color': CLR_TEXT, 'size': 15}},
        number = {'suffix': '/100', 'font': {'color': grade_color, 'size': 48},
                  'valueformat': '.0f'},
        gauge  = {
            'axis'       : {'range': [0,100], 'tickwidth': 1, 'tickcolor': CLR_MUTED,
                            'tickfont': {'size': 13}, 'nticks': 6},
            'bar'        : {'color': grade_color, 'thickness': 0.30},
            'bgcolor'    : CLR_CARD,
            'bordercolor': CLR_BORDER,
            'borderwidth': 2,
            'steps': [
                {'range': [0, 20],  'color': '#1a0a10'},
                {'range': [20, 40], 'color': '#1a140a'},
                {'range': [40, 60], 'color': '#141a0a'},
                {'range': [60, 80], 'color': '#0a1a14'},
                {'range': [80,100], 'color': '#0a1a18'},
            ],
            'threshold': {'line': {'color': 'white', 'width': 3}, 'value': 50}
        }
    ))
    fig.update_layout(
        paper_bgcolor=CLR_BG, font_color=CLR_MUTED,
        height=340, margin=dict(l=40, r=40, t=30, b=80)
    )
    return fig

def prob_gauge(prob_polluted):
    color = CLR_POLLUTED if prob_polluted > 49 else CLR_CLEAN
    fig = go.Figure(go.Indicator(
        mode   = "gauge+number",
        value  = prob_polluted,
        domain = {'x': [0,1], 'y': [0.15, 1]},
        title  = {'text': "Pollution Probability %", 'font': {'color': CLR_TEXT, 'size': 15}},
        number = {'suffix': '%', 'font': {'color': color, 'size': 48},
                  'valueformat': '.1f'},
        gauge  = {
            'axis'       : {'range': [0,100], 'tickcolor': CLR_MUTED,
                            'tickfont': {'size': 13}, 'nticks': 6},
            'bar'        : {'color': color, 'thickness': 0.30},
            'bgcolor'    : CLR_CARD,
            'bordercolor': CLR_BORDER,
            'steps': [
                {'range': [0,40],   'color': '#0a1a14'},
                {'range': [40,60],  'color': '#1a1a0a'},
                {'range': [60,100], 'color': '#1a0a0a'},
            ],
            'threshold': {'line': {'color': 'white', 'width': 3}, 'value': 50}
        }
    ))
    fig.update_layout(
        paper_bgcolor=CLR_BG, font_color=CLR_MUTED,
        height=340, margin=dict(l=40, r=40, t=30, b=80)
    )
    return fig

def radar_chart(indices_dict, station):
    cats   = ['NDWI','NDCI','Turbidity','CDOM','FAI','BOD']
    vals   = [indices_dict['NDWI'], indices_dict['NDCI'],
               indices_dict['turbidity'], indices_dict['CDOM'],
               indices_dict['FAI'], indices_dict['BOD_proxy']]
    ranges = [(-1,1),(-1,1),(0,3),(0,3),(-500,500),(0,3)]
    norm   = [max(0,min(1,(v-lo)/(hi-lo+1e-6))) for v,(lo,hi) in zip(vals,ranges)]
    norm  += [norm[0]]
    cats  += [cats[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=norm, theta=cats, fill='toself',
        fillcolor='rgba(0,170,255,0.12)',
        line=dict(color=CLR_ACCENT, width=2.5),
        name='Current'
    ))
    # Reference "clean" ring
    clean_ref = [0.3]*len(norm)
    fig.add_trace(go.Scatterpolar(
        r=clean_ref, theta=cats, fill='toself',
        fillcolor='rgba(0,229,160,0.05)',
        line=dict(color=CLR_CLEAN, width=1.5, dash='dot'),
        name='Clean Baseline'
    ))
    fig.update_layout(
        polar=dict(
            bgcolor=CLR_CARD,
            radialaxis=dict(visible=True, range=[0,1], gridcolor=CLR_BORDER,
                            tickfont=dict(color=CLR_MUTED, size=12)),
            angularaxis=dict(gridcolor=CLR_BORDER, tickfont=dict(color=CLR_TEXT, size=15))
        ),
        paper_bgcolor=CLR_BG, font_color=CLR_MUTED,
        legend=dict(bgcolor=CLR_CARD2, font=dict(size=13)),
        title=dict(text='Spectral Index Profile', font=dict(color=CLR_TEXT, size=15)),
        height=360, margin=dict(l=60,r=60,t=65,b=30)
    )
    return fig

def monthly_trend_chart(monthly_results, station):
    months = [r['month_num'] for r in monthly_results]
    probs  = [r['prob_polluted'] for r in monthly_results]
    scores = [r.get('wq_score', 50) for r in monthly_results]
    labels = [r['result'] for r in monthly_results]
    colors = [CLR_POLLUTED if l=='Polluted' else CLR_CLEAN for l in labels]

    fig = go.Figure()
    # Shaded risk band
    fig.add_hrect(y0=50, y1=100, fillcolor='rgba(255,59,92,0.04)',
                  line_width=0, annotation_text="Polluted Zone",
                  annotation_position="top right",
                  annotation_font_color=CLR_MUTED)
    fig.add_hrect(y0=0, y1=50, fillcolor='rgba(0,229,160,0.04)',
                  line_width=0, annotation_text="Clean Zone",
                  annotation_position="bottom right",
                  annotation_font_color=CLR_MUTED)
    # Smooth trend line
    fig.add_trace(go.Scatter(
        x=months, y=probs, mode='lines',
        line=dict(color='rgba(0,170,255,0.4)', width=2.5, dash='dot'),
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=months, y=probs, mode='markers',
        marker=dict(color=colors, size=16, line=dict(color='white', width=2)),
        text=[f"<b>{MONTH_NAMES[m-1]}</b><br>P(Polluted): {p:.1f}%<br>WQ Score: {s:.0f}/100<br>Status: {l}"
              for m,p,s,l in zip(months,probs,scores,labels)],
        hovertemplate='%{text}<extra></extra>',
        showlegend=False
    ))
    # WQ Score secondary line
    fig.add_trace(go.Scatter(
        x=months, y=scores, mode='lines+markers',
        name='WQ Score',
        line=dict(color=CLR_ACCENT2, width=2),
        marker=dict(size=8, symbol='diamond'),
        yaxis='y2',
        hovertemplate='WQ Score: %{y:.1f}<extra></extra>'
    ))
    fig.add_hline(y=50, line_dash='dash', line_color='rgba(255,255,255,0.2)',
                  annotation_text='Decision Boundary',
                  annotation_font_color=CLR_MUTED)
    fig.update_xaxes(tickvals=list(range(1,13)), ticktext=MONTH_NAMES, tickfont=dict(size=13))
    fig.update_yaxes(range=[0,105], title_text='P(Polluted) %', tickfont=dict(size=12))
    fig.update_layout(
        yaxis2=dict(title='WQ Score', overlaying='y', side='right',
                    range=[0,105], showgrid=False, tickfont=dict(size=12, color=CLR_ACCENT2)),
        legend=dict(bgcolor=CLR_CARD2, bordercolor=CLR_BORDER, borderwidth=1)
    )
    return dark_layout(fig, f'Monthly Water Quality Trend — {station}', 420)

def comparison_bar_chart(results):
    results_s = sorted(results, key=lambda r: r['prob_polluted'], reverse=True)
    stations  = [r['station'] for r in results_s]
    probs     = [r['prob_polluted'] for r in results_s]
    scores    = [r.get('wq_score', 50) for r in results_s]
    colors    = [CLR_POLLUTED if p>50 else CLR_CLEAN for p in probs]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=probs, y=stations, orientation='h', name='P(Polluted)',
        marker=dict(color=colors, line=dict(color=CLR_BORDER, width=0.5)),
        text=[f"{p:.1f}%" for p in probs], textposition='outside',
        textfont=dict(color=CLR_TEXT, size=13),
        hovertemplate='%{y}: %{x:.1f}%<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=scores, y=stations, mode='markers', name='WQ Score',
        marker=dict(color=CLR_ACCENT, size=14, symbol='diamond',
                    line=dict(color='white', width=1.5)),
        hovertemplate='%{y} WQ Score: %{x:.1f}<extra></extra>',
        xaxis='x2'
    ))
    fig.add_vline(x=50, line_dash='dash', line_color='rgba(255,255,255,0.25)')
    fig.update_xaxes(range=[0,120], title_text='P(Polluted) %', tickfont=dict(size=12))
    fig.update_layout(
        xaxis2=dict(title='WQ Score', overlaying='x', side='top',
                    range=[0,120], showgrid=False, tickfont=dict(size=12, color=CLR_ACCENT)),
    )
    return dark_layout(fig, 'Station Comparison — Pollution vs WQ Score',
                       max(320, len(stations)*46+90))

def index_heatmap_stations(results):
    """Heatmap of indices across stations."""
    index_keys = ['NDCI','turbidity','CDOM','NDWI','FAI','BOD_proxy']
    stations   = [r['station'] for r in results]
    z_vals     = [[r['indices'][k] for k in index_keys] for r in results]
    fig = go.Figure(go.Heatmap(
        z=z_vals, x=index_keys, y=stations,
        colorscale=[[0, CLR_CARD],[0.5,'#1e3a5f'],[1, CLR_POLLUTED]],
        text=[[f"{v:.3f}" for v in row] for row in z_vals],
        texttemplate='%{text}', textfont=dict(size=11),
        hovertemplate='%{y} — %{x}: %{z:.4f}<extra></extra>',
        colorbar=dict(title='Value', tickfont=dict(color=CLR_MUTED, size=11),
                      titlefont=dict(color=CLR_MUTED))
    ))
    return dark_layout(fig, 'Index Heatmap Across Stations', max(300, len(stations)*40+100))

def historical_heatmap(ref_df):
    stations_in = [s for s in RIVER_STATIONS if s in ref_df['Station'].values]
    if not stations_in or 'Binary_True' not in ref_df.columns:
        return None
    rows = []
    for stn in stations_in:
        row = {'Station': stn}
        for m in range(1,13):
            sub = ref_df[(ref_df['Station']==stn)&(ref_df['Month']==m)]
            row[MONTH_NAMES[m-1]] = round((sub['Binary_True']==2).mean()*100,0) if len(sub)>0 else np.nan
        rows.append(row)
    pivot = pd.DataFrame(rows).set_index('Station')
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=MONTH_NAMES, y=pivot.index.tolist(),
        colorscale=[[0,'#0a1a14'],[0.5,'#1a1a0a'],[1,'#3d0a12']],
        zmin=0, zmax=100,
        text=[[f"{v:.0f}%" if not np.isnan(v) else '' for v in row] for row in pivot.values],
        texttemplate='%{text}', textfont=dict(size=11),
        hovertemplate='%{y} — %{x}: %{z:.0f}%<extra></extra>',
        colorbar=dict(title='% Polluted', tickfont=dict(color=CLR_MUTED, size=11),
                      titlefont=dict(color=CLR_MUTED))
    ))
    return dark_layout(fig, 'Historical Pollution Rate — Station × Month',
                       max(420, len(stations_in)*28+110))

def pollution_donut(ref_df):
    if 'Binary_True' not in ref_df.columns:
        return None
    counts  = ref_df['Binary_True'].value_counts()
    clean_n = counts.get(1,0)
    poll_n  = counts.get(2,0)
    fig = go.Figure(go.Pie(
        labels=['Clean','Polluted'], values=[clean_n,poll_n], hole=0.68,
        marker=dict(colors=[CLR_CLEAN, CLR_POLLUTED], line=dict(color=CLR_BG, width=4)),
        textinfo='percent', textfont=dict(color='white', size=14),
        hovertemplate='%{label}: %{value} (%{percent})<extra></extra>'
    ))
    fig.add_annotation(text=f"<b>{clean_n+poll_n}</b><br><span style='font-size:10px'>samples</span>",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(size=18, color=CLR_TEXT))
    fig.update_layout(paper_bgcolor=CLR_BG, font_color=CLR_MUTED,
                      title=dict(text='Training Data Balance', font=dict(color=CLR_TEXT, size=14)),
                      legend=dict(font=dict(color=CLR_TEXT, size=13), bgcolor=CLR_CARD2),
                      height=290, margin=dict(l=15,r=15,t=55,b=15))
    return fig

def station_history_bar(ref_df, station):
    sub = ref_df[ref_df['Station']==station].copy()
    if sub.empty or 'Binary_True' not in sub.columns:
        return None
    sub = sub.sort_values('Month')
    fig = go.Figure()
    for label, val, color in [('Clean',1,CLR_CLEAN),('Polluted',2,CLR_POLLUTED)]:
        months = sub[sub['Binary_True']==val]['Month'].tolist()
        fig.add_trace(go.Bar(
            x=[MONTH_NAMES[m-1] for m in months], y=[1]*len(months),
            name=label, marker_color=color, opacity=0.9
        ))
    fig.update_layout(barmode='stack', yaxis=dict(visible=False),
                      xaxis=dict(tickfont=dict(size=13)))
    return dark_layout(fig, f'Historical Labels — {station}', 220)

def feature_importance_chart(bundle):
    """Show top feature importances from the RF model."""
    model     = bundle.get('model')
    feat_cols = bundle.get('feature_cols', FEATURE_COLS_SAFE)
    if not hasattr(model, 'feature_importances_'):
        return None
    importances = model.feature_importances_
    df_imp = pd.DataFrame({'Feature': feat_cols, 'Importance': importances})
    df_imp = df_imp.sort_values('Importance', ascending=True).tail(15)
    colors = [CLR_ACCENT2 if 'NDCI' in f or 'FAI' in f or 'CDOM' in f
              else CLR_ACCENT for f in df_imp['Feature']]
    fig = go.Figure(go.Bar(
        x=df_imp['Importance'], y=df_imp['Feature'], orientation='h',
        marker=dict(color=colors, line=dict(color=CLR_BORDER, width=0.5)),
        text=[f"{v:.3f}" for v in df_imp['Importance']],
        textposition='outside', textfont=dict(color=CLR_TEXT, size=11),
        hovertemplate='%{y}: %{x:.4f}<extra></extra>'
    ))
    fig.update_xaxes(title_text='Importance', tickfont=dict(size=11))
    fig.update_yaxes(tickfont=dict(size=12))
    return dark_layout(fig, 'Top 15 Model Feature Importances', 480)

def seasonal_box_chart(ref_df):
    """Box plot of NDCI by month across all stations."""
    if 'NDCI' not in ref_df.columns:
        return None
    fig = go.Figure()
    for m in range(1,13):
        sub = ref_df[ref_df['Month']==m]['NDCI'].dropna()
        if len(sub)==0: continue
        fig.add_trace(go.Box(
            y=sub, name=MONTH_NAMES[m-1],
            marker_color=CLR_ACCENT, line_color=CLR_ACCENT,
            fillcolor='rgba(0,170,255,0.15)', boxmean=True
        ))
    return dark_layout(fig, 'NDCI Seasonal Distribution (All Stations)', 360)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
def inject_css():
    st.markdown(f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

      /* ── BASE ── */
      html, body {{ font-size: 16px !important; }}
      body, [class*="css"], .main, .block-container {{
          background-color: {CLR_BG} !important;
          color: {CLR_TEXT} !important;
          font-family: 'Sora', sans-serif !important;
      }}
      .block-container {{
          padding-top: 2rem !important;
          padding-bottom: 2rem !important;
          max-width: 1400px !important;
      }}
      p, span, div, li {{ color: {CLR_TEXT} !important; }}

      /* ── SIDEBAR ── */
      [data-testid="stSidebar"] {{
          background: linear-gradient(180deg, #081220 0%, {CLR_CARD} 100%) !important;
          border-right: 2px solid {CLR_BORDER} !important;
          width: 280px !important;
          min-width: 280px !important;
      }}
      [data-testid="stSidebar"] p,
      [data-testid="stSidebar"] span,
      [data-testid="stSidebar"] div,
      [data-testid="stSidebar"] label {{
          font-size: 15px !important;
          color: {CLR_TEXT} !important;
      }}
      [data-testid="stSidebar"] .stRadio label {{
          font-size: 16px !important;
          font-weight: 500 !important;
          padding: 6px 0 !important;
          color: {CLR_MUTED} !important;
      }}
      [data-testid="stSidebar"] .stRadio [aria-checked="true"] ~ div p {{
          color: {CLR_TEXT} !important;
          font-weight: 700 !important;
      }}

      /* ── METRICS ── */
      [data-testid="stMetric"] {{
          background: {CLR_CARD2} !important;
          border: 1px solid {CLR_BORDER} !important;
          border-radius: 14px !important;
          padding: 20px 24px !important;
      }}
      [data-testid="stMetric"] label,
      [data-testid="stMetricLabel"] p {{
          color: {CLR_MUTED} !important;
          font-size: 13px !important;
          font-weight: 700 !important;
          text-transform: uppercase !important;
          letter-spacing: 0.08em !important;
      }}
      [data-testid="stMetricValue"],
      [data-testid="stMetricValue"] div {{
          color: {CLR_TEXT} !important;
          font-family: 'JetBrains Mono', monospace !important;
          font-size: 2rem !important;
          font-weight: 700 !important;
      }}

      /* ── BUTTONS ── */
      .stButton > button {{
          background: linear-gradient(135deg, #0055bb 0%, #0088dd 100%) !important;
          color: white !important;
          border: none !important;
          border-radius: 10px !important;
          font-weight: 700 !important;
          font-size: 16px !important;
          padding: 14px 32px !important;
          letter-spacing: 0.04em !important;
          box-shadow: 0 4px 20px rgba(0,136,221,0.35) !important;
          width: 100% !important;
      }}
      .stButton > button:hover {{
          box-shadow: 0 6px 28px rgba(0,136,221,0.55) !important;
          background: linear-gradient(135deg, #0066cc 0%, #0099ee 100%) !important;
      }}
      .stButton > button p {{ color: white !important; font-size: 16px !important; }}

      /* ── INPUTS ── */
      .stSelectbox > div > div > div,
      [data-baseweb="select"] > div {{
          background: {CLR_CARD2} !important;
          border: 1.5px solid {CLR_BORDER} !important;
          border-radius: 10px !important;
          color: {CLR_TEXT} !important;
          font-size: 15px !important;
          min-height: 46px !important;
      }}
      .stDateInput input, .stTextInput input {{
          background: {CLR_CARD2} !important;
          border: 1.5px solid {CLR_BORDER} !important;
          border-radius: 10px !important;
          color: {CLR_TEXT} !important;
          font-size: 15px !important;
          padding: 10px 14px !important;
      }}
      .stMultiSelect [data-baseweb="tag"] {{
          background: {CLR_ACCENT} !important;
          border-radius: 6px !important;
      }}
      [data-testid="stWidgetLabel"] p {{
          font-size: 14px !important;
          font-weight: 700 !important;
          color: {CLR_MUTED} !important;
          text-transform: uppercase !important;
          letter-spacing: 0.07em !important;
          margin-bottom: 6px !important;
      }}

      /* ── TABS ── */
      [data-baseweb="tab-list"] {{
          background: {CLR_CARD} !important;
          border-radius: 12px !important;
          padding: 5px !important;
          gap: 4px !important;
          border: 1px solid {CLR_BORDER} !important;
      }}
      [data-baseweb="tab"] {{
          font-size: 15px !important;
          font-weight: 600 !important;
          color: {CLR_MUTED} !important;
          border-radius: 9px !important;
          padding: 12px 24px !important;
          background: transparent !important;
      }}
      [aria-selected="true"][data-baseweb="tab"] {{
          background: {CLR_CARD2} !important;
          color: {CLR_TEXT} !important;
          box-shadow: 0 2px 8px rgba(0,0,0,0.4) !important;
      }}
      [data-baseweb="tab"] p {{ color: inherit !important; font-size: 15px !important; }}

      /* ── DATAFRAME ── */
      [data-testid="stDataFrame"] {{
          border-radius: 12px !important;
          overflow: hidden !important;
          border: 1px solid {CLR_BORDER} !important;
      }}
      [data-testid="stDataFrame"] * {{ font-size: 14px !important; }}

      /* ── EXPANDER ── */
      [data-testid="stExpander"] summary {{
          font-size: 15px !important;
          font-weight: 600 !important;
          color: {CLR_TEXT} !important;
          background: {CLR_CARD2} !important;
          border-radius: 10px !important;
          border: 1px solid {CLR_BORDER} !important;
          padding: 14px 18px !important;
      }}
      [data-testid="stExpander"] summary p {{
          font-size: 15px !important;
          color: {CLR_TEXT} !important;
      }}

      /* ── ALERTS / INFO ── */
      [data-testid="stAlert"] {{
          border-radius: 10px !important;
          font-size: 15px !important;
      }}
      [data-testid="stAlert"] p {{ font-size: 15px !important; }}

      /* ── CAPTION ── */
      [data-testid="stCaptionContainer"] p {{
          font-size: 13px !important;
          color: {CLR_MUTED} !important;
      }}

      /* ── PLOTLY CHARTS — ensure no clipping ── */
      [data-testid="stPlotlyChart"] {{
          border-radius: 12px !important;
          overflow: visible !important;
      }}
      [data-testid="stPlotlyChart"] > div {{
          overflow: visible !important;
      }}

      /* ── DIVIDER ── */
      hr {{ border-color: {CLR_BORDER} !important; margin: 20px 0 !important; }}

      /* ── PROGRESS BAR ── */
      [data-testid="stProgressBar"] > div > div {{
          background: linear-gradient(90deg, {CLR_ACCENT}, {CLR_ACCENT2}) !important;
          border-radius: 4px !important;
      }}

      /* ── CUSTOM CARDS ── */
      .wq-card {{
          background: {CLR_CARD};
          border: 1px solid {CLR_BORDER};
          border-radius: 14px;
          padding: 20px 24px;
          margin-bottom: 14px;
      }}
      .wq-card-title {{
          font-size: 12px;
          font-weight: 700;
          color: {CLR_MUTED};
          text-transform: uppercase;
          letter-spacing: 0.12em;
          margin-bottom: 8px;
      }}
      .wq-card-value {{
          font-family: 'JetBrains Mono', monospace;
          font-size: 1.7rem;
          font-weight: 700;
          color: {CLR_TEXT};
          line-height: 1.2;
      }}

      /* ── HEADINGS ── */
      h1,h2,h3,h4,h5,h6 {{ color: {CLR_TEXT} !important; }}

      /* ── SPINNER ── */
      [data-testid="stSpinner"] p {{ font-size: 15px !important; color: {CLR_MUTED} !important; }}

      /* ── WARNING / SUCCESS ── */
      [data-testid="stNotification"] {{ font-size: 15px !important; }}

      /* ── MARKDOWN TEXT ── */
      .stMarkdown p {{ font-size: 15px !important; line-height: 1.7 !important; }}
    </style>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# UI COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════
def render_result_banner(result):
    is_polluted  = result['result'] == 'Polluted'
    bg           = 'linear-gradient(135deg,#3d0010,#1a0008)' if is_polluted else 'linear-gradient(135deg,#002a18,#001208)'
    bclr         = CLR_POLLUTED if is_polluted else CLR_CLEAN
    icon         = '🔴' if is_polluted else '🟢'
    txt          = result['result'].upper()
    ctx          = STATION_CONTEXT.get(result['station'], {})

    # Pre-compute all values — no expressions inside the HTML f-string
    station_name  = result['station']
    img_date      = result['image_date']
    grade         = result['grade']
    grade_label   = result['grade_label']
    grade_color   = result['grade_color']
    wq_score      = f"{result['wq_score']:.0f}"
    confidence    = f"{result['confidence']:.1f}%"
    industry      = ctx.get('industry', '')
    risk          = ctx.get('risk', '')
    pop           = ctx.get('pop', '')
    risk_color    = CLR_POLLUTED if risk == 'High' else (CLR_WARN if risk == 'Medium' else CLR_CLEAN)
    border_color  = bclr + '44'

    # Build industry row separately as a plain string
    industry_html = ''
    if ctx:
        industry_html = (
            f'<div style="margin-top:20px;padding-top:18px;border-top:1px solid {border_color};'
            f'display:flex;gap:32px;flex-wrap:wrap;">'
            f'<div><div style="font-size:11px;color:{CLR_MUTED};text-transform:uppercase;'
            f'letter-spacing:0.1em;font-weight:700;margin-bottom:4px;">Industry</div>'
            f'<div style="font-size:15px;color:{CLR_TEXT};font-weight:600;">{industry}</div></div>'
            f'<div><div style="font-size:11px;color:{CLR_MUTED};text-transform:uppercase;'
            f'letter-spacing:0.1em;font-weight:700;margin-bottom:4px;">Risk Level</div>'
            f'<div style="font-size:15px;color:{risk_color};font-weight:700;">{risk}</div></div>'
            f'<div><div style="font-size:11px;color:{CLR_MUTED};text-transform:uppercase;'
            f'letter-spacing:0.1em;font-weight:700;margin-bottom:4px;">Population</div>'
            f'<div style="font-size:15px;color:{CLR_TEXT};font-weight:600;">{pop}</div></div>'
            f'<div><div style="font-size:11px;color:{CLR_MUTED};text-transform:uppercase;'
            f'letter-spacing:0.1em;font-weight:700;margin-bottom:4px;">Confidence</div>'
            f'<div style="font-size:15px;color:{CLR_TEXT};font-weight:600;">{confidence}</div></div>'
            f'<div><div style="font-size:11px;color:{CLR_MUTED};text-transform:uppercase;'
            f'letter-spacing:0.1em;font-weight:700;margin-bottom:4px;">Image Date</div>'
            f'<div style="font-size:15px;color:{CLR_TEXT};font-weight:600;">{img_date}</div></div>'
            f'</div>'
        )

    st.markdown(
        f'<div style="background:{bg};border:2px solid {bclr};border-radius:18px;'
        f'padding:32px 38px;margin:16px 0;">'
        f'<div style="display:grid;grid-template-columns:1fr auto;align-items:start;gap:24px;">'
        f'<div>'
        f'<div style="font-size:3.2rem;font-weight:800;color:{bclr};letter-spacing:-1px;line-height:1;">'
        f'{icon} {txt}</div>'
        f'<div style="color:{bclr}99;font-size:1rem;margin-top:10px;font-weight:500;">'
        f'{station_name} · Satellite: {img_date}</div>'
        f'</div>'
        f'<div style="text-align:right;">'
        f'<div style="font-size:4.5rem;font-weight:800;color:{grade_color};'
        f'font-family:JetBrains Mono,monospace;line-height:1;">{grade}</div>'
        f'<div style="font-size:12px;color:{CLR_MUTED};margin-top:6px;font-weight:700;'
        f'text-transform:uppercase;letter-spacing:0.1em;">WQ Grade · {grade_label}</div>'
        f'<div style="font-size:2rem;font-weight:800;color:{grade_color};'
        f'font-family:JetBrains Mono,monospace;margin-top:6px;">'
        f'{wq_score}<span style="font-size:1rem;color:{CLR_MUTED};font-weight:400;">/100</span></div>'
        f'</div>'
        f'</div>'
        f'{industry_html}'
        f'</div>',
        unsafe_allow_html=True
    )

def render_safety_flags(safety):
    items = [
        ('🚰', 'Drinking Water',  safety['drinking']),
        ('🌾', 'Irrigation',      safety['irrigation']),
        ('🐟', 'Aquaculture',     safety['aquaculture']),
        ('🏊', 'Bathing / Swim',  safety['bathing']),
    ]
    cols = st.columns(4)
    for col, (icon, label, ok) in zip(cols, items):
        color  = CLR_CLEAN if ok else CLR_POLLUTED
        status = '✓  SAFE' if ok else '✗  UNSAFE'
        bg     = '#00e5a011' if ok else '#ff3b5c11'
        col.markdown(f"""
        <div style="background:{bg};border:2px solid {color}55;border-radius:14px;
                    padding:22px 16px;text-align:center;">
          <div style="font-size:2.4rem;line-height:1;">{icon}</div>
          <div style="font-size:13px;color:{CLR_MUTED};margin-top:10px;font-weight:700;
                      text-transform:uppercase;letter-spacing:0.09em;">{label}</div>
          <div style="font-size:1.1rem;font-weight:800;color:{color};margin-top:8px;
                      letter-spacing:0.04em;">{status}</div>
        </div>
        """, unsafe_allow_html=True)

def render_pollution_sources(sources):
    if not sources:
        st.markdown(f"""
        <div class="wq-card" style="border-color:{CLR_CLEAN}44;">
          <div class="wq-card-title">Pollution Source Analysis</div>
          <div style="color:{CLR_CLEAN};font-size:1rem;font-weight:600;">
            ✅ No significant pollution sources detected
          </div>
        </div>""", unsafe_allow_html=True)
        return

    st.markdown(f"<div style='font-size:0.85rem;color:{CLR_MUTED};text-transform:uppercase;"
                f"letter-spacing:0.1em;font-weight:700;margin-bottom:10px;'>🔬 Pollution Source Analysis</div>",
                unsafe_allow_html=True)
    for src in sources:
        conf = src['confidence']
        bar_w = conf
        clr   = CLR_POLLUTED if conf > 70 else CLR_WARN
        st.markdown(f"""
        <div class="wq-card" style="border-color:{clr}55;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
            <div style="font-size:1rem;font-weight:700;color:{CLR_TEXT};">{src['source']}</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:1.1rem;
                        font-weight:800;color:{clr};">{conf}%</div>
          </div>
          <div style="background:{CLR_BG};border-radius:4px;height:6px;margin-bottom:10px;">
            <div style="width:{bar_w}%;background:{clr};height:6px;border-radius:4px;
                        transition:width 0.5s ease;"></div>
          </div>
          <div style="font-size:0.88rem;color:{CLR_MUTED};">{src['cause']}</div>
        </div>""", unsafe_allow_html=True)

def render_index_cards(indices):
    items = [
        ('NDWI',       'Water Clarity',        indices['NDWI'],        -1,   1,   True),
        ('NDCI',       'Algae / Chlorophyll',   indices['NDCI'],        -1,   1,   False),
        ('Turbidity',  'Suspended Particles',   indices['turbidity'],    0,   3,   False),
        ('CDOM',       'Dissolved Organics',    indices['CDOM'],         0,   3,   False),
        ('FAI',        'Floating Algae Index',  indices['FAI'],       -500, 500,   False),
        ('BOD Proxy',  'Oxygen Demand',         indices['BOD_proxy'],    0,   3,   False),
    ]
    cols = st.columns(3)
    for i, (name, desc, val, lo, hi, higher_better) in enumerate(items):
        norm = (val - lo) / (hi - lo + 1e-6)
        norm = max(0.0, min(1.0, norm))
        health = norm if higher_better else (1 - norm)
        color = CLR_CLEAN if health > 0.6 else (CLR_WARN if health > 0.35 else CLR_POLLUTED)
        status = 'Good' if health > 0.6 else ('Moderate' if health > 0.35 else 'Poor')
        cols[i%3].markdown(f"""
        <div style="background:{CLR_CARD};border:1.5px solid {color}55;border-radius:14px;
                    padding:20px 22px;margin-bottom:14px;">
          <div style="font-size:12px;font-weight:700;color:{CLR_MUTED};text-transform:uppercase;
                      letter-spacing:0.1em;margin-bottom:6px;">{desc}</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:2rem;font-weight:700;
                      color:{color};line-height:1.1;">{val:.4f}</div>
          <div style="display:flex;justify-content:space-between;align-items:center;margin-top:8px;">
            <div style="font-size:13px;color:{CLR_MUTED};font-weight:600;">{name}</div>
            <div style="font-size:12px;font-weight:700;color:{color};background:{color}22;
                        padding:2px 10px;border-radius:20px;">{status}</div>
          </div>
          <div style="background:{CLR_BG};border-radius:4px;height:5px;margin-top:10px;">
            <div style="width:{norm*100:.0f}%;background:{color};height:5px;border-radius:4px;"></div>
          </div>
        </div>""", unsafe_allow_html=True)

def render_map(results_list):
    """Render an inline OpenStreetMap via iframe with station markers."""
    if not results_list:
        return
    # Build a simple HTML map using Leaflet
    markers_js = ""
    for r in results_list:
        clr = '#ff3b5c' if r['result'] == 'Polluted' else '#00e5a0'
        popup = (f"<b>{r['station']}</b><br>"
                 f"Status: {r['result']}<br>"
                 f"WQ Score: {r.get('wq_score',0):.0f}/100<br>"
                 f"P(Polluted): {r['prob_polluted']:.1f}%")
        markers_js += (f"L.circleMarker([{r['lat']},{r['lon']}],"
                       f"{{radius:14,color:'{clr}',fillColor:'{clr}',fillOpacity:0.8,"
                       f"weight:2}}).addTo(map)"
                       f".bindPopup('{popup}');\n")

    avg_lat = np.mean([r['lat'] for r in results_list])
    avg_lon = np.mean([r['lon'] for r in results_list])
    html = f"""<!DOCTYPE html>
<html><head>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>body{{margin:0}}#map{{height:100vh;background:#060d17}}</style>
</head><body>
<div id="map"></div>
<script>
var map = L.map('map',{{zoomControl:true}}).setView([{avg_lat},{avg_lon}],8);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png',
  {{attribution:'©OpenStreetMap ©CartoDB',maxZoom:19}}).addTo(map);
{markers_js}
</script></body></html>"""
    st.components.v1.html(html, height=420)

# ══════════════════════════════════════════════════════════════════════════════
# PAGES
# ══════════════════════════════════════════════════════════════════════════════
def page_predict(bundle, ref_df):
    st.markdown(f"<h2 style='color:{CLR_TEXT};font-size:2rem;font-weight:800;margin-bottom:4px;'>"
                f"🔍 Single Station Prediction</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{CLR_MUTED};font-size:1.05rem;margin-bottom:24px;'>"
                f"Select a monitoring station and date to fetch Sentinel-2 imagery and predict water quality.</p>",
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        station = st.selectbox("📍 Station", sorted(RIVER_STATIONS.keys()), key='p_stn')
    with c2:
        date = st.date_input("📅 Date", value=datetime(2023,6,1),
                             min_value=datetime(2017,1,1),
                             max_value=datetime.today(), key='p_date')
    with c3:
        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("⚡ Predict", use_container_width=True, type='primary')

    lat, lon = RIVER_STATIONS[station]
    ctx = STATION_CONTEXT.get(station, {})

    # Station info row
    info_items = [f"📌 {lat:.4f}°N, {lon:.4f}°E"]
    if ctx:
        info_items += [f"🏭 {ctx.get('industry','—')}", f"⚠️ Risk: {ctx.get('risk','—')}", f"👥 Pop: {ctx.get('pop','—')}"]
    st.markdown(
        f"<div style='display:flex;gap:28px;flex-wrap:wrap;margin-bottom:20px;'>"
        + "".join(f"<span style='font-size:14px;color:{CLR_MUTED};font-weight:600;background:{CLR_CARD2};"
                  f"padding:8px 16px;border-radius:8px;border:1px solid {CLR_BORDER};'>{it}</span>"
                  for it in info_items)
        + "</div>",
        unsafe_allow_html=True
    )

    if not run:
        return

    with st.spinner("🛰️ Fetching Sentinel-2 imagery from Google Earth Engine..."):
        result = predict_single(station, date.strftime('%Y-%m-%d'), bundle, ref_df)

    if 'error' in result:
        st.error(f"❌ {result['error']}")
        return

    if 'history' not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append(result)

    if result['image_date'] != result['query_date']:
        st.info(f"ℹ️ Closest clear image: **{result['image_date']}** (requested {result['query_date']})")

    render_result_banner(result)

    # ── Gauges Row ──
    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(wq_score_gauge(result['wq_score'], result['grade'],
                                        result['grade_label'], result['grade_color']),
                        use_container_width=True)
    with g2:
        st.plotly_chart(prob_gauge(result['prob_polluted']), use_container_width=True)

    st.markdown("---")

    # ── Safety Flags ──
    st.markdown(f"<h4 style='color:{CLR_TEXT};font-size:1.2rem;margin-bottom:14px;'>"
                f"💧 Water Use Safety Assessment</h4>", unsafe_allow_html=True)
    render_safety_flags(result['safety'])

    st.markdown("---")

    # ── Pollution Sources ──
    render_pollution_sources(result['sources'])

    st.markdown("---")

    # ── Index Cards ──
    st.markdown(f"<h4 style='color:{CLR_TEXT};font-size:1.2rem;margin-bottom:14px;'>"
                f"📊 Spectral Water Quality Indices</h4>", unsafe_allow_html=True)
    render_index_cards(result['indices'])

    # ── Radar + History ──
    r1, r2 = st.columns(2)
    with r1:
        st.plotly_chart(radar_chart(result['indices'], station), use_container_width=True)
    with r2:
        if ref_df is not None:
            hc = station_history_bar(ref_df, station)
            if hc:
                st.plotly_chart(hc, use_container_width=True)

    # ── Raw Bands Expander ──
    with st.expander("🛰️ Raw Sentinel-2 Band Values"):
        bd = result['bands']
        fig_b = go.Figure(go.Bar(
            x=list(bd.keys()), y=list(bd.values()), marker_color=CLR_ACCENT,
            text=[f"{v:.1f}" for v in bd.values()], textposition='outside',
            textfont=dict(size=13)
        ))
        fig_b.update_yaxes(title_text='Surface Reflectance', tickfont=dict(size=12))
        st.plotly_chart(dark_layout(fig_b, '', 240), use_container_width=True)

    # ── Map ──
    with st.expander("🗺️ Station Location Map"):
        render_map([result])

    st.caption("Model: RF 600 trees · 35 features · OA 87.7% · κ 0.720 | Sentinel-2 SR Harmonized")


def page_monthly(bundle, ref_df):
    st.markdown(f"<h2 style='color:{CLR_TEXT};font-size:2rem;font-weight:800;margin-bottom:4px;'>"
                f"📈 Monthly Trend Analysis</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{CLR_MUTED};font-size:1.05rem;margin-bottom:24px;'>"
                f"Analyse all 12 months for a station to detect seasonal pollution patterns.</p>",
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        station = st.selectbox("📍 Station", sorted(RIVER_STATIONS.keys()), key='mt_stn')
    with c2:
        year = st.selectbox("📅 Year", list(range(2024, 2016, -1)), key='mt_yr')
    with c3:
        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("🚀 Run All Months", use_container_width=True, type='primary')

    cache_key = f"trend_{station}_{year}"

    if run:
        results = []
        prog = st.progress(0, text="Starting analysis...")
        for i, month in enumerate(range(1, 13)):
            prog.progress((i+1)/12, text=f"Processing {MONTH_NAMES[month-1]}...")
            r = predict_single(station, f"{year}-{month:02d}-15", bundle, ref_df)
            if 'error' not in r:
                r['month_num'] = month
                results.append(r)
        prog.empty()
        st.session_state[cache_key] = results

    results = st.session_state.get(cache_key, [])
    if not results:
        st.info("👆 Select a station and year, then click **Run All Months**.")
        return

    n_poll  = sum(1 for r in results if r['result'] == 'Polluted')
    avg_p   = np.mean([r['prob_polluted'] for r in results])
    avg_wq  = np.mean([r.get('wq_score',50) for r in results])
    worst   = max(results, key=lambda r: r['prob_polluted'])
    best    = min(results, key=lambda r: r['prob_polluted'])

    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("Months Fetched",   len(results))
    m2.metric("🔴 Polluted",      n_poll)
    m3.metric("🟢 Clean",         len(results)-n_poll)
    m4.metric("Avg P(Polluted)",  f"{avg_p:.1f}%")
    m5.metric("Avg WQ Score",     f"{avg_wq:.0f}/100")

    st.markdown("---")
    st.plotly_chart(monthly_trend_chart(results, station), use_container_width=True)

    # Index trends
    st.markdown(f"<h4 style='color:{CLR_TEXT};margin-bottom:12px;'>📊 Index Trends by Month</h4>",
                unsafe_allow_html=True)
    fig_idx = go.Figure()
    for col, color, name in [
        ('NDCI',       '#f0b429', 'NDCI'),
        ('turbidity',  CLR_POLLUTED, 'Turbidity'),
        ('CDOM',       CLR_ACCENT2, 'CDOM'),
        ('BOD_proxy',  CLR_ACCENT, 'BOD Proxy'),
    ]:
        fig_idx.add_trace(go.Scatter(
            x=[r['month_num'] for r in results],
            y=[r['indices'][col] for r in results],
            mode='lines+markers', name=name,
            line=dict(color=color, width=2.5),
            marker=dict(size=9)
        ))
    fig_idx.update_xaxes(tickvals=list(range(1,13)), ticktext=MONTH_NAMES, tickfont=dict(size=13))
    st.plotly_chart(dark_layout(fig_idx, 'Multi-Index Seasonal Trend', 340),
                    use_container_width=True)

    # Monthly detail table
    st.markdown(f"<h4 style='color:{CLR_TEXT};margin-bottom:12px;'>📋 Monthly Detail Table</h4>",
                unsafe_allow_html=True)
    df_show = pd.DataFrame([{
        'Month'       : MONTH_NAMES[r['month_num']-1],
        'Result'      : r['result'],
        'WQ Score'    : f"{r.get('wq_score',0):.0f}/100",
        'Grade'       : r.get('grade','—'),
        'P(Polluted)' : f"{r['prob_polluted']:.1f}%",
        'NDCI'        : f"{r['indices']['NDCI']:.4f}",
        'Turbidity'   : f"{r['indices']['turbidity']:.4f}",
        'CDOM'        : f"{r['indices']['CDOM']:.4f}",
        'Image Date'  : r['image_date'],
    } for r in results])

    def color_r(v):
        return (f'background-color:#2d0010;color:{CLR_POLLUTED}'
                if v == 'Polluted'
                else f'background-color:#002d1a;color:{CLR_CLEAN}')

    st.dataframe(df_show.style.applymap(color_r, subset=['Result']),
                 use_container_width=True, hide_index=True)

    st.columns(2)[0].success(f"**Best Month:** {MONTH_NAMES[best['month_num']-1]} "
                              f"— WQ {best.get('wq_score',0):.0f}/100 ({best['prob_polluted']:.1f}% polluted)")
    st.columns(2)[1].error(  f"**Worst Month:** {MONTH_NAMES[worst['month_num']-1]} "
                              f"— WQ {worst.get('wq_score',0):.0f}/100 ({worst['prob_polluted']:.1f}% polluted)")


def page_compare(bundle, ref_df):
    st.markdown(f"<h2 style='color:{CLR_TEXT};font-size:2rem;font-weight:800;margin-bottom:4px;'>"
                f"🏭 Multi-Station Comparison</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{CLR_MUTED};font-size:1.05rem;margin-bottom:24px;'>"
                f"Compare water quality across multiple Cauvery basin stations simultaneously.</p>",
                unsafe_allow_html=True)

    c1, c2 = st.columns([3, 1])
    with c1:
        stations = st.multiselect(
            "📍 Stations (2–10)",
            sorted(RIVER_STATIONS.keys()),
            default=['Mettur','Trichy','Erode','Bhavani','Kumbakonam'],
            key='cmp_stns'
        )
    with c2:
        date = st.date_input("📅 Date", value=datetime(2023,6,1),
                             min_value=datetime(2017,1,1),
                             max_value=datetime.today(), key='cmp_date')

    if len(stations) < 2:
        st.warning("Select at least 2 stations.")
        return

    run = st.button("⚡ Compare All", use_container_width=True, type='primary')
    cache_key = f"cmp_{'_'.join(sorted(stations))}_{date}"

    if run:
        results = []
        prog = st.progress(0)
        for i, stn in enumerate(stations):
            prog.progress((i+1)/len(stations), text=f"Fetching {stn}...")
            r = predict_single(stn, date.strftime('%Y-%m-%d'), bundle, ref_df)
            if 'error' not in r:
                results.append(r)
            else:
                st.warning(f"⚠️ {stn}: {r['error']}")
        prog.empty()
        st.session_state[cache_key] = results

    results = st.session_state.get(cache_key, [])
    if not results:
        st.info("👆 Select stations and date, then click **Compare All**.")
        return

    results_s = sorted(results, key=lambda r: r['prob_polluted'], reverse=True)
    n_poll    = sum(1 for r in results_s if r['result'] == 'Polluted')
    avg_wq    = np.mean([r.get('wq_score',50) for r in results_s])

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Stations Analysed", len(results_s))
    m2.metric("🔴 Polluted",       n_poll)
    m3.metric("🟢 Clean",          len(results_s)-n_poll)
    m4.metric("Avg WQ Score",      f"{avg_wq:.0f}/100")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📊 Comparison Charts", "🗺️ Station Map", "📋 Detail Table"])

    with tab1:
        st.plotly_chart(comparison_bar_chart(results_s), use_container_width=True)
        st.plotly_chart(index_heatmap_stations(results_s), use_container_width=True)

        st.markdown(f"<h4 style='color:{CLR_TEXT};margin-bottom:12px;'>📊 Grouped Index Comparison</h4>",
                    unsafe_allow_html=True)
        idx_fig = go.Figure()
        for col, color, name in [
            ('NDCI',      '#f0b429', 'NDCI'),
            ('turbidity', CLR_POLLUTED, 'Turbidity'),
            ('CDOM',      CLR_ACCENT2, 'CDOM'),
            ('NDWI',      CLR_ACCENT, 'NDWI'),
        ]:
            idx_fig.add_trace(go.Bar(
                name=name,
                x=[r['station'] for r in results_s],
                y=[r['indices'][col] for r in results_s],
                marker_color=color, opacity=0.85
            ))
        idx_fig.update_layout(barmode='group')
        st.plotly_chart(dark_layout(idx_fig, '', 360), use_container_width=True)

    with tab2:
        render_map(results_s)

    with tab3:
        tbl = pd.DataFrame([{
            'Station'    : r['station'],
            'Result'     : r['result'],
            'WQ Score'   : f"{r.get('wq_score',0):.0f}/100",
            'Grade'      : r.get('grade','—'),
            'P(Polluted)': f"{r['prob_polluted']:.1f}%",
            'NDCI'       : f"{r['indices']['NDCI']:.4f}",
            'Turbidity'  : f"{r['indices']['turbidity']:.4f}",
            'CDOM'       : f"{r['indices']['CDOM']:.4f}",
            'Image Date' : r['image_date'],
        } for r in results_s])

        def color_r(v):
            return (f'background-color:#2d0010;color:{CLR_POLLUTED}'
                    if v == 'Polluted'
                    else f'background-color:#002d1a;color:{CLR_CLEAN}')

        st.dataframe(tbl.style.applymap(color_r, subset=['Result']),
                     use_container_width=True, hide_index=True)


def page_dashboard(ref_df, bundle):
    st.markdown(f"<h2 style='color:{CLR_TEXT};font-size:2rem;font-weight:800;margin-bottom:4px;'>"
                f"📊 Data Dashboard</h2>", unsafe_allow_html=True)

    if ref_df is None:
        st.warning("Place `tnpcb_extracted_features.csv` in the app folder to enable the dashboard.")
        return

    total   = len(ref_df)
    clean_n = int((ref_df.get('Binary_True', pd.Series())==1).sum())
    poll_n  = int((ref_df.get('Binary_True', pd.Series())==2).sum())

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Total Records",  total)
    k2.metric("Stations",       ref_df['Station'].nunique())
    k3.metric("🟢 Clean",       clean_n)
    k4.metric("🔴 Polluted",    poll_n)
    k5.metric("Model OA",       "87.7%")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📈 Overview", "🌍 Heatmaps", "🤖 Model Insights"])

    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            d = pollution_donut(ref_df)
            if d:
                st.plotly_chart(d, use_container_width=True)
        with col2:
            if 'Binary_True' in ref_df.columns:
                mr = []
                for m in range(1,13):
                    sub = ref_df[ref_df['Month']==m]
                    if len(sub)>0:
                        mr.append({'month': MONTH_NAMES[m-1],
                                   'rate': (sub['Binary_True']==2).mean()*100,
                                   'n': len(sub)})
                mr_df = pd.DataFrame(mr)
                fig_mr = go.Figure(go.Bar(
                    x=mr_df['month'], y=mr_df['rate'],
                    marker_color=[CLR_POLLUTED if r>50 else CLR_CLEAN for r in mr_df['rate']],
                    text=[f"{r:.1f}%" for r in mr_df['rate']], textposition='outside',
                    textfont=dict(size=13),
                    customdata=mr_df['n'],
                    hovertemplate='%{x}: %{y:.1f}%<br>%{customdata} samples<extra></extra>'
                ))
                fig_mr.add_hline(y=50, line_dash='dash', line_color='rgba(255,255,255,0.2)')
                fig_mr.update_yaxes(range=[0,120], title_text='% Polluted', tickfont=dict(size=12))
                st.plotly_chart(dark_layout(fig_mr, 'Monthly Pollution Rate (All Stations)', 310),
                                use_container_width=True)

        if 'Binary_True' in ref_df.columns:
            sr = (ref_df.groupby('Station')['Binary_True']
                        .apply(lambda x: (x==2).mean()*100)
                        .sort_values().reset_index())
            sr.columns = ['Station','rate']
            fig_sr = go.Figure(go.Bar(
                x=sr['rate'], y=sr['Station'], orientation='h',
                marker_color=[CLR_POLLUTED if r>50 else CLR_CLEAN for r in sr['rate']],
                text=[f"{r:.1f}%" for r in sr['rate']], textposition='outside',
                textfont=dict(size=12)
            ))
            fig_sr.add_vline(x=50, line_dash='dash', line_color='rgba(255,255,255,0.2)')
            fig_sr.update_xaxes(range=[0,120], tickfont=dict(size=12))
            fig_sr.update_yaxes(tickfont=dict(size=13))
            st.plotly_chart(dark_layout(fig_sr, 'Per-Station Pollution Rate',
                                        max(340, len(sr)*26+70)),
                            use_container_width=True)

    with tab2:
        hm = historical_heatmap(ref_df)
        if hm:
            st.plotly_chart(hm, use_container_width=True)
        sbox = seasonal_box_chart(ref_df)
        if sbox:
            st.plotly_chart(sbox, use_container_width=True)

    with tab3:
        fi = feature_importance_chart(bundle)
        if fi:
            st.plotly_chart(fi, use_container_width=True)
        else:
            st.info("Feature importance not available for this model type.")

        # Confusion-matrix-style metrics
        st.markdown(f"<h4 style='color:{CLR_TEXT};margin-bottom:16px;'>📐 Model Performance</h4>",
                    unsafe_allow_html=True)
        pm1,pm2,pm3,pm4 = st.columns(4)
        pm1.metric("Overall Accuracy", "87.7%")
        pm2.metric("Cohen's Kappa",    "0.720")
        pm3.metric("Trees",            "600")
        pm4.metric("CV Folds",         "5")
        st.caption("Random Forest trained on Sentinel-2 SR Harmonized imagery · 35 engineered spectral features · 5-fold stratified cross-validation")


def page_history():
    st.markdown(f"<h2 style='color:{CLR_TEXT};font-size:2rem;font-weight:800;margin-bottom:4px;'>"
                f"🕑 Prediction History</h2>", unsafe_allow_html=True)
    hist = st.session_state.get('history', [])
    if not hist:
        st.info("No predictions yet this session. Run a prediction first.")
        return

    df_h = pd.DataFrame([{
        'Station'    : r['station'],
        'Date'       : r['query_date'],
        'Image'      : r['image_date'],
        'Result'     : r['result'],
        'WQ Score'   : r.get('wq_score', 0),
        'Grade'      : r.get('grade', '—'),
        'P(Polluted)': r['prob_polluted'],
        'NDCI'       : r['indices']['NDCI'],
        'Turbidity'  : r['indices']['turbidity'],
    } for r in hist])

    m1,m2,m3 = st.columns(3)
    m1.metric("Total Predictions", len(hist))
    m2.metric("Polluted",   int((df_h['Result']=='Polluted').sum()))
    m3.metric("Avg WQ Score", f"{df_h['WQ Score'].mean():.0f}/100")

    st.markdown("---")

    # Timeline scatter
    fig_h = go.Figure()
    for label, color in [('Clean', CLR_CLEAN), ('Polluted', CLR_POLLUTED)]:
        sub = df_h[df_h['Result']==label]
        fig_h.add_trace(go.Scatter(
            x=sub['Date'], y=sub['P(Polluted)'],
            mode='markers', name=label,
            marker=dict(color=color, size=16, line=dict(color='white', width=2)),
            text=sub['Station'],
            hovertemplate='<b>%{text}</b><br>%{x}<br>P(Polluted): %{y:.1f}%<extra></extra>'
        ))
    fig_h.add_hline(y=50, line_dash='dash', line_color='rgba(255,255,255,0.2)')
    st.plotly_chart(dark_layout(fig_h, 'Prediction Timeline', 340), use_container_width=True)

    # Map of all predicted stations
    with st.expander("🗺️ All Predicted Stations Map"):
        render_map(hist)

    def color_r(v):
        return (f'background-color:#2d0010;color:{CLR_POLLUTED}'
                if v == 'Polluted'
                else f'background-color:#002d1a;color:{CLR_CLEAN}')

    st.dataframe(
        df_h.style.applymap(color_r, subset=['Result'])
                  .format({'P(Polluted)': '{:.1f}%', 'NDCI': '{:.4f}',
                           'Turbidity': '{:.4f}', 'WQ Score': '{:.0f}'}),
        use_container_width=True, hide_index=True
    )

    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(
        page_title="TNPCB · Cauvery Water Quality",
        page_icon="💧", layout="wide",
        initial_sidebar_state="expanded"
    )
    inject_css()

    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center;padding:20px 0 32px;">
          <div style="font-size:3.8rem;line-height:1;margin-bottom:10px;">💧</div>
          <div style="font-size:1.3rem;font-weight:800;color:{CLR_TEXT};letter-spacing:0.5px;
                      line-height:1.2;">TNPCB WQ Monitor</div>
          <div style="font-size:13px;color:{CLR_MUTED};margin-top:8px;line-height:1.7;">
            Sentinel-2 · Random Forest<br>Cauvery Basin · Tamil Nadu
          </div>
        </div>
        """, unsafe_allow_html=True)

        page = st.radio("Navigation", [
            "🔍  Predict",
            "📈  Monthly Trend",
            "🏭  Station Compare",
            "📊  Dashboard",
            "🕑  History",
        ], label_visibility='collapsed')

        st.markdown(f"<hr style='border-color:{CLR_BORDER};margin:20px 0;'>", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="padding:16px 0;">
          <div style="font-size:13px;font-weight:800;color:{CLR_TEXT};text-transform:uppercase;
                      letter-spacing:0.1em;margin-bottom:14px;">Model Metrics</div>
          <div style="display:flex;flex-direction:column;gap:10px;">
            <div style="display:flex;justify-content:space-between;align-items:center;
                        background:{CLR_CARD2};padding:10px 14px;border-radius:8px;
                        border:1px solid {CLR_BORDER};">
              <span style="font-size:14px;color:{CLR_MUTED};font-weight:600;">Overall Accuracy</span>
              <span style="font-size:15px;color:{CLR_CLEAN};font-weight:800;
                           font-family:'JetBrains Mono',monospace;">87.7%</span>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center;
                        background:{CLR_CARD2};padding:10px 14px;border-radius:8px;
                        border:1px solid {CLR_BORDER};">
              <span style="font-size:14px;color:{CLR_MUTED};font-weight:600;">Cohen's κ</span>
              <span style="font-size:15px;color:{CLR_ACCENT};font-weight:800;
                           font-family:'JetBrains Mono',monospace;">0.720</span>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center;
                        background:{CLR_CARD2};padding:10px 14px;border-radius:8px;
                        border:1px solid {CLR_BORDER};">
              <span style="font-size:14px;color:{CLR_MUTED};font-weight:600;">Trees / Feats</span>
              <span style="font-size:15px;color:{CLR_ACCENT2};font-weight:800;
                           font-family:'JetBrains Mono',monospace;">600 / 35</span>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center;
                        background:{CLR_CARD2};padding:10px 14px;border-radius:8px;
                        border:1px solid {CLR_BORDER};">
              <span style="font-size:14px;color:{CLR_MUTED};font-weight:600;">CV Strategy</span>
              <span style="font-size:15px;color:{CLR_ACCENT};font-weight:800;
                           font-family:'JetBrains Mono',monospace;">5-Fold</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        n_hist = len(st.session_state.get('history', []))
        if n_hist:
            st.markdown(f"""
            <div style="margin-top:16px;background:{CLR_CARD2};border:1px solid {CLR_ACCENT}44;
                        border-radius:10px;padding:12px 16px;text-align:center;">
              <div style="font-size:12px;color:{CLR_MUTED};text-transform:uppercase;
                          letter-spacing:0.1em;font-weight:700;">Session Predictions</div>
              <div style="font-size:2rem;color:{CLR_ACCENT};font-weight:800;
                          font-family:'JetBrains Mono',monospace;margin-top:4px;">{n_hist}</div>
            </div>""", unsafe_allow_html=True)

    init_gee()
    bundle = load_model()
    ref_df = load_reference_data()

    if ref_df is None and page not in ["🕑  History"]:
        st.sidebar.warning("⚠️ No reference CSV found")

    if   page == "🔍  Predict":        page_predict(bundle, ref_df)
    elif page == "📈  Monthly Trend":   page_monthly(bundle, ref_df)
    elif page == "🏭  Station Compare": page_compare(bundle, ref_df)
    elif page == "📊  Dashboard":       page_dashboard(ref_df, bundle)
    elif page == "🕑  History":         page_history()

if __name__ == '__main__':
    main()
