"""
================================================================================
AGRITECH SOIL ANALYTICS PLATFORM
================================================================================
Professional soil nutrient prediction system for precision agriculture.
Enterprise-grade ML solution for agricultural advisors and farmers.

© 2024 AgriTech Solutions
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# ================================================================================
# PAGE CONFIGURATION
# ================================================================================
st.set_page_config(
    page_title="AgriTech Soil Analytics",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================================================================================
# PROFESSIONAL CSS - INDUSTRY STANDARD DESIGN
# ================================================================================
st.markdown("""
<style>
    /* Import Professional Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Remove Streamlit default padding */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        max-width: 1400px;
    }
    
    /* Professional Header */
    .main-header {
        background: #ffffff;
        border-bottom: 1px solid #e5e7eb;
        padding: 1.25rem 2rem;
        margin: -1rem -1rem 2rem -1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .logo-section {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    
    .logo-text {
        font-size: 1.5rem;
        font-weight: 700;
        color: #047857;
        letter-spacing: -0.5px;
    }
    
    .tagline {
        color: #6b7280;
        font-size: 0.875rem;
        font-weight: 400;
        margin-top: 0.25rem;
    }
    
    /* Navigation */
    .nav-container {
        display: flex;
        gap: 0.5rem;
        border-bottom: 1px solid #e5e7eb;
        padding: 0;
        margin: 0;
    }
    
    .nav-button {
        padding: 0.75rem 1.5rem;
        border: none;
        background: transparent;
        color: #6b7280;
        font-weight: 500;
        font-size: 0.9375rem;
        cursor: pointer;
        border-bottom: 2px solid transparent;
        transition: all 0.2s;
        position: relative;
    }
    
    .nav-button:hover {
        color: #047857;
        background: #f9fafb;
    }
    
    .nav-button.active {
        color: #047857;
        border-bottom-color: #047857;
        font-weight: 600;
    }
    
    /* Hero Section - Professional */
    .hero-container {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-radius: 16px;
        padding: 4rem 3rem;
        margin-bottom: 3rem;
        border: 1px solid #bbf7d0;
    }
    
    .hero-title {
        font-size: 2.75rem;
        font-weight: 800;
        color: #064e3b;
        margin-bottom: 1rem;
        line-height: 1.2;
        letter-spacing: -1px;
    }
    
    .hero-subtitle {
        font-size: 1.25rem;
        color: #065f46;
        font-weight: 400;
        line-height: 1.6;
    }
    
    /* Feature Cards - Enterprise Style */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .feature-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 2rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .feature-card:hover {
        border-color: #047857;
        box-shadow: 0 10px 25px rgba(4, 120, 87, 0.1);
        transform: translateY(-4px);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    .feature-title {
        font-size: 1.125rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.75rem;
    }
    
    .feature-desc {
        font-size: 0.9375rem;
        color: #6b7280;
        line-height: 1.6;
    }
    
    /* Section Headers - Clean & Professional */
    .section-title {
        font-size: 1.875rem;
        font-weight: 700;
        color: #111827;
        margin: 3rem 0 1.5rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #e5e7eb;
        letter-spacing: -0.5px;
    }
    
    .subsection-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #374151;
        margin: 2rem 0 1rem 0;
    }
    
    /* Form Styling - Modern & Clean */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        border: 1px solid #d1d5db !important;
        border-radius: 8px !important;
        padding: 0.625rem 0.875rem !important;
        font-size: 0.9375rem !important;
        transition: all 0.2s !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #047857 !important;
        box-shadow: 0 0 0 3px rgba(4, 120, 87, 0.1) !important;
    }
    
    /* Buttons - Professional Style */
    .stButton > button {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        color: white;
        font-weight: 600;
        font-size: 0.9375rem;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        border: none;
        box-shadow: 0 4px 6px rgba(4, 120, 87, 0.2);
        transition: all 0.3s;
        letter-spacing: 0.3px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #047857 0%, #065f46 100%);
        box-shadow: 0 6px 12px rgba(4, 120, 87, 0.3);
        transform: translateY(-2px);
    }
    
    /* Metric Cards - Dashboard Style */
    .metric-container {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #6b7280;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #111827;
    }
    
    /* Nutrient Result Cards - Professional */
    .nutrient-result-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem 0;
        transition: all 0.3s;
    }
    
    .nutrient-result-card:hover {
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
    }
    
    .nutrient-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #f3f4f6;
    }
    
    .nutrient-name {
        font-size: 1.125rem;
        font-weight: 600;
        color: #111827;
    }
    
    .nutrient-value-large {
        font-size: 2.25rem;
        font-weight: 700;
        color: #047857;
        margin: 1rem 0;
    }
    
    .nutrient-unit {
        font-size: 1rem;
        color: #6b7280;
        font-weight: 400;
    }
    
    /* Status Badge - Modern */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.875rem;
        letter-spacing: 0.3px;
    }
    
    .status-high {
        background: #d1fae5;
        color: #065f46;
    }
    
    .status-medium {
        background: #fef3c7;
        color: #92400e;
    }
    
    .status-low {
        background: #fee2e2;
        color: #991b1b;
    }
    
    /* Recommendation Box */
    .recommendation-box {
        background: #f9fafb;
        border-left: 4px solid #047857;
        padding: 1.25rem;
        border-radius: 8px;
        margin-top: 1rem;
        font-size: 0.9375rem;
        color: #374151;
        line-height: 1.6;
    }
    
    /* Alert Box - Professional */
    .alert-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 1.25rem;
        margin: 1.5rem 0;
        font-size: 0.9375rem;
        color: #1e40af;
        line-height: 1.6;
    }
    
    .alert-box strong {
        font-weight: 600;
    }
    
    /* Stats Grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    /* Process Steps - Professional */
    .process-step {
        text-align: center;
        padding: 2rem 1rem;
    }
    
    .step-number {
        width: 60px;
        height: 60px;
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        color: white;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0 auto 1.5rem auto;
        box-shadow: 0 4px 12px rgba(4, 120, 87, 0.3);
    }
    
    .step-title {
        font-size: 1.125rem;
        font-weight: 600;
        color: #111827;
        margin-bottom: 0.75rem;
    }
    
    .step-desc {
        font-size: 0.9375rem;
        color: #6b7280;
        line-height: 1.5;
    }
    
    /* Footer - Corporate */
    .footer-container {
        background: #f9fafb;
        border-top: 1px solid #e5e7eb;
        padding: 3rem 2rem;
        margin-top: 4rem;
        text-align: center;
    }
    
    .footer-brand {
        font-size: 1.25rem;
        font-weight: 700;
        color: #047857;
        margin-bottom: 0.5rem;
    }
    
    .footer-text {
        color: #6b7280;
        font-size: 0.875rem;
        line-height: 1.8;
    }
    
    /* Dashboard Card */
    .dashboard-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .dashboard-card-title {
        font-size: 1rem;
        font-weight: 600;
        color: #111827;
        margin-bottom: 1rem;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .feature-grid {
            grid-template-columns: 1fr;
        }
        
        .stats-grid {
            grid-template-columns: repeat(2, 1fr);
        }
        
        .hero-title {
            font-size: 2rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# ================================================================================
# SESSION STATE
# ================================================================================
if 'page' not in st.session_state:
    st.session_state.page = 'Home'
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'input_data' not in st.session_state:
    st.session_state.input_data = None

# ================================================================================
# HELPER FUNCTIONS
# ================================================================================

def load_models(model_dir="saved_model"):
    """Load all trained models"""
    model_mapping = {
        "N,Kg/H": "NKgH_model.pkl",
        "P,Kg/H": "PKgH_model.pkl",
        "K,Kg/H": "KKgH_model.pkl",
        "Zn,ppm": "Znppm_model.pkl",
        "Cu,ppm": "Cuppm_model.pkl",
        "Fe,ppm": "Feppm_model.pkl",
        "S (kg/ha)": "S_(kgha)_model.pkl"
    }
    
    models = {}
    for nutrient, model_file in model_mapping.items():
        model_path = os.path.join(model_dir, model_file)
        if os.path.exists(model_path):
            try:
                models[nutrient] = joblib.load(model_path)
            except Exception as e:
                st.error(f"Error loading {model_file}: {str(e)}")
    
    return models

def categorize_nutrient(value, nutrient_name):
    """Categorize nutrient levels"""
    thresholds = {
        "N,Kg/H": (200, 350),
        "P,Kg/H": (25, 50),
        "K,Kg/H": (120, 200),
        "Zn,ppm": (0.5, 1.0),
        "Cu,ppm": (0.3, 0.6),
        "Fe,ppm": (4.5, 9.0),
        "S (kg/ha)": (10, 20)
    }
    
    low, high = thresholds.get(nutrient_name, (0, 100))
    
    if value < low:
        return "Low", "low"
    elif value < high:
        return "Optimal", "medium"
    else:
        return "High", "high"

def get_gauge_color(status):
    """Get color for gauge based on status"""
    colors = {
        "low": "#dc2626",
        "medium": "#16a34a",
        "high": "#d97706"
    }
    return colors.get(status, "#6b7280")

def create_gauge_chart(value, nutrient_name, nutrient_display):
    """Create professional gauge chart"""
    status, status_class = categorize_nutrient(value, nutrient_name)
    
    thresholds = {
        "N,Kg/H": (200, 350, 500),
        "P,Kg/H": (25, 50, 75),
        "K,Kg/H": (120, 200, 280),
        "Zn,ppm": (0.5, 1.0, 1.5),
        "Cu,ppm": (0.3, 0.6, 1.0),
        "Fe,ppm": (4.5, 9.0, 15),
        "S (kg/ha)": (10, 20, 30)
    }
    
    low, optimal, high = thresholds.get(nutrient_name, (0, 50, 100))
    max_val = high
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': nutrient_display, 'font': {'size': 20, 'color': '#111827', 'family': 'Inter'}},
        number={'font': {'size': 40, 'color': '#047857', 'family': 'Inter'}},
        gauge={
            'axis': {'range': [None, max_val], 'tickwidth': 1, 'tickcolor': "#6b7280"},
            'bar': {'color': get_gauge_color(status_class), 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e5e7eb",
            'steps': [
                {'range': [0, low], 'color': '#fee2e2'},
                {'range': [low, optimal], 'color': '#d1fae5'},
                {'range': [optimal, max_val], 'color': '#fef3c7'}
            ],
            'threshold': {
                'line': {'color': get_gauge_color(status_class), 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='white',
        font={'color': "#111827", 'family': 'Inter'}
    )
    
    return fig

def create_bar_chart(predictions):
    """Create nutrient comparison bar chart"""
    nutrient_names = {
        "N,Kg/H": "Nitrogen",
        "P,Kg/H": "Phosphorus",
        "K,Kg/H": "Potassium",
        "Zn,ppm": "Zinc",
        "Cu,ppm": "Copper",
        "Fe,ppm": "Iron",
        "S (kg/ha)": "Sulfur"
    }
    
    data = []
    colors = []
    
    for nutrient_key, name in nutrient_names.items():
        value = predictions.get(nutrient_key)
        if value is not None:
            status, status_class = categorize_nutrient(value, nutrient_key)
            data.append({
                'Nutrient': name,
                'Value': value,
                'Status': status
            })
            
            color_map = {
                'low': '#dc2626',
                'medium': '#16a34a',
                'high': '#d97706'
            }
            colors.append(color_map[status_class])
    
    df = pd.DataFrame(data)
    
    fig = go.Figure(data=[
        go.Bar(
            x=df['Nutrient'],
            y=df['Value'],
            marker_color=colors,
            text=df['Value'].round(2),
            textposition='outside',
            textfont=dict(size=12, color='#111827', family='Inter'),
            hovertemplate='<b>%{x}</b><br>Value: %{y:.2f}<br>Status: %{text}<extra></extra>',
            hovertext=df['Status']
        )
    ])
    
    fig.update_layout(
        title={
            'text': 'Predicted Nutrient Levels Overview',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#111827', 'family': 'Inter', 'weight': 600}
        },
        xaxis_title='Nutrient',
        yaxis_title='Predicted Value',
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'color': '#111827', 'family': 'Inter'},
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#f3f4f6')
    )
    
    return fig

def create_radar_chart(predictions):
    """Create radar chart for nutrient profile"""
    nutrient_names = {
        "N,Kg/H": ("Nitrogen", 500),
        "P,Kg/H": ("Phosphorus", 75),
        "K,Kg/H": ("Potassium", 280),
        "Zn,ppm": ("Zinc", 1.5),
        "Cu,ppm": ("Copper", 1.0),
        "Fe,ppm": ("Iron", 15),
        "S (kg/ha)": ("Sulfur", 30)
    }
    
    categories = []
    values = []
    
    for nutrient_key, (name, max_val) in nutrient_names.items():
        value = predictions.get(nutrient_key)
        if value is not None:
            categories.append(name)
            # Normalize to percentage
            normalized = (value / max_val) * 100
            values.append(min(normalized, 100))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(4, 120, 87, 0.2)',
        line=dict(color='#047857', width=2),
        marker=dict(size=8, color='#047857'),
        name='Current Status'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                ticksuffix='%',
                showline=False,
                gridcolor='#e5e7eb'
            ),
            angularaxis=dict(
                gridcolor='#e5e7eb'
            )
        ),
        title={
            'text': 'Nutrient Profile (% of Maximum)',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#111827', 'family': 'Inter', 'weight': 600}
        },
        showlegend=False,
        height=450,
        paper_bgcolor='white',
        font={'color': '#111827', 'family': 'Inter'}
    )
    
    return fig

def get_recommendation(nutrient_name, status):
    """Get general non-specific recommendations"""
    recommendations = {
        "N,Kg/H": {
            "Low": "Nitrogen level appears low. Consider improving soil nutrition through proper management and crop planning.",
            "Optimal": "Nitrogen level is within a suitable range. Maintain regular soil care and balanced cropping practices.",
            "High": "Nitrogen level seems high. Avoid unnecessary additions and monitor future crop response."
        },
        "P,Kg/H": {
            "Low": "Phosphorus is on the lower side. Plan practices that improve soil health and nutrient retention.",
            "Optimal": "Phosphorus is stable. Continue with normal soil maintenance and good agricultural practices.",
            "High": "Phosphorus is higher than required. Avoid excess input and monitor before the next cycle."
        },
        "K,Kg/H": {
            "Low": "Potassium is low. Strengthening soil management practices may support plant growth and resilience.",
            "Optimal": "Potassium is balanced. Maintain normal conditions and routine field care.",
            "High": "Potassium levels are high. Avoid adding more and observe plant response in upcoming seasons."
        },
        "Zn,ppm": {
            "Low": "Zinc levels appear low. Review soil health practices that naturally support micronutrient balance.",
            "Optimal": "Zinc is within suitable range. Maintain regular growing practices.",
            "High": "Zinc is higher than needed. No corrective steps required, just monitor routinely."
        },
        "Cu,ppm": {
            "Low": "Copper is low. Proper soil care and long-term management may help improve availability.",
            "Optimal": "Copper levels are appropriate. Continue with normal field practices.",
            "High": "Copper is adequate. No additional input or changes required now."
        },
        "Fe,ppm": {
            "Low": "Iron is low. Observe crop behavior and maintain good soil moisture and organic matter practices.",
            "Optimal": "Iron levels are normal. Current soil conditions are functioning well.",
            "High": "Iron is sufficiently available. Routine monitoring is enough."
        },
        "S (kg/ha)": {
            "Low": "Sulfur level is low. Improving soil structure and organic content may help over time.",
            "Optimal": "Sulfur is balanced well. Maintain standard agronomic practices.",
            "High": "Sulfur level is high. No additional input required, just track future soil reports."
        }
    }
    
    return recommendations.get(nutrient_name, {}).get(status, "General monitoring recommended. Consult an agricultural advisor if needed.")


def preprocess_input(input_dict):
    """Preprocess input data"""
    df = pd.DataFrame([input_dict])
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    
    if cat_cols:
        df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    else:
        df_encoded = df.copy()
    
    return df_encoded

def predict_nutrients(input_data, models):
    """Generate predictions"""
    predictions = {}
    
    try:
        X = preprocess_input(input_data)
    except Exception as e:
        st.error(f"Data processing error: {str(e)}")
        return predictions
    
    for nutrient, model in models.items():
        try:
            if hasattr(model, 'feature_names_in_'):
                model_features = model.feature_names_in_
                for feat in model_features:
                    if feat not in X.columns:
                        X[feat] = 0
                X_aligned = X[model_features]
            else:
                X_aligned = X
            
            pred = model.predict(X_aligned)[0]
            predictions[nutrient] = max(0, pred)
            
        except Exception as e:
            st.error(f"Prediction error for {nutrient}: {str(e)}")
            predictions[nutrient] = None
    
    return predictions

# ================================================================================
# HEADER & NAVIGATION
# ================================================================================
def render_header():
    """Render professional header"""
    st.markdown("""
    <div class="main-header">
        <div class="logo-section">
            <span style="font-size: 2rem;">🌱</span>
            <div>
                <div class="logo-text">AgriTech Soil Analytics</div>
                <div class="tagline">Precision Agriculture Intelligence Platform</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_navigation():
    """Render professional navigation"""
    pages = ['Home', 'Soil Analysis', 'Results', 'Visualization', 'About']
    
    # Create button layout
    cols = st.columns(len(pages))
    
    for idx, page in enumerate(pages):
        with cols[idx]:
            if st.button(
                page, 
                key=f"nav_{page}",
                use_container_width=True,
                type="secondary" if st.session_state.page != page else "primary"
            ):
                st.session_state.page = page
                st.rerun()

# ================================================================================
# PAGE: HOME
# ================================================================================
def render_home():
    """Professional home page"""
    
    # Hero Section
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">Advanced Soil Nutrient Prediction</div>
        <div class="hero-subtitle">
            Leverage machine learning to predict post-harvest soil nutrient status and optimize fertilizer management for sustainable agriculture.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">📊</span>
            <div class="feature-title">Data-Driven Insights</div>
            <div class="feature-desc">
                ML models trained on 80,000+ soil samples from diverse agro-climatic zones deliver accurate nutrient predictions.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">⚡</span>
            <div class="feature-title">Real-Time Analysis</div>
            <div class="feature-desc">
                Instant soil nutrient forecasting with comprehensive recommendations for 7 essential nutrients (NPK + micronutrients).
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">🎯</span>
            <div class="feature-title">Precision Agriculture</div>
            <div class="feature-desc">
                Science-based fertilizer recommendations to optimize crop nutrition, reduce costs, and improve sustainability.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Process Section
    st.markdown('<div class="section-title">How It Works</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="process-step">
            <div class="step-number">1</div>
            <div class="step-title">Input Data</div>
            <div class="step-desc">Enter current soil test parameters and crop history</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="process-step">
            <div class="step-number">2</div>
            <div class="step-title">AI Processing</div>
            <div class="step-desc">ML algorithms analyze patterns and correlations</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="process-step">
            <div class="step-number">3</div>
            <div class="step-title">Predictions</div>
            <div class="step-desc">Receive post-harvest nutrient forecasts</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="process-step">
            <div class="step-number">4</div>
            <div class="step-title">Action Plan</div>
            <div class="step-desc">Get customized fertilizer recommendations</div>
        </div>
        """, unsafe_allow_html=True)
    
    # CTA
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Start Soil Analysis", key="cta_start", use_container_width=True):
            st.session_state.page = 'Soil Analysis'
            st.rerun()

# ================================================================================
# PAGE: SOIL ANALYSIS
# ================================================================================
def render_soil_analysis():
    """Professional soil input form"""
    
    st.markdown('<div class="section-title">Soil Nutrient Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="alert-box">
        <strong>📋 Data Requirements:</strong> Enter your current soil test results and crop information. 
        The system will predict nutrient levels after the next crop cycle based on historical patterns and ML models.
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("soil_analysis_form", clear_on_submit=False):
        
        # Soil Properties
        st.markdown('<div class="subsection-title">Soil Chemical Properties</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ph = st.number_input(
                "Soil pH",
                min_value=4.0,
                max_value=10.0,
                value=7.0,
                step=0.1,
                help="Soil reaction (pH scale 4-10)"
            )
            
            oc = st.number_input(
                "Organic Carbon (%)",
                min_value=0.0,
                max_value=5.0,
                value=0.5,
                step=0.01,
                help="Organic matter content"
            )
        
        with col2:
            ec = st.number_input(
                "Electrical Conductivity (mS/cm)",
                min_value=0.0,
                max_value=5.0,
                value=0.3,
                step=0.01,
                help="Soil salinity indicator"
            )
            
            soil_type = st.selectbox(
                "Soil Texture Class",
                options=["loamy", "sandy", "clay", "sandy loam", "clay loam", "alkali", "sandy clay"],
                help="Primary soil textural classification"
            )
        
        with col3:
            land_type = st.selectbox(
                "Land Classification",
                options=["low land", "upland"],
                help="Topographical land classification"
            )
            
            district = st.selectbox(
                "District",
                options=[
                    "Lakhimpur Kheri",
                    "Hardoi",
                    "Sitapur",
                    "Badaun",
                    "Bhagalpur (Bihar)",
                    "Khagaria (Bihar)"
                ],
                help="Administrative district"
            )
        
        # Crop Information
        st.markdown('<div class="subsection-title">Crop History & Timeline</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            previous_crop = st.selectbox(
                "Previous Crop",
                options=[
                    "vacant", "wheat", "paddy", "sugarcane", "mustard", "potato",
                    "maize", "cotton", "groundnut", "gram", "lentil", "bajra",
                    "jowar", "tomato", "onion", "cucumber", "other"
                ],
                help="Last cultivated crop"
            )
        
        with col2:
            year = st.number_input(
                "Year",
                min_value=2020,
                max_value=2030,
                value=datetime.now().year,
                help="Current year"
            )
        
        with col3:
            month_name = st.selectbox(
                "Month",
                options=["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                help="Current month"
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Submit
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submitted = st.form_submit_button(
                "Generate Prediction Report",
                use_container_width=True
            )
    
    if submitted:
        month_map = {
            "Jan":1, "Feb":2, "Mar":3, "Apr":4, "May":5, "Jun":6,
            "Jul":7, "Aug":8, "Sep":9, "Oct":10, "Nov":11, "Dec":12
        }
        
        input_data = {
            "pH": ph,
            "OC %": oc,
            "Soil Type": soil_type,
            "EC, m S": ec,
            "Land Type": land_type,
            "District": district,
            "Previous Crop": previous_crop,
            "Year": year,
            "Month_num": month_map[month_name]
        }
        
        with st.spinner("Processing soil data through ML pipeline..."):
            models = load_models("saved_model")
        
        if not models:
            st.error("⚠️ Model files not found. Please contact system administrator.")
            return
        
        with st.spinner("Generating nutrient predictions..."):
            predictions = predict_nutrients(input_data, models)
        
        st.session_state.predictions = predictions
        st.session_state.input_data = input_data
        
        st.success("✅ Analysis complete! View results in Results or Visualization tab.")
        st.session_state.page = 'Results'
        st.rerun()

# ================================================================================
# PAGE: RESULTS
# ================================================================================
def render_results():
    """Professional results page"""
    
    if st.session_state.predictions is None:
        st.warning("⚠️ No analysis data available. Please complete soil analysis first.")
        if st.button("Go to Soil Analysis"):
            st.session_state.page = 'Soil Analysis'
            st.rerun()
        return
    
    predictions = st.session_state.predictions
    input_data = st.session_state.input_data
    
    st.markdown('<div class="section-title">Soil Nutrient Prediction Report</div>', unsafe_allow_html=True)
    
    # Input Summary
    st.markdown('<div class="subsection-title">Analysis Parameters</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">pH Level</div>
            <div class="metric-value">{input_data['pH']:.1f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">EC (mS/cm)</div>
            <div class="metric-value">{input_data['EC, m S']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Organic Carbon</div>
            <div class="metric-value">{input_data['OC %']:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Previous Crop</div>
            <div class="metric-value" style="font-size: 1.25rem;">{input_data['Previous Crop'].title()}</div>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Soil Type</div>
            <div class="metric-value" style="font-size: 1.25rem;">{input_data['Soil Type'].title()}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Land Type</div>
            <div class="metric-value" style="font-size: 1.25rem;">{input_data['Land Type'].title()}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">District</div>
            <div class="metric-value" style="font-size: 1.25rem;">{input_data['District']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Predictions
    st.markdown('<div class="subsection-title">Predicted Nutrient Levels (Post-Harvest)</div>', unsafe_allow_html=True)
    
    nutrient_info = {
        "N,Kg/H": ("Nitrogen (N)", "kg/ha", "Primary macronutrient for vegetative growth"),
        "P,Kg/H": ("Phosphorus (P)", "kg/ha", "Essential for root development and energy transfer"),
        "K,Kg/H": ("Potassium (K)", "kg/ha", "Critical for water regulation and disease resistance"),
        "Zn,ppm": ("Zinc (Zn)", "ppm", "Micronutrient for enzyme systems"),
        "Cu,ppm": ("Copper (Cu)", "ppm", "Micronutrient for photosynthesis"),
        "Fe,ppm": ("Iron (Fe)", "ppm", "Essential for chlorophyll formation"),
        "S (kg/ha)": ("Sulfur (S)", "kg/ha", "Required for protein synthesis")
    }
    
    # Primary Nutrients
    st.markdown("**Primary Macronutrients**")
    col1, col2, col3 = st.columns(3)
    
    for idx, (nutrient_key, (name, unit, desc)) in enumerate(list(nutrient_info.items())[:3]):
        col = [col1, col2, col3][idx]
        
        with col:
            value = predictions.get(nutrient_key)
            
            if value is not None:
                status, status_class = categorize_nutrient(value, nutrient_key)
                recommendation = get_recommendation(nutrient_key, status)
                
                st.markdown(f"""
                <div class="nutrient-result-card">
                    <div class="nutrient-header">
                        <div class="nutrient-name">{name}</div>
                        <span class="status-badge status-{status_class}">{status}</span>
                    </div>
                    <div class="nutrient-value-large">{value:.1f} <span class="nutrient-unit">{unit}</span></div>
                    <div class="recommendation-box">{recommendation}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Micronutrients
    st.markdown("**Secondary & Micronutrients**")
    col1, col2, col3, col4 = st.columns(4)
    
    micro_cols = [col1, col2, col3, col4]
    micro_nutrients = list(nutrient_info.items())[3:]
    
    for idx, (nutrient_key, (name, unit, desc)) in enumerate(micro_nutrients):
        with micro_cols[idx]:
            value = predictions.get(nutrient_key)
            
            if value is not None:
                status, status_class = categorize_nutrient(value, nutrient_key)
                recommendation = get_recommendation(nutrient_key, status)
                
                st.markdown(f"""
                <div class="nutrient-result-card">
                    <div class="nutrient-header">
                        <div class="nutrient-name">{name}</div>
                        <span class="status-badge status-{status_class}">{status}</span>
                    </div>
                    <div class="nutrient-value-large" style="font-size: 1.75rem;">{value:.2f} <span class="nutrient-unit">{unit}</span></div>
                    <div class="recommendation-box">{recommendation}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Actions
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("New Analysis", use_container_width=True):
            st.session_state.predictions = None
            st.session_state.input_data = None
            st.session_state.page = 'Soil Analysis'
            st.rerun()
    
    with col2:
        if st.button("View Visualizations", use_container_width=True):
            st.session_state.page = 'Visualization'
            st.rerun()
    
    with col3:
        report_text = f"""AGRITECH SOIL NUTRIENT PREDICTION REPORT
{'='*60}

Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SOIL PARAMETERS:
{'-'*60}
District: {input_data['District']}
Previous Crop: {input_data['Previous Crop'].title()}
Soil Type: {input_data['Soil Type'].title()}
Land Type: {input_data['Land Type'].title()}
pH: {input_data['pH']:.1f}
EC: {input_data['EC, m S']:.2f} mS/cm
Organic Carbon: {input_data['OC %']:.2f}%

PREDICTED NUTRIENT LEVELS (POST-HARVEST):
{'-'*60}
"""
        
        for nutrient_key, (name, unit, desc) in nutrient_info.items():
            value = predictions.get(nutrient_key)
            if value is not None:
                status, _ = categorize_nutrient(value, nutrient_key)
                report_text += f"\n{name}: {value:.2f} {unit} - Status: {status}"
                report_text += f"\nRecommendation: {get_recommendation(nutrient_key, status)}\n"
        
        report_text += f"\n{'='*60}\n© 2024 AgriTech Solutions - Precision Agriculture Platform"
        
        st.download_button(
            label="Download Report",
            data=report_text,
            file_name=f"soil_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )

# ================================================================================
# PAGE: VISUALIZATION
# ================================================================================
def render_visualization():
    """Professional visualization dashboard"""
    
    if st.session_state.predictions is None:
        st.warning("⚠️ No analysis data available. Please complete soil analysis first.")
        if st.button("Go to Soil Analysis"):
            st.session_state.page = 'Soil Analysis'
            st.rerun()
        return
    
    predictions = st.session_state.predictions
    
    st.markdown('<div class="section-title">Nutrient Analysis Dashboard</div>', unsafe_allow_html=True)
    
    # Gauge Charts Section
    st.markdown('<div class="subsection-title">Individual Nutrient Status</div>', unsafe_allow_html=True)
    
    nutrient_display = {
        "N,Kg/H": "Nitrogen (N) - kg/ha",
        "P,Kg/H": "Phosphorus (P) - kg/ha",
        "K,Kg/H": "Potassium (K) - kg/ha",
        "Zn,ppm": "Zinc (Zn) - ppm",
        "Cu,ppm": "Copper (Cu) - ppm",
        "Fe,ppm": "Iron (Fe) - ppm",
        "S (kg/ha)": "Sulfur (S) - kg/ha"
    }
    
    # Primary Nutrients Gauges
    col1, col2, col3 = st.columns(3)
    primary_nutrients = list(nutrient_display.items())[:3]
    
    for idx, (nutrient_key, display_name) in enumerate(primary_nutrients):
        col = [col1, col2, col3][idx]
        value = predictions.get(nutrient_key)
        
        if value is not None:
            with col:
                st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
                fig = create_gauge_chart(value, nutrient_key, display_name)
                st.plotly_chart(fig, use_container_width=True, key=f"gauge_{nutrient_key}")
                
                status, status_class = categorize_nutrient(value, nutrient_key)
                st.markdown(
                    f'<div style="text-align: center; margin-top: -10px;"><span class="status-badge status-{status_class}">{status}</span></div>',
                    unsafe_allow_html=True
                )
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Micronutrients Gauges
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    micro_nutrients = list(nutrient_display.items())[3:]
    
    for idx, (nutrient_key, display_name) in enumerate(micro_nutrients):
        col = [col1, col2, col3, col4][idx]
        value = predictions.get(nutrient_key)
        
        if value is not None:
            with col:
                st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
                fig = create_gauge_chart(value, nutrient_key, display_name)
                st.plotly_chart(fig, use_container_width=True, key=f"gauge_micro_{nutrient_key}")
                
                status, status_class = categorize_nutrient(value, nutrient_key)
                st.markdown(
                    f'<div style="text-align: center; margin-top: -10px;"><span class="status-badge status-{status_class}">{status}</span></div>',
                    unsafe_allow_html=True
                )
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Summary Stats (UNCHANGED)
    st.markdown('<div class="subsection-title">Quick Summary</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    low_count = sum(1 for k, v in predictions.items() if v is not None and categorize_nutrient(v, k)[0] == "Low")
    optimal_count = sum(1 for k, v in predictions.items() if v is not None and categorize_nutrient(v, k)[0] == "Optimal")
    high_count = sum(1 for k, v in predictions.items() if v is not None and categorize_nutrient(v, k)[0] == "High")
    total_analyzed = len([v for v in predictions.values() if v is not None])
    
    with col1:
        st.markdown(f"""
        <div class="metric-container" style="border-left: 4px solid #dc2626;">
            <div class="metric-label">Low Status</div>
            <div class="metric-value" style="color: #dc2626;">{low_count}</div>
            <div style="font-size: 0.75rem; color: #6b7280; margin-top: 0.5rem;">Nutrients</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container" style="border-left: 4px solid #16a34a;">
            <div class="metric-label">Optimal Status</div>
            <div class="metric-value" style="color: #16a34a;">{optimal_count}</div>
            <div style="font-size: 0.75rem; color: #6b7280; margin-top: 0.5rem;">Nutrients</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container" style="border-left: 4px solid #d97706;">
            <div class="metric-label">High Status</div>
            <div class="metric-value" style="color: #d97706;">{high_count}</div>
            <div style="font-size: 0.75rem; color: #6b7280; margin-top: 0.5rem;">Nutrients</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-container" style="border-left: 4px solid #047857;">
            <div class="metric-label">Total Analyzed</div>
            <div class="metric-value" style="color: #047857;">{total_analyzed}</div>
            <div style="font-size: 0.75rem; color: #6b7280; margin-top: 0.5rem;">Nutrients</div>
        </div>
        """, unsafe_allow_html=True)


# ================================================================================
# PAGE: ABOUT
# ================================================================================
def render_about():
    """Professional about page"""
    
    st.markdown('<div class="section-title">About AgriTech Soil Analytics</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Platform Overview
        
        AgriTech Soil Analytics is an enterprise-grade precision agriculture platform that leverages advanced machine learning 
        to predict post-harvest soil nutrient status. Our system enables farmers, agronomists, and agricultural advisors to 
        make data-driven fertilizer management decisions for sustainable and profitable farming.
        
        ### Technology Stack
        
        **Machine Learning Models:**
        - Random Forest Regression algorithms
        - Trained on 80,912 validated soil samples
        - Cross-validated performance metrics
        - Continuous model improvement pipeline
        
        **Coverage:**
        - Multiple agro-climatic zones (Uttar Pradesh & Bihar)
        - 7 essential nutrients (NPK + micronutrients)
        - Diverse soil types and crop rotations
        - Historical data from 2011-2025
        
        ### Key Features
        
        - **Real-time Predictions:** Instant nutrient forecasting using ML models
        - **Comprehensive Analysis:** 7 nutrients including NPK and micronutrients
        - **Interactive Dashboards:** Professional gauge charts and visualizations
        - **Precision Recommendations:** Science-based fertilizer application guidelines
        - **Decision Support:** Data-driven insights for fertilizer optimization
        - **Sustainable Agriculture:** Reduce over-fertilization and environmental impact
        
        ### Use Cases
        
        1. **Pre-season Planning:** Predict nutrient requirements before planting
        2. **Fertilizer Budgeting:** Optimize fertilizer procurement and costs
        3. **Soil Health Monitoring:** Track long-term nutrient trends
        4. **Advisory Services:** Support agricultural extension programs
        5. **Research & Development:** Analyze nutrient dynamics across regions
        """)
    
    with col2:
        st.markdown("""
        <div class="feature-card" style="margin-top: 0;">
            <div class="feature-title">📊 Platform Statistics</div>
            <br>
            <div style="margin: 1.5rem 0;">
                <div style="font-size: 2rem; font-weight: 700; color: #047857;">80,912</div>
                <div style="color: #6b7280; font-size: 0.875rem;">Training Samples</div>
            </div>
            <div style="margin: 1.5rem 0;">
                <div style="font-size: 2rem; font-weight: 700; color: #047857;">7</div>
                <div style="color: #6b7280; font-size: 0.875rem;">Nutrient Predictions</div>
            </div>
            <div style="margin: 1.5rem 0;">
                <div style="font-size: 2rem; font-weight: 700; color: #047857;">6</div>
                <div style="color: #6b7280; font-size: 0.875rem;">Districts Covered</div>
            </div>
            <div style="margin: 1.5rem 0;">
                <div style="font-size: 2rem; font-weight: 700; color: #047857;">15+</div>
                <div style="color: #6b7280; font-size: 0.875rem;">Crop Types</div>
            </div>
        </div>
        
        <div class="alert-box" style="margin-top: 2rem;">
            <strong>🔬 Scientific Validation</strong><br>
            Models validated through 80/20 train-test split with cross-validation protocols following agricultural research standards.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    ### Important Disclaimer
    
    This platform provides predictive insights based on historical patterns and machine learning models. 
    Predictions should be used as decision support tools in conjunction with:
    
    - Regular soil testing and laboratory analysis
    - Local agricultural advisory services
    - Crop-specific management guidelines
    - Regional best practices and farmer experience
    
    For critical agricultural decisions, always consult with certified agronomists and conduct comprehensive soil testing.
    
    ---
    
    ### Technical Support
    
    For technical assistance, model inquiries, or partnership opportunities, please contact your regional agricultural extension office.
    """)

# ================================================================================
# FOOTER
# ================================================================================
def render_footer():
    """Professional footer"""
    st.markdown("""
    <div class="footer-container">
        <div class="footer-brand">🌱 AgriTech Soil Analytics</div>
        <div class="footer-text">
            Precision Agriculture Intelligence Platform<br>
            Machine Learning for Sustainable Farming<br>
            © 2024 AgriTech Solutions. All rights reserved.<br>
            <br>
            <small>Disclaimer: This tool provides predictive insights for decision support. 
            Always consult agricultural experts for critical farming decisions.</small>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ================================================================================
# MAIN APPLICATION
# ================================================================================
def main():
    """Main application"""
    
    render_header()
    render_navigation()
    
    if st.session_state.page == 'Home':
        render_home()
    elif st.session_state.page == 'Soil Analysis':
        render_soil_analysis()
    elif st.session_state.page == 'Results':
        render_results()
    elif st.session_state.page == 'Visualization':
        render_visualization()
    elif st.session_state.page == 'About':
        render_about()
    
    render_footer()

if __name__ == "__main__":
    main()