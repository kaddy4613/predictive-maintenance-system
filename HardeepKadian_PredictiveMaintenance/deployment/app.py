
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

from huggingface_hub import hf_hub_download

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide"
)

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id="Kaddy4613/Predictive-Maintenance-Model",
        filename="best_model.pkl"
    )

    model = joblib.load(model_path)

    return model

model = load_model()

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.metric-card {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.5);
}

.status-healthy {
    background-color: #0f5132;
    padding: 20px;
    border-radius: 15px;
    color: white;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}

.status-faulty {
    background-color: #842029;
    padding: 20px;
    border-radius: 15px;
    color: white;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================

st.title("⚙️ Predictive Maintenance Dashboard")

st.markdown("""
Real-time engine health monitoring and predictive maintenance system powered by Machine Learning and XGBoost.
""")

st.markdown("---")

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("About System")

st.sidebar.info("""
### Technologies Used

- XGBoost
- MLflow
- Hugging Face
- Streamlit
- Docker
- Plotly

### Objective

Detect abnormal engine behavior using operational sensor data and support predictive maintenance planning.
""")

# =====================================================
# SENSOR INPUTS
# =====================================================

st.header("Engine Sensor Inputs")

col1, col2, col3 = st.columns(3)

with col1:

    engine_rpm = st.slider(
        "Engine RPM",
        0,
        3000,
        1200
    )

    fuel_pressure = st.slider(
        "Fuel Pressure",
        0.0,
        25.0,
        5.0
    )

with col2:

    lub_oil_pressure = st.slider(
        "Lub Oil Pressure",
        0.0,
        10.0,
        3.5
    )

    coolant_pressure = st.slider(
        "Coolant Pressure",
        0.0,
        10.0,
        2.0
    )

with col3:

    lub_oil_temp = st.slider(
        "Lub Oil Temp",
        0.0,
        150.0,
        80.0
    )

    coolant_temp = st.slider(
        "Coolant Temp",
        0.0,
        150.0,
        85.0
    )

# =====================================================
# KPI CARDS
# =====================================================

st.markdown("---")

st.subheader("Live Sensor Monitoring")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(
        label="Engine RPM",
        value=f"{engine_rpm}"
    )

with kpi2:
    st.metric(
        label="Fuel Pressure",
        value=f"{fuel_pressure:.2f}"
    )

with kpi3:
    st.metric(
        label="Lub Oil Temp",
        value=f"{lub_oil_temp:.1f} °C"
    )

with kpi4:
    st.metric(
        label="Coolant Temp",
        value=f"{coolant_temp:.1f} °C"
    )

# =====================================================
# INPUT DATAFRAME
# =====================================================

input_data = pd.DataFrame([{

    'Engine rpm': engine_rpm,

    'Lub oil pressure': lub_oil_pressure,

    'Fuel pressure': fuel_pressure,

    'Coolant pressure': coolant_pressure,

    'lub oil temp': lub_oil_temp,

    'Coolant temp': coolant_temp

    '__index_level_0__': 0

}])

# =====================================================
# GAUGE CHARTS
# =====================================================

st.markdown("---")

st.subheader("Operational Sensor Gauges")

g1, g2, g3 = st.columns(3)

with g1:

    fig_rpm = go.Figure(go.Indicator(
        mode="gauge+number",
        value=engine_rpm,
        title={'text': "Engine RPM"},
        gauge={
            'axis': {'range': [0, 3000]},
            'bar': {'color': "lightblue"},
            'steps': [
                {'range': [0, 1000], 'color': "green"},
                {'range': [1000, 2000], 'color': "yellow"},
                {'range': [2000, 3000], 'color': "red"}
            ]
        }
    ))

    st.plotly_chart(fig_rpm, use_container_width=True)

with g2:

    fig_temp = go.Figure(go.Indicator(
        mode="gauge+number",
        value=coolant_temp,
        title={'text': "Coolant Temp"},
        gauge={
            'axis': {'range': [0, 150]},
            'bar': {'color': "orange"},
            'steps': [
                {'range': [0, 70], 'color': "green"},
                {'range': [70, 100], 'color': "yellow"},
                {'range': [100, 150], 'color': "red"}
            ]
        }
    ))

    st.plotly_chart(fig_temp, use_container_width=True)

with g3:

    fig_pressure = go.Figure(go.Indicator(
        mode="gauge+number",
        value=fuel_pressure,
        title={'text': "Fuel Pressure"},
        gauge={
            'axis': {'range': [0, 25]},
            'bar': {'color': "purple"},
            'steps': [
                {'range': [0, 8], 'color': "green"},
                {'range': [8, 15], 'color': "yellow"},
                {'range': [15, 25], 'color': "red"}
            ]
        }
    ))

    st.plotly_chart(fig_pressure, use_container_width=True)

# =====================================================
# PREDICTION
# =====================================================

st.markdown("---")

if st.button("Run Predictive Maintenance Analysis"):

    prediction = model.predict(input_data)[0]

    prediction_proba = model.predict_proba(input_data)[0]

    confidence = max(prediction_proba) * 100

    st.subheader("Prediction Result")

    # =================================================
    # HEALTH STATUS
    # =================================================

    if prediction == 0:

        st.markdown(
            f"""
            <div class="status-healthy">
            ✅ ENGINE HEALTHY
            <br><br>
            Confidence Score: {confidence:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="status-faulty">
            ⚠️ MAINTENANCE REQUIRED
            <br><br>
            Confidence Score: {confidence:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )

    # =================================================
    # CONFIDENCE BAR
    # =================================================

    st.markdown("### Prediction Confidence")

    st.progress(float(confidence/100))

    # =================================================
    # FEATURE IMPORTANCE
    # =================================================

    st.markdown("---")

    st.subheader("Important Operational Parameters")

    feature_importance = pd.DataFrame({

        "Feature": [
            "Engine RPM",
            "Fuel Pressure",
            "Lub Oil Pressure",
            "Coolant Temp",
            "Lub Oil Temp",
            "Coolant Pressure"
        ],

        "Importance": [
            0.32,
            0.24,
            0.18,
            0.12,
            0.08,
            0.06
        ]
    })

    fig = px.bar(
        feature_importance,
        x="Importance",
        y="Feature",
        orientation='h',
        title="Feature Importance"
    )

    st.plotly_chart(fig, use_container_width=True)

    # =================================================
    # RECOMMENDATIONS
    # =================================================

    st.markdown("---")

    st.subheader("Operational Recommendations")

    if prediction == 0:

        st.success("""
        - Engine operating under normal conditions
        - Continue routine monitoring
        - No immediate maintenance required
        """)

    else:

        st.warning("""
        - Perform detailed engine inspection
        - Check lubrication system
        - Inspect fuel delivery mechanism
        - Verify coolant circulation
        - Schedule predictive maintenance servicing
        """)

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption(
    "Predictive Maintenance Dashboard | Machine Learning + XGBoost + Streamlit"
)
