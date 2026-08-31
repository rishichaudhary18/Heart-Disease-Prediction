import streamlit as st
import pandas as pd
import joblib

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="HeartCare | Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #f5f7fb;
}

/* Remove default top padding */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* ---------------- HEADER ---------------- */

.hero {
    background: linear-gradient(135deg, #b31217 0%, #e52d27 55%, #ff5f6d 100%);
    padding: 38px 45px;
    border-radius: 24px;
    color: white;
    margin-bottom: 30px;
    box-shadow: 0 12px 35px rgba(179, 18, 23, 0.20);
}

.hero-content {
    display: flex;
    align-items: center;
    gap: 25px;
}

.heart-icon {
    font-size: 65px;
    background: rgba(255,255,255,0.15);
    width: 100px;
    height: 100px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
}

.hero h1 {
    font-size: 38px;
    font-weight: 800;
    margin: 0;
    letter-spacing: -1px;
}

.hero p {
    font-size: 16px;
    margin-top: 8px;
    opacity: 0.92;
}

/* ---------------- SECTION TITLES ---------------- */

.section-title {
    font-size: 22px;
    font-weight: 700;
    color: #1f2937;
    margin-top: 15px;
    margin-bottom: 5px;
}

.section-subtitle {
    color: #6b7280;
    font-size: 14px;
    margin-bottom: 18px;
}

/* ---------------- CARDS ---------------- */

.input-card {
    background: white;
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 5px 20px rgba(0,0,0,0.06);
    border: 1px solid #edf0f5;
}

.info-card {
    background: white;
    border-radius: 18px;
    padding: 22px;
    box-shadow: 0 5px 20px rgba(0,0,0,0.05);
    border: 1px solid #edf0f5;
    margin-top: 15px;
}

.info-card h3 {
    color: #b31217;
    margin-top: 0;
}

.info-card p {
    color: #6b7280;
    font-size: 14px;
    line-height: 1.6;
}

/* ---------------- STREAMLIT INPUTS ---------------- */

label {
    font-weight: 600 !important;
    color: #374151 !important;
}

div[data-baseweb="select"] > div {
    border-radius: 10px;
}

div[data-testid="stNumberInput"] input {
    border-radius: 10px;
}

div[data-testid="stSlider"] {
    padding-bottom: 8px;
}

/* ---------------- BUTTON ---------------- */

.stButton > button {
    width: 100%;
    height: 52px;
    border-radius: 12px;
    border: none;
    background: linear-gradient(135deg, #b31217, #e52d27);
    color: white;
    font-size: 17px;
    font-weight: 700;
    box-shadow: 0 7px 20px rgba(179,18,23,0.22);
    transition: all 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(179,18,23,0.30);
}

/* ---------------- RESULT CARDS ---------------- */

.result-high {
    background: linear-gradient(135deg, #fff1f2, #ffe4e6);
    border: 2px solid #ef4444;
    border-radius: 18px;
    padding: 28px;
    text-align: center;
    margin-top: 25px;
}

.result-high h2 {
    color: #b91c1c;
    margin-bottom: 8px;
}

.result-high p {
    color: #7f1d1d;
}

.result-low {
    background: linear-gradient(135deg, #ecfdf5, #d1fae5);
    border: 2px solid #22c55e;
    border-radius: 18px;
    padding: 28px;
    text-align: center;
    margin-top: 25px;
}

.result-low h2 {
    color: #15803d;
    margin-bottom: 8px;
}

.result-low p {
    color: #166534;
}

/* ---------------- BADGES ---------------- */

.badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 30px;
    font-size: 12px;
    font-weight: 700;
    background: rgba(255,255,255,0.18);
    margin-top: 10px;
}

/* ---------------- FOOTER ---------------- */

.footer {
    text-align: center;
    color: #9ca3af;
    font-size: 13px;
    margin-top: 45px;
    padding: 20px;
}

/* ---------------- MOBILE ---------------- */

@media (max-width: 768px) {

    .hero {
        padding: 28px;
    }

    .hero-content {
        flex-direction: column;
        text-align: center;
    }

    .hero h1 {
        font-size: 29px;
    }

    .heart-icon {
        width: 75px;
        height: 75px;
        font-size: 45px;
    }

}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HERO HEADER
# ============================================================

st.markdown("""
<div class="hero">
<div class="hero-content">
<div class="heart-icon">❤️</div>
<div>
<h1>HeartCare</h1>
<p>AI-Powered Heart Disease Risk Prediction</p>
<span class="badge">Machine Learning • KNN Model</span>
</div>
</div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# INTRO
# ============================================================

st.markdown("""
<div class="section-title">Patient Information</div>
<div class="section-subtitle">
Enter the patient's health information below to generate a prediction.
</div>
""", unsafe_allow_html=True)


# ============================================================
# INPUT CARDS
# ============================================================

col1, col2 = st.columns(2, gap="large")


# ---------------- LEFT COLUMN ----------------

with col1:

    st.markdown("""
    <div class="input-card">
    <h3>👤 Personal Information</h3>
    </div>
    """, unsafe_allow_html=True)

    age = st.slider(
        "Age",
        18,
        100,
        40
    )

    sex = st.selectbox(
        "Sex",
        ["M", "F"]
    )

    st.markdown("""
    <div class="input-card">
        <h3>🫀 Cardiovascular Information</h3>
    </div>
    """, unsafe_allow_html=True)

    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "TA", "ASY"]
    )

    resting_bp = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        80,
        200,
        120
    )

    cholesterol = st.number_input(
        "Cholesterol (mg/dL)",
        100,
        600,
        200
    )

    fasting_bs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dL",
        [0, 1]
    )


# ---------------- RIGHT COLUMN ----------------

with col2:

    st.markdown("""
    <div class="input-card">
        <h3>📊 Heart Test Results</h3>
    </div>
    """, unsafe_allow_html=True)

    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"]
    )

    max_hr = st.slider(
        "Maximum Heart Rate",
        60,
        220,
        150
    )

    exercise_angina = st.selectbox(
        "Exercise-Induced Angina",
        ["Y", "N"]
    )

    oldpeak = st.slider(
        "Oldpeak (ST Depression)",
        0.0,
        6.0,
        1.0
    )

    st_slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"]
    )

    st.markdown("""
    <div class="info-card">
        <h3>💡 About This Prediction</h3>
        <p>
        This application uses a trained K-Nearest Neighbors (KNN)
        machine learning model to estimate heart disease risk
        based on the information provided above.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

predict_col = st.columns([1, 2, 1])[1]

with predict_col:

    predict = st.button("🔍 Predict Heart Disease Risk")


# ============================================================
# PREDICTION LOGIC
# ============================================================

if predict:

    # Create raw input dictionary
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    # Create dataframe
    input_df = pd.DataFrame([raw_input])

    # Add missing columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns
    input_df = input_df[expected_columns]

    # Scale input
    scaled_input = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(scaled_input)[0]


    # ========================================================
    # RESULT
    # ========================================================

    st.markdown(
        '<div class="section-title">Prediction Result</div>',
        unsafe_allow_html=True
    )

    if prediction == 1:

        st.markdown("""
        <div class="result-high">
            <h2>⚠️ High Risk Detected</h2>
            <p>
            The model predicts a higher likelihood of heart disease
            based on the provided information.
            </p>
            <strong>Please consult a qualified healthcare professional
            for proper medical evaluation.</strong>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class="result-low">
            <h2>✅ Low Risk Detected</h2>
            <p>
            The model predicts a lower likelihood of heart disease
            based on the provided information.
            </p>
            <strong>Continue maintaining a healthy lifestyle and
            regular health checkups.</strong>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    <strong>HeartCare</strong> • Heart Disease Prediction System<br>
    Developed by Rishi • Machine Learning Project<br><br>
    ⚠️ This application is for educational purposes only and is
    not a substitute for professional medical advice.
</div>
""", unsafe_allow_html=True)