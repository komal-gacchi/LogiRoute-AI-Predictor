import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# 1. Page Configuration & Theme Styling
st.set_page_config(
    page_title="LogiRoute Predictor",
    page_icon="🚚",
    layout="centered"
)

# Custom Premium CSS Styling (Matches your dashboard vibe)
st.markdown("""
    <style>
    .main { background-color: #0B132B; color: white; }
    .stButton>button {
        background-color: #10B981; color: white; 
        font-weight: bold; border-radius: 8px; width: 100%;
    }
    .stSelectbox, .stNumberInput { color: black; }
    h1 { color: #22C55E; text-align: center; }
    .result-box {
        background-color: #1E293B; padding: 20px; 
        border-radius: 10px; border-left: 5px solid #10B981;
        text-align: center; margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🚚 LogiRoute AI - Delivery Time Predictor")
st.write("Enter trip details below to forecast delivery latency using the trained Random Forest Model.")

# 2. Artifacts Load Karein (Relative Path for Cloud & Local Compliance)
MODEL_PATH = "logiroute_model.pkl"

@st.cache_resource
def load_artifacts():
    if os.path.exists(MODEL_PATH):
        artifacts = joblib.load(MODEL_PATH)
        return artifacts['model'], artifacts['label_encoders']
    else:
        return None, None

try:
    model, label_encoders = load_artifacts()
    if model is None:
        raise FileNotFoundError
except Exception as e:
    st.error(f"Error: '{MODEL_PATH}' file nahi mili ya corrupt hai. Pehle check karein ki file sahi jagah par hai.")
    st.stop()

# 3. User Input Form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        start_loc = st.selectbox("Start Location", label_encoders['Start_Location'].classes_)
        end_loc = st.selectbox("End Location", label_encoders['End_Location'].classes_)
        distance = st.number_input("Distance (KM)", min_value=0.1, max_value=50.0, value=5.0, step=0.1)
        traffic = st.selectbox("Traffic Level", label_encoders['Traffic_Level'].classes_)
        
    with col2:
        weather = st.selectbox("Weather Condition", label_encoders['Weather'].classes_)
        time_of_day = st.selectbox("Time of Day", label_encoders['Time_of_Day'].classes_)
        order_type = st.selectbox("Order Type", label_encoders['Order_Type'].classes_)
        vehicle = st.selectbox("Vehicle Type", label_encoders['Vehicle_Type'].classes_)
        
    metro_zone = st.radio("Is Metro Construction Zone?", ["No (0)", "Yes (1)"], horizontal=True)
    metro_val = 1 if "Yes" in metro_zone else 0

    submit_btn = st.form_submit_button("Forecast Delivery Time")

# 4. Prediction Logic (With Duplicate Fix and Smart Logic Verification)
if submit_btn:
    if start_loc == end_loc:
        # Agar start aur end location same hai toh model bypass karke alert box dikhao
        st.markdown(f"""
            <div class="result-box" style="border-left: 5px solid #EF4444; background-color: #1E293B;">
                <h3 style="color: white; margin-bottom: 5px;">⏱ nighttime Predicted Delivery Time</h3>
                <h1 style="color: #EF4444; font-size: 35px; margin: 10px 0;">0.0 Minutes</h1>
                <p style="color: #94A3B8; font-size: 13px;">⚠️ Source and Destination are the same location.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        try:
            # Text values ko unke respective label encoder se number me badlein
            input_data = pd.DataFrame([{
                'Start_Location': label_encoders['Start_Location'].transform([start_loc])[0],
                'End_Location': label_encoders['End_Location'].transform([end_loc])[0],
                'Distance_KM': distance,
                'Traffic_Level': label_encoders['Traffic_Level'].transform([traffic])[0],
                'Weather': label_encoders['Weather'].transform([weather])[0],
                'Time_of_Day': label_encoders['Time_of_Day'].transform([time_of_day])[0],
                'Is_Metro_Construction_Zone': metro_val,
                'Order_Type': label_encoders['Order_Type'].transform([order_type])[0],
                'Vehicle_Type': label_encoders['Vehicle_Type'].transform([vehicle])[0]
            }])
            
            # Model se prediction nikalna
            predicted_time = model.predict(input_data)[0]
            
            # Normal Success Display (Green Box)
            st.markdown(f"""
                <div class="result-box">
                    <h3 style="color: white; margin-bottom: 5px;">⏱️ Predicted Delivery Time</h3>
                    <h1 style="color: #10B981; font-size: 40px; margin: 10px 0;">{round(predicted_time, 1)} Minutes</h1>
                    <p style="color: #94A3B8; font-size: 13px;">Model Confidence (R²): 98.78% | Target: Actual_Time_Minutes</p>
                </div>
            """, unsafe_allow_html=True)
            
        except Exception as prediction_error:
            st.error(f"Prediction ke samay unexpected error aaya: {prediction_error}")
