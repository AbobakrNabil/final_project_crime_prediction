import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "full_pipeline.pkl"
METADATA_PATH = BASE_DIR / "models" / "metadata.json"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)

st.set_page_config(page_title="SF Crime Prediction", page_icon="🚓", layout="centered")
st.title("San Francisco Crime Prediction")
st.write("Predict whether a reported crime incident is likely to be **LARCENY/THEFT** or **OTHER_CRIME** using time and location features.")

pipeline = load_model()
metadata = load_metadata()

st.sidebar.header("Incident Inputs")
incident_date = st.sidebar.date_input("Incident date")
hour = st.sidebar.slider("Incident hour", min_value=0, max_value=23, value=18)
pd_district = st.sidebar.selectbox(
    "Police district",
    ["BAYVIEW", "CENTRAL", "INGLESIDE", "MISSION", "NORTHERN", "PARK", "RICHMOND", "SOUTHERN", "TARAVAL", "TENDERLOIN"],
    index=3,
)
address_type = st.sidebar.selectbox("Address type", ["Block", "Intersection"])
x_coord = st.sidebar.number_input("Longitude (X)", value=-122.4194, min_value=-123.0, max_value=-121.0, format="%.6f")
y_coord = st.sidebar.number_input("Latitude (Y)", value=37.7749, min_value=37.0, max_value=38.5, format="%.6f")

day_of_week = incident_date.strftime("%A")
is_weekend = 1 if day_of_week in ["Saturday", "Sunday"] else 0
is_night = 1 if hour >= 22 or hour <= 5 else 0

input_df = pd.DataFrame([
    {
        "DayOfWeek": day_of_week,
        "PdDistrict": pd_district,
        "AddressType": address_type,
        "Hour": hour,
        "Month": incident_date.month,
        "Year": incident_date.year,
        "Day": incident_date.day,
        "IsWeekend": is_weekend,
        "IsNight": is_night,
        "X": x_coord,
        "Y": y_coord,
    }
])

st.subheader("Model Input")
st.dataframe(input_df, use_container_width=True)

if st.button("Predict"):
    prediction = int(pipeline.predict(input_df)[0])
    probability = None
    if hasattr(pipeline, "predict_proba"):
        probability = float(pipeline.predict_proba(input_df)[0][1])

    predicted_label = "LARCENY/THEFT" if prediction == 1 else "OTHER_CRIME"
    st.subheader("Prediction Result")
    st.success(f"Predicted class: {predicted_label}")
    if probability is not None:
        st.metric("Probability of LARCENY/THEFT", f"{probability:.2%}")

    st.caption("This model is intended for educational analysis and should not be used as an operational policing decision system.")

with st.expander("Model details"):
    st.write("Target definition:", metadata.get("target_definition"))
    st.write("Selected model:", metadata.get("best_model_metrics", {}).get("selected_model"))
    st.write("Precision:", metadata.get("best_model_metrics", {}).get("precision"))
    st.write("Recall:", metadata.get("best_model_metrics", {}).get("recall"))
    st.write("F1-score:", metadata.get("best_model_metrics", {}).get("f1_score"))
