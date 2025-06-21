import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ----------- Load Models and Encodings -----------
xgb_model = joblib.load("models/xgb_model.pkl")
scaler = joblib.load("models/scaler.pkl")
venue_encoding_map = joblib.load("models/venue_encoding_map.pkl")
fallback_value = joblib.load("models/fallback_value.pkl")
nn_model = load_model("models/nn_model.keras")

# ----------- Venue Encoding Function -----------
def encode_venue(user_venue, venue_map, fallback_value):
    return venue_map.get(user_venue, fallback_value)

# ----------- Prediction Function -----------
def predict_score(inning, balls_bowled, cumulative_runs, cumulative_wickets, venue, model_choice):
    venue_encoded = encode_venue(venue, venue_encoding_map, fallback_value)

    input_features = np.array([[inning, balls_bowled, cumulative_runs, cumulative_wickets, venue_encoded]])

    if model_choice == "Neural Network":
        input_scaled = scaler.transform(input_features)
        prediction = nn_model.predict(input_scaled)[0][0]
    else:
        prediction = xgb_model.predict(input_features)[0]

    return prediction

# ----------- Streamlit UI -----------
st.set_page_config(page_title="Cricket Score Predictor", layout="centered")
st.title("🏏 IPL Final Score Predictor")
st.markdown("**Predict the final score of an IPL innings using current match stats.**")

# Input fields
inning = st.selectbox("Inning", [1, 2])
balls_bowled = st.number_input("Balls Bowled", min_value=0, max_value=300, value=60)
cumulative_runs = st.number_input("Cumulative Runs", min_value=0, value=80)
cumulative_wickets = st.number_input("Cumulative Wickets", min_value=0, max_value=10, value=3)
venue = st.selectbox("Venue", list(venue_encoding_map.keys()))
model_choice = st.radio("Choose Model", ["XGBoost", "Neural Network"])

# Predict button
if st.button("Predict Final Score"):
    if venue.strip() == "":
        st.warning("Please enter a venue.")
    else:
        prediction = predict_score(inning, balls_bowled, cumulative_runs, cumulative_wickets, venue, model_choice)
        st.success(f"🏏 Predicted Final Score: {prediction:.2f}")
