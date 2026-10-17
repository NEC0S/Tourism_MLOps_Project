# Streamlit frontend for the Wellness Tourism Package prediction model
import os
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

HF_USER = os.getenv("HF_USER", "abhi261204")
MODEL_REPO = f"{HF_USER}/tourism-package-model"

@st.cache_resource
def load_model():
    # Download the registered model from the Hugging Face Hub (falls back to a local file)
    if os.path.exists("best_tourism_model_v1.joblib"):
        return joblib.load("best_tourism_model_v1.joblib")
    path = hf_hub_download(repo_id=MODEL_REPO, filename="best_tourism_model_v1.joblib")
    return joblib.load(path)

model = load_model()

st.title("Wellness Tourism Package - Purchase Predictor")
st.write("Enter the customer details to predict whether they will purchase the Wellness Tourism Package.")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 18, 100, 35)
    type_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    persons = st.number_input("Number of Persons Visiting", 1, 10, 3)
    star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    trips = st.number_input("Number of Trips per Year", 0, 25, 3)
with col2:
    passport = st.selectbox("Has Passport?", ["Yes", "No"])
    own_car = st.selectbox("Owns Car?", ["Yes", "No"])
    children = st.number_input("Children Visiting (below 5)", 0, 5, 0)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    income = st.number_input("Monthly Income", 1000, 200000, 20000)
    satisfaction = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    product = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    followups = st.number_input("Number of Follow-ups", 0, 10, 3)
    duration = st.number_input("Duration of Pitch (min)", 1, 150, 15)

input_df = pd.DataFrame([{
    "Age": age, "TypeofContact": type_contact, "CityTier": city_tier, "DurationOfPitch": duration,
    "Occupation": occupation, "Gender": gender, "NumberOfPersonVisiting": persons,
    "NumberOfFollowups": followups, "ProductPitched": product, "PreferredPropertyStar": star,
    "MaritalStatus": marital, "NumberOfTrips": trips, "Passport": 1 if passport == "Yes" else 0,
    "PitchSatisfactionScore": satisfaction, "OwnCar": 1 if own_car == "Yes" else 0,
    "NumberOfChildrenVisiting": children, "Designation": designation, "MonthlyIncome": income}])

THRESHOLD = 0.45
if st.button("Predict"):
    prob = model.predict_proba(input_df)[0, 1]
    st.metric("Purchase probability", f"{prob:.1%}")
    if prob >= THRESHOLD:
        st.success("Likely to purchase - prioritise this customer for the campaign.")
    else:
        st.warning("Unlikely to purchase - low priority for this campaign.")
