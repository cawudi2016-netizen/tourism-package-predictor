import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Wellness Tourism Package Purchase Predictor", layout="centered")

st.title("🌴 Wellness Tourism Package Predictor")
st.write("Enter customer details below to predict if they are likely to purchase the new Wellness Tourism Package.")

# Path to the saved model
MODEL_PATH = "tourism_project/deployment/model.joblib"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

if model is None:
    st.error(f"Model file not found at '{MODEL_PATH}'. Please ensure the pipeline has successfully trained and saved the model.")
else:
    # Creating form layout matching the dataset columns
    with st.form("prediction_form"):
        st.subheader("Customer Demographics & Information")
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
            city_tier = st.selectbox("City Tier", [1, 2, 3], index=0)
            gender = st.selectbox("Gender", ["Male", "Female"], index=0)
            occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"], index=0)
            designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "Avp", "VP"], index=0)
            monthly_income = st.number_input("Monthly Income", min_value=0, value=25000, step=1000)

        with col2:
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"], index=1)
            passport = st.selectbox("Has Passport?", ["No", "Yes"], index=0)
            own_car = st.selectbox("Owns Car?", ["No", "Yes"], index=1)
            num_trips = st.number_input("Number of Trips Per Year", min_value=0, max_value=20, value=3, step=1)
            num_person = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=3, step=1)
            num_children = st.number_input("Number of Children Visiting (under 5)", min_value=0, max_value=5, value=1, step=1)

        st.subheader("Customer Engagement & Preferences")
        col3, col4 = st.columns(2)

        with col3:
            contact_type = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"], index=0)
            prop_stars = st.selectbox("Preferred Property Star Rating", [3, 4, 5], index=0)
            pitch_duration = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=120, value=15, step=1)

        with col4:
            product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"], index=0)
            num_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=15, value=3, step=1)
            pitch_sat = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3, step=1)

        submitted = st.form_submit_button("Predict Purchase Likelihood")

        if submitted:
            # Process categorical mappings to match training features
            passport_val = 1 if passport == "Yes" else 0
            own_car_val = 1 if own_car == "Yes" else 0

            input_data = pd.DataFrame([{
                'Age': age,
                'TypeofContact': contact_type,
                'CityTier': city_tier,
                'DurationOfPitch': pitch_duration,
                'Occupation': occupation,
                'Gender': gender,
                'NumberOfPersonVisiting': num_person,
                'NumberOfFollowups': num_followups,
                'ProductPitched': product_pitched,
                'PreferredPropertyStar': prop_stars,
                'MaritalStatus': marital_status,
                'NumberOfTrips': num_trips,
                'Passport': passport_val,
                'PitchSatisfactionScore': pitch_sat,
                'OwnCar': own_car_val,
                'NumberOfChildrenVisiting': num_children,
                'Designation': designation,
                'MonthlyIncome': monthly_income
            }])

            try:
                prediction = model.predict(input_data)[0]
                proba = model.predict_proba(input_data)[0][1]

                st.markdown("--- ")
                if prediction == 1:
                    st.success(f"🎉 **High Likelihood!** The customer is likely to purchase the Wellness Tourism Package (Probability: {proba:.2%})")
                else:
                    st.info(f"⚠️ **Low Likelihood.** The customer is unlikely to purchase the Wellness Tourism Package (Probability: {proba:.2%})")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
