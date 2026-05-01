import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the model
try:
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model file 'model.pkl' not found. Please upload it to your repository.")
    st.stop()

st.title("Customer Churn Prediction")

# 1. Get the exact list of columns the model was trained on
# This prevents the errors seen in image_74a0e7.png
try:
    expected_columns = model.feature_names_in_
except AttributeError:
    st.error("Error: The model was not trained with feature names. Please re-train using a DataFrame.")
    st.stop()

# 2. Create the User Interface
# We'll create inputs for the main features identified in your error logs
st.subheader("User Information")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=100, value=30)
    income = st.selectbox("Annual Income Class", ["Low Income", "Middle Income", "High Income"])

with col2:
    frequent_flyer = st.selectbox("Frequent Flyer?", ["No", "Yes", "No Record"])
    booked_hotel = st.selectbox("Booked Hotel Before?", ["No", "Yes"])

if st.button("Predict"):
    # 3. Create a DataFrame with ALL required columns initialized to 0
    # This satisfies the model's requirement for all features shown in your screenshots
    input_df = pd.DataFrame(0, index=[0], columns=expected_columns)
    
    # 4. Map the UI inputs to the specific DataFrame columns
    # We use a helper to match the One-Hot Encoding format (e.g., FeatureName_Value)
    
    if 'Age' in input_df.columns:
        input_df['Age'] = age
        
    # Mapping categorical dropdowns to One-Hot columns
    mappings = {
        f"AnnualIncomeClass_{income}": 1,
        f"FrequentFlyer_{frequent_flyer}": 1,
        f"BookedHotelOrNot_{booked_hotel}": 1
    }
    
    for col, value in mappings.items():
        if col in input_df.columns:
            input_df[col] = value

    try:
        # 5. Make the Prediction
        prediction = model.predict(input_df)
        
        st.divider()
        if prediction[0] == 1:
            st.error("### Result: Customer is likely to churn ❌")
        else:
            st.success("### Result: Customer is likely to stay ✅")
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")

# Helpful sidebar to see what's happening under the hood
with st.sidebar:
    st.write("### Model Requirements")
    st.write(f"This model expects **{len(expected_columns)}** features.")
    if st.checkbox("Show all feature names"):
        st.write(list(expected_columns))
