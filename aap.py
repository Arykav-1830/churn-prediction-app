import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))

st.title("Customer Churn Prediction")

feature1 = st.number_input("Enter Feature 1")
feature2 = st.number_input("Enter Feature 2")

if st.button("Predict"):
    result = model.predict([[feature1, feature2]])
    
    if result == 1:
        st.error("Customer will leave ❌")
    else:
        st.success("Customer will stay ✅")
