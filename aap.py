import streamlit as st
import pickle
import pandas as pd
import numpy as np


model = pickle.load(open('model.pkl', 'rb'))

st.title("Customer Churn Prediction")


feature1 = st.number_input("Enter Feature 1", value=0.0)
feature2 = st.number_input("Enter Feature 2", value=0.0)

if st.button("Predict"):
  
    input_data = pd.DataFrame(
        [[feature1, feature2]], 
        columns=['Column1', 'Column2'] 
    )
    
    try:
        
        result = model.predict(input_data)
        
       
        if result[0] == 1:
            st.error("Customer will leave ❌")
        else:
            st.success("Customer will stay ✅")
            
    except Exception as e:
        st.error(f"Prediction failed: {e}")
