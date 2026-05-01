import streamlit as st
import pickle
import pandas as pd

model = pickle.load(open('model.pkl', 'rb'))

st.title("Customer Churn Prediction")


age = st.number_input("Enter Age", value=30)

income = st.selectbox("Income Class", ["Low Income", "High Income"])

if st.button("Predict"):
  
    expected_columns = [
        'AccountSyncedToSocialMedia_No', 
        'AccountSyncedToSocialMedia_Yes', 
        'Age', 
        'AnnualIncomeClass_High Income', 
        'AnnualIncomeClass_Low Income'
       
    ]
    

    input_df = pd.DataFrame(0, index=[0], columns=expected_columns)
    
   
    input_df['Age'] = age
    if income == "High Income":
        input_df['AnnualIncomeClass_High Income'] = 1
    else:
        input_df['AnnualIncomeClass_Low Income'] = 1
        
    try:
      
        result = model.predict(input_df)
        
        if result[0] == 1:
            st.error("Customer will leave ❌")
        else:
            st.success("Customer will stay ✅")
            
    except Exception as e:
        st.error(f"Prediction failed: {e}")
