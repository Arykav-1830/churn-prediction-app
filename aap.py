import streamlit as st
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open('model.pkl', 'rb'))
columns = pickle.load(open('columns.pkl', 'rb'))

st.title("Customer Churn Prediction")


input_df = pd.DataFrame(columns=columns)


input_df.loc[0, 'Feature1'] = st.number_input("Feature1")
input_df.loc[0, 'Feature2'] = st.number_input("Feature2")


input_df = input_df.fillna(0)

if st.button("Predict"):
    result = model.predict(input_df)

   if result[0] == 1:
        st.error("Customer will leave ❌")
    else:
        st.success("Customer will stay ✅")
