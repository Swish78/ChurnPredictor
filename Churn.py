import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the pre-trained pipeline
with open('churn_model_pipeline.pkl', 'rb') as file:
    pipeline = pickle.load(file)


def main():
    st.title('Churn Predictor')

    st.write('Enter customer information:')

    age = st.number_input('Age', min_value=18, max_value=100)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Miami']
    location = st.selectbox('Location', cities)
    subscription_length = st.number_input('Subscription Length (Months)', min_value=1, max_value=60)
    monthly_bill = st.number_input('Monthly Bill', min_value=0.0, max_value=1000.0)
    total_usage_gb = st.number_input('Total Usage (GB)', min_value=0.0, max_value=1000.0)

    # Create a DataFrame from user input
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Location': [location],
        'Subscription_Length_Months': [subscription_length],
        'Monthly_Bill': [monthly_bill],
        'Total_Usage_GB': [total_usage_gb]
    })

    if st.button('Predict'):
        # Use the pre-trained pipeline to make predictions
        prediction = pipeline.predict(input_data)

        # Display the prediction
        if prediction[0] == 1:
            st.error('The customer is likely to churn.')
        else:
            st.success('The customer is likely to stay.')


if __name__ == '__main__':
    main()
