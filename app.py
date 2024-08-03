import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Ensure the encoder has the correct categories
onehot_encoder_geo.categories_ = [np.array(['France', 'Germany', 'Spain'])]

# Streamlit app title
st.title('Customer Churn Prediction')

# User input
st.header('Please provide the following details:')
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92, value=30)
balance = st.number_input('Balance', value=0.0)
credit_score = st.number_input('Credit Score', value=600)
estimated_salary = st.number_input('Estimated Salary', value=50000)
tenure = st.slider('Tenure', 0, 10, value=3)
num_of_products = st.slider('Number of Products', 1, 4, value=1)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
try:
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # One-hot encode 'Geography'
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()

    # Get feature names for one-hot encoding
    geo_feature_names = onehot_encoder_geo.get_feature_names_out(['Geography'])

    geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_feature_names)

    # Combine one-hot encoded columns with input data
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Add feature names to input data
    input_data.columns = ['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
                          'IsActiveMember', 'EstimatedSalary'] + list(geo_feature_names)

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Predict churn
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    st.subheader(f'Churn Probability: {prediction_proba:.2f}')

    if prediction_proba > 0.5:
        st.error('The customer is likely to churn.')
    else:
        st.success('The customer is not likely to churn.')

except Exception as e:
    st.error(f'An error occurred: {e}')
