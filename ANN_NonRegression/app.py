import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

### Load the trained model, scaler pickle and onehot encodeing pickle
model = tf.keras.models.load_model('model.h5')

## load the encoder and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    label_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


###n Let's use Streamlit App
st.title('Customer Churn Prediction')
st.write('This is a simple Customer Churn Prediction App')

# Create a form to take input from the user
with st.form(key='churn_form'):
    credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=600)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    age = st.number_input('Age', min_value=18, max_value=100, value=30)
    tenure = st.number_input('Tenure', min_value=0, max_value=10, value=5)
    balance = st.number_input('Balance', min_value=0.0, value=10000.0)
    num_of_products = st.number_input('Number of Products', min_value=1, max_value=4, value=1)
    has_cr_card = st.selectbox('Has Credit Card', [0, 1])
    is_active_member = st.selectbox('Is Active Member', [0, 1])
    estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=50000.0)
    geography = st.selectbox('Geography', ['France', 'Germany', 'Spain'])
    
    submit_button = st.form_submit_button(label='Predict')

if submit_button:
    input_data = {
        "CreditScore": credit_score,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_of_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active_member,
        "EstimatedSalary": estimated_salary,
        "Geography": geography
    }
    
    # Process the input data
    input_df = pd.DataFrame([input_data])
    geo_encoded = label_encoder_geo.transform([[input_data['Geography']]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=label_encoder_geo.get_feature_names_out(['Geography']))
    input_df = input_df.drop(columns=['Geography'])
    input_df = pd.concat([input_df, geo_encoded_df], axis=1)
    input_df['Gender'] = label_encoder_gender.transform(input_df['Gender'])
    #I want to show the encoded data
    st.write(input_df)
    input_data_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]
    
    # Display the result
    if prediction_proba > 0.5:
        st.write("The customer is likely to leave the bank")
    else:
        st.write("The customer is likely to stay with the bank")
