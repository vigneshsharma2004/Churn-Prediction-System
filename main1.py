import streamlit as st
import pandas as pd
import pickle

# Load model and base dataset
model = pickle.load(open("model.sav", "rb"))
df_1 = pd.read_csv("first_telc.csv")

st.title("Customer Churn Prediction")

# Input form
with st.form("churn_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0)
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    MonthlyCharges = st.number_input("Monthly Charges")
    TotalCharges = st.text_input("Total Charges")  # Handle empty strings

    submit = st.form_submit_button("Predict")

if submit:
    try:
        TotalCharges = float(TotalCharges)
    except:
        st.error("Total Charges must be a valid number.")
        st.stop()

    # Input data
    data = [[SeniorCitizen, MonthlyCharges, TotalCharges, gender, Partner, Dependents,
             PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup,
             DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract,
             PaperlessBilling, PaymentMethod, tenure]]

    new_df = pd.DataFrame(data, columns=['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender', 
                                         'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                                         'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                         'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                                         'PaymentMethod', 'tenure'])

    # Combine with original dataframe
    df_2 = pd.concat([df_1, new_df], ignore_index=True)
    
    df_2.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
    df_2['tenure'] = df_2['tenure'].fillna(df_2['tenure'].median())


    # Tenure group
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    df_2['tenure_group'] = pd.cut(df_2.tenure.astype(int), range(1, 80, 12), right=False, labels=labels)

    df_2.drop(columns=['tenure'], axis=1, inplace=True)

    # One-hot encode
    df_dummies = pd.get_dummies(df_2[[
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group']])
    
    model_features = ['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender', 
                  'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                  'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                  'DeviceProtection', 'TechSupport', 'StreamingTV', 
                  'StreamingMovies', 'Contract', 'PaperlessBilling', 
                  'PaymentMethod', 'tenure_group']

    # Align columns to match training data
    df_dummies = df_dummies.reindex(columns=model_features, fill_value=0)

    # Get input features (last row)
    input_features = df_dummies.tail(1)

    # Predict
    prediction = model.predict(input_features)[0]
    probability = model.predict_proba(input_features)[0][1]

    # Output
    if prediction == 1:
        st.error(f"⚠️ The customer is **likely to churn**.\n\nConfidence: {probability * 100:.2f}%")
    else:
        st.success(f"✅ The customer is **likely to stay**.\n\nConfidence: {probability * 100:.2f}%")