# coding: utf-8

import pandas as pd
from flask import Flask, request, render_template
import pickle
import traceback

app = Flask("__name__")

# ========== Load Data and Model ==========
try:
    df_1 = pd.read_csv("first_telc.csv")
    model = pickle.load(open("model.sav", "rb"))
except FileNotFoundError as e:
    print(f"File not found: {e}")
except Exception as e:
    print(f"Error loading files: {traceback.format_exc()}")

# ========== Flask Routes ==========
@app.route("/")
def loadPage():
    return render_template('home.html', query="")

@app.route("/", methods=['POST'])
def predict():
    try:
        # ========== Collect Form Data ==========
        form_data = {f"query{i}": request.form[f"query{i}"] for i in range(1, 20)}
        data = [list(form_data.values())]
        
        # ========== Create DataFrame ==========
        columns = [
            'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender', 
            'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
            'PaymentMethod', 'tenure'
        ]
        
        new_df = pd.DataFrame(data, columns=columns)

        # ========== Data Type Conversion ==========
        for col in ['MonthlyCharges', 'TotalCharges', 'tenure']:
            new_df[col] = pd.to_numeric(new_df[col], errors='coerce')
        if new_df.isnull().values.any():
            print(f"Null values found in input data: {new_df.isna().sum()}")

        # ========== Tenure Binning ==========
        labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
        new_df['tenure_group'] = pd.cut(new_df.tenure.fillna(0).astype(int), range(1, 80, 12), right=False, labels=labels)
        new_df.drop(columns=['tenure'], inplace=True)

        # ========== One-Hot Encoding ==========
        new_df_dummies = pd.get_dummies(new_df[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                                                'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                                                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                                                'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group']])

        # ========== Model Prediction ==========
        single = model.predict(new_df_dummies)
        probability = model.predict_proba(new_df_dummies)[:, 1]

        # ========== Generate Output ==========
        if single == 1:
            o1 = "This customer is likely to be churned!!"
        else:
            o1 = "This customer is likely to continue!!"
        
        o2 = f"Confidence: {round(probability[0] * 100, 2)}%"
        
        return render_template('home.html', output1=o1, output2=o2, **form_data)

    except Exception as e:
        print(f"Error during prediction: {traceback.format_exc()}")
        return "Internal Server Error", 500

if __name__ == "__main__":
    app.run(debug=True)
