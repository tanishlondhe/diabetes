import streamlit as st
import pandas as pd
import joblib

model = joblib.load("SVC_Dia2.pkl")
scaler = joblib.load("scaler_d2.pkl")
columns = joblib.load('columns_d2.pkl')


st.title("Diabetes Prediction by tanishh")
st.markdown("Provide the following details to check your diabetes:")

# Input fields
pregnancies = st.number_input("Pregnancies", 0, 17, 1)
glucose = st.number_input("Glucose", 0, 200, 120)
bloodPressure = st.number_input("BloodPressure", 0, 140, 70)
skinThickness = st.number_input("SkinThickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.number_input("Age", 1, 120, 30)

if st.button("Predict"):

    raw_input = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bloodPressure,
        'SkinThickness': skinThickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }

    input_df = pd.DataFrame([raw_input])

    input_df = pd.DataFrame(columns=columns)
    input_df.loc[0] = 0
    input_df['Pregnancies'] = pregnancies
    input_df['Glucose'] = glucose
    input_df['BloodPressure'] = bloodPressure
    input_df['SkinThickness'] = skinThickness
    input_df['Insulin'] = insulin
    input_df['BMI'] = bmi
    input_df['DiabetesPedigreeFunction'] = dpf
    input_df['Age'] = age

    # Scale
    scaled_input = scaler.transform(input_df)

    # Predict
    prediction = model.predict(scaled_input)[0]

    # Output
    if prediction == 1:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")