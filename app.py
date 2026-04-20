import streamlit as st
import pandas as pd
import joblib





st.markdown(
     """
     <style>
     /* Background */
     .stApp {
         background-color: #1f1f1f;
     }
       /* Title styling */
     .main-title {
         font-size: 40px;
         font-weight: 800;
         color: #00c2ff;
         text-align: center;
         padding: 10px;
     }
    

     /* Normal text */
     p, label {
         color: #e0e0e0;
         font-size: 16px;
     }

     /* Input boxes */
     input, textarea {
         background-color: #ffffff !important;
         color: #000000 !important;
         border-radius: 8px;
     }

     /* Number input */
     div[data-baseweb="input"] {
         background-color: #ffffff !important;
         color: #000000 !important;
         border-radius: 8px;
     }
    /* Sub text */
     .sub-text {
         color: #e0e0e0;
         text-align: center;
         font-size: 18px;
     }

     /* Sidebar */
     section[data-testid="stSidebar"] {
         background-color: #2b2b2b;
     }

     /* Buttons */
     .stButton>button {
         background-color: #00c2ff;
         color: white;
         border-radius: 8px;
         font-weight: bold;
     }

     .stButton>button:hover {
         background-color: #0099cc;
     }
     </style>
     """,
     unsafe_allow_html=True
 )



# Bright Title


# Subtitle




model = joblib.load("SVC_diabetes.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load('columns.pkl')

st.title("Diabetes predication by tanishh ")
st.markdown('<div class="main-title">Provide the following details to check your  diabetes:</div>', unsafe_allow_html=True)

# collect th input data 
pregnancies = st.number_input("Pregnancies",0,17,1,1)
glucose = st.number_input("Glucose ",0,200,120)
bloodPressur = st.number_input('BloodPressur',0,140,70)
skinThickness =st.number_input("SkinThickness",0,100,20)
insulin = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.number_input("Age", 1, 120, 30)

if st.button("Predict"):

    raw_input ={
        'Pregnancies':pregnancies,
        "Glucose":glucose,
        "loodPressur": bloodPressur ,
        "SkinThickness":skinThickness,
        "Insulin":insulin,
        "Bmi":bmi,
        "Age":age
    }
    # Create input dataframe
    input_df = pd.DataFrame([raw_input])

    # Fill in missing columns with 0s
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns
    input_df = input_df[columns]

    # Scale the input
    scaled_input = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(scaled_input)[0]

    # Show result
    if prediction == 1:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")