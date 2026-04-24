import os
import pickle
import pandas as pd
import streamlit as st

st.title("Titanic Survival Predictor")
st.write("Enter passenger details to predict survival:")

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "titanic.pkl")
model = pickle.load(open(MODEL_PATH, "rb"))

# Inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=25)
sibspouse = st.number_input("Siblings/Spouse", min_value=0, max_value=10, value=0)
parentchild = st.number_input("Parents/Children", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, value=50.0)
SURVIVAL_THRESHOLD = 0.5
st.write("Survival Probability Threshold:", SURVIVAL_THRESHOLD)
# Predict button
if st.button("Predict"):
    
    input_data = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [sex],
        "Age": [age],
        "sibspouse": [sibspouse],
        "parentchild": [parentchild],
        "Fare": [fare]
    })

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("The passenger is likely to SURVIVE")
    else:
        st.error("The passenger is NOT likely to survive")
