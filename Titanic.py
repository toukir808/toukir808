# titanic_app.py

import streamlit as st
import pandas as pd
import pickle
# from sklearn.linear_model import LogisticRegression

# ----------------------------
# Load Model (train first and save)
# ----------------------------
# Train your model separately and save it:
# with open("titanic_model.pkl", "wb") as f:
#     pickle.dump(model, f)

with open("titanic_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🚢 Titanic Survival Prediction App")

# ----------------------------
# User Inputs
# ----------------------------
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=25)
sibsp = st.number_input("Number of Siblings/Spouses (SibSp)", min_value=0, max_value=8, value=0)
parch = st.number_input("Number of Parents/Children (Parch)", min_value=0, max_value=6, value=0)
fare = st.number_input("Fare", min_value=0.0, value=30.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# ----------------------------
# Preprocess Input
# ----------------------------
# Convert categorical values into same format as training
sex_male = 1 if sex == "male" else 0
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

# Create dataframe with correct column order
input_data = pd.DataFrame({
    "Pclass": [pclass],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare],
    "Sex_male": [sex_male],
    "Embarked_Q": [embarked_Q],
    "Embarked_S": [embarked_S]
})

# ----------------------------
# Make Prediction
# ----------------------------
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"✅ This passenger would have SURVIVED with probability {prob:.2f}")
    else:
        st.error(f"❌ This passenger would NOT have survived (Survival Probability {prob:.2f})")

