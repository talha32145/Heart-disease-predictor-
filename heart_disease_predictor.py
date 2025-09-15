import streamlit as st
import pandas as pd
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import google.generativeai as genai


genai.configure(api_key="AIzaSyAjxcWvjDxmvyy_bJWh6z54Vgusr3yMh-Y")



df = pd.read_csv("heart_statlog_cleveland_hungary_final.csv")


x = df.drop("target", axis=1)
y = df["target"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, random_state=42
)


model = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    class_weight="balanced",
    criterion="gini",
    bootstrap=True,
    max_features="log2",
)
model.fit(x_train, y_train)

pred=model.predict(x_test)
accuracy = accuracy_score(y_test,pred)


st.title("🫀 Heart Disease Predictor with Dr. CardioCare")
st.write(f"**Model Accuracy:** {accuracy * 100:.2f}%")

st.header("Enter Patient Details")

with st.form("patient_form"):
    name = st.text_input("Patient Name")
    age = st.number_input("Age", min_value=1, max_value=120, step=1)
    gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    chestpain = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3, step=1)
    resting_bps = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, step=1)
    cholesterol = st.number_input("Cholesterol Level", min_value=100, max_value=600, step=1)
    fasting = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    resting = st.number_input("Resting ECG (0-2)", min_value=0, max_value=2, step=1)
    maxheart = st.number_input("Max Heart Rate", min_value=60, max_value=250, step=1)
    exercise = st.selectbox("Exercise Induced Angina", [0, 1])
    old = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, step=0.1)
    stslope = st.number_input("ST Slope (0-2)", min_value=0, max_value=2, step=1)

    submitted = st.form_submit_button("Predict")

if submitted:
    
    patient_data = pd.DataFrame({
        "age": [age],
        "sex": [gender],
        "chest pain type": [chestpain],
        "resting bp s": [resting_bps],
        "cholesterol": [cholesterol],
        "fasting blood sugar": [fasting],
        "resting ecg": [resting],
        "max heart rate": [maxheart],
        "exercise angina": [exercise],
        "oldpeak": [old],
        "ST slope": [stslope]
    })

   
    prediction = model.predict(patient_data)
    probability = model.predict_proba(patient_data)

    result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"
    st.subheader(f"🧾 Prediction: {result}")
    st.write(f"**Probability of Heart Disease:** {probability[0][1] * 100:.2f}%")

   
    with st.spinner("🤔 Thinking... Please wait while Dr. CardioCare prepares your advice..."):
        models = genai.GenerativeModel(
            "models/gemini-1.5-flash",
            generation_config={
                "temperature": 0.5,
                "max_output_tokens": 1500
            }
        )

        message = f"""
            You are Dr. CardioCare.
            Probability of heart disease: {probability[0][1]*100:.2f}%
            Patient details: {patient_data.to_dict(orient='records')[0]}
            Please provide clear advice about:
            - Diet
            - Exercise
            - Lifestyle
            - Next steps
        """
        response = models.generate_content(message)

    st.markdown("### 🩺 Dr. CardioCare's Guidance")
    st.write(response.text)

   
    save_data = patient_data.copy()
    save_data["Patient Name"] = name
    save_data["Prediction"] = result
    save_data["Probability (%)"] = probability[0][1] * 100

    with open("Patient_data.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [save_data["Patient Name"].values[0]]
            + save_data.iloc[0, :-3].tolist()
            + [save_data["Prediction"].values[0]]
            + [save_data["Probability (%)"].values[0]]
        )

    st.success("✅ Patient record saved successfully!")
