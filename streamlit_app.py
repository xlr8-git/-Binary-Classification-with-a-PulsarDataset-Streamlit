import streamlit as st
import joblib
import pandas as pd

model_dict = joblib.load('model.pkl')
model = model_dict['model']
model_columns = model_dict['columns']

st.set_page_config(page_title="Pulsar Star Prediction", layout="centered")
st.title("Pulsar Star Classification")
st.write("---")

st.markdown("""
### About the Dataset
This dataset is from the [Kaggle Playground Series: Pulsar Classification](https://www.kaggle.com/competitions/playground-series-s3e10).  
It contains synthetic data generated from a deep learning model trained on pulsar signals.  
Each row represents a star observation, and features describe the shape, intensity, and statistics of the signal.

**Goal:** Predict whether a given observation corresponds to a pulsar.
""")

Mean_Integrated = st.number_input("Mean of Integrated Profile")
SD = st.number_input("Standard Deviation of Integrated Profile")
EK = st.number_input("Excess Kurtosis of Integrated Profile")
Skewness = st.number_input("Skewness of Integrated Profile")
Mean_DMSNR_Curve = st.number_input("Mean of DM-SNR Curve")
SD_DMSNR_Curve = st.number_input("Standard Deviation of DM-SNR Curve")
EK_DMSNR_Curve = st.number_input("Excess Kurtosis of DM-SNR Curve")
Skewness_DMSNR_Curve = st.number_input("Skewness of DM-SNR Curve")

if st.button("Predict"):
    data = {
        'Mean_Integrated': Mean_Integrated,
        'SD': SD,
        'EK': EK,
        'Skewness': Skewness,
        'Mean_DMSNR_Curve': Mean_DMSNR_Curve,
        'SD_DMSNR_Curve': SD_DMSNR_Curve,
        'EK_DMSNR_Curve': EK_DMSNR_Curve,
        'Skewness_DMSNR_Curve': Skewness_DMSNR_Curve
    }

    df = pd.DataFrame([data])
    df = df[model_columns]

    prediction = model.predict(df)[0]
    prediction_text = "Pulsar Star" if prediction == 1 else "Not a Pulsar Star"

    st.success(f"Prediction: {prediction_text}")
