import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

@st.cache
def load_data():
    # Load the complete dataset
    data = pd.read_csv('your_data_full.csv')
    return data

data = load_data()

# Sidebar for dynamic input
st.sidebar.header("Input Parameters")
user_input = {}
for col in data.columns[:-1]:  # Exclude target column
    if data[col].dtype in [np.float64, np.int64]:
        user_input[col] = st.sidebar.slider(col, float(data[col].min()), float(data[col].max()), float(data[col].mean()))
    else:
        user_input[col] = st.sidebar.text_input(col, value=str(data[col].iloc[0]))

user_df = pd.DataFrame([user_input])

@st.cache
def train_model(data):
    X = data.drop(columns=["PDRB Pertanian_2023"])
    y = data["PDRB Pertanian_2023"]
    model = RandomForestRegressor()
    model.fit(X, y)
    return model

model = train_model(data)
prediction = model.predict(user_df)[0]

st.title("Prediksi PDRB Sektor Pertanian")
st.write("Aplikasi ini memprediksi PDRB sektor pertanian menggunakan seluruh data input yang tersedia.")

st.header("Hasil Prediksi")
st.write(f"Prediksi PDRB Tahun 2045: **{prediction:,.2f}**")

st.header("Feature Importance")
feature_importance = pd.Series(model.feature_importances_, index=data.drop(columns=["PDRB Pertanian_2023"]).columns)
plt.figure(figsize=(10, 6))
feature_importance.sort_values().plot(kind="barh")
plt.title("Feature Importance")
plt.xlabel("Kepentingan")
plt.ylabel("Fitur")
st.pyplot(plt)
