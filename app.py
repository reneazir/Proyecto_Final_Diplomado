# app.py
import streamlit as st
import joblib
import numpy as np

DATA_DIR = 'model'

# ==== Cargar modelo ====
@st.cache_resource
def load_model():
    return joblib.load(DATA_DIR+"/modelo_regresion_servicio_poda.pkl")

model = load_model()

# ==== Interfaz ====
st.title("📈 Predicción con Modelo de Regresión para el Servicio de Poda")
st.write("Modelo entrenado con pocos registros entre 2016 y 2025 usando Pipeline + Joblib")

# Entradas del usuario
x1 = st.number_input("Valor de Año 1", value=2026.0)
x2 = st.number_input("Valor de Año 2", value=2027.0)

if st.button("Predecir"):
    try:
        pred = model.predict(np.array([[x1, x2]]))
        st.success(f"Predicción: {pred[0]:.4f}")
    except Exception as e:
        st.error(f"Error en la predicción: {e}")
