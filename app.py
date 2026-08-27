# ============================================================
# app.py — Aplicación Streamlit
# Predicción de riesgo de afectación emocional (mujeres 13-49)
# Proyecto Integrador - Maestría en Ciencia de Datos (UPB)
# Anyi Lorena Arias López · Julio José Bertel Sierra · Juliana Rodríguez Mejía
# ============================================================
import streamlit as st
import pandas as pd
import joblib

# ---------- Cargar el modelo y las herramientas guardadas ----------
modelo = joblib.load('modelo_final.pkl')
scaler = joblib.load('scaler.pkl')
columnas_modelo = joblib.load('columnas_modelo.pkl')

# ---------- Título y descripción ----------
st.title("🧠 Predicción de Riesgo de Afectación Emocional")
st.write("""
Esta aplicación estima el riesgo de afectación emocional severa en mujeres
de 13 a 49 años, a partir de factores sociodemográficos y de estado civil.

**Nota:** es una herramienta académica, no un diagnóstico médico.
""")

st.divider()

# ---------- Formulario de entrada ----------
st.subheader("Ingresa los datos:")

edad = st.slider("Edad", 13, 49, 30)

educacion = st.selectbox(
    "Nivel educativo (código: 1=Ninguno ... valores más altos = mayor nivel)",
    options=[1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17]
)

estado_civil = st.selectbox(
    "Estado civil",
    options=['Casada', 'Union_libre', 'Viuda', 'Separada_divorciada', 'Soltera']
)

actividad = st.selectbox(
    "Actividad principal la semana pasada",
    options=['Trabajando', 'Buscando_trabajo', 'Estudiando',
             'Oficios_hogar', 'Incapacitada', 'Otra']
)

contrato = st.selectbox(
    "Tipo de contrato laboral",
    options=['Escrito_indefinido', 'Escrito_fijo', 'Verbal',
             'Sin_contrato', 'No_aplica']
)

# ---------- Botón de predicción ----------
if st.button("Predecir riesgo"):

    # 1. Armamos un registro con los datos ingresados
    entrada = pd.DataFrame({
        'D3': [edad],
        'G4_NIVEL': [educacion],
        'H1': [estado_civil],
        'G12': [actividad],
        'G24': [contrato]
    })

    # 2. Codificamos igual que en el entrenamiento (One-Hot)
    entrada_cod = pd.get_dummies(entrada, columns=['H1', 'G12', 'G24'])

    # 3. Alineamos con las columnas que el modelo espera
    #    (rellena con 0 las categorías que no aparecen)
    entrada_final = entrada_cod.reindex(columns=columnas_modelo, fill_value=0)

    # 4. Escalamos con el mismo scaler del entrenamiento
    entrada_escalada = scaler.transform(entrada_final)

    # 5. Predecimos
    prediccion = modelo.predict(entrada_escalada)[0]
    probabilidad = modelo.predict_proba(entrada_escalada)[0][1]

    # ---------- Mostrar resultado ----------
    st.divider()
    if prediccion == 1:
        st.error(f"⚠️ RIESGO DETECTADO (probabilidad: {probabilidad:.0%})")
        st.write("El perfil sugiere posible afectación emocional. "
                 "Se recomienda acompañamiento profesional.")
    else:
        st.success(f"✅ Sin riesgo detectado (probabilidad de riesgo: {probabilidad:.0%})")
        st.write("El perfil no sugiere afectación emocional severa.")

    st.caption("Recuerda: herramienta académica, no reemplaza evaluación clínica.")
