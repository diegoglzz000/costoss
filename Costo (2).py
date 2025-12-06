import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.title("Predicción del costo de una actividad")

st.header("Ingresa los datos:")

# ---- Entrada del usuario ----
def user_input():
    presupuesto = st.number_input("Presupuesto:", min_value=0.0, step=1.0)
    tiempo = st.number_input("Tiempo invertido (min):", min_value=0.0, step=1.0)
    tipo = st.number_input("Tipo de actividad:", min_value=0.0, step=1.0)
    momento = st.number_input("Momento del día:", min_value=0.0, step=1.0)
    personas = st.number_input("No. de personas:", min_value=0.0, step=1.0)

    data = {
        'Presupuesto': presupuesto,
        'Tiempo invertido': tiempo,
        'Tipo': tipo,
        'Momento': momento,
        'No. de personas': personas
    }

    return pd.DataFrame(data, index=[0])


df = user_input()

# ---- Cargar datos ----
datos = pd.read_csv("registrosdgg_limpio.csv")

# Nos quedamos solo con las columnas necesarias
cols = ['Presupuesto', 'Tiempo invertido', 'Tipo', 'Momento', 'No. de personas', 'Costo']
datos = datos[cols].dropna()  # Eliminar NaN para evitar errores

# ---- Separar variables ----
X = datos[['Presupuesto', 'Tiempo invertido', 'Tipo', 'Momento', 'No. de personas']]
y = datos['Costo']

# ---- Entrenar modelo ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=1613808
)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

# ---- Obtener coeficientes ----
b0 = modelo.intercept_
b = modelo.coef_

# ---- Predicción manual (sencilla) ----
pred = (
    b0
    + b[0] * df['Presupuesto']
    + b[1] * df['Tiempo invertido']
    + b[2] * df['Tipo']
    + b[3] * df['Momento']
    + b[4] * df['No. de personas']
)

st.subheader("Resultado")
st.write("El costo estimado de la actividad es:", float(pred))



