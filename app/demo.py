import numpy as np
import pickle
import streamlit as st

with open('models/modelo_final.pkl', 'rb') as file:
    modelo = pickle.load(file)

# Título de la página
st.set_page_config(page_title = 'Predictor de Café', page_icon= ":coffee:")
st.title('Calidad del Café:')
st.header('Una Clasificación con Machine Learning')
st.image('https://sepuedeonosepuede.com/wp-content/uploads/2022/10/comer-granos-de-cafe-scaled.jpg',
         caption='Giacomo Salerno')
st.divider()

st.sidebar.title('Guía')
if st.sidebar.button('Países'):
    st.title('Lista de Países')
    st.table(data = {'País': ['Brazil', 'China', 'Colombia', 'Costa Rica', 'El Salvador', 'Ethiopia', 'Guatemala', 'Honduras', 'Indonesia', 'Kenya', 'Malawi', 'Mexico', 'Myanmar', 'Nicaragua', 'Other', 'Taiwan', 'Tanzania, United Republic Of', 'Thailand', 'Uganda', 'Vietnam'],
                         'Número': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]})
    st.divider()
if st.sidebar.button('Variedades'):
    st.title('Lista de Variedades')
    st.table(data = {'Variedad': ['Bourbon', 'Catimor', 'Catuai', 'Caturra', 'Gesha', 'Mandheling', 'Mundo Novo', 'Other', 'Pacamara', 'Pacas', 'Ruiru 11', 'SL14', 'SL28', 'SL34', 'Sumatra', 'Typica', 'Yellow Bourbon'],
                     'Número': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]})
    st.divider()
if st.sidebar.button('Procesado'):
    st.title('Lista de Procesamientos')
    st.table(data = {'Procesamiento': ['Natural / Secado', 'Otro', 'Despulpado natural / Honey', 'Semi-lavado / Semi-despulpado', 'Lavado / Húmedo'],
                     'Número': [0, 1, 2, 3, 4]})
    st.divider()
if st.sidebar.button('Defectos'):
    st.title('Tipos de Defectos')
    col1, col2 = st.columns(2)
    with col1:
        st.header('Categoría 1')
        st.table(data = {'Categoría': ['Ennegrecimiento total', 'Acidificación total', 'Secado', 'Daño por hongos', 'Materia extraña', 'Daño severo por insectos'],
                         'Valor': [1, 1, 1, 1, 1, 5]})
    with col2:
        st.header('Categoría 2')
        st.table(data = {'Categoría': ['Ennegrecimiento parcial', 'Acidificación parcial', 'Pergamino', 'Flotador', 'Inmaduro', 'Averanado', 'Concha', 'Roto/Mordido/Cortado', 'Seco', 'Daño ligero por insectos'],
                         'Valor': [3, 3, 5, 5, 5, 5, 5, 5, 5, 10]})
    st.markdown('Puedes encontrar más información [aquí](https://www.coffeestrategies.com/wp-content/uploads/2020/08/Green-Coffee-Defect-Handbook.pdf)')
    st.divider()

# Parámetros de entrada.
st.header('Introduce los Parámetros:')
pais = st.slider('País', 0, 19)
variedad = st.slider('Variedad', 0, 16)
procesado = st.slider('Procesado', 0, 4)
humedad = st.slider('Humedad', 0.0, 0.2)
year = st.slider('Año de Cosecha', 2011, 2023)
color = st.slider('Color; 0 para verde, 1 para otro', 0, 1)
altitud = st.slider('Altitud', 1, 5000)
def1 = st.slider('Número de Defectos de Categoría 1', 0, 50)
def2 = st.slider('Número de Defectos de Categoría 2', 0, 50)

input = np.array([pais, variedad, procesado, humedad, year, color, altitud, def1, def2]).reshape(1, -1)
pred = modelo.predict(input)[0]

if st.button('¡Espresso!'):
    if pred == 0:
        st.header('Parece que la calidad de tu café es estándar... vamos, por no llamarlo de otra manera.')
    if pred == 1:
        st.header('¡Tienes en tus manos un café de buena calidad! Esperemos que el precio se ajuste a la misma.')
    if pred == 2:
        st.header('¡Felicidades! Tu café es de calidad premium, lo mejor de lo mejor... más te vale no ponerle leche.')
if st.button('Actualizar'):
    st.experimental_rerun()
