# Importo librerías
import numpy as np
import pandas as pd
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Título de la página
st.set_page_config(page_title = 'Calidad del Café', page_icon= ":coffee:")
st.title('Calidad del Café:')
st.header('Una Clasificación con Machine Learning')
st.image('https://sepuedeonosepuede.com/wp-content/uploads/2022/10/comer-granos-de-cafe-scaled.jpg',
         caption='Giacomo Salerno')
st.divider()

# Datasets a trabajar
arabica = pd.read_csv('data/raw/arabica_data_cleaned.csv')
arabicapro = pd.read_csv('data/processed/arabica_processed.csv')
train = pd.read_csv('data/train/arabica_train.csv')
test = pd.read_csv('data/test/arabica_test.csv')

# Sidebar
st.sidebar.title('Contenido')

# ¿Agua Sucia o Café
if st.sidebar.button('¿Agua Sucia o Café?'):
    st.header('¡A nadie le gusta un mal café!')
    st.markdown('Es por eso que nuestra empresa busca siempre lo mejor para nuestros clientes, un café de aroma irresistible, con cuerpo balanceado, que no sea demasiado ácido, y sobre todo, que esté muy bien de precio.')
    st.markdown('La idea principal es crear un clasificador de calidad de café en tres distintas categorías: estándar, bueno y premium')
    st.markdown('Para ello, se utilizan diferentes variables: país de origen, variedad de café, procesado, año de cosecha, humedad, color, defectos e incluso la altura en que ha sido cultivado.')

# Datos 
if st.sidebar.button('Datos'):
    tab0, tab1, tab2, tab3 = st.tabs(['En crudo', 'Procesado', 'Entrenamiento', 'Predicción'])

    with tab0:
        st.header('Datos en crudo')
        arabica
        arabica.shape
        st.divider()
        st.header('Valors Nulos')
        st.image('app/img/missingvalues.png')
        st.divider()
        st.header('Correlaciones')
        st.image('app/img/totalcorr.png')
        st.divider()
        st.header('Distribución de Calidad')
        st.image('app/img/distcalidad.png')
        st.divider()
        st.header('Balanceo')
        st.image('app/img/unbalanced.png')

    with tab1:
        st.header('Procesamiento de Datos')
        arabica
        arabica.shape
        multi = st.multiselect('Acción', ['Raw','Clean', 'Drop', 'Group', 'Encode'])
        
        if 'Clean' in multi:
            arabica.drop(columns=['Unnamed: 0', 'Species', 'Owner', 'Farm.Name', 'Lot.Number', 'Mill', 'ICO.Number',
            'Company', 'Altitude', 'Region', 'Producer', 'Number.of.Bags', 'Bag.Weight',
            'In.Country.Partner', 'Grading.Date', 'Owner.1', 'Aroma', 'Flavor', 'Aftertaste',
            'Acidity', 'Body', 'Balance', 'Uniformity', 'Clean.Cup', 'Sweetness', 'Cupper.Points',
            'Quakers', 'Expiration', 'Certification.Body', 'Certification.Address',
            'Certification.Contact', 'altitude_low_meters', 'altitude_high_meters'])

st.sidebar.button('Modelos')
st.sidebar.button('Conclusiones')
st.sidebar.divider()