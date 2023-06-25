# Importo librerías
from funct import funciones as fun
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import pickle
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
    st.markdown('La idea principal es crear un clasificador de calidad de café en tres distintas categorías: estándar, bueno y premium; para así asegurarnos de siempre comprar productos de calidad por el precio ideal.')
    st.markdown('Para ello, se utilizan diferentes variables: país de origen, variedad de café, procesado, año de cosecha, humedad, color, defectos e incluso la altura en que ha sido cultivado. Todo esto se introduce en un modelo predictivo de machine learning, para así saber qué tan bueno es el producto antes de realizar la compra.')

# Datos
if st.sidebar.button('Datos'):
    st.header('Datos en crudo')
    arabica
    arabica.shape
    st.divider()
    st.header('Variables Categóricas')
    st.image('app/img/category.png')
    st.divider()
    st.header('Valores Nulos')
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

# Procesamiento 
if st.sidebar.button('Procesamiento'):
    tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['En crudo', 'Limpieza', 'Agrupación', 'Correcciones', 'Etiquetado', 'Balanceado', 'Final'])

    with tab0:
        st.header('Datos en crudo')
        arabica
        arabica.shape
        st.divider()
        

    with tab1:
        st.header('Limpieza de Datos')
        st.code("""
        # Columnas con las que me quiero quedar.
        arabica = arabica[['Country.of.Origin', 'Variety', 'Processing.Method', 'Moisture',
                        'Harvest.Year', 'Color','unit_of_measurement', 'altitude_mean_meters',
                        'Category.One.Defects', 'Category.Two.Defects','Total.Cup.Points']]

        # Drop missing values.
        arabica = arabica.dropna()
        """)
        st.divider()
        st.subheader('Resultado')
        arabica = fun.clean_data(arabica)
        arabica
        arabica.shape
        st.divider()

    with tab2:
        st.header('Agrupación de Minorías')
        st.code("""
        # Agrupar valores minoritarios.
        otros = arabica['Country.of.Origin'].value_counts() <= 5
        for i in range(len(otros.index)):
            if(otros[i]):
                arabica.loc[arabica["Country.of.Origin"] == otros.index[i], "Country.of.Origin"] = "Other"

        otros = arabica['Variety'].value_counts() == 1
        for i in range(len(otros.index)):
            if(otros[i]):
                arabica.loc[arabica["Variety"] == otros.index[i], "Variety"] = "Other"
        
        color = {'Green': 'None',
                 'Blue' : ['Blue-Green', 'Bluish-Green']}
        arabica['Color'] = arabica['Color'].map(lambda x: next((k for k, v in color.items() if x in v), x))
        """)
        st.divider()
        st.header('Resultado')

        column1, column2 = st.columns(2)

        with column1:
            st.subheader('Antes')
            arabica['Country.of.Origin']
            unico = arabica['Country.of.Origin'].nunique()
            unico
            st.divider()
            arabica['Variety']
            unico = arabica['Variety'].nunique()
            unico
            st.divider()
            arabica['Color']
            unico = arabica['Color'].nunique()
            unico
        
        with column2:
            st.subheader('Después')
            arabica = fun.group(arabica)
            arabica['Country.of.Origin']
            unico = arabica['Country.of.Origin'].nunique()
            unico
            st.divider()
            arabica['Variety']
            unico = arabica['Variety'].nunique()
            unico
            st.divider()
            arabica['Color']
            unico = arabica['Color'].nunique()
            unico

    with tab3:
        st.header('Otras Correcciones')
        st.code("""
        # Corregir año de cosecha.
        year = {'2015/2016' : 2016,
                '2013/2014' : 2014,
                '2017 / 2018' : 2018,
                '2014/2015' : 2015,
                '2011/2012' : 2012,
                '2016 / 2017' : 2017}
        arabica['Harvest.Year'] = arabica['Harvest.Year'].replace(year).astype(int)

        # Convertir altitudes a mismas unidades.
        mask = arabica['unit_of_measurement'].eq('ft')
        arabica.loc[mask, ['altitude_mean_meters']] /= 3.281
        arabica= arabica.drop(columns= "unit_of_measurement")

        # Eliminar outliers irreales en altitud
        arabica= arabica[~(arabica["altitude_mean_meters"] > 9000)]
        """)
        st.divider()
        st.header('Resultado')

        column1, column2 = st.columns(2)

        with column1:
            st.subheader('Antes')
            valor = arabica['Harvest.Year'].unique()
            st.markdown('Valores únicos de [Harvest.Year]:')
            st.markdown(valor)
            st.divider()
            valor = arabica['unit_of_measurement'].unique()
            st.markdown('Unidades de altitud:')
            st.markdown(valor)
            st.image('app/img/outlierspre.png')

        with column2:
            st.subheader('Después')
            arabica = fun.correct(arabica)
            valor = arabica['Harvest.Year'].unique()
            st.markdown('Valores únicos de [Harvest.Year]:')
            st.markdown(valor)
            st.write("")
            st.write("")
            st.write("")
            st.divider()
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.image('app/img/outlierspost.png')
    
    with tab4:
        st.header('Etiquetado de Categorías')
        st.code("""
        # Binarizar los colores.
        lb = LabelBinarizer()

        arabica.Color = lb.fit_transform(arabica.Color)

        # Encoding de otras variables categóricas.
        le = LabelEncoder()

        columnas = ["Country.of.Origin", "Variety", "Processing.Method"]

        for columna in columnas:
            arabica[columna] = le.fit_transform(arabica[columna])

        # Defino la columna target.
        calidad = [0, 1, 2]
        calif = [0, 80, 85, 100]
        arabica['Calidad'] = pd.cut(arabica['Total.Cup.Points'], bins=calif, labels=calidad)
        """)
        st.divider()
        st.header('Resultados')

        column1, column2 = st.columns(2)

        with column1:
            st.subheader('Antes')
            arabica['Color']
            st.divider()
            st.dataframe(arabica[['Country.of.Origin', 'Variety', 'Processing.Method']])

        with column2:
            st.subheader('Después')
            arabica = fun.label(arabica)
            arabica['Color']
            st.divider()
            st.dataframe(arabica[['Country.of.Origin', 'Variety', 'Processing.Method']])

    with tab5:
        st.header('Balanceado de Datos')
        st.code("""
        # Balanceo el dataframe.
        ros = RandomOverSampler(random_state=5)
        X_resampled, y_resampled = ros.fit_resample(arabica.loc[:, 'Country.of.Origin':'Total.Cup.Points'], arabica['Calidad'])
        df_resampled = pd.DataFrame(X_resampled, columns=arabica.loc[:, 'Country.of.Origin':'Total.Cup.Points'].columns)
        df_resampled['Calidad'] = y_resampled
        df_balanced = pd.concat([arabica, df_resampled], ignore_index=True)

        # Quito la "chuleta"
        df_balanced.drop(columns='Total.Cup.Points', inplace=True)
        """)
        st.divider()
        st.header('Resultados')
        column1, column2 = st.columns(2)
        
        with column1:
            st.subheader('Antes')
            arabica
            arabica.shape
            st.image('app/img/unbalanced.png')

        with column2:
            st.subheader('Después')
            arabica = fun.balance(arabica)
            arabica
            arabica.shape
            st.image('app/img/balanced.png')
    
    with tab6:
        st.header('Dataframe Procesado')
        arabicapro
        arabicapro.shape
        st.divider()
        st.header('Train')
        train
        train.shape
        st.header('Test')
        test
        test.shape
        st.balloons()

st.sidebar.button('Modelos')
st.sidebar.button('Conclusiones')
st.sidebar.divider()