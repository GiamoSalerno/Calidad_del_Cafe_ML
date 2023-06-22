# Importar librerías
import pandas as pd
from sklearn.preprocessing import LabelEncoder, LabelBinarizer

# Leer y limpiar los datos
arabica = pd.read_csv('data/raw/arabica_data_cleaned.csv')

arabica = arabica[['Country.of.Origin', 'Variety','Processing.Method', 'Moisture', 'Harvest.Year','Color',
                   'unit_of_measurement', 'altitude_mean_meters', 'Category.One.Defects', 'Category.Two.Defects',
                   'Total.Cup.Points']]

# Drop missing values.
arabica = arabica.dropna()

# Agrupar valores minoritarios.
otros = arabica['Country.of.Origin'].value_counts() <= 5
for i in range(len(otros.index)):
    if(otros[i]):
        arabica.loc[arabica["Country.of.Origin"] == otros.index[i], "Country.of.Origin"] = "Other"

otros = arabica['Variety'].value_counts() == 1
for i in range(len(otros.index)):
    if(otros[i]):
        arabica.loc[arabica["Variety"] == otros.index[i], "Variety"] = "Other"

# Corregir año de cosecha.
year = {'2015/2016' : 2016,
        '2013/2014' : 2014,
        '2017 / 2018' : 2018,
        '2014/2015' : 2015,
        '2011/2012' : 2012,
        '2016 / 2017' : 2017}
arabica['Harvest.Year'] = arabica['Harvest.Year'].replace(year).astype(int)

# Agrupar colores.
color = {'Green': 'None',
         'Blue' : ['Blue-Green', 'Bluish-Green']}
arabica['Color'] = arabica['Color'].map(lambda x: next((k for k, v in color.items() if x in v), x))

# Convertir altitudes a mismas unidades.
mask = arabica['unit_of_measurement'].eq('ft')
arabica.loc[mask, ['altitude_mean_meters']] /= 3.281
arabica= arabica.drop(columns= "unit_of_measurement")

# Eliminar outliers irreales en altitud
arabica= arabica[~(arabica["altitude_mean_meters"] > 9000)]

# Binarizar los colores.
lb = LabelBinarizer()

arabica.Color = lb.fit_transform(arabica.Color)

# Encoding de otras variables categóricas.
le = LabelEncoder()

columnas = ["Country.of.Origin", "Variety", "Processing.Method"]

for columna in columnas:
    arabica[columna] = le.fit_transform(arabica[columna])

# Guardar en carpeta correspondiente.
arabica.to_csv('data/processed/arabica_processed.csv')