from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from sklearn.preprocessing import LabelEncoder, LabelBinarizer

def clean_data(df:pd.DataFrame):
    '''
    Función para eliminar columnas no deseadas del dataframe "arabica".

    Args:
        df (pd.DataFrame): Dataframe a trabajar (arabica).
    
    Returns:
        df (pd.DataFrame): Dataframe limpio.
    '''
    df = df.drop(columns=['Unnamed: 0', 'Species', 'Owner', 'Farm.Name', 'Lot.Number', 'Mill', 'ICO.Number',
                          'Company', 'Altitude', 'Region', 'Producer', 'Number.of.Bags', 'Bag.Weight',
                          'In.Country.Partner', 'Grading.Date', 'Owner.1', 'Aroma', 'Flavor', 'Aftertaste',
                          'Acidity', 'Body', 'Balance', 'Uniformity', 'Clean.Cup', 'Sweetness', 'Cupper.Points',
                          'Quakers', 'Expiration', 'Certification.Body', 'Certification.Address',
                          'Certification.Contact', 'altitude_low_meters', 'altitude_high_meters'])
    df = df.dropna()
    return df

def group(df:pd.DataFrame):
    '''
    Función para agrupar valores minoritarios del dataframe "arabica".

    Args:
        df (pd.DataFrame): Dataframe a trabajar (arabica).
    
    Returns:
        df (pd.DataFrame): Dataframe con valores agrupados.
    '''
    otros = df['Country.of.Origin'].value_counts() <= 5
    for i in range(len(otros.index)):
        if(otros[i]):
            df.loc[df["Country.of.Origin"] == otros.index[i], "Country.of.Origin"] = "Other"

    otros = df['Variety'].value_counts() == 1
    for i in range(len(otros.index)):
        if(otros[i]):
            df.loc[df["Variety"] == otros.index[i], "Variety"] = "Other"
    
    color = {'Green': 'None',
             'Blue' : ['Blue-Green', 'Bluish-Green']}
    df['Color'] = df['Color'].map(lambda x: next((k for k, v in color.items() if x in v), x))
    return df

def correct(df:pd.DataFrame):
    '''
    Función para corregir años y altitudes del dataframe "arabica".

    Args:
        df (pd.DataFrame): Dataframe a trabajar (arabica).
    
    Returns:
        df (pd.DataFrame): Dataframe corregido.
    '''
    year = {'2015/2016' : 2016,
            '2013/2014' : 2014,
            '2017 / 2018' : 2018,
            '2014/2015' : 2015,
            '2011/2012' : 2012,
            '2016 / 2017' : 2017}
    df['Harvest.Year'] = df['Harvest.Year'].replace(year).astype(int)

    mask = df['unit_of_measurement'].eq('ft')
    df.loc[mask, ['altitude_mean_meters']] /= 3.281
    df= df.drop(columns= "unit_of_measurement")

    df= df[~(df["altitude_mean_meters"] > 9000)]

    return df

def label(df:pd.DataFrame):
    '''
    Función para etiquetar variables categóricas del dataframe "arabica".

    Args:
        df (pd.DataFrame): Dataframe a trabajar (arabica).
    
    Returns:
        df (pd.DataFrame): Dataframe etiquetado.
    '''
    lb = LabelBinarizer()
    df.Color = lb.fit_transform(df.Color)

    le = LabelEncoder()
    columnas = ["Country.of.Origin", "Variety", "Processing.Method"]

    for columna in columnas:
        df[columna] = le.fit_transform(df[columna])

    calidad = [0, 1, 2]
    calif = [0, 80, 85, 100]
    df['Calidad'] = pd.cut(df['Total.Cup.Points'], bins=calif, labels=calidad)

    return df

def balance(df:pd.DataFrame):
    '''
    Función para balancear los registros del dataframe "arabica".

    Args:
        df (pd.DataFrame): Dataframe a trabajar (arabica).
    
    Returns:
        df (pd.DataFrame): Dataframe balanceado.
    '''
    ros = RandomOverSampler(random_state=5)
    X_resampled, y_resampled = ros.fit_resample(df.loc[:, 'Country.of.Origin':'Total.Cup.Points'], df['Calidad'])
    df_resampled = pd.DataFrame(X_resampled, columns=df.loc[:, 'Country.of.Origin':'Total.Cup.Points'].columns)
    df_resampled['Calidad'] = y_resampled
    df_balanced = pd.concat([df, df_resampled], ignore_index=True)

    df_balanced.drop(columns='Total.Cup.Points', inplace=True)

    return df_balanced