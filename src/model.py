# Importo librerías.
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import pickle

# Leo los datos.
arabica = pd.read_csv('data/train/arabica_train.csv')
arabica.drop(columns=['Unnamed: 0'], inplace=True)

# Defino X e Y.
X = arabica.drop(columns=['Total.Cup.Points'])
y = arabica['Total.Cup.Points']

# Divido en train y test.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state= 5)

# Defino el modelo.
pipe = Pipeline(steps=[("scaler", StandardScaler()),
                       ('pca', PCA(8)),
                       ('regressor', LinearRegression())])

linear_params = {'scaler' : [StandardScaler()],
                 'regressor': [LinearRegression()]}

en_params = {'scaler' : [StandardScaler()],
             'regressor': [ElasticNet()]}

dt_params = {'scaler' : [StandardScaler(), None],
             'regressor': [DecisionTreeRegressor()]}

knn_params = {'scaler' : [StandardScaler()],
              'regressor': [KNeighborsRegressor()],
              'regressor__n_neighbors': [3,9,15]}

svr_params = {'scaler' : [StandardScaler()],
              'regressor': [SVR()],
              'regressor__C': [1]}

search_space = [linear_params, en_params, dt_params, knn_params, knn_params, svr_params]

reg = GridSearchCV(estimator = pipe,
                   param_grid = search_space,
                   cv=3,
                   scoring='r2',
                   n_jobs=-1)

# Entreno el modelo.
reg.fit(X_train, y_train)

# Guardo el modelo.
with open('models/trained_model1.pkl', 'wb') as file:
    pickle.dump(reg, file)