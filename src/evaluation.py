# Importo librerías.
import pickle, pandas as pd, numpy as np
from sklearn import metrics

# Importo el modelo preentrenado.
with open('models/trained_model1.pkl', 'rb') as file:
    modelo1 = pickle.load(file)

# Importo el test.
arabica = pd.read_csv('data/test/arabica_test.csv')
arabica.drop(columns=['Unnamed: 0'], inplace=True)

X = arabica.drop(columns= ['Total.Cup.Points'])
y = arabica['Total.Cup.Points']

predictions = modelo1.predict(X)

print('MAE:', metrics.mean_absolute_error(predictions, y))
print('MSE:', metrics.mean_squared_error(predictions, y))
print('RMSE:', np.sqrt(metrics.mean_squared_error(predictions, y)))
print('R2:', metrics.r2_score(predictions, y))