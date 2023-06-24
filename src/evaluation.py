# Importo librerías.
import pandas as pd, pickle
from sklearn import metrics

# Importo el modelo preentrenado.
with open('models/modelo_final.pkl', 'rb') as file:
    modelo = pickle.load(file)

# Datos de predicción.
ruta = 'data/test/arabica_test.csv'
test = pd.read_csv(ruta)
X = test.drop(columns='Calidad')
y = test.Calidad

# Predicción y evaluación.
print('Evaluando', ruta)
predicciones = modelo.predict(X)
print('Accuracy:', metrics.accuracy_score(predicciones, y))
print('Precision:', metrics.precision_score(predicciones, y, average=None))
print('Recall:', metrics.recall_score(predicciones, y, average=None))