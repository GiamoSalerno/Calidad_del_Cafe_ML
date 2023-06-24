# Importo librerías necesarias
import pandas as pd, pickle
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline

# Modelo final a entrenar
best_params = {'selectkbest__k': 9,
               'pca__n_components': 9,
               'classifier': GradientBoostingClassifier(max_depth=7, max_features=2)}

best_model = Pipeline(steps=[("selectkbest", SelectKBest(k=best_params['selectkbest__k'])),
                             ("pca", PCA(n_components=best_params['pca__n_components'])),
                             ("classifier", best_params['classifier'])])

# Datos de entrenamiento
train = pd.read_csv('data/train/arabica_train.csv')
X = train.drop(columns='Calidad')
y = train.Calidad

# Entrenamiento
best_model.fit(X, y)

# Guardado del modelo entrenado
with open('models/modelo_final.pkl', 'wb') as file:
    pickle.dump(best_model, file)

# Mensaje final
print('El modelo ha sido guardado correctamente')