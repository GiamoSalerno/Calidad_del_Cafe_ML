# Importo librerías necesarias
import pandas as pd, pickle, yaml
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline

# Defino el configurador.
with open('models/model_config.yaml', 'r') as file:
    yam = yaml.safe_load(file)

# Datos de entrenamiento
train = pd.read_csv(yam['data']['train_file'])
X = train.drop(columns = yam['data']['target_column'])
y = train[yam['data']['target_column']]

# Modelo final a entrenar
best_params = {'selectkbest__k': yam['model']['pipeline']['steps'][0]['params']['k'],
               'pca__n_components': yam['model']['pipeline']['steps'][1]['params']['n_components'],
               'classifier': GradientBoostingClassifier(max_depth = yam['model']['pipeline']['steps'][2]['params']['max_depth'],
                                                        max_features = yam['model']['pipeline']['steps'][2]['params']['max_features'])}

best_model = Pipeline(steps=[("selectkbest", SelectKBest(k=best_params['selectkbest__k'])),
                             ("pca", PCA(n_components=best_params['pca__n_components'])),
                             ("classifier", best_params['classifier'])])

# Entrenamiento
best_model.fit(X, y)

# Guardado del modelo entrenado
with open(yam['output_file'], 'wb') as file:
    pickle.dump(best_model, file)

# Mensaje final
print('El modelo ha sido guardado correctamente')