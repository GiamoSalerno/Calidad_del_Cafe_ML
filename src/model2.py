# Importo librerías
import pandas as pd, pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV

# Leo los datos.
arabica = pd.read_csv('data/train/arabica_train2.csv')
arabica.drop(columns=['Unnamed: 0'], inplace=True)

# Defino X e Y.
X = arabica.drop(columns=['Total.Cup.Points'])
y = arabica['Total.Cup.Points']

# Divido en train y test.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state= 5)

# Defino el modelo.
pipe = Pipeline(steps=[("scaler", StandardScaler()),
                       ("selectkbest", SelectKBest()),
                       ("pca", PCA()),
                       ('classifier', RandomForestClassifier())])

logistic_params = {'selectkbest__k' : [1,2],
                   'pca__n_components': [1,2],
                   'classifier': [LogisticRegression(solver='liblinear')],
                   'classifier__penalty': ['l1','l2']}

rf_params = {'scaler' : [StandardScaler(), None],
             'selectkbest__k' : [1,2],
             'pca__n_components': [1,2],
             'classifier': [RandomForestClassifier()],
             'classifier__max_features': [2,3,4],
             'classifier__max_depth': [3,5,7]}

gb_params = {'scaler' : [StandardScaler(), None],
             'selectkbest__k' : [1,2],
             'pca__n_components': [1,2],
             'classifier': [GradientBoostingClassifier()],
             'classifier__max_features': [2,3,4],
             'classifier__max_depth': [3,5,7]}

knn_params = {'selectkbest__k' : [1,2],
              'pca__n_components': [1,2],
              'classifier': [KNeighborsClassifier()],
              'classifier__n_neighbors': [1,2,3]}

svm_params = {'selectkbest__k' : [1,2],
              'pca__n_components': [1,2],
              'classifier': [SVC()],
              'classifier__C': [20, 40, 60]}

search_space = [logistic_params,
                rf_params,
                gb_params,
                knn_params,
                svm_params]

clf = GridSearchCV(estimator = pipe,
                   param_grid = search_space,
                   cv=3,
                   scoring="accuracy",
                   n_jobs=-1)

# Entreno el modelo.
clf.fit(X_train, y_train)

# Guardo el modelo.
with open('models/trained_classifier_model1.pkl', 'wb') as file:
    pickle.dump(clf, file)