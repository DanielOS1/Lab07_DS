import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def cargar_datos(ruta):
    data = pd.read_csv(ruta)
    X = data.drop(['id', 'diagnosis'], axis=1)
    y = data['diagnosis']
    return X, y

def dividir_datos(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

def normalizar_datos(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def buscar_hiperparametros(X_train, y_train, search_type='grid'):
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs'],
        'penalty': ['l2']
    }
    if search_type == 'grid':
        search = GridSearchCV(LogisticRegression(max_iter=5000), param_grid, cv=5, scoring='accuracy')
    else:
        search = RandomizedSearchCV(LogisticRegression(max_iter=5000), param_grid, n_iter=100, cv=5, scoring='accuracy')
    search.fit(X_train, y_train)
    print(f"Mejores hiperparámetros con {search_type.capitalize()} Search:", search.best_params_)
    return search.best_estimator_

def evaluar_modelo(modelo, X_test, y_test):
    predicciones = modelo.predict(X_test)
    print("Matriz de Confusión:\n", confusion_matrix(y_test, predicciones))
    print("Informe de Clasificación:\n", classification_report(y_test, predicciones))

def visualizar_matriz_confusion(y_test, predicciones):
    matriz_conf = confusion_matrix(y_test, predicciones)
    sns.heatmap(matriz_conf, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicciones')
    plt.ylabel('Valores Reales')
    plt.title('Matriz de Confusión')
    plt.show()

def visualizar_caracteristicas_importantes(modelo, X):
    coeficientes = pd.Series(modelo.coef_[0], index=X.columns)
    coeficientes.sort_values().plot(kind='barh')
    plt.title('Características Importantes en la Regresión Logística')
    plt.show()

def visualizar_distribucion_caracteristicas(X, y, caracteristicas):
    df = X.copy()
    df['Clase'] = y
    for caracteristica in caracteristicas:
        plt.figure(figsize=(6, 4))
        sns.histplot(data=df, x=caracteristica, hue='Clase', kde=True, element='step')
        plt.title(f'Distribución de {caracteristica} por Clase')
        plt.show()


X, y = cargar_datos('breast-cancer.csv')
X_train, X_test, y_train, y_test = dividir_datos(X, y)
X_train_scaled, X_test_scaled = normalizar_datos(X_train, X_test)

model = LogisticRegression(max_iter=5000)
model.fit(X_train_scaled, y_train)

# Búsqueda de hiperparámetros y evaluación
best_model_grid = buscar_hiperparametros(X_train_scaled, y_train, 'grid')
best_model_random = buscar_hiperparametros(X_train_scaled, y_train, 'random')

evaluar_modelo(best_model_grid, X_test_scaled, y_test)
evaluar_modelo(best_model_random, X_test_scaled, y_test)

# Visualizaciones
predicciones_grid = best_model_grid.predict(X_test_scaled)
visualizar_matriz_confusion(y_test, predicciones_grid)
visualizar_caracteristicas_importantes(best_model_grid, pd.DataFrame(X_train_scaled, columns=X.columns))

visualizar_distribucion_caracteristicas(pd.DataFrame(X_train_scaled, columns=X.columns), y_train, ['radius_mean', 'texture_mean'])