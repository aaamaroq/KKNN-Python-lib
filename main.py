from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from managerCsv import *
import argparse

def knn(datos_entrenamiento, etiquetas_entrenamiento, datos_prueba, etiquetas_prueba, k):
    # Creamos el objeto clasificador KNN con el valor de k especificado
    knn = KNeighborsClassifier(n_neighbors=k)
    # Entrenamos el clasificador KNN con los datos de entrenamiento
    knn.fit(datos_entrenamiento, etiquetas_entrenamiento)
    # Predecimos las etiquetas de los datos de prueba Por que utilizmaos el algoritmo para predicción
    etiquetas_predichas = knn.predict(datos_prueba)
    # Evaluamos la precisión del modelo utilizando la función accuracy_score
    precision = accuracy_score(etiquetas_prueba, etiquetas_predichas)
    # Devolvemos la precisión del modelo
    return precision





# Definimos los argumentos del programa
parser = argparse.ArgumentParser(description='Importar datos,etiquetas desde archivo CSV y k desde consola')
parser.add_argument('data_file', type=str, help='Nombre del archivo CSV con los datos')
parser.add_argument('target_file', type=str, help='Nombre del archivo CSV con las etiquetas')
parser.add_argument('k', type=int, help='numero de vecinos a visitar')

args = parser.parse_args()

# Leemos los argumentos del programa
#Leemos los argumentos del programa
datos = import_from_csv(args.data_file)
etiquetas = import_from_csv(args.target_file)
#NOTA: Se tiene en cuenta que k pueda ser igual a 0
k = args.k

# Dividimos los datos en entrenamiento y prueba utilizando la función train_test_split
datos_entrenamiento, datos_prueba, etiquetas_entrenamiento, etiquetas_prueba = train_test_split(datos, etiquetas, test_size=0.2, random_state=42)

# Utilizamos la función knn para entrenar un modelo y calcular su precisión
precision = knn(datos_entrenamiento, etiquetas_entrenamiento, datos_prueba, etiquetas_prueba, k)

# Imprimimos la precisión del modelo
print("Precisión del modelo:", precision)
