from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from Utils import *
import os

def knn(datos_entrenamiento, etiquetas_entrenamiento, k):
    # Creamos el objeto clasificador KNN con el valor de k especificado
    knn = KNeighborsClassifier(n_neighbors=k)
    # Entrenamos el clasificador KNN con los datos de entrenamiento
    knn.fit(datos_entrenamiento, etiquetas_entrenamiento)

    #devolvemos el modelo ya entrenado
    return knn






