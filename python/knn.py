from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import os
from colorama import Fore
import pandas as pd

class MyKNeighborsClassifier:
    """
    Clasificador k-vecinos más cercanos personalizado.
    """

    def __init__(self, n_neighbors, metric):
        """
        Inicializa el clasificador.

        Parámetros:
        - n_neighbors: número de vecinos a considerar.
        - metric: métrica de distancia a utilizar ('euclidean' o 'manhattan').
        """
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.X_train = None
        self.y_train = None

    def fit(self, datos_entrenamiento, etiquetas_entrenamiento):
        """
        Ajusta el modelo a los datos de entrenamiento.

        Parámetros:
        - datos_entrenamiento: matriz de características de los datos de entrenamiento.
        - etiquetas_entrenamiento: etiquetas de los datos de entrenamiento.
        """
        self.X_train = np.array(datos_entrenamiento)
        self.y_train = np.array(etiquetas_entrenamiento)

    def score(self, datos_prueba, etiquetas_prueba):
        """
        Calcula la precisión del modelo en los datos de prueba.

        Parámetros:
        - datos_prueba: matriz de características de los datos de prueba.
        - etiquetas_prueba: etiquetas de los datos de prueba.

        Retorna:
        - accuracy: precisión del modelo en los datos de prueba.
        """
        X_test = np.array(datos_prueba)
        y_test = np.array(etiquetas_prueba)
        y_pred = self.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        return accuracy

    def predict(self, datos_prueba):
        """
        Realiza predicciones para los datos de prueba.

        Parámetros:
        - datos_prueba: matriz de características de los datos de prueba.

        Retorna:
        - y_pred: etiquetas predichas para los datos de prueba.
        """
        X_test = np.array(datos_prueba)  # Convierte los datos de prueba a una matriz NumPy
        y_pred = []  # Lista para almacenar las etiquetas predichas

        # Itera sobre cada dato de prueba
        for i in range(X_test.shape[0]):
            distances = self.calculate_distances(X_test[i])  # Calcula las distancias entre el punto de prueba y los puntos de entrenamiento
            indices = np.argsort(distances)[:self.n_neighbors]  # Obtiene los índices de los vecinos más cercanos
            neighbors_labels = self.y_train[indices]  # Obtiene las etiquetas de los vecinos más cercanos

            # Cuenta las ocurrencias de cada etiqueta
            unique_labels, counts = np.unique(neighbors_labels, return_counts=True)

            # Encuentra la etiqueta más frecuente entre los vecinos más cercanos
            predicted_label = unique_labels[np.argmax(counts)]

            # Agrega la etiqueta predicha a la lista
            y_pred.append(predicted_label)

        return np.array(y_pred)  # Convierte la lista en un arreglo NumPy y lo retorna


    def kneighbors(self, datos, n_neighbors=1, return_distance=False):
        """
        Encuentra los k vecinos más cercanos para los datos dados.

        Parámetros:
        - datos: matriz de características de los datos.
        - n_neighbors: número de vecinos a considerar (por defecto es 1).
        - return_distance: indica si se deben devolver las distancias también (por defecto es False).

        Retorna:
        - Si return_distance es True:
            - distances: matriz de distancias a los k vecinos más cercanos.
        - Si return_distance es False:
            - indices: matriz de índices de los k vecinos más cercanos.
        """
        X = np.array(datos)  # Convierte los datos a una matriz NumPy
        distances = []  # Lista para almacenar las distancias
        indices = []  # Lista para almacenar los índices de los vecinos

        # Itera sobre cada dato
        for i in range(X.shape[0]):
            distances_i = self.calculate_distances(X[i])  # Calcula las distancias entre el dato y los puntos de entrenamiento
            indices_i = np.argsort(distances_i)[:n_neighbors]  # Obtiene los índices de los vecinos más cercanos

            distances.append(distances_i[indices_i])  # Agrega las distancias de los vecinos más cercanos
            indices.append(indices_i)  # Agrega los índices de los vecinos más cercanos

        if return_distance:
            return np.array(distances)  # Retorna una matriz de distancias
        else:
            return np.array(indices)  # Retorna una matriz de índices


    def calculate_distances(self, x):
        """
        Calcula las distancias entre un punto dado y los puntos de entrenamiento.

        Parámetros:
        - x: punto para el cual se calcularán las distancias.
        
        Retorna:
        - distances: matriz de distancias entre el punto dado y los puntos de entrenamiento.
        """

        distances = []
        for i in range(self.X_train.shape[0]):
            if self.metric == 'euclidean':
                print(self.X_train[i])
                distance = np.sqrt(np.sum((self.X_train[i] - x) ** 2))  # Distancia Euclidiana
            elif self.metric == 'manhattan':
                distance = np.sum(np.abs(self.X_train[i] - x))  # Distancia Manhattan
            elif self.metric == 'minkowski':
                p = 3  # Parámetro p para la distancia Minkowski (puedes ajustar su valor según necesites)
                distance = np.power(np.sum(np.power(np.abs(self.X_train[i] - x), p)), 1/p)  # Distancia Minkowski
            elif self.metric == 'chebyshev':
                distance = np.max(np.abs(self.X_train[i] - x))  # Distancia Chebyshev
            else:
                raise ValueError("Unsupported metric: {}".format(self.metric))

            distances.append(distance)

        return np.array(distances)


def getKnn(datos_entrenamiento, etiquetas_entrenamiento, k, metric='manhattan', model="myModel"):
    # Creamos el objeto clasificador KNN con el valor de k especificado

    if model == "myModel" :
        knn = MyKNeighborsClassifier(n_neighbors=k, metric=metric)
        # Entrenamos el clasificador KNN con los datos de entrenamiento
        knn.fit(datos_entrenamiento, etiquetas_entrenamiento)
        #devolvemos el modelo ya entrenado
        return knn
    elif model == "librery":
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
        # Entrenamos el clasificador KNN con los datos de entrenamiento
        knn.fit(datos_entrenamiento, etiquetas_entrenamiento)
        #devolvemos el modelo ya entrenado
        return knn
    
    raise ValueError("llegó aquí por ",model)




def export_to_csv(data, filename):
    """
    Exporta un objeto pandas.DataFrame a un archivo CSV con el nombre especificado.
    """
    data.to_csv(filename, index=False)

def import_from_csv(filename):
    """
    Importa un archivo CSV y retorna un objeto pandas.DataFrame con su contenido.
    """
    try:
        data = pd.read_csv(filename)
    except FileNotFoundError:
        print(Fore.RED + f"Error: El archivo {filename} no existe.")
        exit(1)
    except pd.errors.ParserError:
        print(Fore.RED + f"Error: El archivo {filename} no tiene un formato válido.")
        exit(1)

    if data is None:
        print(Fore.RED + "Error: El archivo {filename} está vacio.")
        exit(1)

    return data

def getDataSets():
    path= os.path.abspath("./dataset")
    datasets = {}

    if not os.listdir(path):
        print(Fore.RED + "Error: No hay ningún dataset para cargar." )
        exit(1)

    for foldername in os.listdir(path):
        folderpath = os.path.join(path, foldername)
        if os.path.isdir(folderpath):
            data = import_from_csv(folderpath + '/data.csv')
            target =  import_from_csv(folderpath + '/target.csv')
            datasets[foldername] = (data, target)
    return datasets


# Ejemplo de uso con el dataset Iris
#from sklearn.datasets import load_iris

#iris = load_iris()

# Exportar iris.target a un archivo CSV
#export_to_csv(pd.DataFrame(iris.target), "iris_target.csv")

# Exportar iris.data a un archivo CSV
#export_to_csv(pd.DataFrame(iris.data), "iris_data.csv")

# Importar iris.target desde un archivo CSV
#target = import_from_csv("iris_target.csv")

# Importar iris.data desde un archivo CSV
#data = import_from_csv("iris_data.csv")




