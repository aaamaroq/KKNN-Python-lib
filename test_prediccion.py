from knn import *

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def test_prediccion():
    # Obtenemos todos los dataset
    datasets = getDataSets()

    #definimos k en el rango 1 hasta 10
    for k in range(1, 11):


        claves = list(datasets.keys())
        # Para cada dataset ejecutamos las pruebas
        for clave in claves:
            datos,etiquetas = datasets[clave]
            print("DATASET: "+clave + " K: "+str(k))
        
            # Dividimos los datos en entrenamiento y prueba utilizando la función train_test_split
            datos_entrenamiento, datos_prueba, etiquetas_entrenamiento, etiquetas_prueba = train_test_split(datos, etiquetas, test_size=0.2, random_state=42)

            knn = getKnn(datos_entrenamiento, etiquetas_entrenamiento,k)

            # Predecimos las etiquetas de los datos de prueba Por que utilizmaos el algoritmo para predicción
            etiquetas_predichas = knn.predict(datos_prueba)
            # Evaluamos la precisión del modelo utilizando la función accuracy_score
            precision = accuracy_score(etiquetas_prueba, etiquetas_predichas)

            #El modelo precide el 100 de las veces
            print("PRECISIÓN: "+str(precision))
            assert precision > 0.9

