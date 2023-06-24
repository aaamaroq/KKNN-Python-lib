
# Generated by CodiumAI
from knn import getDataSets
from sklearn.model_selection import train_test_split
from knn import getKnn
from sklearn.metrics import accuracy_score
from sklearn.metrics import pairwise_distances
from sklearn.metrics import mean_absolute_error
import time
import pytest
import csv

"""
Entradas:
- datos_entrenamiento: datos de entrenamiento como un arreglo numpy o un DataFrame de pandas.
- etiquetas_entrenamiento: etiquetas de entrenamiento como un arreglo numpy o una Serie de pandas.
- k: número de vecinos a considerar para la clasificación (entero).
- métrica: métrica de distancia a utilizar para calcular la distancia entre puntos (por defecto es 'manhattan').

Flujo:
1. Crear un objeto clasificador KNN con el número especificado de vecinos y la métrica de distancia.
2. Entrenar el clasificador con los datos de entrenamiento y las etiquetas proporcionadas.
3. Devolver el modelo entrenado.

Salidas:
- knn: objeto clasificador KNN entrenado.

Aspectos adicionales:
- La métrica de distancia por defecto es 'manhattan', pero se pueden especificar otras métricas.
- La función asume que los datos de entrenamiento y las etiquetas están preprocesados y en el formato correcto.
"""

class TestGetknn:
    
    # Prueba que la función getKnn devuelve un modelo KNN entrenado con entradas válidas, incluido el valor especificado de k y la métrica de distancia.
    @pytest.mark.parametrize("model",["librery","myModel"])
    def test_getKnn_validInputs(self,model):
        # Obtenemos todos los dataset
        datasets = getDataSets()
        k=3

        claves = list(datasets.keys())
        # Para cada dataset ejecutamos las pruebas
        for clave in claves:
            datos,etiquetas = datasets[clave]
            print("DATASET: "+clave + " K: "+str(k))
        
            # Dividimos los datos en entrenamiento y prueba utilizando la función train_test_split
            datos_entrenamiento, datos_prueba, etiquetas_entrenamiento, etiquetas_prueba = train_test_split(datos, etiquetas, test_size=0.2, random_state=42)

            knn = getKnn(datos_entrenamiento, etiquetas_entrenamiento,k,model=model)

            # Verificamos que el modelo haya sido entrenado correctamente
            assert knn.score(datos_prueba, etiquetas_prueba) > 0.0

            # Verificamos que el modelo haya sido entrenado con el valor de k especificado
            assert knn.n_neighbors == k

            # Verificamos que el modelo haya sido entrenado con la métrica de distancia especificada
            assert knn.metric == 'manhattan' or knn.metric == 'euclidean' or knn.metric == 'minkowski' or knn.metric == 'chebyshev' or knn.metric == 'manhattan_distance' or knn.metric == 'mahalanobis'

    # Prueba la funcionalidad básica de la función getKnn entrenando un modelo KNN en el conjunto de datos de Iris y evaluando su precisión.
    @pytest.mark.parametrize("model",["librery","myModel"])
    def test_Basic_functionality_(self,model):
        # Obtenemos el dataset Iris
        from sklearn.datasets import load_iris
        iris = load_iris()
        
        k=3
        
        datos_entrenamiento, datos_prueba, etiquetas_entrenamiento, etiquetas_prueba = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

        knn = getKnn(datos_entrenamiento, etiquetas_entrenamiento,k,model=model)

        # Predecimos las etiquetas de los datos de prueba Por que utilizmaos el algoritmo para predicción
        etiquetas_predichas = knn.predict(datos_prueba)
        # Evaluamos la precisión del modelo utilizando la función accuracy_score
        precision = accuracy_score(etiquetas_prueba, etiquetas_predichas)

        #El modelo precide el 100 de las veces
        print("PRECISIÓN: "+str(precision))
        assert precision > 0.9

    # Pruebe con el conjunto de datos más pequeño posible y verifique que el algoritmo pueda manejar estos casos especiales correctamente.
    @pytest.mark.parametrize("model",["librery","myModel"])
    def test_SmallestDataset(self,model):
        # Obtener el conjunto de datos más pequeño
        datos = [[1, 2], [3, 4], [5, 6]]
        etiquetas = [0, 1, 0]
        k = 1

        print("DATASET: Smallest Dataset" + " K: " + str(k))

        knn = getKnn(datos, etiquetas, k,model=model)

        # Predecir las etiquetas para el mismo conjunto de datos
        etiquetas_predichas = knn.predict(datos)

        # Verificar si las etiquetas predichas son iguales a las etiquetas reales
        assert all(etiquetas_predichas == etiquetas)


    # Para un conjunto de datos de prueba específico, verifique si los vecinos más cercanos devueltos son de hecho los puntos más cercanos según la métrica de distancia elegida.
    @pytest.mark.parametrize("model",["librery","myModel"])
    def test_Nearest_neighbors_validity_(self,model):
        # Obtenemos todos los dataset
        datasets = getDataSets()
        k=3

        claves = list(datasets.keys())
        # Para cada dataset ejecutamos las pruebas
        for clave in claves:
            datos,etiquetas = datasets[clave]
            print("DATASET: "+clave + " K: "+str(k))
        

            knn = getKnn(datos, etiquetas,k,model=model)

            # Obtener los vecinos más cercanos de cada punto de prueba
            closest_neighbors = knn.kneighbors(datos, n_neighbors=1, return_distance=False)

            # Calcular las distancias entre los datos de prueba y los demás puntos del conjunto de datos utilizando la métrica de distancia elegida por el modelo
            distances = pairwise_distances(datos, metric=knn.metric)

            
            # Verificar si los vecinos más cercanos devueltos son realmente los puntos más cercanos según la métrica de distancia
            correct = 0
            for i, neighbor_index in enumerate(closest_neighbors):
                if neighbor_index == distances[i].argmin():
                    correct += 1

            # Calcular la precisión (accuracy) dividiendo el número de predicciones correctas entre el número total de puntos de prueba
            accuracy = correct / len(closest_neighbors)
            
            # Imprimir la precisión para el conjunto de datos actual
            print(f"Accuracy for {clave}: {accuracy}")

            # Verificar que todos los vecinos más cercanos devueltos son realmente los puntos más cercanos
            assert correct == len(closest_neighbors), "¡Error! Los vecinos más cercanos no son los puntos más cercanos según la métrica de distancia."

    # Pruebe el algoritmo con diferentes valores de k y verifique si la precisión cambian como se esperaba. 
    @pytest.mark.parametrize("model",["librery","myModel"])
    def test_K_values(self,model):
        from sklearn.datasets import load_iris
        iris = load_iris()

        # Dividir los datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

        # Abrir archivo CSV para escritura
        ruta='log/precision_python_'+model+'.csv'
        with open(ruta, mode='w', newline='') as archivo_csv:
            writer = csv.writer(archivo_csv)
            writer.writerow(['k', 'Precisión', 'Cumple assert'])
        
            # Probar el algoritmo con diferentes valores de k y comprobar su precisión
            for k in range(1, 10):
                knn = getKnn(X_train, y_train, k,model=model)
                accuracy = knn.score(X_test, y_test)
                
                # Comprobar si se cumple el assert
                assert_status = accuracy >= 0.9
                
                # Escribir en el archivo CSV
                writer.writerow([k, accuracy, assert_status])


    @pytest.mark.parametrize("model", ["librery", "myModel"])
    def test_Distance_metric_variation_(self,model):
        from sklearn.datasets import load_iris
        iris = load_iris()
        k = 3
        datos_entrenamiento, datos_prueba, etiquetas_entrenamiento, etiquetas_prueba = train_test_split(
            iris.data, iris.target, test_size=0.2, random_state=42
        )

        metrics = ["euclidean", "manhattan", "minkowski", "chebyshev"]
        precisions = []

        for metric in metrics:
            knn = getKnn(datos_entrenamiento, etiquetas_entrenamiento, k, metric=metric, model=model)
            etiquetas_predichas = knn.predict(datos_prueba)
            precision = accuracy_score(etiquetas_prueba, etiquetas_predichas)
            precisions.append(precision)

        with open("log/precision_metrics_python_"+model+".csv", mode="w", newline="") as csv_file:
            fieldnames = ["Metric", "Precision"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for metric, precision in zip(metrics, precisions):
                writer.writerow({"Metric": metric, "Precision": precision})
                assert precision > 0.9




  

    # Si el algoritmo se utiliza para regresión, verificar si los valores predichos son cercanos a los valores reales y si el error es aceptable.
    # Nota esta opción falla si los valores de entrenamientos son pequeños
    @pytest.mark.parametrize("model",["librery","myModel"])
    def test_Prueba_de_regresión(self,model):
        # Creamos un conjunto de datos de prueba para regresión
        datos_entrenamiento = [[1, 2], [2, 4], [3, 6], [4, 8], [5, 10], [6, 12], [7, 14], [8, 16]]
        etiquetas_entrenamiento = [2, 4, 6, 8, 10, 12, 14, 16]
        k = 2

        knn = getKnn(datos_entrenamiento, etiquetas_entrenamiento, k,model=model)

        # Predecimos las etiquetas de los datos de prueba
        datos_prueba = [[5, 10], [6, 12]]
        etiquetas_predichas = knn.predict(datos_prueba)


        # Verificamos si el error es aceptable mediante el error absoluto medio 
        error = mean_absolute_error(etiquetas_predichas, [10,12])
        assert error < 2.5


    @pytest.mark.parametrize("model",["librery","myModel"])
    def test_Performance_test(self,model):
        # Obtenemos todos los dataset
        datasets = getDataSets()
        k=3

        claves = list(datasets.keys())
        
        # Abrir archivo CSV para escritura
        ruta='log/time_python_'+model+'.csv'
        with open(ruta, mode='w', newline='') as archivo_csv:
            writer = csv.writer(archivo_csv)
            writer.writerow(['Dataset', 'Tiempo de ejecución', 'Cumple assert'])
        
            # Para cada dataset ejecutamos las pruebas
            for clave in claves:
                datos, etiquetas = datasets[clave]
                print("DATASET: "+clave + " K: "+str(k))
            
                # Dividimos los datos en entrenamiento y prueba utilizando la función train_test_split
                datos_entrenamiento, datos_prueba, etiquetas_entrenamiento, etiquetas_prueba = train_test_split(datos, etiquetas, test_size=0.2, random_state=42)

                start_time = time.time()
                knn = getKnn(datos_entrenamiento, etiquetas_entrenamiento,k,model=model)
                end_time = time.time()

                # Predecimos las etiquetas de los datos de prueba Por que utilizmaos el algoritmo para predicción
                knn.predict(datos_prueba)

                # Evaluate runtime performance
                total_time = end_time - start_time
                print("Tiempo de ejecución: ", total_time)
                
                # Comprobar si se cumple el assert
                assert_status = total_time < 10.0
                
                # Escribir en el archivo CSV
                writer.writerow([clave, total_time, assert_status])


if __name__ == "__main__":
    pytest.main()