from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def knn(datos_entrenamiento, etiquetas_entrenamiento, datos_prueba, etiquetas_prueba, k):
    # Creamos el objeto clasificador KNN con el valor de k especificado
    knn = KNeighborsClassifier(n_neighbors=k)
    # Entrenamos el clasificador KNN con los datos de entrenamiento
    knn.fit(datos_entrenamiento, etiquetas_entrenamiento)
    # Predecimos las etiquetas de los datos de prueba
    etiquetas_predichas = knn.predict(datos_prueba)
    # Evaluamos la precisión del modelo utilizando la función accuracy_score
    precision = accuracy_score(etiquetas_prueba, etiquetas_predichas)
    # Devolvemos la precisión del modelo
    return precision

