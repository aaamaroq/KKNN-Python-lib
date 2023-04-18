import pandas as pd
from colorama import Fore

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

