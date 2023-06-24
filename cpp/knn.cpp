#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <map>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <vector>
#include <string>


using namespace std;

struct PuntoDatos {
    vector<float> datos;
    int etiqueta;
};

class Distancia {
public:
    virtual double calcularDistancia(PuntoDatos p1, PuntoDatos p2) = 0;
};

class DistanciaEuclidiana : public Distancia {
public:
    double calcularDistancia(PuntoDatos p1, PuntoDatos p2) override {
        double sumaCuadrados = 0.0;
        for (size_t i = 0; i < p1.datos.size(); ++i) {
            double diff = p1.datos[i] - p2.datos[i];
            sumaCuadrados += diff * diff;
        }
        return sqrt(sumaCuadrados);
    }
};


// Asume que las funciones y clases previamente mostradas ya están definidas

class DistanciaMinkowski : public Distancia {
public:
    double calcularDistancia(PuntoDatos p1, PuntoDatos p2) override {
        double sumaPotencias = 0.0;
        int p = 3;
        for (size_t i = 0; i < p1.datos.size(); ++i) {
            sumaPotencias += pow(abs(p1.datos[i] - p2.datos[i]), p);
        }
        return pow(sumaPotencias, 1.0 / p);
    }
};

class DistanciaChebyshev : public Distancia {
public:
    double calcularDistancia(PuntoDatos p1, PuntoDatos p2) override {
        double maximo = 0.0;
        for (size_t i = 0; i < p1.datos.size(); ++i) {
            double diff = abs(p1.datos[i] - p2.datos[i]);
            if (diff > maximo) {
                maximo = diff;
            }
        }
        return maximo;
    }
};


class DistanciaManhattan : public Distancia {
public:
    double calcularDistancia(PuntoDatos p1, PuntoDatos p2) override {
        double sumaDistancias = 0.0;
        for (size_t i = 0; i < p1.datos.size(); ++i) {
            sumaDistancias += abs(p1.datos[i] - p2.datos[i]);
        }
        return sumaDistancias;
    }
};

// Función para clasificar un punto en función de sus vecinos más cercanos
int clasificarPunto(const vector<PuntoDatos>& conjuntoDatos, PuntoDatos punto, int k, Distancia& metrica) {
    vector<pair<double, int>> distancias;
    
    // Calcular las distancias entre el punto y todos los puntos del conjunto de datos
    for (const auto& dato : conjuntoDatos) {
        double distancia = metrica.calcularDistancia(punto, dato);
        distancias.push_back(make_pair(distancia, dato.etiqueta));
    }
    
    // Ordenar las distancias de menor a mayor
    sort(distancias.begin(), distancias.end());
    
    // Contar las etiquetas de los k vecinos más cercanos
    map<int, int> conteoEtiquetas;
    for (int i = 0; i < k; ++i) {
        conteoEtiquetas[distancias[i].second]++;
    }
    
    // Encontrar la etiqueta más común entre los vecinos más cercanos
    int etiquetaMasComun;
    int conteoMaximo = 0;
    for (const auto& par : conteoEtiquetas) {
        if (par.second > conteoMaximo) {
            conteoMaximo = par.second;
            etiquetaMasComun = par.first;
        }
    }
    
    return etiquetaMasComun;
}




std::vector<PuntoDatos> cargarDataSet(const std::string& archivoData, const std::string& archivoTarget) {
    std::vector<PuntoDatos> dataset;
    
    // Abrir el archivo data.csv
    std::ifstream archivoDataStream(archivoData);
    if (!archivoDataStream.is_open()) {
        std::cerr << "Error al abrir el archivo " << archivoData << std::endl;
        return dataset;
    }
    
    // Abrir el archivo target.csv
    std::ifstream archivoTargetStream(archivoTarget);
    if (!archivoTargetStream.is_open()) {
        std::cerr << "Error al abrir el archivo " << archivoTarget << std::endl;
        archivoDataStream.close();
        return dataset;
    }
    
    std::string lineaData, lineaTarget;
    
    // Leer la primera línea de data.csv (encabezados)
    std::getline(archivoDataStream, lineaData);
    
    // Leer la primera línea de target.csv (encabezados)
    std::getline(archivoTargetStream, lineaTarget);
    
    while (std::getline(archivoDataStream, lineaData) && std::getline(archivoTargetStream, lineaTarget)) {
        std::stringstream ssData(lineaData);
        std::stringstream ssTarget(lineaTarget);
        std::string valorData, valorTarget;
        
        PuntoDatos punto;
        
        // Leer los datos del archivo data.csv
        while (std::getline(ssData, valorData, ',')) {
            punto.datos.push_back(std::stod(valorData));
        }
        
        // Leer la etiqueta del archivo target.csv
        if (std::getline(ssTarget, valorTarget, ',')) {
            punto.etiqueta = std::stoi(valorTarget);
        }
        
        dataset.push_back(punto);
    }
    
    archivoDataStream.close();
    archivoTargetStream.close();
    
    return dataset;
}



map<string, vector<vector<PuntoDatos>>> cargarTodosLosDataset() {
    string carpetaDataset = "dataset";

    map<string, vector<vector<PuntoDatos>>> datasets;

    // Iterar sobre las carpetas en la carpeta dataset
    for (const auto& carpeta : filesystem::directory_iterator(carpetaDataset)) {
        if (carpeta.is_directory()) {
            string nombreCarpeta = carpeta.path().filename().string();
            string archivoData = carpeta.path().string() + "/data.csv";
            string archivoTarget = carpeta.path().string() + "/target.csv";

            // Cargar el dataset de la carpeta actual
            vector<PuntoDatos> dataset = cargarDataSet(archivoData, archivoTarget);

            datasets[nombreCarpeta].push_back(dataset);
        }
    }

    return datasets;
}








