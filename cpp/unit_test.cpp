#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <utility>
#include "knn.cpp"

//Obtenemos el dataset y dividimos el 80% para entrenamiento y el resto para pruebas
void dividirDatos(const vector<PuntoDatos>& dataset, vector<PuntoDatos>& entrenamiento, vector<PuntoDatos>& prueba, double ratio = 0.8) {
    size_t dataSize = dataset.size();
    size_t trainSize = static_cast<size_t>(dataSize * ratio);

    for (size_t i = 0; i < dataSize; ++i) {
        if (i < trainSize) {
            entrenamiento.push_back(dataset[i]);
        } else {
            prueba.push_back(dataset[i]);
        }
    }
}

// Compara los datos obtenidos con los datos que se conocen son ciertos
double calcularPrecision(const vector<PuntoDatos>& entrenamiento, const vector<PuntoDatos>& prueba, int k, Distancia& metrica) {
    int aciertos = 0;

    for (const auto& puntoPrueba : prueba) {
        int etiquetaPredicha = clasificarPunto(entrenamiento, puntoPrueba, k, metrica);
        if (etiquetaPredicha == puntoPrueba.etiqueta) {
            aciertos++;
        }
    }
    return static_cast<double>(aciertos) / prueba.size();
}


void test_K_values() {
    // Obtenemos los dataset
    auto datasets = cargarTodosLosDataset();
    const std::string nombreDatasetBuscado = "Iris";
    
    // De ellos obtenemos el dataset Iris (si no está no hacemos nada)
    if (datasets.find(nombreDatasetBuscado) != datasets.end()) {
        const auto& dataset = datasets[nombreDatasetBuscado].front();

        // Separamos los datos en prueba y entrenamiento
        vector<PuntoDatos> entrenamiento, prueba;
        dividirDatos(dataset, entrenamiento, prueba);
        
        //Preparamos los datos y el archivo de salida
        std::ofstream resultados("log/precision_cpp.csv");
        resultados << "k,Precision,Cumple assert\n";
        DistanciaEuclidiana distEuclidiana;

        // Para cada k de 1 a 10 calculamos su precisión con la distancia euclidiana
        for (int k = 1; k <= 10; ++k) {
            double precision = calcularPrecision(entrenamiento, prueba, k, distEuclidiana);
            bool assert_status = precision >= 0.9;
            resultados << k << "," << precision << "," << assert_status << "\n";
        }

        // se cierra el archivo
        resultados.close();

    } else {
        std::cerr << "Dataset " << nombreDatasetBuscado << " no encontrado." << std::endl;
    }
}

void test_Performance_test() {
    auto datasets = cargarTodosLosDataset();
    std::ofstream resultados("log/time_cpp.csv");
    resultados << "Dataset,Tiempo de ejecucion,Cumple assert\n";

    // Para cada dataset
    for (const auto& par : datasets) {
        const auto& nombreDataset = par.first;
        const auto& dataset = par.second.front();

        // Obtener datso de entrenamiento y prueba
        vector<PuntoDatos> entrenamiento, prueba;
        dividirDatos(dataset, entrenamiento, prueba);

        DistanciaEuclidiana distEuclidiana;
        int k = 3;

        // medir tiempo y clasificar puntos
        auto start = std::chrono::high_resolution_clock::now();
        for (const auto& puntoPrueba : prueba) {
            clasificarPunto(entrenamiento, puntoPrueba, k, distEuclidiana);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        double total_time = diff.count();

        bool assert_status = total_time < 10.0;
        resultados << nombreDataset << "," << total_time << "," << assert_status << "\n";
    }

    resultados.close();
}


void test_Metricas() {
    // Carga todos los dataset
    auto datasets = cargarTodosLosDataset();
    const std::string nombreDatasetBuscado = "Iris";

    // Busca el dataset Iris
    if (datasets.find(nombreDatasetBuscado) != datasets.end()) {
        const auto& dataset = datasets[nombreDatasetBuscado].front();

        vector<PuntoDatos> entrenamiento, prueba;
        dividirDatos(dataset, entrenamiento, prueba);

        std::ofstream resultados("log/precision_metricas_cpp.csv");
        resultados << "Metrica,k,Precision\n";

        // Lista de métricas a evaluar
        std::vector<std::pair<std::string, Distancia*>> metricas = {
            {"Euclidiana", new DistanciaEuclidiana()},
            {"Manhattan", new DistanciaManhattan()},
            {"Chebyshov", new DistanciaChebyshev()}
        };

        int k = 3;

        // Calcular la precisión para cada métrica
        for (const auto& par : metricas) {
            const auto& nombreMetrica = par.first;
            Distancia* metrica = par.second;

            double precision = calcularPrecision(entrenamiento, prueba, k, *metrica);
            resultados << nombreMetrica << "," << k << "," << precision << "\n";

            delete metrica; // Liberar la memoria de la métrica
        }

        resultados.close();
    } else {
        std::cerr << "Dataset " << nombreDatasetBuscado << " no encontrado." << std::endl;
    }
}

int main(){

    test_K_values();
    test_Performance_test();
    test_Metricas() ;

    return 0;

}