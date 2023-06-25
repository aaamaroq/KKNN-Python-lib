
pub mod knn {

    // Importamos las bibliotecas necesarias
    use std::cmp::Ordering;
    use std::collections::HashMap;

    // Definimos la estructura MyClasificadorKNN
    pub struct MyClasificadorKNN {
        k: usize,
        puntosEntrenamiento: Vec<(Vec<f64>, String)>,
    }

    impl MyClasificadorKNN {
        // Función constructor
        pub fn new(k: usize, puntosEntrenamiento: Vec<(Vec<f64>, String)>) -> Self {
            MyClasificadorKNN { k, puntosEntrenamiento }
        }

        // Función predecir
        pub fn predecir(&self, punto: &Vec<f64>,distancia_fn: fn(&Vec<f64>, &Vec<f64>) -> f64) -> String {
            // Calcular las distancias entre el punto y los puntosEntrenamiento en el clasificador
            let mut distancias: Vec<(f64, &String)> = self
                .puntosEntrenamiento
                .iter()
                .map(|(x, clase)| ((distancia_fn)(x, punto), clase))
                .collect();

            // Ordenar las distancias de menor a mayor
            distancias.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

            // Obtener las k clases más cercanas
            let clases_cercanas = distancias.iter().take(self.k).map(|&(_, clase)| clase);

            // Contar las apariciones de cada clase y devolver la clase con mayor frecuencia
            let mut conteo = HashMap::new();
            for clase in clases_cercanas {
                *conteo.entry(clase).or_insert(0) += 1;
            }

            conteo
                .into_iter()
                .max_by_key(|&(_, count)| count)
                .map(|(clase, _)| clase.clone())
                .unwrap()
        }
    }

    // Función de distancia euclidiana entre dos puntos
    pub fn euclidean_distance(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    // Función de distancia de Manhattan entre dos puntos
    pub fn manhattan_distance(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .sum()
    }

    // Función de distancia de Chebyshov entre dos puntos
    pub fn chebyshev_distance(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(f64::MIN, f64::max)
    }

}





