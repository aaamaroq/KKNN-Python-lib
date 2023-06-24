use ndarray::{Array2, Array1};
use ndarray::prelude::*;
use std::cmp::Ordering;

pub struct ClasificadorKVecinos {
    num_vecinos: usize,
    metrica: String,
    x_entrenamiento: Option<Array2<f64>>,
    y_entrenamiento: Option<Array1<usize>>,
}

impl ClasificadorKVecinos {
    // Crea una nueva instancia del clasificador k-NN
    pub fn nuevo(num_vecinos: usize, metrica: &str) -> Self {
        ClasificadorKVecinos {
            num_vecinos,
            metrica: String::from(metrica),
            x_entrenamiento: None,
            y_entrenamiento: None,
        }
    }

    // Ajusta el clasificador con los datos de entrenamiento y sus respectivas etiquetas
    pub fn ajustar(&mut self, datos_entrenamiento: &Array2<f64>, etiquetas_entrenamiento: &Array1<usize>) {
        self.x_entrenamiento = Some(datos_entrenamiento.clone());
        self.y_entrenamiento = Some(etiquetas_entrenamiento.clone());
    }

    // Calcula la precisión del clasificador utilizando datos de prueba y sus respectivas etiquetas
    pub fn puntuacion(&self, datos_prueba: &Array2<f64>, etiquetas_prueba: &Array1<usize>) -> f64 {
        let y_prediccion = self.predecir(datos_prueba);
        let precision = y_prediccion.iter().zip(etiquetas_prueba).filter(|&(x, y)| x == y).count() as f64 / y_prediccion.len() as f64;
        precision
    }

    // Predice las etiquetas de los datos de prueba utilizando el algoritmo k-NN
    pub fn predecir(&self, datos_prueba: &Array2<f64>) -> Array1<usize> {
        datos_prueba
            .axis_iter(Axis(0))
            .map(|x_prueba| {
                // Calcula las distancias entre el dato de prueba y los datos de entrenamiento
                let distancias = self.calcular_distancias(&x_prueba);
                
                // Ordena los índices de los datos de entrenamiento según las distancias
                let mut indices: Vec<usize> = (0..distancias.len()).collect();
                indices.sort_by(|a, b| distancias[*a].partial_cmp(&distancias[*b]).unwrap_or(Ordering::Equal));
                
                // Selecciona las etiquetas de los k vecinos más cercanos
                let etiquetas_vecinos: Vec<usize> = indices
                    .iter()
                    .take(self.num_vecinos)
                    .map(|&idx| self.y_entrenamiento.as_ref().unwrap()[idx])
                    .collect();
                
                // Encuentra la etiqueta más común entre los vecinos y la asigna como predicción
                let (etiqueta_predicha, _) = etiquetas_vecinos.into_iter().fold(std::collections::HashMap::new(), |mut acum, x| {
                    *acum.entry(x).or_insert(0) += 1;
                    acum
                }).into_iter().max_by_key(|&(_, conteo)| conteo).unwrap();
                etiqueta_predicha
            })
            .collect()
    }

    // Calcula las distancias entre un punto de prueba y todos los puntos de entrenamiento utilizando la métrica especificada
    fn calcular_distancias(&self, x: &Array1<f64>) -> Array1<f64> {
        self.x_entrenamiento.as_ref().unwrap().axis_iter(Axis(0)).map(|x_entrenamiento_i| {
            match self.metrica.as_str() {
                "euclidean" => x_entrenamiento_i.sq_l2_dist(&x).unwrap().sqrt(),
                "manhattan" => x_entrenamiento_i.l1_dist(&x).unwrap(),
                "minkowski" => {
                    let p = 3;
                    x_entrenamiento_i.iter().zip(x.iter()).map(|(&a, &b)| (a - b).abs().powi(p)).sum::<f64>().powf(1.0 / p as f64)
                }
                "chebyshev" => x_entrenamiento_i.iter().zip(x.iter()).map(|(&a, &b)| (a - b).abs()).fold(f64::MIN, f64::max),
                _ => panic!("Métrica no soportada: {}", self.metrica),
            }
        }).collect()
    }
}