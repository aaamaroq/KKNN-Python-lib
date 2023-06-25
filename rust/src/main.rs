


#[cfg(test)]
mod tests {

    use std::collections::HashMap;
    use std::fs::{self, File};
    use std::io::{BufRead, BufReader, Write};
    use std::path::Path;
    use std::time::Instant;
    mod knn;
    use knn::knn::{MyClasificadorKNN, euclidean_distance, manhattan_distance, chebyshev_distance};

    const PRECISION_THRESHOLD: f64 = 0.9;


    #[test]
    fn test_performance() -> Result<(), Box<dyn std::error::Error>> {
        let mut datasets = cargar_datasets()?;
        let k = 3;
        let mut output_file = File::create("../log/time_rust.csv")?;
        writeln!(output_file, "Dataset,Tiempo de ejecucion,Cumple assert")?;

        for (dataset_name, dataset) in datasets.iter() {
            let (train_data, test_data) = dividir_datos(dataset[..].to_vec(), 0.8);

            let start = Instant::now();
            let _ = calcular_precision(train_data[..].to_vec(), test_data[..].to_vec(), k, euclidean_distance);
            let elapsed_time = start.elapsed().as_secs_f64();

            let assert_passed = if elapsed_time < 10.0 { "Sí" } else { "No" };
            writeln!(output_file, "{},{},{}", dataset_name, elapsed_time, assert_passed)?;
        }

        Ok(())
    }

    #[test]
    fn test_metricas() -> Result<(), Box<dyn std::error::Error>> {
        let mut datasets = cargar_datasets()?;
        let iris_dataset = match datasets.get("Iris") {
            Some(dataset) => dataset,
            None => return Err("Iris dataset not found".into()),
        };

        let (train_data, test_data) = dividir_datos(iris_dataset[..].to_vec(), 0.8);
        let k = 3;

        let mut output_file = File::create("../log/precision_metricas_rust.csv")?;
        writeln!(output_file, "Metrica,k,Precision")?;

        let metrics: Vec<(&str, fn(&Vec<f64>, &Vec<f64>) -> f64)> = vec![
            ("Euclidean", euclidean_distance),
            ("Manhattan", manhattan_distance),
            ("Chebyshev", chebyshev_distance),
        ];

        for (metric_name, metric_fn) in metrics {
            let precision = calcular_precision(train_data[..].to_vec(), test_data[..].to_vec(), k, metric_fn);
            writeln!(output_file, "{},{},{}", metric_name, k, precision)?;
        }

        Ok(())
    }

    #[test]
    // Función test_K_values
    fn test_K_values() -> Result<(), Box<dyn std::error::Error>> {
        let mut datasets = cargar_datasets()?;
        let iris_dataset = match datasets.get("Iris") {
            Some(dataset) => dataset,
            None => return Err("Iris dataset not found".into()),
        };

        let (train_data, test_data) = dividir_datos(iris_dataset[..].to_vec(), 0.8);

        let mut output_file = File::create("../log/precision_rust.csv")?;
        writeln!(output_file, "k,Precision,Cumple assert")?;

        for k in 1..=10 {

            let precision = calcular_precision(train_data[..].to_vec(),test_data[..].to_vec(),k,euclidean_distance);

            let assert_passed = if precision >= PRECISION_THRESHOLD { "Si" } else { "No" };
            writeln!(output_file, "{},{},{}", k, precision, assert_passed)?;
        }

        Ok(())
    }

    // Función para divir los datos en entrenamiento y prueba
    pub fn dividir_datos(datos: Vec<(Vec<f64>, String)>, porcentaje_entrenamiento: f64) -> (Vec<(Vec<f64>, String)>, Vec<(Vec<f64>, String)>) {
        use rand::seq::SliceRandom;
        use rand::thread_rng;

        let mut rng = thread_rng();
        let cantidad_total = datos.len();
        let cantidad_entrenamiento = (cantidad_total as f64 * porcentaje_entrenamiento).round() as usize;

        let mut datos = datos;
        datos.shuffle(&mut rng);

        let (entrenamiento, prueba) = datos.split_at(cantidad_entrenamiento);

        (entrenamiento.to_vec(), prueba.to_vec())
    }


    pub fn calcular_precision(
        datos_entrenamiento: Vec<(Vec<f64>, String)>,
        datos_prueba: Vec<(Vec<f64>, String)>,
        k: usize,
        distancia_fn: fn(&Vec<f64>, &Vec<f64>) -> f64,
    ) -> f64 {
        let clasificador = MyClasificadorKNN::new(k, datos_entrenamiento);

        let mut aciertos = 0;

        for (punto, clase_real) in &datos_prueba {
            let prediccion = clasificador.predecir(&punto, distancia_fn);
            if *prediccion == *clase_real {
                aciertos += 1;
            }
        }

        aciertos as f64 / datos_prueba.len() as f64
    }




    // Función para cargar los conjuntos de datos de las subcarpetas en la carpeta "dataset"
    fn cargar_datasets() -> Result<HashMap<String, Vec<(Vec<f64>, String)>>, Box<dyn std::error::Error>> {
        let mut datasets = HashMap::new();
        let dataset_dir = Path::new("../dataset");

        // Leer las subcarpetas en la carpeta "dataset"
        for entry in fs::read_dir(dataset_dir)? {
            let entry = entry?;
            let path = entry.path();

            // Si la entrada es una carpeta, leer los archivos "data.csv" y "target.csv"
            if path.is_dir() {
                let dataset_name = path.file_name().unwrap().to_string_lossy().into_owned();
                let data_path = path.join("data.csv");
                let target_path = path.join("target.csv");

                // Leer los archivos CSV y crear el vector de tuplas (datos, etiqueta)
                let data = leer_csv_numerico(data_path)?;
                let target = leer_csv_categorico(target_path)?;
                let dataset = data.into_iter().zip(target).collect();

                // Almacenar el conjunto de datos en el mapa con la clave del nombre de la subcarpeta
                datasets.insert(dataset_name, dataset);
            }
        }

        Ok(datasets)
    }


    // Función auxiliar para leer un archivo CSV de números de punto flotante
    fn leer_csv_numerico(path: impl AsRef<Path>) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
        let file = fs::File::open(path)?;
        let reader = BufReader::new(file);
        let mut data = Vec::new();

        // Leer cada línea del archivo CSV y convertir las entradas en números de punto flotante
        for line in reader.lines() {
            let line = line?;
            let row: Vec<f64> = line
                .split(',')
                .map(|s| s.parse().unwrap_or_default())
                .collect();

            data.push(row);
        }

        Ok(data)
    }

    // Función auxiliar para leer un archivo CSV de etiquetas (cadenas de caracteres)
    fn leer_csv_categorico(path: impl AsRef<Path>) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let file = fs::File::open(path)?;
        let reader = BufReader::new(file);
        let mut data = Vec::new();

        // Leer cada línea del archivo CSV y almacenar las etiquetas como cadenas de caracteres
        for line in reader.lines() {
            let line = line?;
            data.push(line);
        }

        Ok(data)
    }



}


