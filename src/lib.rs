pub struct KNN {
    k: usize,
    training_data: Vec<(Vec<f64>, String)>,
}

impl KNN {
    pub fn new(k: usize) -> Self {
        KNN {
            k,
            training_data: Vec::new(),
        }
    }

    pub fn fit(&mut self, data: Vec<(Vec<f64>, String)>) {
        self.training_data = data;
    }

    pub fn predict(&self, point: Vec<f64>) -> String {
        let mut distances: Vec<(f64, &String)> = self.training_data
            .iter()
            .map(|(train_point, label)| {
                let distance = euclidean_distance(&point, train_point);
                (distance, label)
            })
            .collect();

        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut label_counts = std::collections::HashMap::new();
        for i in 0..self.k {
            let label = distances[i].1;
            *label_counts.entry(label).or_insert(0) += 1;
        }

        label_counts.into_iter().max_by_key(|&(_, count)| count).unwrap().0.clone()
    }
}

fn euclidean_distance(point1: &Vec<f64>, point2: &Vec<f64>) -> f64 {
    point1.iter().zip(point2.iter())
        .map(|(x1, x2)| (x1 - x2).powi(2))
        .sum::<f64>()
        .sqrt()
}
