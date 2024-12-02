## 谢谢打赏
![图片](./qc.png)
开发一个简单的K-Nearest Neighbors (KNN) 库可以帮助你理解这个算法的基本原理。以下是一个用Rust编写的KNN库的示例。

首先，创建一个新的Rust项目：

```sh
cargo new knn_lib
cd knn_lib
```

在`src/lib.rs`中，编写以下代码：

```rust
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
```

这个库定义了一个`KNN`结构体，并实现了`fit`和`predict`方法。`fit`方法用于存储训练数据，`predict`方法用于预测新数据点的标签。

接下来，编写一个测试来验证这个库。在`tests`目录下创建一个新的测试文件`knn_tests.rs`：

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knn() {
        let mut knn = KNN::new(3);
        let training_data = vec![
            (vec![1.0, 2.0], "A".to_string()),
            (vec![2.0, 3.0], "A".to_string()),
            (vec![3.0, 3.0], "B".to_string()),
            (vec![6.0, 6.0], "B".to_string()),
        ];
        knn.fit(training_data);

        let test_point = vec![2.0, 2.0];
        let predicted_label = knn.predict(test_point);

        assert_eq!(predicted_label, "A");
    }
}
```

这个测试创建了一个KNN实例，使用一些训练数据进行训练，并预测一个新数据点的标签。

最后，运行测试来验证库的功能：

```sh
cargo test
```

这个简单的KNN库展示了如何用Rust实现基本的KNN算法。你可以根据需要扩展和优化这个库。
