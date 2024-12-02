#[cfg(test)]
mod tests {
    use knn_lib::KNN;

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
