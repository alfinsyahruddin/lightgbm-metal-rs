use lightgbm_metal_rs::{train, Backend, DenseMatrix, Model, TrainingConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x: Vec<_> = (0..96).map(|v| v as f64 / 16.0).collect();
    let labels: Vec<_> = x.iter().map(|x| if *x < 3.0 { 1.0 } else { 5.0 }).collect();
    let matrix = DenseMatrix::new(&x, x.len(), 1)?;
    let backend = if std::env::args().any(|a| a == "--metal") {
        Backend::Metal
    } else {
        Backend::Cpu
    };
    let config = TrainingConfig {
        backend,
        iterations: 10,
        num_leaves: 4,
        min_data_in_leaf: 3,
        max_bin: 63,
        ..Default::default()
    };
    let trained = train(matrix, &labels, &config)?;
    let predictions = trained.model.predict(matrix)?;
    trained.model.save("target/demo-model.json")?;
    let loaded = Model::load("target/demo-model.json")?;
    assert_eq!(predictions, loaded.predict(matrix)?);
    println!("{}", serde_json::to_string_pretty(&trained.report)?);
    println!(
        "First prediction: {}; last: {}",
        predictions[0],
        predictions[predictions.len() - 1]
    );
    println!("Native model saved and verified: target/demo-model.json");
    Ok(())
}
