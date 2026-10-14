use lightgbm_metal_rs::{DenseMatrix, Model};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let fixture: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
        "validation/fixtures/quality_regression_l2.json",
    )?)?;
    let cols = fixture["cols"].as_u64().ok_or("missing cols")? as usize;
    let train_rows = fixture["train_rows"].as_u64().ok_or("missing train_rows")? as usize;
    let x: Vec<_> = fixture["features"]
        .as_array()
        .ok_or("missing features")?
        .iter()
        .map(|v| v.as_f64().unwrap_or(f64::NAN))
        .collect();
    let values = &x[train_rows * cols..];
    let model = Model::load("target/metal-model.json")?;
    let expected: Vec<f64> =
        serde_json::from_str(&std::fs::read_to_string("target/metal-predictions.json")?)?;
    let actual = model.predict_raw(DenseMatrix::new(values, values.len() / cols, cols)?)?;
    if actual != expected {
        return Err("cross-build predictions changed".into());
    }
    println!(
        "CPU-only inference exactly matches {} Metal-trained predictions",
        actual.len()
    );
    Ok(())
}
