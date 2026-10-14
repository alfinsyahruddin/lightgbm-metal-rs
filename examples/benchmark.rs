use lightgbm_metal_rs::{train, Backend, DenseMatrix, Objective, TrainingConfig};
use serde::{Deserialize, Serialize};
use std::{path::Path, time::Instant};

#[derive(Deserialize)]
struct Case {
    name: String,
    rows: usize,
    cols: usize,
    train_rows: usize,
    features: String,
    labels: String,
    objective: Objective,
    max_bin: usize,
}

#[derive(Serialize)]
struct Run {
    phase: &'static str,
    preprocessing_seconds: f64,
    backend_setup_seconds: f64,
    training_seconds: f64,
    end_to_end_seconds: f64,
    quality: f64,
    gpu_dispatches: u64,
    peak_histogram_bytes: usize,
    histogram_wall_seconds: f64,
    histogram_gpu_seconds: f64,
    histogram_calls: u64,
    small_leaf_histogram_calls: u64,
}

fn read_doubles(path: impl AsRef<Path>) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() % 8 != 0 {
        return Err("invalid float64 input length".into());
    }
    let mut values = Vec::with_capacity(bytes.len() / 8);
    for chunk in bytes.chunks_exact(8) {
        let mut value = [0_u8; 8];
        value.copy_from_slice(chunk);
        values.push(f64::from_le_bytes(value));
    }
    Ok(values)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let manifest = args
        .next()
        .ok_or("usage: benchmark CASE_JSON cpu|metal [measured_runs]")?;
    let backend = match args.next().as_deref() {
        Some("cpu") => Backend::Cpu,
        Some("metal") => Backend::Metal,
        _ => return Err("backend must be cpu or metal".into()),
    };
    let measured: usize = args.next().unwrap_or_else(|| "5".into()).parse()?;
    if measured == 0 {
        return Err("need at least one measured warm run".into());
    }
    let case: Case = serde_json::from_str(&std::fs::read_to_string(manifest)?)?;
    let features = read_doubles(&case.features)?;
    let labels = read_doubles(&case.labels)?;
    if labels.len() != case.rows
        || features.len()
            != case
                .rows
                .checked_mul(case.cols)
                .ok_or("dimension overflow")?
        || case.train_rows == 0
        || case.train_rows >= case.rows
    {
        return Err("invalid benchmark dimensions".into());
    }
    let config = TrainingConfig {
        backend,
        objective: case.objective,
        max_bin: case.max_bin,
        ..Default::default()
    };
    let validation = DenseMatrix::new(
        &features[case.train_rows * case.cols..],
        case.rows - case.train_rows,
        case.cols,
    )?;
    let mut runs = Vec::new();
    for run in 0..measured + 2 {
        let started = Instant::now();
        let matrix = DenseMatrix::new(
            &features[..case.train_rows * case.cols],
            case.train_rows,
            case.cols,
        )?;
        let matrix_validation = started.elapsed().as_secs_f64();
        let trained = train(matrix, &labels[..case.train_rows], &config)?;
        let elapsed = started.elapsed().as_secs_f64();
        let raw = trained.model.predict_raw(validation)?;
        let actual_labels = &labels[case.train_rows..];
        let quality = match case.objective {
            Objective::RegressionL2 => (raw
                .iter()
                .zip(actual_labels)
                .map(|(p, y)| (p - y).powi(2))
                .sum::<f64>()
                / raw.len() as f64)
                .sqrt(),
            Objective::BinaryLogloss => {
                raw.iter()
                    .zip(actual_labels)
                    .map(|(p, y)| p.max(0.0) - p * y + (-p.abs()).exp().ln_1p())
                    .sum::<f64>()
                    / raw.len() as f64
            }
        };
        let phase = if run == 0 {
            "cold_process"
        } else if run == 1 {
            "warmup"
        } else {
            "measured_warm"
        };
        eprintln!(
            "{} {backend:?} {phase}: {elapsed:.3}s, quality {quality:.6}",
            case.name
        );
        runs.push(Run {
            phase,
            preprocessing_seconds: trained.report.preprocessing_seconds + matrix_validation,
            backend_setup_seconds: trained.report.backend_setup_seconds,
            training_seconds: trained.report.training_seconds,
            end_to_end_seconds: elapsed,
            quality,
            gpu_dispatches: trained.report.gpu_histogram_dispatches,
            peak_histogram_bytes: trained.report.peak_histogram_workspace_bytes,
            histogram_wall_seconds: trained.report.histogram_wall_seconds,
            histogram_gpu_seconds: trained.report.histogram_gpu_seconds,
            histogram_calls: trained.report.histogram_calls,
            small_leaf_histogram_calls: trained.report.small_leaf_histogram_calls,
        });
    }
    println!(
        "{}",
        serde_json::to_string(&serde_json::json!({"case":case.name,"backend":backend,
        "rows":case.rows,"cols":case.cols,"train_rows":case.train_rows,"max_bin":case.max_bin,
        "cpu_threads":1,"raw_input_bytes":(features.len()+labels.len())*8,
        "binned_dataset_bytes":case.train_rows*case.cols,"histogram_budget_bytes":config.histogram_workspace_bytes,
        "config":config,"runs":runs}))?
    );
    Ok(())
}
