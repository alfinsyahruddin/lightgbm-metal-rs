use lightgbm_metal_rs::{backend_available, Backend};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let backend = if std::env::args().any(|a| a == "--metal") {
        Backend::Metal
    } else {
        Backend::Cpu
    };
    backend_available(backend)?;
    println!("{backend:?} backend available");
    Ok(())
}
