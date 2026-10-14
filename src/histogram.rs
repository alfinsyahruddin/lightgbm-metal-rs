use crate::{binning::BinnedData, Error};
use crate::{Backend, Result};

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
mod metal;

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Stat {
    pub gradient: f64,
    pub hessian: f64,
    pub count: u32,
}

impl Stat {
    pub fn add(self, other: Self) -> Self {
        Self {
            gradient: self.gradient + other.gradient,
            hessian: self.hessian + other.hessian,
            count: self.count + other.count,
        }
    }
    pub fn subtract(self, other: Self) -> Self {
        Self {
            gradient: self.gradient - other.gradient,
            hessian: self.hessian - other.hessian,
            count: self.count - other.count,
        }
    }
}

pub(crate) trait HistogramBackend {
    fn begin_iteration(&mut self, _gradients: &[f32], _hessians: &[f32]) -> Result<()> {
        Ok(())
    }
    fn construct(
        &mut self,
        data: &BinnedData,
        rows: &[u32],
        gradients: &[f32],
        hessians: &[f32],
        held: usize,
    ) -> Result<Vec<Stat>>;
    fn record_workspace(&mut self, bytes: usize) -> Result<()>;
    fn peak_bytes(&self) -> usize;
    fn auxiliary_bytes(&self) -> usize {
        0
    }
    fn profile(&self) -> Profile {
        Profile::default()
    }
    fn dispatches(&self) -> u64 {
        0
    }
}

pub(crate) struct CpuHistograms {
    pub budget: usize,
    pub peak: usize,
    pub profile: Profile,
}

#[derive(Clone, Copy, Default)]
pub(crate) struct Profile {
    pub wall_seconds: f64,
    pub gpu_seconds: f64,
    pub calls: u64,
    pub small_leaf_calls: u64,
}

pub(crate) fn histogram_bytes(data: &BinnedData) -> Result<usize> {
    data.maps
        .len()
        .checked_mul(data.stride)
        .and_then(|n| n.checked_mul(std::mem::size_of::<Stat>()))
        .ok_or_else(|| Error::InvalidData("histogram dimensions overflow".into()))
}

impl HistogramBackend for CpuHistograms {
    fn construct(
        &mut self,
        data: &BinnedData,
        rows: &[u32],
        gradients: &[f32],
        hessians: &[f32],
        held: usize,
    ) -> Result<Vec<Stat>> {
        let started = std::time::Instant::now();
        self.record_workspace(
            held.checked_add(histogram_bytes(data)?)
                .ok_or_else(|| Error::InvalidData("workspace size overflow".into()))?,
        )?;
        let mut histogram = vec![Stat::default(); data.maps.len() * data.stride];
        for col in 0..data.maps.len() {
            let values = &data.values[col * data.rows..(col + 1) * data.rows];
            for &row in rows {
                let row = row as usize;
                let stat = &mut histogram[col * data.stride + values[row] as usize];
                stat.gradient += gradients[row] as f64;
                stat.hessian += hessians[row] as f64;
                stat.count += 1;
            }
        }
        if histogram
            .iter()
            .any(|s| !s.gradient.is_finite() || !s.hessian.is_finite())
        {
            return Err(Error::Numerical("histogram sums are nonfinite".into()));
        }
        self.profile.wall_seconds += started.elapsed().as_secs_f64();
        self.profile.calls += 1;
        self.profile.small_leaf_calls += u64::from(rows.len() <= 4096);
        Ok(histogram)
    }

    fn record_workspace(&mut self, bytes: usize) -> Result<()> {
        if bytes > self.budget {
            return Err(Error::Workspace {
                required: bytes,
                available: self.budget,
            });
        }
        self.peak = self.peak.max(bytes);
        Ok(())
    }

    fn peak_bytes(&self) -> usize {
        self.peak
    }
    fn profile(&self) -> Profile {
        self.profile
    }
}

pub(crate) fn create_backend(
    backend: Backend,
    budget: usize,
    _data: &BinnedData,
) -> Result<Box<dyn HistogramBackend>> {
    match backend {
        Backend::Cpu => Ok(Box::new(CpuHistograms {
            budget,
            peak: 0,
            profile: Profile::default(),
        })),
        Backend::Metal => {
            #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
            {
                Ok(Box::new(metal::MetalHistograms::new(_data, budget)?))
            }
            #[cfg(not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")))]
            {
                Err(Error::BackendUnavailable(
                    "Metal requires the metal feature on Apple Silicon macOS".into(),
                ))
            }
        }
    }
}

pub(crate) fn check_backend(backend: Backend) -> Result<()> {
    match backend {
        Backend::Cpu => Ok(()),
        Backend::Metal => {
            #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
            {
                metal::check_device()
            }
            #[cfg(not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")))]
            {
                Err(crate::Error::BackendUnavailable(
                    "Metal requires the metal feature on Apple Silicon macOS".into(),
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_is_available_without_framework_initialization() {
        assert!(check_backend(Backend::Cpu).is_ok());
    }

    #[test]
    fn cpu_histograms_preserve_subset_counts_missing_and_empty_bins() {
        let values = [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, f64::NAN];
        let data = BinnedData::new(crate::DenseMatrix::new(&values, 10, 1).unwrap(), 63).unwrap();
        let g = [1000.0, -1000.0, 0.125, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, -3.0];
        let h = [1.0; 10];
        let mut backend = CpuHistograms {
            budget: 4096,
            peak: 0,
            profile: Profile::default(),
        };
        let stats = backend
            .construct(&data, &[0, 1, 2, 6, 9], &g, &h, 0)
            .unwrap();
        assert_eq!(stats.iter().map(|s| s.count).sum::<u32>(), 5);
        assert_eq!(stats[data.bin(0, 0)].gradient, 0.125);
        assert_eq!(stats[data.bin(9, 0)].count, 1);
        assert_eq!(stats[data.bin(9, 0)].hessian, 1.0);
        assert_eq!(stats[data.bin(3, 0)].count, 0);
    }

    #[test]
    #[cfg(not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")))]
    fn metal_request_without_support_returns_error() {
        assert!(matches!(
            check_backend(Backend::Metal),
            Err(crate::Error::BackendUnavailable(_))
        ));
    }
}
