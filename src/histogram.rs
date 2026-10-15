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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct HistHandle(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct LeafRange {
    pub offset: u32,
    pub count: u32,
    pub buf_idx: u8,
}

#[derive(Clone, Copy)]
pub(crate) struct SplitLeaf<'a> {
    pub rows: &'a [u32],
    #[allow(dead_code)]
    pub range: LeafRange,
    pub total: Stat,
    pub held: usize,
}

/// Leaf output assignment for resident score updates.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct LeafOutput {
    /// Starting offset in row indices buffer.
    pub offset: u32,
    /// Number of rows in leaf.
    pub count: u32,
    /// Scaled leaf output value (learning_rate * leaf_val).
    pub value: f32,
    /// Buffer index (0 for rows_a, 1 for rows_b).
    pub buf_idx: u32,
}

pub(crate) trait HistogramBackend {
    fn init_objective(
        &mut self,
        labels: &[f64],
        initial_score: f64,
        objective: crate::Objective,
    ) -> Result<Stat>;
    fn update_scores_and_derivatives(
        &mut self,
        leaves: &[LeafOutput],
        cpu_leaf_rows: &[Vec<u32>],
        objective: crate::Objective,
    ) -> Result<Stat>;
    #[allow(dead_code)]
    fn read_scores(&mut self) -> Result<Vec<f64>>;
    #[allow(dead_code)]
    fn begin_iteration(&mut self, _gradients: &[f32], _hessians: &[f32]) -> Result<()> {
        Ok(())
    }
    fn begin_tree(&mut self) -> Result<()> {
        Ok(())
    }
    fn resident_rows(&self) -> bool {
        false
    }
    fn partition(
        &mut self,
        data: &BinnedData,
        parent_rows: &[u32],
        parent_range: LeafRange,
        split: &crate::training::Split,
    ) -> Result<(Vec<u32>, Vec<u32>, LeafRange, LeafRange)>;
    fn construct(
        &mut self,
        data: &BinnedData,
        rows: &[u32],
        gradients: &[f32],
        hessians: &[f32],
        held: usize,
    ) -> Result<HistHandle>;
    fn construct_and_find_split(
        &mut self,
        data: &BinnedData,
        leaf: &SplitLeaf<'_>,
        gradients: &[f32],
        hessians: &[f32],
        config: &crate::TrainingConfig,
    ) -> Result<(Option<crate::training::Split>, HistHandle)> {
        let handle = self.construct(data, leaf.rows, gradients, hessians, leaf.held)?;
        let best = self.find_best_split(data, handle, leaf.total, config)?;
        Ok((best, handle))
    }
    fn find_best_split(
        &mut self,
        data: &BinnedData,
        handle: HistHandle,
        total: Stat,
        config: &crate::TrainingConfig,
    ) -> Result<Option<crate::training::Split>>;
    fn subtract(&mut self, parent: HistHandle, smaller: HistHandle) -> Result<Option<HistHandle>>;
    fn subtract_and_find_split(
        &mut self,
        data: &BinnedData,
        parent: HistHandle,
        smaller: HistHandle,
        total: Stat,
        config: &crate::TrainingConfig,
    ) -> Result<Option<(Option<crate::training::Split>, HistHandle)>> {
        if let Some(sibling) = self.subtract(parent, smaller)? {
            let split = self.find_best_split(data, sibling, total, config)?;
            Ok(Some((split, sibling)))
        } else {
            Ok(None)
        }
    }
    fn free(&mut self, handle: HistHandle);
    fn clear(&mut self);
    #[allow(dead_code)]
    fn read_histogram(&self, handle: HistHandle) -> Result<Vec<Stat>>;
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
    storage: Vec<Option<Vec<Stat>>>,
    pub scores: Vec<f64>,
    pub labels: Vec<f64>,
    pub gradients: Vec<f32>,
    pub hessians: Vec<f32>,
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
    fn init_objective(
        &mut self,
        labels: &[f64],
        initial_score: f64,
        objective: crate::Objective,
    ) -> Result<Stat> {
        self.labels = labels.to_vec();
        self.scores = vec![initial_score; labels.len()];
        let (gradients, hessians) =
            crate::objective::derivatives(&self.scores, &self.labels, objective)?;
        let stat = crate::training::totals_all(&gradients, &hessians);
        self.gradients = gradients;
        self.hessians = hessians;
        Ok(stat)
    }

    fn update_scores_and_derivatives(
        &mut self,
        leaves: &[LeafOutput],
        cpu_leaf_rows: &[Vec<u32>],
        objective: crate::Objective,
    ) -> Result<Stat> {
        for (leaf, rows) in leaves.iter().zip(cpu_leaf_rows) {
            let delta = leaf.value as f64;
            for &row in rows {
                self.scores[row as usize] += delta;
            }
        }
        for &score in &self.scores {
            if !score.is_finite() {
                return Err(Error::Numerical("training scores overflowed".into()));
            }
        }
        let (gradients, hessians) =
            crate::objective::derivatives(&self.scores, &self.labels, objective)?;
        let stat = crate::training::totals_all(&gradients, &hessians);
        self.gradients = gradients;
        self.hessians = hessians;
        Ok(stat)
    }

    fn read_scores(&mut self) -> Result<Vec<f64>> {
        Ok(self.scores.clone())
    }

    fn begin_iteration(&mut self, gradients: &[f32], hessians: &[f32]) -> Result<()> {
        self.gradients = gradients.to_vec();
        self.hessians = hessians.to_vec();
        Ok(())
    }

    fn partition(
        &mut self,
        data: &BinnedData,
        parent_rows: &[u32],
        _parent_range: LeafRange,
        split: &crate::training::Split,
    ) -> Result<(Vec<u32>, Vec<u32>, LeafRange, LeafRange)> {
        let mut left_rows = Vec::with_capacity(split.left.count as usize);
        let mut right_rows = Vec::with_capacity(split.right.count as usize);
        let missing_bin = data.maps[split.feature].bounds.len() - 1;
        for &row in parent_rows {
            let bin = data.bin(row as usize, split.feature);
            let is_missing = data.maps[split.feature].missing && bin == missing_bin;
            let goes_left = if is_missing {
                split.default_left
            } else {
                bin <= split.threshold_bin
            };
            if goes_left {
                left_rows.push(row);
            } else {
                right_rows.push(row);
            }
        }
        if left_rows.len() != split.left.count as usize
            || right_rows.len() != split.right.count as usize
        {
            return Err(Error::Numerical(
                "row partition does not match histogram counts".into(),
            ));
        }
        let left_range = LeafRange {
            offset: 0,
            count: left_rows.len() as u32,
            buf_idx: 0,
        };
        let right_range = LeafRange {
            offset: 0,
            count: right_rows.len() as u32,
            buf_idx: 0,
        };
        Ok((left_rows, right_rows, left_range, right_range))
    }

    fn construct(
        &mut self,
        data: &BinnedData,
        rows: &[u32],
        gradients: &[f32],
        hessians: &[f32],
        held: usize,
    ) -> Result<HistHandle> {
        let started = std::time::Instant::now();
        self.record_workspace(
            held.checked_add(histogram_bytes(data)?)
                .ok_or_else(|| Error::InvalidData("workspace size overflow".into()))?,
        )?;
        let grads = if !gradients.is_empty() {
            gradients
        } else {
            &self.gradients
        };
        let hesses = if !hessians.is_empty() {
            hessians
        } else {
            &self.hessians
        };
        let mut histogram = vec![Stat::default(); data.maps.len() * data.stride];
        for col in 0..data.maps.len() {
            let values = &data.values[col * data.rows..(col + 1) * data.rows];
            for &row in rows {
                let row = row as usize;
                let stat = &mut histogram[col * data.stride + values[row] as usize];
                stat.gradient += grads[row] as f64;
                stat.hessian += hesses[row] as f64;
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
        let slot = if let Some(idx) = self.storage.iter().position(|s| s.is_none()) {
            self.storage[idx] = Some(histogram);
            idx
        } else {
            let idx = self.storage.len();
            self.storage.push(Some(histogram));
            idx
        };
        Ok(HistHandle(slot as u32))
    }

    fn find_best_split(
        &mut self,
        data: &BinnedData,
        handle: HistHandle,
        total: Stat,
        config: &crate::TrainingConfig,
    ) -> Result<Option<crate::training::Split>> {
        let hist = self
            .storage
            .get(handle.0 as usize)
            .and_then(|s| s.as_ref())
            .ok_or_else(|| Error::InvalidData("invalid histogram handle".into()))?;
        crate::training::best_split(data, hist, total, config)
    }

    fn subtract(&mut self, parent: HistHandle, smaller: HistHandle) -> Result<Option<HistHandle>> {
        let parent_hist = self
            .storage
            .get(parent.0 as usize)
            .and_then(|s| s.as_ref())
            .ok_or_else(|| Error::InvalidData("invalid parent histogram handle".into()))?;
        let smaller_hist = self
            .storage
            .get(smaller.0 as usize)
            .and_then(|s| s.as_ref())
            .ok_or_else(|| Error::InvalidData("invalid smaller histogram handle".into()))?;
        if let Some(diff) = crate::training::subtract_histogram(parent_hist, smaller_hist) {
            let slot = if let Some(idx) = self.storage.iter().position(|s| s.is_none()) {
                self.storage[idx] = Some(diff);
                idx
            } else {
                let idx = self.storage.len();
                self.storage.push(Some(diff));
                idx
            };
            Ok(Some(HistHandle(slot as u32)))
        } else {
            Ok(None)
        }
    }

    fn free(&mut self, handle: HistHandle) {
        if let Some(slot) = self.storage.get_mut(handle.0 as usize) {
            *slot = None;
        }
    }

    fn clear(&mut self) {
        self.storage.clear();
    }

    fn read_histogram(&self, handle: HistHandle) -> Result<Vec<Stat>> {
        self.storage
            .get(handle.0 as usize)
            .and_then(|s| s.clone())
            .ok_or_else(|| Error::InvalidData("invalid histogram handle".into()))
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

impl CpuHistograms {
    pub(crate) fn new(budget: usize) -> Self {
        Self {
            budget,
            peak: 0,
            profile: Profile::default(),
            storage: Vec::new(),
            scores: Vec::new(),
            labels: Vec::new(),
            gradients: Vec::new(),
            hessians: Vec::new(),
        }
    }
}

pub(crate) fn create_backend(
    backend: Backend,
    budget: usize,
    _data: &BinnedData,
) -> Result<Box<dyn HistogramBackend>> {
    match backend {
        Backend::Cpu => Ok(Box::new(CpuHistograms::new(budget))),
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
        let mut backend = CpuHistograms::new(4096);
        let handle = backend
            .construct(&data, &[0, 1, 2, 6, 9], &g, &h, 0)
            .unwrap();
        let stats = backend.read_histogram(handle).unwrap();
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
