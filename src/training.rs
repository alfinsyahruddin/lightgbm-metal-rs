use crate::{
    binning::BinnedData,
    histogram::{self, HistogramBackend, Stat},
    model::{Node, Tree},
    objective, DenseMatrix, Error, Model, Objective, Result, StageProfile, TrainingConfig,
    TrainingReport,
};
use std::time::Instant;

/// A successfully trained model and its observable execution report.
#[derive(Debug)]
pub struct TrainingResult {
    /// Portable model usable for CPU prediction and native persistence.
    pub model: Model,
    /// Backend, dispatch counts, tracked workspace, and timing information.
    pub report: TrainingReport,
}

/// Train a leaf-wise histogram GBDT model on dense numeric data.
///
/// Labels must be finite, with exactly one per row; binary labels are zero or one.
/// Configuration and input errors are returned before backend work.
///
/// ```
/// use lightgbm_metal_rs::{train, DenseMatrix, TrainingConfig};
/// let values = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
/// let labels = [0.0, 0.0, 0.0, 2.0, 2.0, 2.0];
/// let matrix = DenseMatrix::new(&values, 6, 1)?;
/// let config = TrainingConfig { min_data_in_leaf: 1, iterations: 3, ..Default::default() };
/// let trained = train(matrix, &labels, &config)?;
/// assert_eq!(trained.model.predict(matrix)?.len(), 6);
/// # Ok::<(), lightgbm_metal_rs::Error>(())
/// ```
pub fn train(
    matrix: DenseMatrix<'_>,
    labels: &[f64],
    config: &TrainingConfig,
) -> Result<TrainingResult> {
    let started = Instant::now();
    config.validate()?;
    matrix.validate_labels(labels, config.objective)?;
    let data = BinnedData::new(matrix, config.max_bin)?;
    let initial = objective::initial_score(labels, config.objective)?;
    let prepared = Instant::now();
    let mut backend =
        histogram::create_backend(config.backend, config.histogram_workspace_bytes, &data)?;
    let initialized = Instant::now();
    let mut trees = Vec::new();
    let mut stage_profile = StageProfile::default();
    let single_class =
        config.objective == Objective::BinaryLogloss && labels.iter().all(|&y| y == labels[0]);
    if !single_class {
        let t_deriv = Instant::now();
        let mut root_stat = backend.init_objective(labels, initial, config.objective)?;
        stage_profile.derivatives_seconds += t_deriv.elapsed().as_secs_f64();
        for _ in 0..config.iterations {
            let (tree, leaf_outputs, cpu_leaf_rows) = fit_tree(
                &data,
                root_stat,
                config,
                backend.as_mut(),
                &mut stage_profile,
            )?;
            if tree.nodes.len() == 1 {
                break;
            }
            let t_scores = Instant::now();
            root_stat = backend.update_scores_and_derivatives(
                &leaf_outputs,
                &cpu_leaf_rows,
                config.objective,
            )?;
            stage_profile.score_update_seconds += t_scores.elapsed().as_secs_f64();
            trees.push(tree);
        }
    }
    let fitted = Instant::now();
    let report = TrainingReport {
        backend: config.backend,
        gpu_histogram_dispatches: backend.dispatches(),
        peak_histogram_workspace_bytes: backend.peak_bytes(),
        trees: trees.len(),
        preprocessing_seconds: prepared.duration_since(started).as_secs_f64(),
        backend_setup_seconds: initialized.duration_since(prepared).as_secs_f64(),
        training_seconds: fitted.duration_since(initialized).as_secs_f64(),
        histogram_wall_seconds: backend.profile().wall_seconds,
        histogram_gpu_seconds: backend.profile().gpu_seconds,
        histogram_calls: backend.profile().calls,
        small_leaf_histogram_calls: backend.profile().small_leaf_calls,
        stage_profile: Some(stage_profile),
    };
    Ok(TrainingResult {
        model: Model::new(config.objective, matrix.cols(), initial, trees)?,
        report,
    })
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Split {
    pub(crate) feature: usize,
    pub(crate) threshold_bin: usize,
    pub(crate) threshold: f64,
    pub(crate) default_left: bool,
    pub(crate) gain: f64,
    pub(crate) left: Stat,
    pub(crate) right: Stat,
}

struct Leaf {
    node: usize,
    depth: usize,
    rows: Vec<u32>,
    range: histogram::LeafRange,
    best: Option<Split>,
    histogram: Option<histogram::HistHandle>,
}

pub(crate) fn soft_threshold(gradient: f64, penalty: f64) -> f64 {
    if gradient > penalty {
        gradient - penalty
    } else if gradient < -penalty {
        gradient + penalty
    } else {
        0.0
    }
}

pub(crate) fn output(stat: Stat, config: &TrainingConfig) -> Result<f64> {
    let value = -soft_threshold(stat.gradient, config.lambda_l1)
        / (stat.hessian + config.lambda_l2 + objective::EPSILON);
    if value.is_finite() {
        Ok(value)
    } else {
        Err(Error::Numerical("leaf output is nonfinite".into()))
    }
}

pub(crate) fn gain(stat: Stat, config: &TrainingConfig) -> f64 {
    let gradient = soft_threshold(stat.gradient, config.lambda_l1);
    gradient * gradient / (stat.hessian + config.lambda_l2 + objective::EPSILON)
}

pub(crate) fn totals_all(gradients: &[f32], hessians: &[f32]) -> Stat {
    let mut total = Stat::default();
    for (&g, &h) in gradients.iter().zip(hessians) {
        total.gradient += g as f64;
        total.hessian += h as f64;
        total.count += 1;
    }
    total
}

pub(crate) fn best_split(
    data: &BinnedData,
    histogram: &[Stat],
    total: Stat,
    config: &TrainingConfig,
) -> Result<Option<Split>> {
    let parent_gain = gain(total, config);
    if !parent_gain.is_finite() {
        return Err(Error::Numerical("split gain overflowed".into()));
    }
    let mut best: Option<Split> = None;
    for (feature, mapper) in data.maps.iter().enumerate() {
        if !mapper.splittable {
            continue;
        }
        let stats = &histogram[feature * data.stride..feature * data.stride + mapper.bounds.len()];
        let numeric_bins = stats.len() - usize::from(mapper.missing);
        let missing = if mapper.missing {
            stats[numeric_bins]
        } else {
            Stat::default()
        };
        let thresholds = if mapper.missing {
            numeric_bins
        } else {
            numeric_bins.saturating_sub(1)
        };
        // Match upstream's reverse-first scan; exact ties retain the first candidate.
        for missing_left in [true, false] {
            if !mapper.missing && !missing_left {
                continue;
            }
            let mut cumulative = stats[..thresholds]
                .iter()
                .fold(Stat::default(), |sum, &s| sum.add(s));
            for threshold_bin in (0..thresholds).rev() {
                let left = if missing_left {
                    cumulative.add(missing)
                } else {
                    cumulative
                };
                cumulative = cumulative.subtract(stats[threshold_bin]);
                if left.count > total.count {
                    return Err(Error::Numerical("histogram count exceeds leaf rows".into()));
                }
                let right = total.subtract(left);
                if (left.count as usize) < config.min_data_in_leaf
                    || (right.count as usize) < config.min_data_in_leaf
                    || left.hessian < config.min_sum_hessian_in_leaf
                    || right.hessian < config.min_sum_hessian_in_leaf
                {
                    continue;
                }
                let split_gain = gain(left, config) + gain(right, config) - parent_gain;
                if !split_gain.is_finite() {
                    return Err(Error::Numerical("split gain is nonfinite".into()));
                }
                if split_gain <= config.min_gain_to_split
                    || best.is_some_and(|s| split_gain <= s.gain)
                {
                    continue;
                }
                let threshold = mapper.bounds[threshold_bin].min(1e300);
                best = Some(Split {
                    feature,
                    threshold_bin,
                    threshold,
                    default_left: if mapper.missing {
                        missing_left
                    } else {
                        0.0 <= threshold
                    },
                    gain: split_gain,
                    left,
                    right,
                });
            }
        }
    }
    Ok(best)
}

struct FitContext<'a> {
    data: &'a BinnedData,
    config: &'a TrainingConfig,
    backend: &'a mut dyn HistogramBackend,
    stage_profile: &'a mut StageProfile,
}

impl FitContext<'_> {
    fn candidate(
        &mut self,
        rows: &[u32],
        range: histogram::LeafRange,
        depth: usize,
        held: usize,
        prepared: Option<histogram::HistHandle>,
        known_total: Option<Stat>,
    ) -> Result<(Option<Split>, Option<histogram::HistHandle>)> {
        let count = if rows.is_empty() {
            range.count as usize
        } else {
            rows.len()
        };
        if self.config.max_depth.is_some_and(|limit| depth >= limit)
            || count / 2 < self.config.min_data_in_leaf
            || !self.data.maps.iter().any(|m| m.splittable)
        {
            if let Some(h) = prepared {
                self.backend.free(h);
            }
            return Ok((None, None));
        }
        let total = match known_total {
            Some(t) => t,
            None => {
                return Err(Error::InvalidData("missing leaf total".into()));
            }
        };
        if total.hessian / 2.0 < self.config.min_sum_hessian_in_leaf {
            if let Some(h) = prepared {
                self.backend.free(h);
            }
            return Ok((None, None));
        }
        let (best, handle) = match prepared {
            Some(handle) => {
                let t_split = Instant::now();
                let best = self
                    .backend
                    .find_best_split(self.data, handle, total, self.config)?;
                self.stage_profile.split_search_seconds += t_split.elapsed().as_secs_f64();
                (best, handle)
            }
            None => {
                let t_split = Instant::now();
                let leaf = histogram::SplitLeaf {
                    rows,
                    range,
                    total,
                    held,
                };
                let (best, handle) = self.backend.construct_and_find_split(
                    self.data,
                    &leaf,
                    &[],
                    &[],
                    self.config,
                )?;
                self.stage_profile.split_search_seconds += t_split.elapsed().as_secs_f64();
                (best, handle)
            }
        };
        if best.is_some() {
            Ok((best, Some(handle)))
        } else {
            self.backend.free(handle);
            Ok((None, None))
        }
    }
}

pub(crate) fn subtract_histogram(parent: &[Stat], smaller: &[Stat]) -> Option<Vec<Stat>> {
    for (&parent, &small) in parent.iter().zip(smaller) {
        if small.count > parent.count {
            return None;
        }
        let difference = parent.subtract(small);
        if difference.count == 0 {
            continue;
        }
        let gradient_scale = parent.gradient.abs() + small.gradient.abs();
        let hessian_scale = parent.hessian.abs() + small.hessian.abs();
        if !difference.gradient.is_finite()
            || !difference.hessian.is_finite()
            || difference.hessian <= 1e-4 * hessian_scale
            || (gradient_scale > 1e-12 && difference.gradient.abs() <= 1e-4 * gradient_scale)
        {
            return None;
        }
    }
    Some(
        parent
            .iter()
            .zip(smaller)
            .map(|(&parent, &small)| {
                let difference = parent.subtract(small);
                if difference.count == 0 {
                    Stat::default()
                } else {
                    difference
                }
            })
            .collect(),
    )
}

fn fit_tree(
    data: &BinnedData,
    root: Stat,
    config: &TrainingConfig,
    backend: &mut dyn HistogramBackend,
    stage_profile: &mut StageProfile,
) -> Result<(Tree, Vec<histogram::LeafOutput>, Vec<Vec<u32>>)> {
    backend.begin_tree()?;
    let root_range = histogram::LeafRange {
        offset: 0,
        count: data.rows as u32,
        buf_idx: 0,
    };
    let root_rows = if backend.resident_rows() {
        Vec::new()
    } else {
        (0..data.rows as u32).collect()
    };
    let mut context = FitContext {
        data,
        config,
        backend,
        stage_profile,
    };
    let (root_best, root_histogram) =
        context.candidate(&root_rows, root_range, 0, 0, None, Some(root))?;
    let mut leaves = vec![Leaf {
        node: 0,
        depth: 0,
        rows: root_rows,
        range: root_range,
        best: root_best,
        histogram: root_histogram,
    }];
    let mut tree = Tree {
        nodes: vec![Node::Leaf {
            value: output(root, config)? * config.learning_rate,
        }],
    };
    for _ in 1..config.num_leaves {
        let selected = leaves
            .iter()
            .enumerate()
            .filter_map(|(i, leaf)| leaf.best.map(|split| (i, split)))
            .max_by(|(a, sa), (b, sb)| sa.gain.total_cmp(&sb.gain).then_with(|| b.cmp(a)));
        let Some((leaf_index, split)) = selected else {
            break;
        };
        let parent = &mut leaves[leaf_index];
        let mut parent_histogram = parent.histogram.take();
        let (node, depth) = (parent.node, parent.depth + 1);
        let t_part = Instant::now();
        let (left_rows, right_rows, left_range, right_range) =
            context
                .backend
                .partition(context.data, &parent.rows, parent.range, &split)?;
        context.stage_profile.partitioning_seconds += t_part.elapsed().as_secs_f64();
        let left_node = tree.nodes.len();
        let right_node = left_node + 1;
        tree.nodes[node] = Node::Split {
            feature: split.feature,
            threshold: split.threshold,
            default_left: split.default_left,
            left: left_node,
            right: right_node,
        };
        tree.nodes.extend([
            Node::Leaf {
                value: output(split.left, config)? * config.learning_rate,
            },
            Node::Leaf {
                value: output(split.right, config)? * config.learning_rate,
            },
        ]);
        let bytes = histogram::histogram_bytes(data)?;
        let mut held = leaves
            .iter()
            .filter(|leaf| leaf.histogram.is_some())
            .count()
            .saturating_mul(bytes);
        let auxiliary = context.backend.auxiliary_bytes();
        if held
            .saturating_add(bytes.saturating_mul(3))
            .saturating_add(auxiliary)
            > config.histogram_workspace_bytes
        {
            for leaf in &mut leaves {
                if let Some(h) = leaf.histogram.take() {
                    context.backend.free(h);
                }
            }
            held = 0;
        }
        if bytes.saturating_mul(3).saturating_add(auxiliary) > config.histogram_workspace_bytes {
            if let Some(h) = parent_histogram.take() {
                context.backend.free(h);
            }
        }
        let small_left = left_range.count <= right_range.count;
        let (small_rows, large_rows) = if small_left {
            (&left_rows, &right_rows)
        } else {
            (&right_rows, &left_rows)
        };
        let (small_range, large_range) = if small_left {
            (left_range, right_range)
        } else {
            (right_range, left_range)
        };
        let (small_total, large_total) = if small_left {
            (split.left, split.right)
        } else {
            (split.right, split.left)
        };
        let final_split = leaves.len() + 1 == config.num_leaves;
        let (small_best, mut small_histogram) = if final_split {
            if let Some(h) = parent_histogram.take() {
                context.backend.free(h);
            }
            (None, None)
        } else {
            context.candidate(
                small_rows,
                small_range,
                depth,
                held + usize::from(parent_histogram.is_some()) * bytes,
                None,
                Some(small_total),
            )?
        };
        let (large_best, large_histogram) = if final_split {
            (None, None)
        } else if let (Some(parent), Some(smaller)) = (parent_histogram.take(), small_histogram) {
            context.backend.record_workspace(held + 3 * bytes)?;
            let t_sub = Instant::now();
            let outcome = context.backend.subtract_and_find_split(
                context.data,
                parent,
                smaller,
                large_total,
                context.config,
            )?;
            context.backend.free(parent);
            context.stage_profile.split_search_seconds += t_sub.elapsed().as_secs_f64();
            if let Some((best, handle)) = outcome {
                if best.is_some() {
                    (best, Some(handle))
                } else {
                    context.backend.free(handle);
                    (None, None)
                }
            } else {
                if held
                    .saturating_add(bytes.saturating_mul(2))
                    .saturating_add(auxiliary)
                    > config.histogram_workspace_bytes
                {
                    if let Some(h) = small_histogram.take() {
                        context.backend.free(h);
                    }
                }
                context.candidate(
                    large_rows,
                    large_range,
                    depth,
                    held + usize::from(small_histogram.is_some()) * bytes,
                    None,
                    Some(large_total),
                )?
            }
        } else {
            if let Some(h) = parent_histogram.take() {
                context.backend.free(h);
            }
            if held
                .saturating_add(bytes.saturating_mul(2))
                .saturating_add(auxiliary)
                > config.histogram_workspace_bytes
            {
                if let Some(h) = small_histogram.take() {
                    context.backend.free(h);
                }
            }
            context.candidate(
                large_rows,
                large_range,
                depth,
                held + usize::from(small_histogram.is_some()) * bytes,
                None,
                Some(large_total),
            )?
        };
        if held
            .saturating_add(bytes.saturating_mul(2))
            .saturating_add(auxiliary)
            > config.histogram_workspace_bytes
        {
            if let Some(h) = small_histogram.take() {
                context.backend.free(h);
            }
        }
        let ((left_best, left_histogram), (right_best, right_histogram)) = if small_left {
            ((small_best, small_histogram), (large_best, large_histogram))
        } else {
            ((large_best, large_histogram), (small_best, small_histogram))
        };
        leaves[leaf_index] = Leaf {
            node: left_node,
            depth,
            rows: left_rows,
            range: left_range,
            best: left_best,
            histogram: left_histogram,
        };
        leaves.push(Leaf {
            node: right_node,
            depth,
            rows: right_rows,
            range: right_range,
            best: right_best,
            histogram: right_histogram,
        });
    }
    let mut leaf_outputs = Vec::with_capacity(leaves.len());
    let mut cpu_leaf_rows = Vec::with_capacity(leaves.len());
    for leaf in &mut leaves {
        let value = match tree.nodes[leaf.node] {
            Node::Leaf { value } => value as f32,
            _ => unreachable!(),
        };
        leaf_outputs.push(histogram::LeafOutput {
            offset: leaf.range.offset,
            count: leaf.range.count,
            value,
            buf_idx: leaf.range.buf_idx as u32,
        });
        cpu_leaf_rows.push(std::mem::take(&mut leaf.rows));
        if let Some(h) = leaf.histogram.take() {
            backend.free(h);
        }
    }
    backend.clear();
    Ok((tree, leaf_outputs, cpu_leaf_rows))
}
