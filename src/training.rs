use crate::{
    binning::BinnedData,
    histogram::{self, HistogramBackend, Stat},
    model::{Node, Tree},
    objective, DenseMatrix, Error, Model, Objective, Result, TrainingConfig, TrainingReport,
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
    let mut scores = vec![initial; labels.len()];
    let mut trees = Vec::new();
    let single_class =
        config.objective == Objective::BinaryLogloss && labels.iter().all(|&y| y == labels[0]);
    if !single_class {
        for _ in 0..config.iterations {
            let (gradients, hessians) = objective::derivatives(&scores, labels, config.objective)?;
            backend.begin_iteration(&gradients, &hessians)?;
            let tree = fit_tree(&data, &gradients, &hessians, config, backend.as_mut())?;
            if tree.nodes.len() == 1 {
                break;
            }
            for (row, score) in scores.iter_mut().enumerate() {
                *score += tree.predict(matrix, row);
                if !score.is_finite() {
                    return Err(Error::Numerical("training scores overflowed".into()));
                }
            }
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
    };
    Ok(TrainingResult {
        model: Model::new(config.objective, matrix.cols(), initial, trees)?,
        report,
    })
}

#[derive(Debug, Clone, Copy)]
struct Split {
    feature: usize,
    threshold_bin: usize,
    threshold: f64,
    default_left: bool,
    gain: f64,
    left: Stat,
    right: Stat,
}

struct Leaf {
    node: usize,
    depth: usize,
    rows: Vec<u32>,
    best: Option<Split>,
    histogram: Option<Vec<Stat>>,
}

fn soft_threshold(gradient: f64, penalty: f64) -> f64 {
    if gradient > penalty {
        gradient - penalty
    } else if gradient < -penalty {
        gradient + penalty
    } else {
        0.0
    }
}

fn output(stat: Stat, config: &TrainingConfig) -> Result<f64> {
    let value = -soft_threshold(stat.gradient, config.lambda_l1)
        / (stat.hessian + config.lambda_l2 + objective::EPSILON);
    if value.is_finite() {
        Ok(value)
    } else {
        Err(Error::Numerical("leaf output is nonfinite".into()))
    }
}

fn gain(stat: Stat, config: &TrainingConfig) -> f64 {
    let gradient = soft_threshold(stat.gradient, config.lambda_l1);
    gradient * gradient / (stat.hessian + config.lambda_l2 + objective::EPSILON)
}

fn totals(rows: &[u32], gradients: &[f32], hessians: &[f32]) -> Stat {
    let mut total = Stat::default();
    for &row in rows {
        total.gradient += gradients[row as usize] as f64;
        total.hessian += hessians[row as usize] as f64;
        total.count += 1;
    }
    total
}

fn best_split(
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
    gradients: &'a [f32],
    hessians: &'a [f32],
    config: &'a TrainingConfig,
    backend: &'a mut dyn HistogramBackend,
}

impl FitContext<'_> {
    fn candidate(
        &mut self,
        rows: &[u32],
        depth: usize,
        held: usize,
        prepared: Option<Vec<Stat>>,
    ) -> Result<(Option<Split>, Option<Vec<Stat>>)> {
        if self.config.max_depth.is_some_and(|limit| depth >= limit)
            || rows.len() / 2 < self.config.min_data_in_leaf
            || !self.data.maps.iter().any(|m| m.splittable)
        {
            return Ok((None, None));
        }
        let total = totals(rows, self.gradients, self.hessians);
        if total.hessian / 2.0 < self.config.min_sum_hessian_in_leaf {
            return Ok((None, None));
        }
        let histogram = match prepared {
            Some(histogram) => histogram,
            None => self
                .backend
                .construct(self.data, rows, self.gradients, self.hessians, held)?,
        };
        let best = best_split(self.data, &histogram, total, self.config)?;
        Ok((
            best,
            if best.is_some() {
                Some(histogram)
            } else {
                None
            },
        ))
    }
}

fn subtract_histogram(parent: &[Stat], smaller: &[Stat]) -> Option<Vec<Stat>> {
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
    gradients: &[f32],
    hessians: &[f32],
    config: &TrainingConfig,
    backend: &mut dyn HistogramBackend,
) -> Result<Tree> {
    let rows: Vec<_> = (0..data.rows as u32).collect();
    let root = totals(&rows, gradients, hessians);
    let mut context = FitContext {
        data,
        gradients,
        hessians,
        config,
        backend,
    };
    let (root_best, root_histogram) = context.candidate(&rows, 0, 0, None)?;
    let mut leaves = vec![Leaf {
        node: 0,
        depth: 0,
        rows,
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
        let mut left_rows = Vec::with_capacity(split.left.count as usize);
        let mut right_rows = Vec::with_capacity(split.right.count as usize);
        let missing_bin = data.maps[split.feature].bounds.len() - 1;
        for row in std::mem::take(&mut parent.rows) {
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
                leaf.histogram = None;
            }
            held = 0;
        }
        if bytes.saturating_mul(3).saturating_add(auxiliary) > config.histogram_workspace_bytes {
            parent_histogram = None;
        }
        let small_left = left_rows.len() <= right_rows.len();
        let (small_rows, large_rows) = if small_left {
            (&left_rows, &right_rows)
        } else {
            (&right_rows, &left_rows)
        };
        let final_split = leaves.len() + 1 == config.num_leaves;
        let (small_best, mut small_histogram) = if final_split {
            (None, None)
        } else {
            context.candidate(
                small_rows,
                depth,
                held + usize::from(parent_histogram.is_some()) * bytes,
                None,
            )?
        };
        let sibling = if let (Some(parent), Some(smaller)) = (&parent_histogram, &small_histogram) {
            context.backend.record_workspace(held + 3 * bytes)?;
            subtract_histogram(parent, smaller)
        } else {
            None
        };
        drop(parent_histogram);
        if held
            .saturating_add(bytes.saturating_mul(2))
            .saturating_add(auxiliary)
            > config.histogram_workspace_bytes
        {
            small_histogram = None;
        }
        let (large_best, large_histogram) = if final_split {
            (None, None)
        } else {
            context.candidate(
                large_rows,
                depth,
                held + usize::from(small_histogram.is_some()) * bytes,
                sibling,
            )?
        };
        let ((left_best, left_histogram), (right_best, right_histogram)) = if small_left {
            ((small_best, small_histogram), (large_best, large_histogram))
        } else {
            ((large_best, large_histogram), (small_best, small_histogram))
        };
        leaves[leaf_index] = Leaf {
            node: left_node,
            depth,
            rows: left_rows,
            best: left_best,
            histogram: left_histogram,
        };
        leaves.push(Leaf {
            node: right_node,
            depth,
            rows: right_rows,
            best: right_best,
            histogram: right_histogram,
        });
    }
    Ok(tree)
}
