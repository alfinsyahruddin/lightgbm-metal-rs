use super::{histogram_bytes, HistHandle, HistogramBackend, Profile, Stat};
use crate::{binning::BinnedData, Error, Result};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::*;
use std::{ffi::c_void, mem, ptr::NonNull};

type Device = Retained<ProtocolObject<dyn MTLDevice>>;
type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {}
const SOURCE: &str = include_str!("../../kernels/histogram.metal");
#[repr(C)]
#[derive(Clone, Copy)]
struct GpuStat {
    gradient: f32,
    hessian: f32,
    count: u32,
}
#[repr(C)]
struct Params {
    rows: u32,
    stride: u32,
    features: u32,
    leaf_rows: u32,
    chunks: u32,
    row_chunk: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct FeatureMeta {
    bounds_len: u32,
    missing: u32,
    splittable: u32,
    _pad: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct SplitParams {
    stride: u32,
    num_features: u32,
    min_data_in_leaf: u32,
    total_count: u32,
    total_gradient: f32,
    total_hessian: f32,
    parent_gain: f32,
    lambda_l1: f32,
    lambda_l2: f32,
    min_sum_hessian_in_leaf: f32,
    min_gain_to_split: f32,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct SplitCandidate {
    gain: f32,
    left_gradient: f32,
    left_hessian: f32,
    left_count: u32,
    threshold_bin: u32,
    missing_left: u32,
    valid: u32,
    feature: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct PartitionParams {
    num_rows: u32,
    leaf_rows: u32,
    feature: u32,
    threshold_bin: u32,
    missing_bin: u32,
    has_missing: u32,
    default_left: u32,
    total_left_count: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct ObjectiveParams {
    num_rows: u32,
    objective: u32,
}

fn device() -> Result<Device> {
    // SAFETY: The framework returns an owned default-device reference or null.
    let device = unsafe { Retained::from_raw(MTLCreateSystemDefaultDevice()) }
        .ok_or_else(|| Error::BackendUnavailable("no usable Metal device".into()))?;
    if !device.hasUnifiedMemory() {
        return Err(Error::BackendUnavailable(
            "Metal backend requires unified memory on Apple Silicon".into(),
        ));
    }
    Ok(device)
}
pub(super) fn check_device() -> Result<()> {
    device().map(|_| ())
}
fn compile(device: &ProtocolObject<dyn MTLDevice>, source: &str) -> Result<CachedPipelines> {
    let options = MTLCompileOptions::new();
    options.setFastMathEnabled(false);
    options.setLanguageVersion(MTLLanguageVersion::MTLLanguageVersion2_2);
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&options))
        .map_err(|e| Error::BackendExecution(format!("Metal compilation: {e}")))?;
    let pipeline = |name| -> Result<Pipeline> {
        let function = library
            .newFunctionWithName(&NSString::from_str(name))
            .ok_or_else(|| Error::BackendExecution(format!("missing Metal function {name}")))?;
        device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| Error::BackendExecution(format!("Metal pipeline: {e}")))
    };
    let partial = pipeline("partial_histograms")?;
    if partial.threadExecutionWidth() != 32 || partial.maxTotalThreadsPerThreadgroup() < 128 {
        return Err(Error::BackendUnavailable(
            "histogram kernel requires a 32-lane SIMD group and at least 128 threads per threadgroup".into(),
        ));
    }
    let reduce = pipeline("reduce_histograms")?;
    let subtract = pipeline("subtract_histograms")?;
    let find_splits = pipeline("find_best_splits")?;
    let init_rows = pipeline("init_row_indices")?;
    let partition_single = pipeline("partition_single_block")?;
    let partition_count = pipeline("partition_count_blocks")?;
    let partition_prefix = pipeline("partition_prefix_blocks")?;
    let partition_scatter = pipeline("partition_scatter_blocks")?;
    let init_scores = pipeline("init_scores_and_derivatives")?;
    let update_leaf_scores = pipeline("update_leaf_scores")?;
    let compute_derivatives = pipeline("compute_derivatives")?;
    Ok(CachedPipelines {
        partial,
        reduce,
        subtract,
        find_splits,
        init_rows,
        partition_single,
        partition_count,
        partition_prefix,
        partition_scatter,
        init_scores,
        update_leaf_scores,
        compute_derivatives,
    })
}

#[derive(Clone)]
struct CachedPipelines {
    partial: Pipeline,
    reduce: Pipeline,
    subtract: Pipeline,
    find_splits: Pipeline,
    init_rows: Pipeline,
    partition_single: Pipeline,
    partition_count: Pipeline,
    partition_prefix: Pipeline,
    partition_scatter: Pipeline,
    init_scores: Pipeline,
    update_leaf_scores: Pipeline,
    compute_derivatives: Pipeline,
}
unsafe impl Send for CachedPipelines {}
unsafe impl Sync for CachedPipelines {}

static PIPELINE_CACHE: std::sync::OnceLock<CachedPipelines> = std::sync::OnceLock::new();

fn get_cached_pipelines(device: &ProtocolObject<dyn MTLDevice>) -> Result<CachedPipelines> {
    if let Some(cached) = PIPELINE_CACHE.get() {
        return Ok(cached.clone());
    }
    let compiled = compile(device, SOURCE)?;
    let _ = PIPELINE_CACHE.set(compiled.clone());
    Ok(compiled)
}
fn allocated<T>(value: Option<T>) -> Result<T> {
    value.ok_or_else(|| Error::BackendExecution("Metal allocation failed".into()))
}
fn new_buffer(device: &ProtocolObject<dyn MTLDevice>, bytes: usize) -> Result<Buffer> {
    if bytes == 0 || bytes > device.maxBufferLength() {
        return Err(Error::InvalidData(format!(
            "unsupported Metal buffer length: {bytes}"
        )));
    }
    allocated(
        device.newBufferWithLength_options(bytes, MTLResourceOptions::MTLResourceStorageModeShared),
    )
}
fn write<T: Copy>(buffer: &ProtocolObject<dyn MTLBuffer>, values: &[T]) -> Result<()> {
    let bytes = mem::size_of_val(values);
    if bytes > buffer.length() {
        return Err(Error::InvalidData(
            "Metal buffer write exceeds allocation".into(),
        ));
    }
    // SAFETY: Shared storage has at least `bytes` bytes, the source is initialized,
    // and callers only write before submission or after the preceding command completed.
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr().cast::<u8>(),
            buffer.contents().as_ptr().cast::<u8>(),
            bytes,
        );
    }
    Ok(())
}
fn complete(status: MTLCommandBufferStatus, error: Option<String>) -> Result<()> {
    if status != MTLCommandBufferStatus::Completed || error.is_some() {
        return Err(Error::BackendExecution(format!(
            "Metal command did not complete: {status:?}, {error:?}"
        )));
    }
    Ok(())
}
fn required_buffer(slot: &Option<Buffer>) -> Result<&ProtocolObject<dyn MTLBuffer>> {
    slot.as_deref()
        .ok_or_else(|| Error::BackendExecution("histogram buffer was not allocated".into()))
}
pub(super) struct MetalHistograms {
    device: Device,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    partial_pipeline: Pipeline,
    reduce_pipeline: Pipeline,
    subtract_pipeline: Pipeline,
    find_splits_pipeline: Pipeline,
    init_rows_pipeline: Pipeline,
    partition_single_pipeline: Pipeline,
    partition_count_pipeline: Pipeline,
    partition_prefix_pipeline: Pipeline,
    partition_scatter_pipeline: Pipeline,
    init_scores_pipeline: Pipeline,
    update_leaf_scores_pipeline: Pipeline,
    compute_derivatives_pipeline: Pipeline,
    bins: Buffer,
    gradients: Buffer,
    hessians: Buffer,
    scores: Buffer,
    score_errs: Buffer,
    labels: Buffer,
    obj_block_totals: Buffer,
    obj_error_flag: Buffer,
    rows: Buffer,
    rows_a: Buffer,
    rows_b: Buffer,
    rows_count: usize,
    block_counts: Buffer,
    block_offsets: Buffer,
    splittable_features: Buffer,
    splittable_count: usize,
    feature_meta: Buffer,
    feature_count: usize,
    candidates: Buffer,
    cancel_flag: Buffer,
    partials: Option<Buffer>,
    hist_buffers: Vec<Option<Buffer>>,
    free_slots: Vec<usize>,
    width: usize,
    gpu_bytes: usize,
    budget: usize,
    peak: usize,
    dispatches: u64,
    profile: Profile,
    last_partition_command: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    rows_a_initialized: bool,
}

impl MetalHistograms {
    pub fn new(data: &BinnedData, budget: usize) -> Result<Self> {
        let device = device()?;
        let pipelines = get_cached_pipelines(&device)?;
        let queue = allocated(device.newCommandQueue())?;
        let bins = new_buffer(&device, data.values.len())?;
        write(&bins, &data.values)?;
        let row_bytes = data
            .rows
            .checked_mul(mem::size_of::<u32>())
            .ok_or_else(|| Error::InvalidData("row buffer size overflow".into()))?;
        let gradients = new_buffer(&device, row_bytes)?;
        let hessians = new_buffer(&device, row_bytes)?;
        let rows = new_buffer(&device, row_bytes)?;
        let rows_a = new_buffer(&device, row_bytes)?;
        let rows_b = new_buffer(&device, row_bytes)?;
        let rows_count = data.rows;
        let max_blocks = data.rows.div_ceil(256).max(1);
        let block_bytes = (max_blocks + 1)
            .checked_mul(mem::size_of::<u32>())
            .ok_or_else(|| Error::InvalidData("partition block buffer overflow".into()))?;
        let block_counts = new_buffer(&device, block_bytes)?;
        let block_offsets = new_buffer(&device, block_bytes)?;
        let splittable: Vec<u32> = data
            .maps
            .iter()
            .enumerate()
            .filter_map(|(i, m)| if m.splittable { Some(i as u32) } else { None })
            .collect();
        let splittable_bytes = splittable
            .len()
            .max(1)
            .checked_mul(mem::size_of::<u32>())
            .ok_or_else(|| Error::InvalidData("splittable features size overflow".into()))?;
        let splittable_features = new_buffer(&device, splittable_bytes)?;
        if !splittable.is_empty() {
            write(&splittable_features, &splittable)?;
        }
        let splittable_count = splittable.len();

        let metas: Vec<FeatureMeta> = data
            .maps
            .iter()
            .map(|m| FeatureMeta {
                bounds_len: m.bounds.len() as u32,
                missing: u32::from(m.missing),
                splittable: u32::from(m.splittable),
                _pad: 0,
            })
            .collect();
        let meta_bytes = metas
            .len()
            .max(1)
            .checked_mul(mem::size_of::<FeatureMeta>())
            .ok_or_else(|| Error::InvalidData("feature meta size overflow".into()))?;
        let feature_meta = new_buffer(&device, meta_bytes)?;
        if !metas.is_empty() {
            write(&feature_meta, &metas)?;
        }
        let feature_count = metas.len();

        let candidate_bytes = feature_count
            .max(1)
            .checked_mul(mem::size_of::<SplitCandidate>())
            .ok_or_else(|| Error::InvalidData("candidates size overflow".into()))?;
        let candidates = new_buffer(&device, candidate_bytes)?;

        let cancel_flag = new_buffer(&device, mem::size_of::<u32>())?;

        let scores = new_buffer(&device, row_bytes)?;
        let score_errs = new_buffer(&device, row_bytes)?;
        let labels = new_buffer(&device, row_bytes)?;
        let num_obj_groups = data.rows.div_ceil(256).max(1);
        let obj_block_totals = new_buffer(
            &device,
            num_obj_groups
                .checked_mul(2 * mem::size_of::<f32>())
                .ok_or_else(|| Error::InvalidData("obj block totals size overflow".into()))?,
        )?;
        let obj_error_flag = new_buffer(&device, mem::size_of::<u32>())?;

        let width = data
            .maps
            .len()
            .checked_mul(data.stride)
            .filter(|&n| n <= u32::MAX as usize)
            .ok_or_else(|| Error::InvalidData("Metal histogram width exceeds u32".into()))?;
        let gpu_bytes = width
            .checked_mul(mem::size_of::<GpuStat>())
            .ok_or_else(|| Error::InvalidData("Metal histogram size overflow".into()))?;

        Ok(Self {
            device,
            queue,
            partial_pipeline: pipelines.partial,
            reduce_pipeline: pipelines.reduce,
            subtract_pipeline: pipelines.subtract,
            find_splits_pipeline: pipelines.find_splits,
            init_rows_pipeline: pipelines.init_rows,
            partition_single_pipeline: pipelines.partition_single,
            partition_count_pipeline: pipelines.partition_count,
            partition_prefix_pipeline: pipelines.partition_prefix,
            partition_scatter_pipeline: pipelines.partition_scatter,
            init_scores_pipeline: pipelines.init_scores,
            update_leaf_scores_pipeline: pipelines.update_leaf_scores,
            compute_derivatives_pipeline: pipelines.compute_derivatives,
            bins,
            gradients,
            hessians,
            scores,
            score_errs,
            labels,
            obj_block_totals,
            obj_error_flag,
            rows,
            rows_a,
            rows_b,
            rows_count,
            block_counts,
            block_offsets,
            splittable_features,
            splittable_count,
            feature_meta,
            feature_count,
            candidates,
            cancel_flag,
            partials: None,
            hist_buffers: Vec::new(),
            free_slots: Vec::new(),
            width,
            gpu_bytes,
            budget,
            peak: 0,
            dispatches: 0,
            profile: Profile::default(),
            last_partition_command: None,
            rows_a_initialized: false,
        })
    }

    fn check_last_command(&mut self) -> Result<()> {
        if let Some(cmd) = self.last_partition_command.take() {
            unsafe { cmd.waitUntilCompleted() };
            complete(cmd.status(), unsafe { cmd.error() }.map(|e| e.to_string()))?;
        }
        Ok(())
    }

    fn acquire_slot(&mut self) -> Result<HistHandle> {
        if let Some(slot) = self.free_slots.pop() {
            Ok(HistHandle(slot as u32))
        } else {
            let slot = self.hist_buffers.len();
            let buf = new_buffer(&self.device, self.gpu_bytes)?;
            if self.splittable_count < self.feature_count {
                write(
                    &buf,
                    &vec![
                        GpuStat {
                            gradient: 0.0,
                            hessian: 0.0,
                            count: 0,
                        };
                        self.width
                    ],
                )?;
            }
            self.hist_buffers.push(Some(buf));
            Ok(HistHandle(slot as u32))
        }
    }
}

impl HistogramBackend for MetalHistograms {
    fn init_objective(
        &mut self,
        labels: &[f64],
        initial_score: f64,
        objective: crate::Objective,
    ) -> Result<Stat> {
        let labels_f32: Vec<f32> = labels.iter().map(|&y| y as f32).collect();
        write(&self.labels, &labels_f32)?;

        let num_rows = self.rows_count as u32;
        let objective_code = match objective {
            crate::Objective::RegressionL2 => 0u32,
            crate::Objective::BinaryLogloss => 1u32,
        };
        let params = ObjectiveParams {
            num_rows,
            objective: objective_code,
        };

        let init_s = initial_score as f32;
        let init_e = (init_s as f64 - initial_score) as f32;

        // SAFETY: obj_error_flag is allocated with sizeof(u32) and host writes before command buffer submission.
        unsafe {
            *self.obj_error_flag.contents().as_ptr().cast::<u32>() = 0;
        }

        let command = allocated(self.queue.commandBuffer())?;
        let encoder = allocated(command.computeCommandEncoder())?;
        encoder.setComputePipelineState(&self.init_scores_pipeline);

        // SAFETY: Buffer bindings match kernel signature in histogram.metal; params/scalars are valid for the call.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&self.scores), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&self.score_errs), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&self.labels), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&self.gradients), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&self.hessians), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(&self.obj_block_totals), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(&self.obj_error_flag), 0, 6);
            encoder.setBytes_length_atIndex(
                NonNull::from(&params).cast::<c_void>(),
                mem::size_of::<ObjectiveParams>(),
                7,
            );
            encoder.setBytes_length_atIndex(
                NonNull::from(&init_s).cast::<c_void>(),
                mem::size_of::<f32>(),
                8,
            );
            encoder.setBytes_length_atIndex(
                NonNull::from(&init_e).cast::<c_void>(),
                mem::size_of::<f32>(),
                9,
            );
        }

        let num_groups = self.rows_count.div_ceil(256).max(1);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: num_groups,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );

        encoder.setComputePipelineState(&self.init_rows_pipeline);
        // SAFETY: rows_a has length rows_count * sizeof(u32); num_rows is scalar.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&self.rows_a), 0, 0);
            encoder.setBytes_length_atIndex(
                NonNull::from(&num_rows).cast::<c_void>(),
                mem::size_of::<u32>(),
                1,
            );
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: self.rows_count.div_ceil(256),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();

        self.check_last_command()?;
        command.commit();
        self.rows_a_initialized = true;
        // SAFETY: Command buffer is committed and we wait synchronously for its completion.
        unsafe { command.waitUntilCompleted() };
        complete(
            command.status(),
            unsafe { command.error() }.map(|e| e.to_string()),
        )?;

        // SAFETY: Command completed; obj_error_flag was written atomically by the GPU kernel.
        let error_flag = unsafe { *self.obj_error_flag.contents().as_ptr().cast::<u32>() };
        if error_flag != 0 {
            return Err(Error::Numerical(
                "objective derivatives are nonfinite".into(),
            ));
        }

        // SAFETY: Command completed; obj_block_totals has at least num_groups * 2 floats.
        let totals_slice = unsafe {
            std::slice::from_raw_parts(
                self.obj_block_totals.contents().as_ptr().cast::<f32>(),
                num_groups * 2,
            )
        };
        let mut sum_g = 0.0f64;
        let mut sum_h = 0.0f64;
        for i in 0..num_groups {
            sum_g += totals_slice[i * 2] as f64;
            sum_h += totals_slice[i * 2 + 1] as f64;
        }

        self.dispatches += 1;
        Ok(Stat {
            gradient: sum_g,
            hessian: sum_h,
            count: num_rows,
        })
    }

    fn update_scores_and_derivatives(
        &mut self,
        leaves: &[super::LeafOutput],
        _cpu_leaf_rows: &[Vec<u32>],
        objective: crate::Objective,
    ) -> Result<Stat> {
        self.check_last_command()?;

        let command = allocated(self.queue.commandBuffer())?;

        // 1. Update scores from leaf assignments
        if !leaves.is_empty() {
            let encoder = allocated(command.computeCommandEncoder())?;
            encoder.setComputePipelineState(&self.update_leaf_scores_pipeline);

            for leaf in leaves {
                if leaf.count == 0 {
                    continue;
                }
                let rows_buf = if leaf.buf_idx == 0 {
                    &self.rows_a
                } else {
                    &self.rows_b
                };
                // SAFETY: Buffer bindings match update_leaf_scores signature; leaf struct is copied by value.
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(&self.scores), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&self.score_errs), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(rows_buf), 0, 2);
                    encoder.setBytes_length_atIndex(
                        NonNull::from(leaf).cast::<c_void>(),
                        mem::size_of::<super::LeafOutput>(),
                        3,
                    );
                }
                let count = leaf.count as usize;
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: count.div_ceil(256),
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 256,
                        height: 1,
                        depth: 1,
                    },
                );
            }
            encoder.endEncoding();
        }

        // 2. Compute derivatives and block totals
        // SAFETY: obj_error_flag is allocated with sizeof(u32) and host writes before command buffer submission.
        unsafe {
            *self.obj_error_flag.contents().as_ptr().cast::<u32>() = 0;
        }

        let num_rows = self.rows_count as u32;
        let objective_code = match objective {
            crate::Objective::RegressionL2 => 0u32,
            crate::Objective::BinaryLogloss => 1u32,
        };
        let params = ObjectiveParams {
            num_rows,
            objective: objective_code,
        };

        let encoder2 = allocated(command.computeCommandEncoder())?;
        encoder2.setComputePipelineState(&self.compute_derivatives_pipeline);
        // SAFETY: Buffer bindings match compute_derivatives signature; params struct is valid for the call.
        unsafe {
            encoder2.setBuffer_offset_atIndex(Some(&self.scores), 0, 0);
            encoder2.setBuffer_offset_atIndex(Some(&self.score_errs), 0, 1);
            encoder2.setBuffer_offset_atIndex(Some(&self.labels), 0, 2);
            encoder2.setBuffer_offset_atIndex(Some(&self.gradients), 0, 3);
            encoder2.setBuffer_offset_atIndex(Some(&self.hessians), 0, 4);
            encoder2.setBuffer_offset_atIndex(Some(&self.obj_block_totals), 0, 5);
            encoder2.setBuffer_offset_atIndex(Some(&self.obj_error_flag), 0, 6);
            encoder2.setBytes_length_atIndex(
                NonNull::from(&params).cast::<c_void>(),
                mem::size_of::<ObjectiveParams>(),
                7,
            );
        }

        let num_groups = self.rows_count.div_ceil(256).max(1);
        encoder2.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: num_groups,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        encoder2.setComputePipelineState(&self.init_rows_pipeline);
        // SAFETY: rows_a has length rows_count * sizeof(u32); num_rows is scalar.
        unsafe {
            encoder2.setBuffer_offset_atIndex(Some(&self.rows_a), 0, 0);
            encoder2.setBytes_length_atIndex(
                NonNull::from(&num_rows).cast::<c_void>(),
                mem::size_of::<u32>(),
                1,
            );
        }
        encoder2.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: self.rows_count.div_ceil(256),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        encoder2.endEncoding();

        command.commit();
        self.rows_a_initialized = true;
        // SAFETY: Command buffer is committed and we wait synchronously for its completion.
        unsafe { command.waitUntilCompleted() };
        complete(
            command.status(),
            unsafe { command.error() }.map(|e| e.to_string()),
        )?;

        // SAFETY: Command completed; obj_error_flag was written atomically by GPU kernel.
        let error_flag = unsafe { *self.obj_error_flag.contents().as_ptr().cast::<u32>() };
        if error_flag != 0 {
            return Err(Error::Numerical(
                "objective derivatives are nonfinite".into(),
            ));
        }

        // SAFETY: Command completed; obj_block_totals has at least num_groups * 2 floats.
        let totals_slice = unsafe {
            std::slice::from_raw_parts(
                self.obj_block_totals.contents().as_ptr().cast::<f32>(),
                num_groups * 2,
            )
        };
        let mut sum_g = 0.0f64;
        let mut sum_h = 0.0f64;
        for i in 0..num_groups {
            sum_g += totals_slice[i * 2] as f64;
            sum_h += totals_slice[i * 2 + 1] as f64;
        }

        self.dispatches += 2;
        Ok(Stat {
            gradient: sum_g,
            hessian: sum_h,
            count: num_rows,
        })
    }

    fn read_scores(&mut self) -> Result<Vec<f64>> {
        self.check_last_command()?;
        // SAFETY: self.scores and self.score_errs have length rows_count * sizeof(f32).
        let scores_slice = unsafe {
            std::slice::from_raw_parts(
                self.scores.contents().as_ptr().cast::<f32>(),
                self.rows_count,
            )
        };
        let score_errs_slice = unsafe {
            std::slice::from_raw_parts(
                self.score_errs.contents().as_ptr().cast::<f32>(),
                self.rows_count,
            )
        };
        let result = scores_slice
            .iter()
            .zip(score_errs_slice)
            .map(|(&s, &e)| (s as f64) - (e as f64))
            .collect();
        Ok(result)
    }

    fn begin_iteration(&mut self, gradients: &[f32], hessians: &[f32]) -> Result<()> {
        write(&self.gradients, gradients)?;
        write(&self.hessians, hessians)
    }

    fn begin_tree(&mut self) -> Result<()> {
        if self.rows_a_initialized {
            self.rows_a_initialized = false;
            return Ok(());
        }
        let command = allocated(self.queue.commandBuffer())?;
        let encoder = allocated(command.computeCommandEncoder())?;
        encoder.setComputePipelineState(&self.init_rows_pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&self.rows_a), 0, 0);
            let num_rows = self.rows_count as u32;
            encoder.setBytes_length_atIndex(
                NonNull::from(&num_rows).cast::<c_void>(),
                mem::size_of::<u32>(),
                1,
            );
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: self.rows_count.div_ceil(256),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        self.check_last_command()?;
        command.commit();
        self.last_partition_command = Some(command);
        self.dispatches += 1;
        Ok(())
    }

    fn resident_rows(&self) -> bool {
        true
    }

    fn partition(
        &mut self,
        data: &BinnedData,
        _parent_rows: &[u32],
        parent_range: super::LeafRange,
        split: &crate::training::Split,
    ) -> Result<(Vec<u32>, Vec<u32>, super::LeafRange, super::LeafRange)> {
        let left_count = split.left.count;
        let right_count = split.right.count;
        let left_range = super::LeafRange {
            offset: parent_range.offset,
            count: left_count,
            buf_idx: 1 - parent_range.buf_idx,
        };
        let right_range = super::LeafRange {
            offset: parent_range.offset + left_count,
            count: right_count,
            buf_idx: 1 - parent_range.buf_idx,
        };

        if parent_range.count == 0 {
            return Ok((Vec::new(), Vec::new(), left_range, right_range));
        }
        self.check_last_command()?;

        let parent_rows_buf = if parent_range.buf_idx == 0 {
            &self.rows_a
        } else {
            &self.rows_b
        };
        let output_rows_buf = if parent_range.buf_idx == 0 {
            &self.rows_b
        } else {
            &self.rows_a
        };

        let mapper = &data.maps[split.feature];
        let missing_bin = mapper.bounds.len().saturating_sub(1) as u32;
        let params = PartitionParams {
            num_rows: data.rows as u32,
            leaf_rows: parent_range.count,
            feature: split.feature as u32,
            threshold_bin: split.threshold_bin as u32,
            missing_bin,
            has_missing: u32::from(mapper.missing),
            default_left: u32::from(split.default_left),
            total_left_count: left_count,
        };

        let byte_offset = (parent_range.offset as usize) * 4;
        let command = allocated(self.queue.commandBuffer())?;

        if parent_range.count <= 1024 {
            let encoder = allocated(command.computeCommandEncoder())?;
            encoder.setComputePipelineState(&self.partition_single_pipeline);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&self.bins), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(parent_rows_buf), byte_offset, 1);
                encoder.setBuffer_offset_atIndex(Some(output_rows_buf), byte_offset, 2);
                encoder.setBytes_length_atIndex(
                    NonNull::from(&params).cast::<c_void>(),
                    mem::size_of::<PartitionParams>(),
                    3,
                );
            }
            let threads = (parent_range.count.div_ceil(32) * 32).max(32);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: threads as usize,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
            self.dispatches += 1;
        } else {
            let num_blocks = parent_range.count.div_ceil(256);

            // Pass 1: Count
            let encoder1 = allocated(command.computeCommandEncoder())?;
            encoder1.setComputePipelineState(&self.partition_count_pipeline);
            unsafe {
                encoder1.setBuffer_offset_atIndex(Some(&self.bins), 0, 0);
                encoder1.setBuffer_offset_atIndex(Some(parent_rows_buf), byte_offset, 1);
                encoder1.setBuffer_offset_atIndex(Some(&self.block_counts), 0, 2);
                encoder1.setBytes_length_atIndex(
                    NonNull::from(&params).cast::<c_void>(),
                    mem::size_of::<PartitionParams>(),
                    3,
                );
            }
            encoder1.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: num_blocks as usize,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            encoder1.endEncoding();

            // Pass 2: Prefix
            let encoder2 = allocated(command.computeCommandEncoder())?;
            encoder2.setComputePipelineState(&self.partition_prefix_pipeline);
            unsafe {
                encoder2.setBuffer_offset_atIndex(Some(&self.block_counts), 0, 0);
                encoder2.setBuffer_offset_atIndex(Some(&self.block_offsets), 0, 1);
                encoder2.setBytes_length_atIndex(
                    NonNull::from(&num_blocks).cast::<c_void>(),
                    mem::size_of::<u32>(),
                    2,
                );
            }
            encoder2.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            encoder2.endEncoding();

            // Pass 3: Scatter
            let encoder3 = allocated(command.computeCommandEncoder())?;
            encoder3.setComputePipelineState(&self.partition_scatter_pipeline);
            unsafe {
                encoder3.setBuffer_offset_atIndex(Some(&self.bins), 0, 0);
                encoder3.setBuffer_offset_atIndex(Some(parent_rows_buf), byte_offset, 1);
                encoder3.setBuffer_offset_atIndex(Some(&self.block_offsets), 0, 2);
                encoder3.setBuffer_offset_atIndex(Some(output_rows_buf), byte_offset, 3);
                encoder3.setBytes_length_atIndex(
                    NonNull::from(&params).cast::<c_void>(),
                    mem::size_of::<PartitionParams>(),
                    4,
                );
            }
            encoder3.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: num_blocks as usize,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            encoder3.endEncoding();
            self.dispatches += 3;
        }

        command.commit();
        self.last_partition_command = Some(command);

        Ok((Vec::new(), Vec::new(), left_range, right_range))
    }

    fn construct_and_find_split(
        &mut self,
        data: &BinnedData,
        leaf: &super::SplitLeaf<'_>,
        _gradients: &[f32],
        _hessians: &[f32],
        config: &crate::TrainingConfig,
    ) -> Result<(Option<crate::training::Split>, HistHandle)> {
        let started = std::time::Instant::now();
        let result =
            objc2::rc::autoreleasepool(|_| self.construct_and_find_split_inner(data, leaf, config));
        if result.is_ok() {
            self.profile.wall_seconds += started.elapsed().as_secs_f64();
            self.profile.calls += 1;
            let count = if leaf.rows.is_empty() {
                leaf.range.count as usize
            } else {
                leaf.rows.len()
            };
            self.profile.small_leaf_calls += u64::from(count <= 4096);
        }
        result
    }

    fn construct(
        &mut self,
        data: &BinnedData,
        rows: &[u32],
        _gradients: &[f32],
        _hessians: &[f32],
        held: usize,
    ) -> Result<HistHandle> {
        let started = std::time::Instant::now();
        let result = objc2::rc::autoreleasepool(|_| self.construct_inner(data, rows, held));
        if result.is_ok() {
            self.profile.wall_seconds += started.elapsed().as_secs_f64();
            self.profile.calls += 1;
            self.profile.small_leaf_calls += u64::from(rows.len() <= 4096);
        }
        result
    }

    fn find_best_split(
        &mut self,
        data: &BinnedData,
        handle: HistHandle,
        total: Stat,
        config: &crate::TrainingConfig,
    ) -> Result<Option<crate::training::Split>> {
        let hist_buf = self
            .hist_buffers
            .get(handle.0 as usize)
            .and_then(|b| b.as_ref())
            .ok_or_else(|| Error::InvalidData("invalid histogram handle".into()))?;

        let parent_gain = crate::training::gain(total, config);
        if !parent_gain.is_finite() {
            return Err(Error::Numerical("split gain overflowed".into()));
        }

        let params = SplitParams {
            stride: data.stride as u32,
            num_features: data.maps.len() as u32,
            min_data_in_leaf: config.min_data_in_leaf as u32,
            total_count: total.count,
            total_gradient: total.gradient as f32,
            total_hessian: total.hessian as f32,
            parent_gain: parent_gain as f32,
            lambda_l1: config.lambda_l1 as f32,
            lambda_l2: config.lambda_l2 as f32,
            min_sum_hessian_in_leaf: config.min_sum_hessian_in_leaf as f32,
            min_gain_to_split: config.min_gain_to_split as f32,
        };

        let command = allocated(self.queue.commandBuffer())?;
        let encoder = allocated(command.computeCommandEncoder())?;
        encoder.setComputePipelineState(&self.find_splits_pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(hist_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&self.feature_meta), 0, 1);
            encoder.setBytes_length_atIndex(
                NonNull::from(&params).cast::<c_void>(),
                mem::size_of::<SplitParams>(),
                2,
            );
            encoder.setBuffer_offset_atIndex(Some(&self.candidates), 0, 3);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: data.maps.len().div_ceil(32),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        unsafe {
            command.commit();
            command.waitUntilCompleted();
        }
        complete(
            command.status(),
            unsafe { command.error() }.map(|e| e.to_string()),
        )?;
        self.dispatches += 1;

        self.parse_split_candidates(data, total, parent_gain, config)
    }

    fn subtract(&mut self, parent: HistHandle, smaller: HistHandle) -> Result<Option<HistHandle>> {
        let sibling_handle = self.acquire_slot()?;
        let parent_buf = self
            .hist_buffers
            .get(parent.0 as usize)
            .and_then(|b| b.as_ref())
            .ok_or_else(|| Error::InvalidData("invalid parent histogram handle".into()))?;
        let smaller_buf = self
            .hist_buffers
            .get(smaller.0 as usize)
            .and_then(|b| b.as_ref())
            .ok_or_else(|| Error::InvalidData("invalid smaller histogram handle".into()))?;
        let sibling_buf = self
            .hist_buffers
            .get(sibling_handle.0 as usize)
            .and_then(|b| b.as_ref())
            .unwrap();

        write(&self.cancel_flag, &[0u32])?;

        let params = Params {
            rows: 0,
            stride: (self.width / self.feature_count) as u32,
            features: self.feature_count as u32,
            leaf_rows: 0,
            chunks: 0,
            row_chunk: 0,
        };

        let command = allocated(self.queue.commandBuffer())?;
        let encoder = allocated(command.computeCommandEncoder())?;
        encoder.setComputePipelineState(&self.subtract_pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(parent_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(smaller_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(sibling_buf), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&self.cancel_flag), 0, 3);
            encoder.setBytes_length_atIndex(
                NonNull::from(&params).cast::<c_void>(),
                mem::size_of::<Params>(),
                4,
            );
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: self.width.div_ceil(32),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        unsafe {
            command.commit();
            command.waitUntilCompleted();
        }
        complete(
            command.status(),
            unsafe { command.error() }.map(|e| e.to_string()),
        )?;
        self.dispatches += 1;

        let flag = unsafe { *(self.cancel_flag.contents().as_ptr().cast::<u32>()) };
        if flag == 0 {
            Ok(Some(sibling_handle))
        } else {
            self.free(sibling_handle);
            Ok(None)
        }
    }

    fn subtract_and_find_split(
        &mut self,
        data: &BinnedData,
        parent: HistHandle,
        smaller: HistHandle,
        total: Stat,
        config: &crate::TrainingConfig,
    ) -> Result<Option<(Option<crate::training::Split>, HistHandle)>> {
        let started = std::time::Instant::now();
        let result = objc2::rc::autoreleasepool(|_| {
            self.subtract_and_find_split_inner(data, parent, smaller, total, config)
        });
        if result.is_ok() {
            self.profile.wall_seconds += started.elapsed().as_secs_f64();
            self.profile.calls += 1;
        }
        result
    }

    fn free(&mut self, handle: HistHandle) {
        let slot = handle.0 as usize;
        if slot < self.hist_buffers.len() && !self.free_slots.contains(&slot) {
            self.free_slots.push(slot);
        }
    }

    fn clear(&mut self) {
        let _ = self.check_last_command();
        self.free_slots.clear();
        for i in 0..self.hist_buffers.len() {
            self.free_slots.push(i);
        }
    }

    fn read_histogram(&self, handle: HistHandle) -> Result<Vec<Stat>> {
        if let Some(cmd) = &self.last_partition_command {
            unsafe { cmd.waitUntilCompleted() };
        }
        let slot = handle.0 as usize;
        let buf = self
            .hist_buffers
            .get(slot)
            .and_then(|b| b.as_ref())
            .ok_or_else(|| Error::InvalidData("invalid histogram handle".into()))?;
        let results = unsafe {
            std::slice::from_raw_parts(buf.contents().as_ptr().cast::<GpuStat>(), self.width)
        };
        let mut histogram = Vec::with_capacity(self.width);
        for stat in results {
            if !stat.gradient.is_finite() || !stat.hessian.is_finite() {
                return Err(Error::Numerical("Metal histogram is nonfinite".into()));
            }
            histogram.push(Stat {
                gradient: stat.gradient as f64,
                hessian: stat.hessian as f64,
                count: stat.count,
            });
        }
        Ok(histogram)
    }

    fn record_workspace(&mut self, bytes: usize) -> Result<()> {
        let required = bytes
            .checked_add(self.auxiliary_bytes())
            .ok_or_else(|| Error::InvalidData("workspace accounting overflow".into()))?;
        if required > self.budget {
            return Err(Error::Workspace {
                required,
                available: self.budget,
            });
        }
        self.peak = self.peak.max(required);
        Ok(())
    }

    fn profile(&self) -> Profile {
        self.profile
    }

    fn peak_bytes(&self) -> usize {
        self.peak
    }

    fn dispatches(&self) -> u64 {
        self.dispatches
    }

    fn auxiliary_bytes(&self) -> usize {
        self.partials.as_ref().map_or(0, |b| b.length()) + self.gpu_bytes
    }
}

impl MetalHistograms {
    fn parse_split_candidates(
        &self,
        data: &BinnedData,
        total: Stat,
        parent_gain: f64,
        config: &crate::TrainingConfig,
    ) -> Result<Option<crate::training::Split>> {
        let candidates_slice = unsafe {
            std::slice::from_raw_parts(
                self.candidates.contents().as_ptr().cast::<SplitCandidate>(),
                data.maps.len(),
            )
        };

        let mut best: Option<crate::training::Split> = None;
        for (feature, &cand) in candidates_slice.iter().enumerate() {
            if cand.valid == 0 {
                continue;
            }
            let mapper = &data.maps[feature];
            let left = Stat {
                gradient: cand.left_gradient as f64,
                hessian: cand.left_hessian as f64,
                count: cand.left_count,
            };
            let right = total.subtract(left);
            let split_gain = crate::training::gain(left, config)
                + crate::training::gain(right, config)
                - parent_gain;
            if !split_gain.is_finite() {
                return Err(Error::Numerical("split gain is nonfinite".into()));
            }
            if split_gain <= config.min_gain_to_split || best.is_some_and(|s| split_gain <= s.gain)
            {
                continue;
            }
            let threshold_bin = cand.threshold_bin as usize;
            let threshold = mapper.bounds[threshold_bin].min(1e300);
            best = Some(crate::training::Split {
                feature,
                threshold_bin,
                threshold,
                default_left: if mapper.missing {
                    cand.missing_left != 0
                } else {
                    0.0 <= threshold
                },
                gain: split_gain,
                left,
                right,
            });
        }
        Ok(best)
    }

    fn subtract_and_find_split_inner(
        &mut self,
        data: &BinnedData,
        parent: HistHandle,
        smaller: HistHandle,
        total: Stat,
        config: &crate::TrainingConfig,
    ) -> Result<Option<(Option<crate::training::Split>, HistHandle)>> {
        let parent_gain = crate::training::gain(total, config);
        if !parent_gain.is_finite() {
            return Err(Error::Numerical("split gain overflowed".into()));
        }
        let sibling_handle = self.acquire_slot()?;
        let parent_buf = self
            .hist_buffers
            .get(parent.0 as usize)
            .and_then(|b| b.as_ref())
            .ok_or_else(|| Error::InvalidData("invalid parent histogram handle".into()))?;
        let smaller_buf = self
            .hist_buffers
            .get(smaller.0 as usize)
            .and_then(|b| b.as_ref())
            .ok_or_else(|| Error::InvalidData("invalid smaller histogram handle".into()))?;
        let sibling_buf = self
            .hist_buffers
            .get(sibling_handle.0 as usize)
            .and_then(|b| b.as_ref())
            .unwrap();

        write(&self.cancel_flag, &[0u32])?;

        let params = Params {
            rows: 0,
            stride: (self.width / self.feature_count) as u32,
            features: self.feature_count as u32,
            leaf_rows: 0,
            chunks: 0,
            row_chunk: 0,
        };
        let split_params = SplitParams {
            stride: data.stride as u32,
            num_features: data.maps.len() as u32,
            min_data_in_leaf: config.min_data_in_leaf as u32,
            total_count: total.count,
            total_gradient: total.gradient as f32,
            total_hessian: total.hessian as f32,
            parent_gain: parent_gain as f32,
            lambda_l1: config.lambda_l1 as f32,
            lambda_l2: config.lambda_l2 as f32,
            min_sum_hessian_in_leaf: config.min_sum_hessian_in_leaf as f32,
            min_gain_to_split: config.min_gain_to_split as f32,
        };

        let command = allocated(self.queue.commandBuffer())?;

        let sub_encoder = allocated(command.computeCommandEncoder())?;
        sub_encoder.setComputePipelineState(&self.subtract_pipeline);
        unsafe {
            sub_encoder.setBuffer_offset_atIndex(Some(parent_buf), 0, 0);
            sub_encoder.setBuffer_offset_atIndex(Some(smaller_buf), 0, 1);
            sub_encoder.setBuffer_offset_atIndex(Some(sibling_buf), 0, 2);
            sub_encoder.setBuffer_offset_atIndex(Some(&self.cancel_flag), 0, 3);
            sub_encoder.setBytes_length_atIndex(
                NonNull::from(&params).cast::<c_void>(),
                mem::size_of::<Params>(),
                4,
            );
        }
        sub_encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: self.width.div_ceil(32),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
        sub_encoder.endEncoding();

        let find_encoder = allocated(command.computeCommandEncoder())?;
        find_encoder.setComputePipelineState(&self.find_splits_pipeline);
        unsafe {
            find_encoder.setBuffer_offset_atIndex(Some(sibling_buf), 0, 0);
            find_encoder.setBuffer_offset_atIndex(Some(&self.feature_meta), 0, 1);
            find_encoder.setBytes_length_atIndex(
                NonNull::from(&split_params).cast::<c_void>(),
                mem::size_of::<SplitParams>(),
                2,
            );
            find_encoder.setBuffer_offset_atIndex(Some(&self.candidates), 0, 3);
        }
        find_encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: data.maps.len().div_ceil(32),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
        find_encoder.endEncoding();

        unsafe {
            command.commit();
            command.waitUntilCompleted();
        }
        complete(
            command.status(),
            unsafe { command.error() }.map(|e| e.to_string()),
        )?;
        self.check_last_command()?;
        self.dispatches += 2;
        let gpu_elapsed = unsafe { command.GPUEndTime() - command.GPUStartTime() };
        if gpu_elapsed.is_finite() && gpu_elapsed >= 0.0 {
            self.profile.gpu_seconds += gpu_elapsed;
        }

        let flag = unsafe { *(self.cancel_flag.contents().as_ptr().cast::<u32>()) };
        if flag == 0 {
            let best = self.parse_split_candidates(data, total, parent_gain, config)?;
            Ok(Some((best, sibling_handle)))
        } else {
            self.free(sibling_handle);
            Ok(None)
        }
    }

    fn construct_and_find_split_inner(
        &mut self,
        data: &BinnedData,
        leaf: &super::SplitLeaf<'_>,
        config: &crate::TrainingConfig,
    ) -> Result<(Option<crate::training::Split>, HistHandle)> {
        let parent_gain = crate::training::gain(leaf.total, config);
        if !parent_gain.is_finite() {
            return Err(Error::Numerical("split gain overflowed".into()));
        }
        let host_bytes = histogram_bytes(data)?;
        let gpu_bytes = self.gpu_bytes;
        let required = leaf
            .held
            .checked_add(host_bytes)
            .and_then(|n| n.checked_add(gpu_bytes.checked_mul(2)?))
            .ok_or_else(|| Error::InvalidData("Metal workspace size overflow".into()))?;
        if required > self.budget {
            return Err(Error::Workspace {
                required,
                available: self.budget,
            });
        }
        if !leaf.rows.is_empty() {
            write(&self.rows, leaf.rows)?;
        }
        let (byte_offset, leaf_rows) = if leaf.rows.is_empty() {
            ((leaf.range.offset as usize) * 4, leaf.range.count as usize)
        } else {
            (0, leaf.rows.len())
        };

        let max_partial = self.budget - leaf.held - host_bytes - gpu_bytes;
        let desired_chunks = leaf_rows.div_ceil(4096);
        let max_chunks = (max_partial / gpu_bytes).min(desired_chunks).max(1);
        let row_chunk = leaf_rows.div_ceil(max_chunks);
        let chunks = leaf_rows.div_ceil(row_chunk);
        let partial_bytes = gpu_bytes
            .checked_mul(chunks)
            .ok_or_else(|| Error::InvalidData("partial histogram size overflow".into()))?;
        if chunks > 1 {
            if self.partials.as_ref().is_some_and(|buffer| {
                buffer.length() < partial_bytes || buffer.length() > max_partial
            }) {
                self.partials = None;
            }
            if self.partials.is_none() {
                self.partials = Some(new_buffer(&self.device, partial_bytes)?);
            }
        }
        self.record_workspace(leaf.held + host_bytes)?;
        let handle = self.acquire_slot()?;
        let output = self.hist_buffers[handle.0 as usize].as_ref().unwrap();

        let params = Params {
            rows: data.rows as u32,
            stride: data.stride as u32,
            features: data.maps.len() as u32,
            leaf_rows: leaf_rows as u32,
            chunks: chunks as u32,
            row_chunk: row_chunk as u32,
        };
        let split_params = SplitParams {
            stride: data.stride as u32,
            num_features: data.maps.len() as u32,
            min_data_in_leaf: config.min_data_in_leaf as u32,
            total_count: leaf.total.count,
            total_gradient: leaf.total.gradient as f32,
            total_hessian: leaf.total.hessian as f32,
            parent_gain: parent_gain as f32,
            lambda_l1: config.lambda_l1 as f32,
            lambda_l2: config.lambda_l2 as f32,
            min_sum_hessian_in_leaf: config.min_sum_hessian_in_leaf as f32,
            min_gain_to_split: config.min_gain_to_split as f32,
        };

        let command = allocated(self.queue.commandBuffer())?;
        let encoder = allocated(command.computeCommandEncoder())?;
        encoder.setComputePipelineState(&self.partial_pipeline);
        let target_buffer = if chunks == 1 {
            output
        } else {
            required_buffer(&self.partials)?
        };
        let row_buf = if leaf.rows.is_empty() {
            if leaf.range.buf_idx == 0 {
                &*self.rows_a
            } else {
                &*self.rows_b
            }
        } else {
            &*self.rows
        };
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&self.bins), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&self.gradients), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&self.hessians), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(row_buf), byte_offset, 3);
            encoder.setBuffer_offset_atIndex(Some(target_buffer), 0, 4);
            encoder.setBytes_length_atIndex(
                NonNull::from(&params).cast::<c_void>(),
                mem::size_of::<Params>(),
                5,
            );
            encoder.setBuffer_offset_atIndex(Some(&self.splittable_features), 0, 6);
        }
        if self.splittable_count > 0 {
            let simd_groups = data.stride.div_ceil(64).clamp(1, 4);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: self.splittable_count,
                    height: chunks,
                    depth: 1,
                },
                MTLSize {
                    width: simd_groups * 32,
                    height: 1,
                    depth: 1,
                },
            );
        }
        encoder.endEncoding();
        if chunks > 1 {
            let partials = required_buffer(&self.partials)?;
            let reducer = allocated(command.computeCommandEncoder())?;
            reducer.setComputePipelineState(&self.reduce_pipeline);
            unsafe {
                reducer.setBuffer_offset_atIndex(Some(partials), 0, 0);
                reducer.setBuffer_offset_atIndex(Some(output), 0, 1);
                reducer.setBytes_length_atIndex(
                    NonNull::from(&params).cast::<c_void>(),
                    mem::size_of::<Params>(),
                    2,
                );
            }
            reducer.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: self.width.div_ceil(32),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 32,
                    height: 1,
                    depth: 1,
                },
            );
            reducer.endEncoding();
        }

        let finder = allocated(command.computeCommandEncoder())?;
        finder.setComputePipelineState(&self.find_splits_pipeline);
        unsafe {
            finder.setBuffer_offset_atIndex(Some(output), 0, 0);
            finder.setBuffer_offset_atIndex(Some(&self.feature_meta), 0, 1);
            finder.setBytes_length_atIndex(
                NonNull::from(&split_params).cast::<c_void>(),
                mem::size_of::<SplitParams>(),
                2,
            );
            finder.setBuffer_offset_atIndex(Some(&self.candidates), 0, 3);
        }
        finder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: data.maps.len().div_ceil(32),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
        finder.endEncoding();

        unsafe {
            command.commit();
            command.waitUntilCompleted();
        }
        complete(
            command.status(),
            unsafe { command.error() }.map(|e| e.to_string()),
        )?;
        self.check_last_command()?;
        self.dispatches += if chunks == 1 { 2 } else { 3 };
        let gpu_elapsed = unsafe { command.GPUEndTime() - command.GPUStartTime() };
        if gpu_elapsed.is_finite() && gpu_elapsed >= 0.0 {
            self.profile.gpu_seconds += gpu_elapsed;
        }

        let best = self.parse_split_candidates(data, leaf.total, parent_gain, config)?;
        Ok((best, handle))
    }

    fn construct_inner(
        &mut self,
        data: &BinnedData,
        rows: &[u32],
        held: usize,
    ) -> Result<HistHandle> {
        let host_bytes = histogram_bytes(data)?;
        let gpu_bytes = self.gpu_bytes;
        let required = held
            .checked_add(host_bytes)
            .and_then(|n| n.checked_add(gpu_bytes.checked_mul(2)?))
            .ok_or_else(|| Error::InvalidData("Metal workspace size overflow".into()))?;
        if required > self.budget {
            return Err(Error::Workspace {
                required,
                available: self.budget,
            });
        }
        if !rows.is_empty() {
            write(&self.rows, rows)?;
        }
        let leaf_rows = if rows.is_empty() {
            self.rows_count
        } else {
            rows.len()
        };

        let max_partial = self.budget - held - host_bytes - gpu_bytes;
        let desired_chunks = leaf_rows.div_ceil(4096);
        let max_chunks = (max_partial / gpu_bytes).min(desired_chunks).max(1);
        let row_chunk = leaf_rows.div_ceil(max_chunks);
        let chunks = leaf_rows.div_ceil(row_chunk);
        let partial_bytes = gpu_bytes
            .checked_mul(chunks)
            .ok_or_else(|| Error::InvalidData("partial histogram size overflow".into()))?;
        if chunks > 1 {
            if self.partials.as_ref().is_some_and(|buffer| {
                buffer.length() < partial_bytes || buffer.length() > max_partial
            }) {
                self.partials = None;
            }
            if self.partials.is_none() {
                self.partials = Some(new_buffer(&self.device, partial_bytes)?);
            }
        }
        self.record_workspace(held + host_bytes)?;
        let handle = self.acquire_slot()?;
        let output = self.hist_buffers[handle.0 as usize].as_ref().unwrap();

        let params = Params {
            rows: data.rows as u32,
            stride: data.stride as u32,
            features: data.maps.len() as u32,
            leaf_rows: leaf_rows as u32,
            chunks: chunks as u32,
            row_chunk: row_chunk as u32,
        };
        let command = allocated(self.queue.commandBuffer())?;
        let encoder = allocated(command.computeCommandEncoder())?;
        encoder.setComputePipelineState(&self.partial_pipeline);
        let target_buffer = if chunks == 1 {
            output
        } else {
            required_buffer(&self.partials)?
        };
        let row_buf = if rows.is_empty() {
            &*self.rows_a
        } else {
            &*self.rows
        };
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&self.bins), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&self.gradients), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&self.hessians), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(row_buf), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(target_buffer), 0, 4);
            encoder.setBytes_length_atIndex(
                NonNull::from(&params).cast::<c_void>(),
                mem::size_of::<Params>(),
                5,
            );
            encoder.setBuffer_offset_atIndex(Some(&self.splittable_features), 0, 6);
        }
        if self.splittable_count > 0 {
            let simd_groups = data.stride.div_ceil(64).clamp(1, 4);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: self.splittable_count,
                    height: chunks,
                    depth: 1,
                },
                MTLSize {
                    width: simd_groups * 32,
                    height: 1,
                    depth: 1,
                },
            );
        }
        encoder.endEncoding();
        if chunks > 1 {
            let partials = required_buffer(&self.partials)?;
            let reducer = allocated(command.computeCommandEncoder())?;
            reducer.setComputePipelineState(&self.reduce_pipeline);
            // SAFETY: Encoders execute in order; the reducer guards accesses by width.
            unsafe {
                reducer.setBuffer_offset_atIndex(Some(partials), 0, 0);
                reducer.setBuffer_offset_atIndex(Some(output), 0, 1);
                reducer.setBytes_length_atIndex(
                    NonNull::from(&params).cast::<c_void>(),
                    mem::size_of::<Params>(),
                    2,
                );
            }
            reducer.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: self.width.div_ceil(32),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 32,
                    height: 1,
                    depth: 1,
                },
            );
            reducer.endEncoding();
        }
        // SAFETY: Encoders are ended and all referenced buffers outlive the command.
        unsafe {
            command.commit();
            command.waitUntilCompleted();
        }
        // SAFETY: Error access is valid after waitUntilCompleted.
        complete(
            command.status(),
            unsafe { command.error() }.map(|e| e.to_string()),
        )?;
        self.dispatches += if chunks == 1 { 1 } else { 2 };
        // SAFETY: GPU timestamps are read only after successful completion.
        let gpu_elapsed = unsafe { command.GPUEndTime() - command.GPUStartTime() };
        if gpu_elapsed.is_finite() && gpu_elapsed >= 0.0 {
            self.profile.gpu_seconds += gpu_elapsed;
        }
        Ok(handle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::histogram::CpuHistograms;
    #[test]
    fn injected_allocation_and_command_errors_are_explicit() {
        assert!(allocated::<Buffer>(None).is_err());
        assert!(complete(
            MTLCommandBufferStatus::Error,
            Some("injected execution error".into())
        )
        .is_err());
    }
    #[test]
    #[ignore = "requires real Apple Silicon Metal access; run the hardware validation command"]
    fn metal_histograms_match_cpu_on_real_gpu() {
        for max_bin in [2, 63, 255] {
            let values: Vec<_> = (0..8197)
                .map(|i| {
                    if i % 17 == 0 {
                        f64::NAN
                    } else {
                        ((i % 101) as f64 - 50.0) / 10.0
                    }
                })
                .collect();
            let data = BinnedData::new(
                crate::DenseMatrix::new(&values, values.len(), 1).unwrap(),
                max_bin,
            )
            .unwrap();
            let gradients: Vec<_> = (0..values.len())
                .map(|i| if i % 2 == 0 { 1.0001_f32 } else { -1.0 })
                .collect();
            let hessians = vec![0.25_f32; values.len()];
            let rows: Vec<_> = (0..values.len() as u32).filter(|i| i % 3 != 0).collect();
            let mut cpu = CpuHistograms::new(1024 * 1024);
            let mut metal = MetalHistograms::new(&data, 1024 * 1024).unwrap();
            metal.begin_iteration(&gradients, &hessians).unwrap();
            for membership in [&rows[..], &rows[..13], &rows[..]] {
                let expected_h = cpu
                    .construct(&data, membership, &gradients, &hessians, 0)
                    .unwrap();
                let expected = cpu.read_histogram(expected_h).unwrap();
                let actual_h = metal
                    .construct(&data, membership, &gradients, &hessians, 0)
                    .unwrap();
                let actual = metal.read_histogram(actual_h).unwrap();
                for (bin, (a, e)) in actual.iter().zip(&expected).enumerate() {
                    let norm = membership
                        .iter()
                        .filter(|&&row| data.bin(row as usize, 0) == bin)
                        .map(|&row| gradients[row as usize].abs() as f64)
                        .sum::<f64>();
                    assert_eq!(a.count, e.count);
                    assert!((a.gradient - e.gradient).abs() <= 1e-5 + 1e-4 * norm);
                    assert!((a.hessian - e.hessian).abs() <= 1e-5 + 1e-4 * e.hessian);
                }
            }
            assert_eq!(metal.dispatches(), 5);
            assert!(metal.peak_bytes() <= 1024 * 1024);
            let minimum = histogram_bytes(&data).unwrap() * 2;
            let mut bounded = MetalHistograms::new(&data, minimum).unwrap();
            bounded.begin_iteration(&gradients, &hessians).unwrap();
            assert!(bounded
                .construct(&data, &rows, &gradients, &hessians, 0)
                .is_ok());
            assert!(bounded.peak_bytes() <= minimum);
            assert!(bounded
                .construct(&data, &rows, &gradients, &hessians, minimum)
                .is_err());
        }
    }
    #[test]
    #[ignore = "requires real Apple Silicon Metal access; run the hardware validation command"]
    fn invalid_metal_source_fails_without_histogram_results() {
        assert!(compile(&device().unwrap(), "invalid Metal source").is_err());
    }

    #[test]
    #[ignore = "requires real Apple Silicon Metal access; run the hardware validation command"]
    fn metal_partition_matches_cpu_on_real_gpu() {
        if check_device().is_ok() {
            let n_rows = 2500;
            let mut values = Vec::with_capacity(n_rows * 2);
            for i in 0..n_rows {
                let v0 = if i % 17 == 0 {
                    f64::NAN
                } else {
                    ((i * 7 + 13) % 100) as f64
                };
                let v1 = if i % 11 == 0 {
                    f64::NAN
                } else {
                    ((i * 13 + 5) % 50) as f64
                };
                values.push(v0);
                values.push(v1);
            }
            let data = crate::binning::BinnedData::new(
                crate::DenseMatrix::new(&values, n_rows, 2).unwrap(),
                64,
            )
            .unwrap();
            let mut cpu = CpuHistograms::new(1024 * 1024);
            let mut metal = MetalHistograms::new(&data, 1024 * 1024).unwrap();
            metal.begin_tree().unwrap();

            // Multi-block partition test (> 1024 rows)
            let root_range = crate::histogram::LeafRange {
                offset: 0,
                count: n_rows as u32,
                buf_idx: 0,
            };
            let parent_rows: Vec<u32> = (0..n_rows as u32).collect();
            let missing_bin_0 = data.maps[0].bounds.len() - 1;
            let mut left_count = 0u32;
            let mut right_count = 0u32;
            for &r in &parent_rows {
                let b = data.bin(r as usize, 0);
                let is_m = data.maps[0].missing && b == missing_bin_0;
                let goes_left = if is_m { true } else { b <= 30 };
                if goes_left {
                    left_count += 1;
                } else {
                    right_count += 1;
                }
            }
            let split_f0 = crate::training::Split {
                feature: 0,
                threshold_bin: 30,
                threshold: 45.0,
                default_left: true,
                gain: 1.0,
                left: crate::histogram::Stat {
                    count: left_count,
                    ..Default::default()
                },
                right: crate::histogram::Stat {
                    count: right_count,
                    ..Default::default()
                },
            };

            let (cpu_left, cpu_right, _, _) = cpu
                .partition(&data, &parent_rows, root_range, &split_f0)
                .unwrap();
            let (_, _, metal_left_range, metal_right_range) =
                metal.partition(&data, &[], root_range, &split_f0).unwrap();

            assert_eq!(metal_left_range.offset, 0);
            assert_eq!(metal_left_range.count, left_count);
            assert_eq!(metal_left_range.buf_idx, 1);
            assert_eq!(metal_right_range.offset, left_count);
            assert_eq!(metal_right_range.count, right_count);
            assert_eq!(metal_right_range.buf_idx, 1);

            metal.check_last_command().unwrap();
            let metal_left_rows = unsafe {
                std::slice::from_raw_parts(
                    metal.rows_b.contents().as_ptr().cast::<u32>(),
                    left_count as usize,
                )
            };
            assert_eq!(metal_left_rows, cpu_left.as_slice());

            let metal_right_rows = unsafe {
                std::slice::from_raw_parts(
                    metal
                        .rows_b
                        .contents()
                        .as_ptr()
                        .cast::<u32>()
                        .add(left_count as usize),
                    right_count as usize,
                )
            };
            assert_eq!(metal_right_rows, cpu_right.as_slice());

            // Single-block partition test (<= 1024 rows) on child
            let (target_range, target_cpu_rows) = if metal_left_range.count <= 1024 {
                (metal_left_range, &cpu_left)
            } else {
                (metal_right_range, &cpu_right)
            };
            let missing_bin_1 = data.maps[1].bounds.len() - 1;
            let mut left_count_1 = 0u32;
            let mut right_count_1 = 0u32;
            for &r in target_cpu_rows.iter() {
                let b = data.bin(r as usize, 1);
                let is_m = data.maps[1].missing && b == missing_bin_1;
                let goes_left = if is_m { false } else { b <= 20 };
                if goes_left {
                    left_count_1 += 1;
                } else {
                    right_count_1 += 1;
                }
            }
            let split_f1 = crate::training::Split {
                feature: 1,
                threshold_bin: 20,
                threshold: 25.0,
                default_left: false,
                gain: 1.0,
                left: crate::histogram::Stat {
                    count: left_count_1,
                    ..Default::default()
                },
                right: crate::histogram::Stat {
                    count: right_count_1,
                    ..Default::default()
                },
            };

            let (cpu_left_1, cpu_right_1, _, _) = cpu
                .partition(&data, target_cpu_rows, target_range, &split_f1)
                .unwrap();
            let (_, _, metal_left_1, metal_right_1) = metal
                .partition(&data, &[], target_range, &split_f1)
                .unwrap();

            assert_eq!(metal_left_1.count, left_count_1);
            assert_eq!(metal_right_1.count, right_count_1);
            assert_eq!(metal_left_1.buf_idx, 0);
            assert_eq!(metal_right_1.buf_idx, 0);

            metal.check_last_command().unwrap();
            let out_buf = &metal.rows_a;
            let metal_left_rows_1 = unsafe {
                std::slice::from_raw_parts(
                    out_buf
                        .contents()
                        .as_ptr()
                        .cast::<u32>()
                        .add(metal_left_1.offset as usize),
                    left_count_1 as usize,
                )
            };
            assert_eq!(metal_left_rows_1, cpu_left_1.as_slice());

            let metal_right_rows_1 = unsafe {
                std::slice::from_raw_parts(
                    out_buf
                        .contents()
                        .as_ptr()
                        .cast::<u32>()
                        .add(metal_right_1.offset as usize),
                    right_count_1 as usize,
                )
            };
            assert_eq!(metal_right_rows_1, cpu_right_1.as_slice());
        }
    }

    #[test]
    #[ignore = "requires real Apple Silicon Metal access; run the hardware validation command"]
    fn metal_objective_and_scores_match_cpu_on_real_gpu() {
        if check_device().is_ok() {
            for objective in [
                crate::Objective::RegressionL2,
                crate::Objective::BinaryLogloss,
            ] {
                let n_rows = 2048;
                let mut values = Vec::with_capacity(n_rows);
                let mut labels = Vec::with_capacity(n_rows);
                for i in 0..n_rows {
                    values.push((i % 50) as f64);
                    let label = if objective == crate::Objective::RegressionL2 {
                        (i as f64 * 0.1).sin() * 10.0
                    } else {
                        (i % 3 == 0) as u32 as f64
                    };
                    labels.push(label);
                }
                let data = crate::binning::BinnedData::new(
                    crate::DenseMatrix::new(&values, n_rows, 1).unwrap(),
                    64,
                )
                .unwrap();
                let initial = crate::objective::initial_score(&labels, objective).unwrap();
                let mut cpu = CpuHistograms::new(1024 * 1024);
                let mut metal = MetalHistograms::new(&data, 1024 * 1024).unwrap();

                let cpu_stat = cpu.init_objective(&labels, initial, objective).unwrap();
                let metal_stat = metal.init_objective(&labels, initial, objective).unwrap();

                let norm =
                    labels.iter().map(|y| y.abs()).sum::<f64>() + (n_rows as f64) * initial.abs();
                assert_eq!(cpu_stat.count, metal_stat.count);
                assert!(
                    (cpu_stat.gradient - metal_stat.gradient).abs() <= 1e-4 * norm.max(1.0),
                    "init gradient mismatch: cpu={}, metal={}, norm={}",
                    cpu_stat.gradient,
                    metal_stat.gradient,
                    norm
                );
                assert!(
                    (cpu_stat.hessian - metal_stat.hessian).abs()
                        <= 1e-4 * cpu_stat.hessian.abs().max(1.0),
                    "init hessian mismatch"
                );

                let cpu_scores = cpu.read_scores().unwrap();
                let metal_scores = metal.read_scores().unwrap();
                for (c, m) in cpu_scores.iter().zip(&metal_scores) {
                    assert!((c - m).abs() <= 1e-6);
                }

                metal.begin_tree().unwrap();
                let half = (n_rows / 2) as u32;
                let leaves = vec![
                    crate::histogram::LeafOutput {
                        offset: 0,
                        count: half,
                        value: 0.125,
                        buf_idx: 0,
                    },
                    crate::histogram::LeafOutput {
                        offset: half,
                        count: n_rows as u32 - half,
                        value: -0.0625,
                        buf_idx: 0,
                    },
                ];
                let cpu_leaf_rows = vec![
                    (0..half).collect::<Vec<_>>(),
                    (half..n_rows as u32).collect::<Vec<_>>(),
                ];

                for _round in 0..10 {
                    let cpu_updated = cpu
                        .update_scores_and_derivatives(&leaves, &cpu_leaf_rows, objective)
                        .unwrap();
                    let metal_updated = metal
                        .update_scores_and_derivatives(&leaves, &cpu_leaf_rows, objective)
                        .unwrap();

                    assert_eq!(cpu_updated.count, metal_updated.count);
                    assert!(
                        (cpu_updated.gradient - metal_updated.gradient).abs()
                            <= 1e-3 * cpu_updated.gradient.abs().max(1.0),
                        "round gradient mismatch"
                    );
                    assert!(
                        (cpu_updated.hessian - metal_updated.hessian).abs()
                            <= 1e-3 * cpu_updated.hessian.abs().max(1.0),
                        "round hessian mismatch"
                    );

                    let c_s = cpu.read_scores().unwrap();
                    let m_s = metal.read_scores().unwrap();
                    for (c, m) in c_s.iter().zip(&m_s) {
                        assert!((c - m).abs() <= 1e-5, "accumulated score error");
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "requires real Apple Silicon Metal access; run the hardware validation command"]
    fn metal_subtraction_and_split_search_matches_cpu_on_real_gpu() {
        if check_device().is_ok() {
            let n_rows = 1500;
            let mut values = Vec::with_capacity(n_rows * 2);
            for i in 0..n_rows {
                let v0 = if i % 13 == 0 {
                    f64::NAN
                } else {
                    ((i * 17 + 5) % 80) as f64
                };
                let v1 = if i % 19 == 0 {
                    f64::NAN
                } else {
                    ((i * 23 + 11) % 60) as f64
                };
                values.push(v0);
                values.push(v1);
            }
            let data = crate::binning::BinnedData::new(
                crate::DenseMatrix::new(&values, n_rows, 2).unwrap(),
                64,
            )
            .unwrap();
            let gradients: Vec<f32> = (0..n_rows).map(|i| (i as f32 * 0.05).sin() * 2.0).collect();
            let hessians: Vec<f32> = vec![1.0; n_rows];

            let mut cpu = CpuHistograms::new(1024 * 1024);
            let mut metal = MetalHistograms::new(&data, 1024 * 1024).unwrap();
            cpu.begin_iteration(&gradients, &hessians).unwrap();
            metal.begin_iteration(&gradients, &hessians).unwrap();

            let parent_rows: Vec<u32> = (0..n_rows as u32).collect();
            let child_count = 600;
            let child_rows: Vec<u32> = (0..child_count).collect();

            let parent_cpu_h = cpu
                .construct(&data, &parent_rows, &gradients, &hessians, 0)
                .unwrap();
            let parent_metal_h = metal
                .construct(&data, &parent_rows, &gradients, &hessians, 0)
                .unwrap();

            let child_cpu_h = cpu
                .construct(&data, &child_rows, &gradients, &hessians, 0)
                .unwrap();
            let child_metal_h = metal
                .construct(&data, &child_rows, &gradients, &hessians, 0)
                .unwrap();

            // Test Metal subtract
            let diff_cpu_h = cpu.subtract(parent_cpu_h, child_cpu_h).unwrap().unwrap();
            let diff_metal_h = metal
                .subtract(parent_metal_h, child_metal_h)
                .unwrap()
                .unwrap();

            let diff_cpu_hist = cpu.read_histogram(diff_cpu_h).unwrap();
            let diff_metal_hist = metal.read_histogram(diff_metal_h).unwrap();

            assert_eq!(diff_cpu_hist.len(), diff_metal_hist.len());
            for (c, m) in diff_cpu_hist.iter().zip(&diff_metal_hist) {
                assert_eq!(c.count, m.count);
                assert!((c.gradient - m.gradient).abs() <= 1e-4 * (c.count as f64).max(1.0));
                assert!((c.hessian - m.hessian).abs() <= 1e-4 * (c.count as f64).max(1.0));
            }

            // Test subtract cancellation on negative counts (invalid subtraction: child - parent)
            let invalid_cpu = cpu.subtract(child_cpu_h, parent_cpu_h).unwrap();
            let invalid_metal = metal.subtract(child_metal_h, parent_metal_h).unwrap();
            assert!(invalid_cpu.is_none());
            assert!(invalid_metal.is_none());

            // Test split finding on subtracted sibling
            let mut sibling_total = Stat::default();
            for &r in &parent_rows[child_count as usize..] {
                sibling_total.count += 1;
                sibling_total.gradient += gradients[r as usize] as f64;
                sibling_total.hessian += hessians[r as usize] as f64;
            }
            let config = crate::TrainingConfig::default();
            let cpu_best = cpu
                .find_best_split(&data, diff_cpu_h, sibling_total, &config)
                .unwrap();
            let (metal_best, sibling_h) = metal
                .subtract_and_find_split(
                    &data,
                    parent_metal_h,
                    child_metal_h,
                    sibling_total,
                    &config,
                )
                .unwrap()
                .unwrap();

            match (cpu_best, metal_best) {
                (Some(c), Some(m)) => {
                    assert_eq!(c.feature, m.feature);
                    assert_eq!(c.threshold_bin, m.threshold_bin);
                    assert_eq!(c.default_left, m.default_left);
                    assert!((c.gain - m.gain).abs() <= 1e-4 * c.gain.abs().max(1.0));
                    assert_eq!(c.left.count, m.left.count);
                    assert_eq!(c.right.count, m.right.count);
                }
                (None, None) => {}
                _ => panic!("CPU and Metal best split eligibility mismatch"),
            }
            metal.free(sibling_h);
        }
    }

    #[test]
    #[ignore = "requires real Apple Silicon Metal access; run the hardware validation command"]
    fn metal_concurrent_sessions_are_isolated_on_real_gpu() {
        if check_device().is_ok() {
            let num_threads = 4;
            let handles: Vec<_> = (0..num_threads)
                .map(|t| {
                    std::thread::spawn(move || {
                        let n_rows = 500;
                        let mut values = Vec::with_capacity(n_rows * 2);
                        let mut labels = Vec::with_capacity(n_rows);
                        for i in 0..n_rows {
                            values.push(((i * 7 + t * 3) % 40) as f64);
                            values.push(((i * 11 + t * 5) % 30) as f64);
                            labels.push(((i + t) % 2) as f64);
                        }
                        let matrix = crate::DenseMatrix::new(&values, n_rows, 2).unwrap();
                        let config = crate::TrainingConfig {
                            backend: crate::Backend::Metal,
                            iterations: 5,
                            max_depth: Some(3),
                            num_leaves: 7,
                            ..Default::default()
                        };
                        let trained = crate::train(matrix, &labels, &config).unwrap();
                        assert!(trained.model.tree_count() > 0);
                        let preds = trained
                            .model
                            .predict_raw(crate::DenseMatrix::new(&values[..20], 10, 2).unwrap())
                            .unwrap();
                        assert_eq!(preds.len(), 10);
                        for p in preds {
                            assert!(p.is_finite());
                        }
                    })
                })
                .collect();

            for h in handles {
                h.join().expect("thread should not panic");
            }
        }
    }
}
