use super::{histogram_bytes, HistogramBackend, Profile, Stat};
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
fn compile(device: &ProtocolObject<dyn MTLDevice>, source: &str) -> Result<(Pipeline, Pipeline)> {
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
    if partial.threadExecutionWidth() != 32 || partial.maxTotalThreadsPerThreadgroup() < 32 {
        return Err(Error::BackendUnavailable(
            "histogram kernel requires a 32-lane SIMD group".into(),
        ));
    }
    Ok((partial, pipeline("reduce_histograms")?))
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
    bins: Buffer,
    gradients: Buffer,
    hessians: Buffer,
    rows: Buffer,
    partials: Option<Buffer>,
    output: Option<Buffer>,
    budget: usize,
    peak: usize,
    dispatches: u64,
    profile: Profile,
}
impl MetalHistograms {
    pub fn new(data: &BinnedData, budget: usize) -> Result<Self> {
        let device = device()?;
        let (partial_pipeline, reduce_pipeline) = compile(&device, SOURCE)?;
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
        Ok(Self {
            device,
            queue,
            partial_pipeline,
            reduce_pipeline,
            bins,
            gradients,
            hessians,
            rows,
            partials: None,
            output: None,
            budget,
            peak: 0,
            dispatches: 0,
            profile: Profile::default(),
        })
    }
}
impl HistogramBackend for MetalHistograms {
    fn begin_iteration(&mut self, gradients: &[f32], hessians: &[f32]) -> Result<()> {
        write(&self.gradients, gradients)?;
        write(&self.hessians, hessians)
    }
    fn construct(
        &mut self,
        data: &BinnedData,
        rows: &[u32],
        gradients: &[f32],
        hessians: &[f32],
        held: usize,
    ) -> Result<Vec<Stat>> {
        let started = std::time::Instant::now();
        // Drain temporary command/encoder objects after their retained references have
        // completed; resident buffers and returned Rust statistics keep their own ownership.
        let result = objc2::rc::autoreleasepool(|_| {
            self.construct_inner(data, rows, gradients, hessians, held)
        });
        if result.is_ok() {
            self.profile.wall_seconds += started.elapsed().as_secs_f64();
            self.profile.calls += 1;
            self.profile.small_leaf_calls += u64::from(rows.len() <= 4096);
        }
        result
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
        self.partials.as_ref().map_or(0, |b| b.length())
            + self.output.as_ref().map_or(0, |b| b.length())
    }
}
impl MetalHistograms {
    fn construct_inner(
        &mut self,
        data: &BinnedData,
        rows: &[u32],
        _gradients: &[f32],
        _hessians: &[f32],
        held: usize,
    ) -> Result<Vec<Stat>> {
        let host_bytes = histogram_bytes(data)?;
        let width = data
            .maps
            .len()
            .checked_mul(data.stride)
            .filter(|&n| n <= u32::MAX as usize)
            .ok_or_else(|| Error::InvalidData("Metal histogram width exceeds u32".into()))?;
        let gpu_bytes = width
            .checked_mul(mem::size_of::<GpuStat>())
            .ok_or_else(|| Error::InvalidData("Metal histogram size overflow".into()))?;
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
        let max_partial = self.budget - held - host_bytes - gpu_bytes;
        let desired_chunks = rows.len().div_ceil(4096);
        let max_chunks = (max_partial / gpu_bytes).min(desired_chunks).max(1);
        let row_chunk = rows.len().div_ceil(max_chunks);
        let chunks = rows.len().div_ceil(row_chunk);
        let partial_bytes = gpu_bytes
            .checked_mul(chunks)
            .ok_or_else(|| Error::InvalidData("partial histogram size overflow".into()))?;
        if self
            .partials
            .as_ref()
            .is_some_and(|buffer| buffer.length() < partial_bytes || buffer.length() > max_partial)
        {
            self.partials = None;
        }
        if self.partials.is_none() {
            self.partials = Some(new_buffer(&self.device, partial_bytes)?);
        }
        if self.output.is_none() {
            self.output = Some(new_buffer(&self.device, gpu_bytes)?);
        }
        self.record_workspace(held + host_bytes)?;
        write(&self.rows, rows)?;
        let params = Params {
            rows: data.rows as u32,
            stride: data.stride as u32,
            features: data.maps.len() as u32,
            leaf_rows: rows.len() as u32,
            chunks: chunks as u32,
            row_chunk: row_chunk as u32,
        };
        let command = allocated(self.queue.commandBuffer())?;
        let encoder = allocated(command.computeCommandEncoder())?;
        encoder.setComputePipelineState(&self.partial_pipeline);
        let partials = required_buffer(&self.partials)?;
        let output = required_buffer(&self.output)?;
        // SAFETY: Offsets are zero, kernel accesses follow checked dimensions, buffers
        // outlive completion, and setBytes copies the initialized parameter structure.
        unsafe {
            for (index, buffer) in [
                &*self.bins,
                &*self.gradients,
                &*self.hessians,
                &*self.rows,
                partials,
            ]
            .into_iter()
            .enumerate()
            {
                encoder.setBuffer_offset_atIndex(Some(buffer), 0, index);
            }
            encoder.setBytes_length_atIndex(
                NonNull::from(&params).cast::<c_void>(),
                mem::size_of::<Params>(),
                5,
            );
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: data.maps.len(),
                height: chunks,
                depth: data.stride.div_ceil(16),
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
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
                width: width.div_ceil(32),
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
        self.dispatches += 2;
        // SAFETY: GPU timestamps are read only after successful completion.
        let gpu_elapsed = unsafe { command.GPUEndTime() - command.GPUStartTime() };
        if gpu_elapsed.is_finite() && gpu_elapsed >= 0.0 {
            self.profile.gpu_seconds += gpu_elapsed;
        }
        // SAFETY: Successful completion initialized `width` elements in shared storage;
        // the GPU is no longer writing, and the storage is aligned for GpuStat.
        let results = unsafe {
            std::slice::from_raw_parts(output.contents().as_ptr().cast::<GpuStat>(), width)
        };
        let mut histogram = Vec::with_capacity(width);
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
            let mut cpu = CpuHistograms {
                budget: 1024 * 1024,
                peak: 0,
                profile: Profile::default(),
            };
            let mut metal = MetalHistograms::new(&data, 1024 * 1024).unwrap();
            metal.begin_iteration(&gradients, &hessians).unwrap();
            for membership in [&rows[..], &rows[..13], &rows[..]] {
                let expected = cpu
                    .construct(&data, membership, &gradients, &hessians, 0)
                    .unwrap();
                let actual = metal
                    .construct(&data, membership, &gradients, &hessians, 0)
                    .unwrap();
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
            assert_eq!(metal.dispatches(), 6);
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
}
