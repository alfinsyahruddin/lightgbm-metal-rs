#include <metal_stdlib>
using namespace metal;
struct Stat { float gradient; float hessian; uint count; };
struct Params { uint rows, stride, features, leaf_rows, chunks, row_chunk; };

// One SIMD group per feature, row chunk, and 16-bin tile. Lanes own their sums.
kernel void partial_histograms(device const uchar* bins [[buffer(0)]],
                               device const float* gradients [[buffer(1)]],
                               device const float* hessians [[buffer(2)]],
                               device const uint* row_indices [[buffer(3)]],
                               device Stat* partials [[buffer(4)]],
                               constant Params& p [[buffer(5)]],
                               uint3 group [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_threadgroup]]) {
    float g[16] = {}, h[16] = {};
    uint counts[16] = {};
    uint feature = group.x, chunk = group.y, bin_start = group.z * 16;
    ulong start = ulong(chunk) * p.row_chunk;
    ulong end = start + p.row_chunk;
    if (end > p.leaf_rows) end = p.leaf_rows;
    for (ulong i = start + lane; i < end; i += 32) {
        uint row = row_indices[i];
        uint bin = bins[ulong(feature) * p.rows + row];
        if (bin >= bin_start && bin < bin_start + 16) {
            uint local = bin - bin_start;
            g[local] += gradients[row];
            h[local] += hessians[row];
            counts[local] += 1;
        }
    }
    for (uint b = 0; b < 16; ++b) {
        float sum_g = simd_sum(g[b]), sum_h = simd_sum(h[b]);
        uint sum_count = simd_sum(counts[b]);
        if (lane == 0 && bin_start + b < p.stride) {
            ulong index = (ulong(chunk) * p.features + feature) * p.stride + bin_start + b;
            partials[index] = {sum_g, sum_h, sum_count};
        }
    }
}

kernel void reduce_histograms(device const Stat* partials [[buffer(0)]],
                              device Stat* output [[buffer(1)]],
                              constant Params& p [[buffer(2)]],
                              uint index [[thread_position_in_grid]]) {
    uint width = p.features * p.stride;
    if (index >= width) return;
    float g = 0, h = 0;
    uint count = 0;
    for (uint chunk = 0; chunk < p.chunks; ++chunk) {
        Stat s = partials[ulong(chunk) * width + index];
        g += s.gradient;
        h += s.hessian;
        count += s.count;
    }
    output[index] = {g, h, count};
}
