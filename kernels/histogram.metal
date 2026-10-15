#include <metal_stdlib>
using namespace metal;
struct Stat { float gradient; float hessian; uint count; };
struct Params { uint rows, stride, features, leaf_rows, chunks, row_chunk; };

// Each threadgroup handles one feature and one row chunk.
// Up to 4 SIMD groups (128 threads) accumulate bins in parallel (64 bins per SIMD group),
// covering up to 256 bins in a single pass without repeated tile scans.
kernel void partial_histograms(device const uchar* bins [[buffer(0)]],
                               device const float* gradients [[buffer(1)]],
                               device const float* hessians [[buffer(2)]],
                               device const uint* row_indices [[buffer(3)]],
                               device Stat* partials [[buffer(4)]],
                               constant Params& p [[buffer(5)]],
                               device const uint* splittable_features [[buffer(6)]],
                               uint2 group [[threadgroup_position_in_grid]],
                               uint simd_id [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
    float g[64] = {}, h[64] = {};
    uint counts[64] = {};
    uint feature = splittable_features[group.x];
    uint chunk = group.y;
    uint bin_start = simd_id * 64;
    if (bin_start >= p.stride) return;
    ulong start = ulong(chunk) * p.row_chunk;
    ulong end = start + p.row_chunk;
    if (end > p.leaf_rows) end = p.leaf_rows;
    for (ulong i = start + lane; i < end; i += 32) {
        uint row = row_indices[i];
        uint bin = bins[ulong(feature) * p.rows + row];
        if (bin >= bin_start && bin < bin_start + 64) {
            uint local = bin - bin_start;
            g[local] += gradients[row];
            h[local] += hessians[row];
            counts[local] += 1;
        }
    }
    for (uint b = 0; b < 64; ++b) {
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

kernel void subtract_histograms(device const Stat* parent [[buffer(0)]],
                                device const Stat* smaller [[buffer(1)]],
                                device Stat* sibling [[buffer(2)]],
                                device atomic_uint* cancel_flag [[buffer(3)]],
                                constant Params& p [[buffer(4)]],
                                uint index [[thread_position_in_grid]]) {
    uint width = p.features * p.stride;
    if (index >= width) return;
    Stat p_stat = parent[index];
    Stat s_stat = smaller[index];
    if (s_stat.count > p_stat.count) {
        atomic_store_explicit(cancel_flag, 1, memory_order_relaxed);
        return;
    }
    float g = p_stat.gradient - s_stat.gradient;
    float h = p_stat.hessian - s_stat.hessian;
    uint c = p_stat.count - s_stat.count;
    if (c == 0) {
        sibling[index] = {0.0f, 0.0f, 0};
        return;
    }
    float grad_scale = abs(p_stat.gradient) + abs(s_stat.gradient);
    float hess_scale = abs(p_stat.hessian) + abs(s_stat.hessian);
    if (!isnan(g) && !isnan(h) && (h > 1e-4f * hess_scale) && (grad_scale <= 1e-12f || abs(g) > 1e-4f * grad_scale)) {
        sibling[index] = {g, h, c};
    } else {
        atomic_store_explicit(cancel_flag, 1, memory_order_relaxed);
    }
}

struct FeatureMeta {
    uint bounds_len;
    uint missing;
    uint splittable;
    uint _pad;
};

struct SplitParams {
    uint stride;
    uint num_features;
    uint min_data_in_leaf;
    uint total_count;
    float total_gradient;
    float total_hessian;
    float parent_gain;
    float lambda_l1;
    float lambda_l2;
    float min_sum_hessian_in_leaf;
    float min_gain_to_split;
};

struct SplitCandidate {
    float gain;
    float left_gradient;
    float left_hessian;
    uint left_count;
    uint threshold_bin;
    uint missing_left;
    uint valid;
    uint feature;
};

inline float soft_threshold(float gradient, float penalty) {
    if (gradient > penalty) {
        return gradient - penalty;
    } else if (gradient < -penalty) {
        return gradient + penalty;
    } else {
        return 0.0f;
    }
}

inline float calc_gain(float gradient, float hessian, float lambda_l1, float lambda_l2) {
    float g = soft_threshold(gradient, lambda_l1);
    return (g * g) / (hessian + lambda_l2 + 1e-12f);
}

kernel void find_best_splits(device const Stat* histogram [[buffer(0)]],
                             device const FeatureMeta* meta [[buffer(1)]],
                             constant SplitParams& p [[buffer(2)]],
                             device SplitCandidate* candidates [[buffer(3)]],
                             uint feature [[thread_position_in_grid]]) {
    if (feature >= p.num_features) return;
    FeatureMeta f_meta = meta[feature];
    if (f_meta.splittable == 0) {
        candidates[feature] = {0.0f, 0.0f, 0.0f, 0, 0, 0, 0, feature};
        return;
    }
    uint bounds_len = f_meta.bounds_len;
    bool has_missing = f_meta.missing != 0;
    uint numeric_bins = bounds_len - (has_missing ? 1 : 0);
    uint thresholds = has_missing ? numeric_bins : (numeric_bins > 0 ? numeric_bins - 1 : 0);
    ulong feat_offset = ulong(feature) * p.stride;
    Stat missing = has_missing ? histogram[feat_offset + numeric_bins] : Stat{0.0f, 0.0f, 0};

    SplitCandidate best;
    best.gain = p.min_gain_to_split;
    best.left_gradient = 0.0f;
    best.left_hessian = 0.0f;
    best.left_count = 0;
    best.threshold_bin = 0;
    best.missing_left = 0;
    best.valid = 0;
    best.feature = feature;

    for (int m_left = 1; m_left >= 0; --m_left) {
        if (!has_missing && m_left == 0) continue;
        Stat cumulative = {0.0f, 0.0f, 0};
        for (uint b = 0; b < thresholds; ++b) {
            Stat s = histogram[feat_offset + b];
            cumulative.gradient += s.gradient;
            cumulative.hessian += s.hessian;
            cumulative.count += s.count;
        }
        for (int tb = int(thresholds) - 1; tb >= 0; --tb) {
            Stat left = cumulative;
            if (m_left == 1) {
                left.gradient += missing.gradient;
                left.hessian += missing.hessian;
                left.count += missing.count;
            }
            Stat cur_bin = histogram[feat_offset + tb];
            cumulative.gradient -= cur_bin.gradient;
            cumulative.hessian -= cur_bin.hessian;
            cumulative.count -= cur_bin.count;

            if (left.count > p.total_count) continue;
            uint right_count = p.total_count - left.count;
            float right_hessian = p.total_hessian - left.hessian;
            float right_gradient = p.total_gradient - left.gradient;

            if (left.count < p.min_data_in_leaf || right_count < p.min_data_in_leaf) continue;
            if (left.hessian < p.min_sum_hessian_in_leaf || right_hessian < p.min_sum_hessian_in_leaf) continue;

            float gain_left = calc_gain(left.gradient, left.hessian, p.lambda_l1, p.lambda_l2);
            float gain_right = calc_gain(right_gradient, right_hessian, p.lambda_l1, p.lambda_l2);
            float split_gain = gain_left + gain_right - p.parent_gain;

            if (isnan(split_gain) || isinf(split_gain)) continue;

            if (split_gain > best.gain) {
                best.gain = split_gain;
                best.threshold_bin = tb;
                best.missing_left = m_left;
                best.left_count = left.count;
                best.left_gradient = left.gradient;
                best.left_hessian = left.hessian;
                best.valid = 1;
            }
        }
    }
    candidates[feature] = best;
}

struct PartitionParams {
    uint num_rows;
    uint leaf_rows;
    uint feature;
    uint threshold_bin;
    uint missing_bin;
    uint has_missing;
    uint default_left;
    uint total_left_count;
};

kernel void init_row_indices(device uint* rows [[buffer(0)]],
                             constant uint& num_rows [[buffer(1)]],
                             uint id [[thread_position_in_grid]]) {
    if (id < num_rows) {
        rows[id] = id;
    }
}

kernel void partition_single_block(device const uchar* bins [[buffer(0)]],
                                   device const uint* parent_rows [[buffer(1)]],
                                   device uint* output_rows [[buffer(2)]],
                                   constant PartitionParams& p [[buffer(3)]],
                                   uint tid [[thread_position_in_threadgroup]],
                                   uint simd_id [[simdgroup_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint simd_totals[32];
    threadgroup uint simd_offsets[32];

    if (simd_id == 0) {
        simd_totals[lane] = 0;
        simd_offsets[lane] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint pred = 0;
    uint row = 0;
    if (tid < p.leaf_rows) {
        row = parent_rows[tid];
        uint bin = bins[ulong(p.feature) * p.num_rows + row];
        bool is_missing = (p.has_missing != 0) && (bin == p.missing_bin);
        bool goes_left = is_missing ? (p.default_left != 0) : (bin <= p.threshold_bin);
        pred = goes_left ? 1 : 0;
    }

    uint simd_left = simd_prefix_exclusive_sum(pred);
    if (lane == 31) {
        simd_totals[simd_id] = simd_left + pred;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        uint num_simds = (p.leaf_rows + 31) / 32;
        uint s_val = (lane < num_simds) ? simd_totals[lane] : 0;
        uint s_off = simd_prefix_exclusive_sum(s_val);
        simd_offsets[lane] = s_off;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < p.leaf_rows) {
        uint local_left = simd_offsets[simd_id] + simd_left;
        uint local_right = tid - local_left;
        if (pred == 1) {
            output_rows[local_left] = row;
        } else {
            output_rows[p.total_left_count + local_right] = row;
        }
    }
}

kernel void partition_count_blocks(device const uchar* bins [[buffer(0)]],
                                   device const uint* parent_rows [[buffer(1)]],
                                   device uint* block_counts [[buffer(2)]],
                                   constant PartitionParams& p [[buffer(3)]],
                                   uint gid [[thread_position_in_grid]],
                                   uint group_id [[threadgroup_position_in_grid]],
                                   uint simd_id [[simdgroup_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint simd_totals[8];
    uint pred = 0;
    if (gid < p.leaf_rows) {
        uint row = parent_rows[gid];
        uint bin = bins[ulong(p.feature) * p.num_rows + row];
        bool is_missing = (p.has_missing != 0) && (bin == p.missing_bin);
        bool goes_left = is_missing ? (p.default_left != 0) : (bin <= p.threshold_bin);
        pred = goes_left ? 1 : 0;
    }
    uint simd_sum_val = simd_sum(pred);
    if (lane == 0) {
        simd_totals[simd_id] = simd_sum_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0 && lane == 0) {
        uint block_sum = 0;
        for (uint i = 0; i < 8; ++i) {
            block_sum += simd_totals[i];
        }
        block_counts[group_id] = block_sum;
    }
}

kernel void partition_prefix_blocks(device const uint* block_counts [[buffer(0)]],
                                    device uint* block_offsets [[buffer(1)]],
                                    constant uint& num_blocks [[buffer(2)]],
                                    uint tid [[thread_position_in_threadgroup]]) {
    if (tid == 0) {
        uint running = 0;
        for (uint i = 0; i < num_blocks; ++i) {
            block_offsets[i] = running;
            running += block_counts[i];
        }
    }
}

kernel void partition_scatter_blocks(device const uchar* bins [[buffer(0)]],
                                     device const uint* parent_rows [[buffer(1)]],
                                     device const uint* block_offsets [[buffer(2)]],
                                     device uint* output_rows [[buffer(3)]],
                                     constant PartitionParams& p [[buffer(4)]],
                                     uint gid [[thread_position_in_grid]],
                                     uint group_id [[threadgroup_position_in_grid]],
                                     uint simd_id [[simdgroup_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint simd_totals[8];
    threadgroup uint simd_offsets[8];

    uint pred = 0;
    uint row = 0;
    if (gid < p.leaf_rows) {
        row = parent_rows[gid];
        uint bin = bins[ulong(p.feature) * p.num_rows + row];
        bool is_missing = (p.has_missing != 0) && (bin == p.missing_bin);
        bool goes_left = is_missing ? (p.default_left != 0) : (bin <= p.threshold_bin);
        pred = goes_left ? 1 : 0;
    }

    uint simd_left = simd_prefix_exclusive_sum(pred);
    if (lane == 31) {
        simd_totals[simd_id] = simd_left + pred;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        uint s_val = (lane < 8) ? simd_totals[lane] : 0;
        uint s_off = simd_prefix_exclusive_sum(s_val);
        if (lane < 8) {
            simd_offsets[lane] = s_off;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (gid < p.leaf_rows) {
        uint block_left_base = block_offsets[group_id];
        uint local_left = simd_offsets[simd_id] + simd_left;
        uint global_left = block_left_base + local_left;
        uint global_right = gid - global_left;
        if (pred == 1) {
            output_rows[global_left] = row;
        } else {
            output_rows[p.total_left_count + global_right] = row;
        }
    }
}

struct ObjectiveParams {
    uint num_rows;
    uint objective; // 0: RegressionL2, 1: BinaryLogloss
};

struct LeafOutput {
    uint offset;
    uint count;
    float value;
    uint buf_idx;
};

kernel void init_scores_and_derivatives(device float* scores [[buffer(0)]],
                                       device float* score_errs [[buffer(1)]],
                                       device const float* labels [[buffer(2)]],
                                       device float* gradients [[buffer(3)]],
                                       device float* hessians [[buffer(4)]],
                                       device float* block_totals [[buffer(5)]],
                                       device atomic_uint* error_flag [[buffer(6)]],
                                       constant ObjectiveParams& p [[buffer(7)]],
                                       constant float& init_score [[buffer(8)]],
                                       constant float& init_err [[buffer(9)]],
                                       uint gid [[thread_position_in_grid]],
                                       uint group_id [[threadgroup_position_in_grid]],
                                       uint simd_id [[simdgroup_index_in_threadgroup]],
                                       uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float simd_grads[8];
    threadgroup float simd_hesses[8];

    float g = 0.0f;
    float h = 0.0f;
    if (gid < p.num_rows) {
        scores[gid] = init_score;
        score_errs[gid] = init_err;
        float label = labels[gid];
        if (p.objective == 0) { // RegressionL2
            g = init_score - label;
            h = 1.0f;
        } else { // BinaryLogloss
            if (init_score >= 0.0f) {
                float e = precise::exp(-init_score);
                float s = 1.0f / (1.0f + e);
                float s_neg = e / (1.0f + e);
                g = (label == 1.0f) ? -s_neg : s;
            } else {
                float e = precise::exp(init_score);
                float s = e / (1.0f + e);
                float s_neg = 1.0f / (1.0f + e);
                g = (label == 1.0f) ? -s_neg : s;
            }
            float mag = abs(g);
            h = mag * (1.0f - mag);
        }
        if (!isfinite(g) || !isfinite(h) || !isfinite(init_score)) {
            atomic_store_explicit(error_flag, 1, memory_order_relaxed);
        }
        gradients[gid] = g;
        hessians[gid] = h;
    }

    float sum_g = simd_sum(g);
    float sum_h = simd_sum(h);
    if (lane == 0) {
        simd_grads[simd_id] = sum_g;
        simd_hesses[simd_id] = sum_h;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0 && lane == 0) {
        float group_g = 0.0f;
        float group_h = 0.0f;
        for (uint i = 0; i < 8; ++i) {
            group_g += simd_grads[i];
            group_h += simd_hesses[i];
        }
        block_totals[group_id * 2] = group_g;
        block_totals[group_id * 2 + 1] = group_h;
    }
}

kernel void update_leaf_scores(device float* scores [[buffer(0)]],
                               device float* score_errs [[buffer(1)]],
                               device const uint* rows [[buffer(2)]],
                               constant LeafOutput& leaf [[buffer(3)]],
                               uint tid [[thread_position_in_grid]]) {
    if (tid < leaf.count) {
        uint row = rows[leaf.offset + tid];
        float delta = leaf.value;
        float y = delta - score_errs[row];
        float t = scores[row] + y;
        score_errs[row] = (t - scores[row]) - y;
        scores[row] = t;
    }
}

kernel void compute_derivatives(device const float* scores [[buffer(0)]],
                                device const float* score_errs [[buffer(1)]],
                                device const float* labels [[buffer(2)]],
                                device float* gradients [[buffer(3)]],
                                device float* hessians [[buffer(4)]],
                                device float* block_totals [[buffer(5)]],
                                device atomic_uint* error_flag [[buffer(6)]],
                                constant ObjectiveParams& p [[buffer(7)]],
                                uint gid [[thread_position_in_grid]],
                                uint group_id [[threadgroup_position_in_grid]],
                                uint simd_id [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float simd_grads[8];
    threadgroup float simd_hesses[8];

    float g = 0.0f;
    float h = 0.0f;
    if (gid < p.num_rows) {
        float score = scores[gid] - score_errs[gid];
        float label = labels[gid];
        if (p.objective == 0) { // RegressionL2
            g = score - label;
            h = 1.0f;
        } else { // BinaryLogloss
            if (score >= 0.0f) {
                float e = precise::exp(-score);
                float s = 1.0f / (1.0f + e);
                float s_neg = e / (1.0f + e);
                g = (label == 1.0f) ? -s_neg : s;
            } else {
                float e = precise::exp(score);
                float s = e / (1.0f + e);
                float s_neg = 1.0f / (1.0f + e);
                g = (label == 1.0f) ? -s_neg : s;
            }
            float mag = abs(g);
            h = mag * (1.0f - mag);
        }
        if (!isfinite(g) || !isfinite(h) || !isfinite(score)) {
            atomic_store_explicit(error_flag, 1, memory_order_relaxed);
        }
        gradients[gid] = g;
        hessians[gid] = h;
    }

    float sum_g = simd_sum(g);
    float sum_h = simd_sum(h);
    if (lane == 0) {
        simd_grads[simd_id] = sum_g;
        simd_hesses[simd_id] = sum_h;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0 && lane == 0) {
        float group_g = 0.0f;
        float group_h = 0.0f;
        for (uint i = 0; i < 8; ++i) {
            group_g += simd_grads[i];
            group_h += simd_hesses[i];
        }
        block_totals[group_id * 2] = group_g;
        block_totals[group_id * 2 + 1] = group_h;
    }
}
