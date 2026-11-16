//! Four-column Q4K storage with the original block size and canonical GGUF bytes.
//!
//! Nibbles are rearranged within each scale group to share activation loads across columns.
//! The packed layout is used for eligible AArch64 dotprod linears; raw data is deinterleaved
//! before serialization, and unsupported shapes and platforms keep the existing kernels.
//! I8MM builds pair activation rows for prefill; single-row decode retains the SDOT tile.

use super::k_quants::{BlockQ4K, GgmlType, K_SCALE_SIZE, QK_K};
use super::{GgmlDType, QStorage, QuantizedType};
use crate::Result;
use half::f16;
use std::borrow::Cow;

#[cfg(all(target_arch = "aarch64", target_feature = "neon", target_endian = "little"))]
use super::k_quants::BlockQ8K;
#[cfg(all(target_arch = "aarch64", target_feature = "neon", target_endian = "little"))]
use core::arch::aarch64::*;

const COLUMNS: usize = 4;
const RUN: usize = 8;

/// Each 32-value scale group holds two eight-byte runs per column. Its nibbles encode the
/// first and second 16-value halves, so one load serves both halves with the same scale.
#[derive(Clone)]
#[repr(C)]
struct BlockQ4Kx4 {
    scales: [f16; COLUMNS],
    minimums: [f16; COLUMNS],
    metadata: [[u8; COLUMNS]; K_SCALE_SIZE],
    quants: [u8; QK_K / 2 * COLUMNS],
}

const _: () =
    assert!(std::mem::size_of::<BlockQ4Kx4>() == COLUMNS * std::mem::size_of::<BlockQ4K>());

impl BlockQ4Kx4 {
    fn interleave(source: [&BlockQ4K; COLUMNS]) -> Self {
        let mut packed = Self {
            scales: [f16::ZERO; COLUMNS],
            minimums: [f16::ZERO; COLUMNS],
            metadata: [[0; COLUMNS]; K_SCALE_SIZE],
            quants: [0; QK_K / 2 * COLUMNS],
        };
        for (column, block) in source.iter().enumerate() {
            packed.scales[column] = block.d;
            packed.minimums[column] = block.dmin;
            for (index, byte) in block.scales.iter().enumerate() {
                packed.metadata[index][column] = *byte;
            }
            for group in 0..QK_K / 32 {
                let shift = group % 2 * 4;
                for segment in 0..2 {
                    let input = group / 2 * 32 + segment * RUN;
                    let output = ((group * 2 + segment) * COLUMNS + column) * RUN;
                    for offset in 0..RUN {
                        let low = (block.qs[input + offset] >> shift) & 15;
                        let high = (block.qs[input + offset + 16] >> shift) & 15;
                        packed.quants[output + offset] = low | (high << 4);
                    }
                }
            }
        }
        packed
    }

    fn deinterleave(&self, destination: &mut [BlockQ4K; COLUMNS]) {
        for (column, block) in destination.iter_mut().enumerate() {
            block.d = self.scales[column];
            block.dmin = self.minimums[column];
            for (index, byte) in block.scales.iter_mut().enumerate() {
                *byte = self.metadata[index][column];
            }
            block.qs.fill(0);
            for group in 0..QK_K / 32 {
                let shift = group % 2 * 4;
                for segment in 0..2 {
                    let input = ((group * 2 + segment) * COLUMNS + column) * RUN;
                    let output = group / 2 * 32 + segment * RUN;
                    for offset in 0..RUN {
                        let byte = self.quants[input + offset];
                        block.qs[output + offset] |= (byte & 15) << shift;
                        block.qs[output + offset + 16] |= (byte >> 4) << shift;
                    }
                }
            }
        }
    }
}

pub struct Q4Kx4 {
    blocks: Vec<BlockQ4Kx4>,
    columns: usize,
    depth_blocks: usize,
}

pub fn is_eligible(dimensions: &[usize]) -> bool {
    matches!(dimensions, [columns, depth]
        if *columns > 0 && *depth > 0 && columns % COLUMNS == 0 && depth % QK_K == 0)
}

impl Q4Kx4 {
    pub fn from_q4k(source: &[BlockQ4K], columns: usize, depth: usize) -> Result<Self> {
        if !is_eligible(&[columns, depth]) {
            crate::bail!("shape [{columns}, {depth}] cannot be interleaved as q4k_x4")
        }
        let depth_blocks = depth / QK_K;
        let block_count = columns
            .checked_mul(depth_blocks)
            .ok_or_else(|| crate::Error::msg("q4k_x4: weight dimensions overflow"))?;
        if source.len() != block_count {
            crate::bail!("q4k_x4: expected {block_count} blocks, got {}", source.len())
        }
        let mut blocks = Vec::with_capacity(columns / COLUMNS * depth_blocks);
        for column in (0..columns).step_by(COLUMNS) {
            for depth_index in 0..depth_blocks {
                blocks.push(BlockQ4Kx4::interleave(std::array::from_fn(|offset| {
                    &source[(column + offset) * depth_blocks + depth_index]
                })));
            }
        }
        Ok(Self { blocks, columns, depth_blocks })
    }

    pub fn unpack(&self) -> Vec<BlockQ4K> {
        let mut output = vec![BlockQ4K::zeros(); self.columns * self.depth_blocks];
        let mut temporary = std::array::from_fn(|_| BlockQ4K::zeros());
        for group in 0..self.columns / COLUMNS {
            for depth_index in 0..self.depth_blocks {
                self.blocks[group * self.depth_blocks + depth_index].deinterleave(&mut temporary);
                for (column, block) in temporary.iter().enumerate() {
                    output[(group * COLUMNS + column) * self.depth_blocks + depth_index] =
                        block.clone();
                }
            }
        }
        output
    }
}

impl QuantizedType for Q4Kx4 {
    fn dtype(&self) -> GgmlDType {
        GgmlDType::Q4K
    }

    fn block_size(&self) -> usize {
        QK_K
    }

    fn size(&self) -> usize {
        self.blocks.len() * std::mem::size_of::<BlockQ4Kx4>()
    }

    fn storage_size_in_bytes(&self) -> usize {
        self.size()
    }

    fn as_ptr(&self) -> *const u8 {
        self.blocks.as_ptr().cast()
    }

    fn raw_data(&self) -> Result<Cow<'_, [u8]>> {
        let plain = self.unpack();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                plain.as_ptr().cast::<u8>(),
                std::mem::size_of_val(plain.as_slice()),
            )
        };
        Ok(Cow::Owned(bytes.to_vec()))
    }

    fn dequantize(&self, element_count: usize) -> Result<Vec<f32>> {
        let mut values = vec![0.0; element_count];
        BlockQ4K::to_float(&self.unpack(), &mut values)?;
        Ok(values)
    }

    fn from_float(&mut self, values: &[f32]) -> Result<()> {
        let mut plain = vec![BlockQ4K::zeros(); self.columns * self.depth_blocks];
        BlockQ4K::from_float(values, &mut plain)?;
        *self = Self::from_q4k(&plain, self.columns, self.depth_blocks * QK_K)?;
        Ok(())
    }

    fn matmul_t(&self, shape: (usize, usize, usize), lhs: &[f32], dst: &mut [f32]) -> Result<()> {
        let (rows, depth, columns) = shape;
        if columns != self.columns || depth != self.depth_blocks * QK_K {
            crate::bail!("q4k_x4: weight shape mismatch for {shape:?}")
        }
        let input_count = rows
            .checked_mul(depth)
            .ok_or_else(|| crate::Error::msg("q4k_x4: input dimensions overflow"))?;
        let output_count = rows
            .checked_mul(columns)
            .ok_or_else(|| crate::Error::msg("q4k_x4: output dimensions overflow"))?;
        if lhs.len() != input_count || dst.len() < output_count {
            crate::bail!("q4k_x4: invalid buffers for {shape:?}")
        }
        if rows == 0 {
            return Ok(());
        }
        #[cfg(all(target_arch = "aarch64", target_feature = "neon", target_endian = "little"))]
        {
            let activations = quantize_lhs(rows, depth, lhs)?;
            matmul_interleaved(&self.blocks, &activations, dst, rows, columns, self.depth_blocks);
            Ok(())
        }
        #[cfg(not(all(
            target_arch = "aarch64",
            target_feature = "neon",
            target_endian = "little"
        )))]
        super::k_quants::matmul(shape, lhs, &self.unpack(), dst)
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon", target_endian = "little"))]
#[tracing::instrument(name = "q4k-x4-quantize-lhs", skip_all, fields(rows, depth))]
fn quantize_lhs(rows: usize, depth: usize, values: &[f32]) -> Result<Vec<BlockQ8K>> {
    let blocks = depth / QK_K;
    let mut output = vec![BlockQ8K::zeros(); rows * blocks];
    for row in 0..rows {
        BlockQ8K::from_float(
            &values[row * depth..(row + 1) * depth],
            &mut output[row * blocks..(row + 1) * blocks],
        )?;
    }
    Ok(output)
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon", target_endian = "little"))]
#[tracing::instrument(name = "q4k-x4-matmul", skip_all, fields(rows, columns, depth = depth_blocks * QK_K))]
fn matmul_interleaved(
    weights: &[BlockQ4Kx4],
    lhs: &[BlockQ8K],
    dst: &mut [f32],
    rows: usize,
    columns: usize,
    depth_blocks: usize,
) {
    let job = Job {
        weights: weights.as_ptr() as usize,
        activations: lhs.as_ptr() as usize,
        output: dst.as_mut_ptr() as usize,
        rows,
        columns,
        depth_blocks,
        #[cfg(target_feature = "i8mm")]
        i8mm: i8mm_enabled(),
    };
    let groups = columns / COLUMNS;
    if groups <= 4 {
        unsafe { job.run(0, groups) };
        return;
    }
    crate::threadpool::dispatch(|worker, workers| {
        let count = groups.div_ceil(workers);
        let first = (worker * count).min(groups);
        let last = (first + count).min(groups);
        if first < last {
            unsafe { job.run(first, last) };
        }
    });
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon", target_endian = "little"))]
struct Job {
    weights: usize,
    activations: usize,
    output: usize,
    rows: usize,
    columns: usize,
    depth_blocks: usize,
    #[cfg(target_feature = "i8mm")]
    i8mm: bool,
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon", target_endian = "little"))]
impl Job {
    /// # Safety
    /// Buffers must cover the configured matrix dimensions, and concurrent calls must own
    /// disjoint four-column group ranges. The caller keeps all three buffers alive.
    unsafe fn run(&self, first: usize, last: usize) {
        let weights = self.weights as *const BlockQ4Kx4;
        let activations = self.activations as *const BlockQ8K;
        let output = self.output as *mut f32;
        for group in first..last {
            let weight = unsafe { weights.add(group * self.depth_blocks) };
            let column = group * COLUMNS;
            let mut row = 0;
            while row + 4 <= self.rows {
                #[cfg(target_feature = "i8mm")]
                if self.i8mm {
                    unsafe {
                        tile_i8mm::<4>(
                            weight,
                            activations.add(row * self.depth_blocks),
                            self.depth_blocks,
                            output.add(row * self.columns + column),
                            self.columns,
                        );
                    }
                    row += 4;
                    continue;
                }
                unsafe {
                    tile::<4>(
                        weight,
                        activations.add(row * self.depth_blocks),
                        self.depth_blocks,
                        output.add(row * self.columns + column),
                        self.columns,
                    );
                }
                row += 4;
            }
            #[cfg(target_feature = "i8mm")]
            if self.i8mm && row + 2 <= self.rows {
                unsafe {
                    tile_i8mm::<2>(
                        weight,
                        activations.add(row * self.depth_blocks),
                        self.depth_blocks,
                        output.add(row * self.columns + column),
                        self.columns,
                    );
                }
                row += 2;
            }
            while row < self.rows {
                unsafe {
                    tile::<1>(
                        weight,
                        activations.add(row * self.depth_blocks),
                        self.depth_blocks,
                        output.add(row * self.columns + column),
                        self.columns,
                    );
                }
                row += 1;
            }
        }
    }
}

#[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon",
    target_feature = "i8mm",
    target_endian = "little"
))]
/// `XN_Q4K_I8MM=0` keeps the packed SDOT path for prefill comparisons.
fn i8mm_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("XN_Q4K_I8MM").as_deref() != Ok("0"))
}

#[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon",
    target_feature = "i8mm",
    target_endian = "little"
))]
#[inline(always)]
unsafe fn smmla(accumulator: int32x4_t, activations: int8x16_t, weights: int8x16_t) -> int32x4_t {
    let mut accumulator = accumulator;
    unsafe {
        core::arch::asm!(
            "smmla {accumulator:v}.4s, {activations:v}.16b, {weights:v}.16b",
            accumulator = inout(vreg) accumulator,
            activations = in(vreg) activations,
            weights = in(vreg) weights,
            options(pure, nomem, nostack),
        );
    }
    accumulator
}

#[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon",
    target_feature = "i8mm",
    target_endian = "little"
))]
/// # Safety
/// `ROWS` must be two or four. Weights must contain `depth_blocks` packed blocks, activations
/// must contain `ROWS` rows at that block stride, and output must cover four columns per row.
unsafe fn tile_i8mm<const ROWS: usize>(
    weights: *const BlockQ4Kx4,
    activations: *const BlockQ8K,
    depth_blocks: usize,
    output: *mut f32,
    output_stride: usize,
) {
    unsafe {
        let zero = vdupq_n_s32(0);
        let mut totals = [vdupq_n_f32(0.0); ROWS];
        let nibble_mask = vdupq_n_u8(15);
        for block_index in 0..depth_blocks {
            let block = &*weights.add(block_index);
            let mut integer_sums = [zero; ROWS];
            let mut minimum_sums = [zero; ROWS];
            for group in 0..QK_K / 32 {
                let (scales, minimums) = scales_and_minimums(block, group);
                let mut first_pair = [zero; ROWS];
                let mut second_pair = [zero; ROWS];
                for segment in 0..2 {
                    let offset = (group * 2 + segment) * RUN * COLUMNS;
                    let first_bits = vld1q_u8(block.quants.as_ptr().add(offset));
                    let second_bits = vld1q_u8(block.quants.as_ptr().add(offset + 16));
                    let first_low = vreinterpretq_s8_u8(vandq_u8(first_bits, nibble_mask));
                    let second_low = vreinterpretq_s8_u8(vandq_u8(second_bits, nibble_mask));
                    let first_high = vreinterpretq_s8_u8(vshrq_n_u8(first_bits, 4));
                    let second_high = vreinterpretq_s8_u8(vshrq_n_u8(second_bits, 4));
                    for pair in 0..ROWS / 2 {
                        let first_activation =
                            &*activations.add(pair * 2 * depth_blocks + block_index);
                        let second_activation =
                            &*activations.add((pair * 2 + 1) * depth_blocks + block_index);
                        let start = group * 32 + segment * RUN;
                        let low = vcombine_s8(
                            vld1_s8(first_activation.qs.as_ptr().add(start)),
                            vld1_s8(second_activation.qs.as_ptr().add(start)),
                        );
                        let high = vcombine_s8(
                            vld1_s8(first_activation.qs.as_ptr().add(start + 16)),
                            vld1_s8(second_activation.qs.as_ptr().add(start + 16)),
                        );
                        first_pair[pair] = smmla(first_pair[pair], low, first_low);
                        second_pair[pair] = smmla(second_pair[pair], low, second_low);
                        first_pair[pair] = smmla(first_pair[pair], high, first_high);
                        second_pair[pair] = smmla(second_pair[pair], high, second_high);
                    }
                }
                for pair in 0..ROWS / 2 {
                    let first_row = vcombine_s32(
                        vget_low_s32(first_pair[pair]),
                        vget_low_s32(second_pair[pair]),
                    );
                    let second_row = vcombine_s32(
                        vget_high_s32(first_pair[pair]),
                        vget_high_s32(second_pair[pair]),
                    );
                    integer_sums[pair * 2] = vmlaq_s32(integer_sums[pair * 2], first_row, scales);
                    integer_sums[pair * 2 + 1] =
                        vmlaq_s32(integer_sums[pair * 2 + 1], second_row, scales);
                }
                for row in 0..ROWS {
                    let activation = &*activations.add(row * depth_blocks + block_index);
                    let block_sum = i32::from(activation.bsums[group * 2])
                        + i32::from(activation.bsums[group * 2 + 1]);
                    minimum_sums[row] = vmlaq_n_s32(minimum_sums[row], minimums, block_sum);
                }
            }
            let scales = load_f16x4(block.scales.as_ptr());
            let minimums = load_f16x4(block.minimums.as_ptr());
            for row in 0..ROWS {
                let activation = &*activations.add(row * depth_blocks + block_index);
                let activation_scale = vdupq_n_f32(activation.d);
                let scale = vmulq_f32(activation_scale, scales);
                let minimum = vmulq_f32(activation_scale, minimums);
                totals[row] =
                    vsubq_f32(totals[row], vmulq_f32(minimum, vcvtq_f32_s32(minimum_sums[row])));
                totals[row] =
                    vaddq_f32(totals[row], vmulq_f32(scale, vcvtq_f32_s32(integer_sums[row])));
            }
        }
        for (row, total) in totals.iter().enumerate() {
            vst1q_f32(output.add(row * output_stride), *total);
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon", target_endian = "little"))]
#[inline(always)]
unsafe fn load_f16x4(values: *const f16) -> float32x4_t {
    let packed = unsafe { vld1_u16(values.cast()) };
    let output: float32x4_t;
    unsafe {
        core::arch::asm!(
            "fcvtl {output:v}.4s, {input:v}.4h",
            output = out(vreg) output,
            input = in(vreg) packed,
            options(pure, nomem, nostack),
        );
    }
    output
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon", target_endian = "little"))]
#[inline(always)]
unsafe fn load_u8x4(values: &[u8; COLUMNS]) -> uint32x4_t {
    unsafe {
        let bytes = vcreate_u8(u64::from(u32::from_le_bytes(*values)));
        vmovl_u16(vget_low_u16(vmovl_u8(bytes)))
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon", target_endian = "little"))]
#[inline(always)]
unsafe fn scales_and_minimums(block: &BlockQ4Kx4, group: usize) -> (int32x4_t, int32x4_t) {
    unsafe {
        let mask_six = vdupq_n_u32(63);
        if group < 4 {
            return (
                vreinterpretq_s32_u32(vandq_u32(load_u8x4(&block.metadata[group]), mask_six)),
                vreinterpretq_s32_u32(vandq_u32(load_u8x4(&block.metadata[group + 4]), mask_six)),
            );
        }
        let mixed = load_u8x4(&block.metadata[group + 4]);
        let scale_high = vshlq_n_u32(vshrq_n_u32(load_u8x4(&block.metadata[group - 4]), 6), 4);
        let minimum_high = vshlq_n_u32(vshrq_n_u32(load_u8x4(&block.metadata[group]), 6), 4);
        (
            vreinterpretq_s32_u32(vorrq_u32(vandq_u32(mixed, vdupq_n_u32(15)), scale_high)),
            vreinterpretq_s32_u32(vorrq_u32(vshrq_n_u32(mixed, 4), minimum_high)),
        )
    }
}

#[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon",
    target_endian = "little",
    target_feature = "dotprod"
))]
#[inline(always)]
unsafe fn sdot(accumulator: int32x4_t, weights: int8x16_t, activations: int8x16_t) -> int32x4_t {
    let mut accumulator = accumulator;
    unsafe {
        core::arch::asm!(
            "sdot {accumulator:v}.4s, {weights:v}.16b, {activations:v}.16b",
            accumulator = inout(vreg) accumulator,
            weights = in(vreg) weights,
            activations = in(vreg) activations,
            options(pure, nomem, nostack),
        );
    }
    accumulator
}

#[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon",
    target_endian = "little",
    not(target_feature = "dotprod")
))]
#[inline(always)]
unsafe fn sdot(accumulator: int32x4_t, weights: int8x16_t, activations: int8x16_t) -> int32x4_t {
    unsafe {
        let low = vmull_s8(vget_low_s8(weights), vget_low_s8(activations));
        let high = vmull_s8(vget_high_s8(weights), vget_high_s8(activations));
        vaddq_s32(accumulator, vpaddq_s32(vpaddlq_s16(low), vpaddlq_s16(high)))
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon", target_endian = "little"))]
/// # Safety
/// Weights must contain `depth_blocks` packed blocks, activations must contain `ROWS` rows
/// with that block stride, and output must cover `ROWS` rows of four values at `output_stride`.
unsafe fn tile<const ROWS: usize>(
    weights: *const BlockQ4Kx4,
    activations: *const BlockQ8K,
    depth_blocks: usize,
    output: *mut f32,
    output_stride: usize,
) {
    unsafe {
        let zero = vdupq_n_s32(0);
        let mut totals = [vdupq_n_f32(0.0); ROWS];
        let nibble_mask = vdupq_n_u8(15);
        for block_index in 0..depth_blocks {
            let block = &*weights.add(block_index);
            let mut integer_sums = [zero; ROWS];
            let mut minimum_sums = [zero; ROWS];
            for group in 0..QK_K / 32 {
                let (scales, minimums) = scales_and_minimums(block, group);
                let mut first_pair = [zero; ROWS];
                let mut second_pair = [zero; ROWS];
                for segment in 0..2 {
                    let offset = (group * 2 + segment) * RUN * COLUMNS;
                    let first_bits = vld1q_u8(block.quants.as_ptr().add(offset));
                    let second_bits = vld1q_u8(block.quants.as_ptr().add(offset + 16));
                    let first_low = vreinterpretq_s8_u8(vandq_u8(first_bits, nibble_mask));
                    let second_low = vreinterpretq_s8_u8(vandq_u8(second_bits, nibble_mask));
                    let first_high = vreinterpretq_s8_u8(vshrq_n_u8(first_bits, 4));
                    let second_high = vreinterpretq_s8_u8(vshrq_n_u8(second_bits, 4));
                    for row in 0..ROWS {
                        let activation = &*activations.add(row * depth_blocks + block_index);
                        let bytes = vld1_s8(activation.qs.as_ptr().add(group * 32 + segment * RUN));
                        let duplicated = vcombine_s8(bytes, bytes);
                        first_pair[row] = sdot(first_pair[row], first_low, duplicated);
                        second_pair[row] = sdot(second_pair[row], second_low, duplicated);
                        let bytes =
                            vld1_s8(activation.qs.as_ptr().add(group * 32 + segment * RUN + 16));
                        let duplicated = vcombine_s8(bytes, bytes);
                        first_pair[row] = sdot(first_pair[row], first_high, duplicated);
                        second_pair[row] = sdot(second_pair[row], second_high, duplicated);
                    }
                }
                for row in 0..ROWS {
                    integer_sums[row] = vmlaq_s32(
                        integer_sums[row],
                        vpaddq_s32(first_pair[row], second_pair[row]),
                        scales,
                    );
                    let activation = &*activations.add(row * depth_blocks + block_index);
                    let block_sum = i32::from(activation.bsums[group * 2])
                        + i32::from(activation.bsums[group * 2 + 1]);
                    minimum_sums[row] = vmlaq_n_s32(minimum_sums[row], minimums, block_sum);
                }
            }
            let scales = load_f16x4(block.scales.as_ptr());
            let minimums = load_f16x4(block.minimums.as_ptr());
            for row in 0..ROWS {
                let activation = &*activations.add(row * depth_blocks + block_index);
                let activation_scale = vdupq_n_f32(activation.d);
                let scale = vmulq_f32(activation_scale, scales);
                let minimum = vmulq_f32(activation_scale, minimums);
                totals[row] =
                    vsubq_f32(totals[row], vmulq_f32(minimum, vcvtq_f32_s32(minimum_sums[row])));
                totals[row] =
                    vaddq_f32(totals[row], vmulq_f32(scale, vcvtq_f32_s32(integer_sums[row])));
            }
        }
        for (row, total) in totals.iter().enumerate() {
            vst1q_f32(output.add(row * output_stride), *total);
        }
    }
}

fn enabled(columns: usize) -> bool {
    static MINIMUM_COLUMNS: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    match *MINIMUM_COLUMNS.get_or_init(|| match std::env::var("XN_Q4K_REPACK").as_deref() {
        Ok("0") => None,
        Ok("1") => Some(COLUMNS),
        _ if cfg!(all(
            target_arch = "aarch64",
            target_feature = "neon",
            target_feature = "dotprod",
            target_endian = "little"
        )) =>
        {
            Some(128)
        }
        _ => None,
    }) {
        Some(minimum) => columns >= minimum,
        None => false,
    }
}

fn interleaved(source: &[BlockQ4K], dimensions: &[usize]) -> Option<QStorage> {
    if is_eligible(dimensions)
        && enabled(dimensions[0])
        && let Ok(packed) = Q4Kx4::from_q4k(source, dimensions[0], dimensions[1])
    {
        return Some(QStorage::Cpu(Box::new(packed)));
    }
    None
}

pub fn q4k_storage(source: &[BlockQ4K], dimensions: &[usize]) -> QStorage {
    interleaved(source, dimensions).unwrap_or_else(|| QStorage::Cpu(Box::new(source.to_vec())))
}

pub fn q4k_storage_owned(source: Vec<BlockQ4K>, dimensions: &[usize]) -> QStorage {
    interleaved(&source, dimensions).unwrap_or_else(|| QStorage::Cpu(Box::new(source)))
}
