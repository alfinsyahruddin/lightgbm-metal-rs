// Numeric binning translated from the bundled LightGBM src/io/bin.cpp.
// Copyright (c) Microsoft Corporation and the LightGBM developers. MIT license.
use crate::{DenseMatrix, Error, Result};

const ZERO: f64 = 1e-35_f32 as f64;

#[derive(Debug)]
pub(crate) struct BinMapper {
    pub bounds: Vec<f64>,
    pub missing: bool,
    pub splittable: bool,
}

impl BinMapper {
    pub fn new(column: &[f64], max_bin: usize) -> Result<Self> {
        let missing_count = column.iter().filter(|v| v.is_nan()).count();
        let missing = missing_count > 0;
        let mut sorted: Vec<_> = column
            .iter()
            .copied()
            .filter(|v| v.is_finite() && v.abs() > ZERO)
            .collect();
        sorted.sort_by(f64::total_cmp);
        let zeros = column.len() - sorted.len() - missing_count;
        let mut distinct = Vec::new();
        let mut counts: Vec<usize> = Vec::new();
        if sorted.is_empty() || (sorted[0] > 0.0 && zeros > 0) {
            distinct.push(0.0);
            counts.push(zeros);
        }
        for (index, &value) in sorted.iter().enumerate() {
            if index > 0 && value <= upper(sorted[index - 1]) {
                if let Some(last) = distinct.last_mut() {
                    *last = value;
                }
                if let Some(last) = counts.last_mut() {
                    *last += 1;
                }
            } else {
                if index > 0 && sorted[index - 1] < 0.0 && value > 0.0 {
                    distinct.push(0.0);
                    counts.push(zeros);
                }
                distinct.push(value);
                counts.push(1);
            }
        }
        if sorted.last().is_some_and(|v| *v < 0.0) && zeros > 0 {
            distinct.push(0.0);
            counts.push(zeros);
        }
        let total = column.len() - missing_count;
        let numeric_bins = max_bin - usize::from(missing);
        let mut bounds = zero_bins(&distinct, &counts, numeric_bins, total);
        if bounds.iter().any(|v| v.is_nan()) {
            return Err(Error::Numerical("numeric bin boundaries overflowed".into()));
        }
        // Upstream stores the numeric enum marker MissingType::NaN (2) in the dummy bin.
        // This value is metadata, never a candidate numeric threshold.
        if missing {
            bounds.push(2.0);
        }
        let occupied = counts.iter().filter(|&&n| n > 0).count() + usize::from(missing_count > 0);
        Ok(Self {
            bounds,
            missing,
            splittable: occupied > 1,
        })
    }

    pub fn bin(&self, value: f64) -> u8 {
        let numeric_count = self.bounds.len() - usize::from(self.missing);
        if value.is_nan() && self.missing {
            return (self.bounds.len() - 1) as u8;
        }
        let value = if value.is_nan() { 0.0 } else { value };
        self.bounds[..numeric_count]
            .partition_point(|&bound| value > bound)
            .min(numeric_count - 1) as u8
    }
}

fn upper(value: f64) -> f64 {
    if value == f64::INFINITY {
        return value;
    }
    if value == 0.0 {
        return f64::from_bits(1);
    }
    f64::from_bits(if value > 0.0 {
        value.to_bits() + 1
    } else {
        value.to_bits() - 1
    })
}

fn push_boundary(bounds: &mut Vec<f64>, value: f64) {
    if bounds.last().map_or(true, |&last| value > upper(last)) {
        bounds.push(value);
    }
}

fn greedy(values: &[f64], counts: &[usize], mut max_bin: usize, total: usize) -> Vec<f64> {
    let mut bounds = Vec::new();
    if values.len() <= max_bin {
        let mut current = 0;
        for i in 0..values.len().saturating_sub(1) {
            current += counts[i];
            if current >= 3 {
                let previous = bounds.len();
                push_boundary(&mut bounds, upper((values[i] + values[i + 1]) / 2.0));
                if bounds.len() > previous {
                    current = 0;
                }
            }
        }
    } else {
        max_bin = max_bin.min(total / 3).max(1);
        let mut mean = total as f64 / max_bin as f64;
        let mut rest_bins = max_bin as isize;
        let mut rest_rows = total as isize;
        let big: Vec<_> = counts.iter().map(|&n| n as f64 >= mean).collect();
        for (&n, &large) in counts.iter().zip(&big) {
            if large {
                rest_bins -= 1;
                rest_rows -= n as isize;
            }
        }
        mean = rest_rows as f64 / rest_bins as f64;
        let mut uppers = vec![f64::INFINITY; max_bin];
        let mut lowers = vec![f64::INFINITY; max_bin];
        let mut bin = 0;
        lowers[0] = values[0];
        let mut current = 0;
        if max_bin > 1 {
            for i in 0..values.len() - 1 {
                if !big[i] {
                    rest_rows -= counts[i] as isize;
                }
                current += counts[i];
                if big[i]
                    || current as f64 >= mean
                    || (big[i + 1] && current as f64 >= (mean * 0.5).max(1.0))
                {
                    uppers[bin] = values[i];
                    bin += 1;
                    lowers[bin] = values[i + 1];
                    if bin >= max_bin - 1 {
                        break;
                    }
                    current = 0;
                    if !big[i] {
                        rest_bins -= 1;
                        mean = rest_rows as f64 / rest_bins as f64;
                    }
                }
            }
        }
        for i in 0..bin {
            push_boundary(&mut bounds, upper((uppers[i] + lowers[i + 1]) / 2.0));
        }
    }
    bounds.push(f64::INFINITY);
    bounds
}

fn zero_bins(values: &[f64], counts: &[usize], max_bin: usize, total: usize) -> Vec<f64> {
    let left_count = values.partition_point(|&v| v <= -ZERO);
    let right_start = values.partition_point(|&v| v <= ZERO);
    let left_rows: usize = counts[..left_count].iter().sum();
    let zero_rows: usize = counts[left_count..right_start].iter().sum();
    let right_rows: usize = counts[right_start..].iter().sum();
    let mut bounds = Vec::new();
    if left_count > 0 && max_bin > 1 {
        let left_max = ((left_rows as f64 / (total - zero_rows) as f64 * (max_bin - 1) as f64)
            as usize)
            .max(1);
        bounds = greedy(
            &values[..left_count],
            &counts[..left_count],
            left_max,
            left_rows,
        );
        if let Some(last) = bounds.last_mut() {
            *last = -ZERO;
        }
    }
    let right_max = max_bin.saturating_sub(1 + bounds.len());
    if right_start < values.len() && right_max > 0 {
        bounds.push(ZERO);
        bounds.extend(greedy(
            &values[right_start..],
            &counts[right_start..],
            right_max,
            right_rows,
        ));
    } else {
        bounds.push(f64::INFINITY);
    }
    bounds
}

pub(crate) struct BinnedData {
    pub rows: usize,
    pub stride: usize,
    pub maps: Vec<BinMapper>,
    pub values: Vec<u8>,
}

impl BinnedData {
    pub fn new(matrix: DenseMatrix<'_>, max_bin: usize) -> Result<Self> {
        let mut values = Vec::new();
        values
            .try_reserve_exact(matrix.values().len())
            .map_err(|e| Error::InvalidData(format!("binned data allocation: {e}")))?;
        let mut maps = Vec::with_capacity(matrix.cols());
        for col in 0..matrix.cols() {
            let column: Vec<_> = (0..matrix.rows())
                .map(|row| matrix.value(row, col))
                .collect();
            let mapper = BinMapper::new(&column, max_bin)?;
            values.extend(column.iter().map(|&v| mapper.bin(v)));
            maps.push(mapper);
        }
        Ok(Self {
            rows: matrix.rows(),
            stride: max_bin,
            maps,
            values,
        })
    }

    pub fn bin(&self, row: usize, col: usize) -> usize {
        self.values[col * self.rows + row] as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_bins_match_frozen_upstream_mapper() {
        for name in [
            "regression",
            "two_bins",
            "constant",
            "all_missing",
            "unseen_nan",
        ] {
            let path = format!(
                "{}/validation/fixtures/{name}.json",
                env!("CARGO_MANIFEST_DIR")
            );
            let fixture: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
            let x: Vec<f64> = fixture["features"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap_or(f64::NAN))
                .collect();
            let mapper =
                BinMapper::new(&x, fixture["config"]["max_bin"].as_u64().unwrap() as usize)
                    .unwrap();
            let expected = &fixture["binning"];
            assert_eq!(
                mapper.missing,
                expected["missing"].as_bool().unwrap(),
                "{name}"
            );
            assert_eq!(
                mapper.bounds.len(),
                expected["upper_bounds"].as_array().unwrap().len(),
                "{name}"
            );
            for (&bound, value) in mapper
                .bounds
                .iter()
                .zip(expected["upper_bounds"].as_array().unwrap())
            {
                if let Some(value) = value.as_f64() {
                    assert_eq!(bound, value, "{name}");
                } else {
                    assert!(!bound.is_finite(), "{name}");
                }
            }
            for (&value, expected) in x.iter().zip(expected["bins"].as_array().unwrap()) {
                assert_eq!(
                    mapper.bin(value) as u64,
                    expected.as_u64().unwrap(),
                    "{name}"
                );
            }
        }
    }
}
