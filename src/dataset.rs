use crate::{Error, Objective, Result};

/// Validated, borrowed row-major numeric matrix. NaNs are missing values; zeros are numeric.
///
/// Empty matrices are valid for prediction. Training separately requires at least one row.
///
/// ```
/// use lightgbm_metal_rs::DenseMatrix;
/// let values = [1.0, f64::NAN, 2.0, 0.0];
/// let matrix = DenseMatrix::new(&values, 2, 2)?;
/// assert_eq!(matrix.rows(), 2);
/// # Ok::<(), lightgbm_metal_rs::Error>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct DenseMatrix<'a> {
    values: &'a [f64],
    rows: usize,
    cols: usize,
}

impl<'a> DenseMatrix<'a> {
    /// Validate dimensions and values without copying the matrix.
    pub fn new(values: &'a [f64], rows: usize, cols: usize) -> Result<Self> {
        if cols == 0 || rows > u32::MAX as usize || cols > u32::MAX as usize {
            return Err(Error::InvalidData(
                "feature count must be positive and dimensions must fit u32".into(),
            ));
        }
        let size = rows
            .checked_mul(cols)
            .ok_or_else(|| Error::InvalidData("matrix dimensions overflow".into()))?;
        if size != values.len() {
            return Err(Error::InvalidData(format!(
                "expected {size} feature values, got {}",
                values.len()
            )));
        }
        if values.iter().any(|v| v.is_infinite()) {
            return Err(Error::InvalidData("infinite feature value".into()));
        }
        Ok(Self { values, rows, cols })
    }

    /// Number of rows.
    pub fn rows(self) -> usize {
        self.rows
    }

    /// Number of features per row.
    pub fn cols(self) -> usize {
        self.cols
    }

    /// Row-major feature values.
    pub fn values(self) -> &'a [f64] {
        self.values
    }

    pub(crate) fn value(self, row: usize, col: usize) -> f64 {
        self.values[row * self.cols + col]
    }

    pub(crate) fn validate_labels(self, labels: &[f64], objective: Objective) -> Result<()> {
        if self.rows == 0 || labels.len() != self.rows {
            return Err(Error::InvalidData(
                "training requires nonempty rows and one label per row".into(),
            ));
        }
        if labels.iter().any(|y| !y.is_finite()) {
            return Err(Error::InvalidData("labels must be finite".into()));
        }
        if objective == Objective::BinaryLogloss && labels.iter().any(|&y| y != 0.0 && y != 1.0) {
            return Err(Error::InvalidData(
                "binary labels must be exactly zero or one".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_shapes_and_feature_values_without_large_allocations() {
        assert!(DenseMatrix::new(&[], usize::MAX, 2).is_err());
        assert!(DenseMatrix::new(&[1.0], 1, 2).is_err());
        assert!(DenseMatrix::new(&[f64::INFINITY], 1, 1).is_err());
        assert!(DenseMatrix::new(&[f64::NAN], 1, 1).is_ok());
    }

    #[test]
    fn validates_training_labels_and_empty_prediction_data() {
        let empty = DenseMatrix::new(&[], 0, 1).unwrap();
        assert!(empty.validate_labels(&[], Objective::RegressionL2).is_err());
        let matrix = DenseMatrix::new(&[0.0, 1.0], 2, 1).unwrap();
        assert!(matrix
            .validate_labels(&[0.0], Objective::RegressionL2)
            .is_err());
        assert!(matrix
            .validate_labels(&[0.0, f64::NAN], Objective::RegressionL2)
            .is_err());
        assert!(matrix
            .validate_labels(&[0.0, 0.5], Objective::BinaryLogloss)
            .is_err());
    }
}
