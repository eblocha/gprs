use libm::exp;

use super::errors::{CovarianceParamLengthError, InvalidKernelError};
use super::kernel::Kernel;

/// Radial Basis Function kernel
///
/// `K = exp(-||x - x'||^2 / (2 * l^2))`
///
/// where `||x - x'||` is the euclidean distance between vectors x and x',
/// and `l` is the length scale
#[derive(Debug)]
pub struct RBF {
    gamma: Vec<f64>,
}

impl RBF {
    pub fn new(length_scale: &Vec<f64>) -> Result<Self, InvalidKernelError> {
        let gamma = Self::gamma(length_scale)?;
        Ok(RBF { gamma })
    }

    fn gamma(length_scale: &Vec<f64>) -> Result<Vec<f64>, InvalidKernelError> {
        if length_scale.len() == 0 {
            return Err(InvalidKernelError::EmptyLengthScale);
        }

        let all_positive = length_scale.iter().all(|l| *l > 0.0);

        if !all_positive {
            return Err(InvalidKernelError::NonPositiveLengthScale);
        }

        return Ok(length_scale.iter().map(|v| -0.5 / (*v * *v)).collect());
    }

    /// Set the length scale of the kernel
    pub fn set_length_scale(&mut self, length_scale: &Vec<f64>) -> Result<(), InvalidKernelError> {
        self.gamma = Self::gamma(length_scale)?;
        Ok(())
    }
}

impl Kernel for RBF {
    fn call(&self, x: &[f64], y: &[f64]) -> Result<f64, CovarianceParamLengthError> {
        let x_len = x.len();
        let y_len = y.len();

        if x_len != self.gamma.len() || y_len != self.gamma.len() {
            return Err(CovarianceParamLengthError {
                x_len,
                y_len,
                expected: self.gamma.len(),
            });
        }

        let diffs = self
            .gamma
            .iter()
            .enumerate()
            .map(|(index, g)| {
                let diff = x[index] - y[index];
                diff * diff * g
            })
            .sum();

        return Ok(exp(diffs));
    }
}
