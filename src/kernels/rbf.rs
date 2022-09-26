use super::errors::{CovarianceParamLengthError, InvalidKernelError};
use super::kernel::{CovarianceResult, Kernel, KernelResult};

/// Radial Basis Function kernel
///
/// `K = exp(-||x - x'||^2 / (2 * l^2))`
///
/// where `||x - x'||` is the euclidean distance between vectors x and x',
/// and `l` is the length scale
///
/// # Examples
/// ```
/// use gprs::kernels::RBF;
/// // create a 2-d RBF kernel
/// let kern = RBF::new(vec![1.0, 2.0]);
/// // estimate covariance between 2 points
/// let K = kern.call(vec![1.8, 5.5], vec![2.2, 3.0])?;
/// ```
#[derive(Debug)]
pub struct RBF {
    gamma: Vec<f64>,
}

impl RBF {
    pub fn new(length_scale: &Vec<f64>) -> KernelResult<Self> {
        let gamma = Self::gamma(length_scale)?;
        Ok(RBF { gamma })
    }

    fn gamma(length_scale: &Vec<f64>) -> KernelResult<Vec<f64>> {
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
    pub fn set_length_scale(&mut self, length_scale: &Vec<f64>) -> KernelResult<()> {
        self.gamma = Self::gamma(length_scale)?;
        Ok(())
    }
}

impl Kernel for RBF {
    fn call(&self, x: &Vec<f64>, y: &Vec<f64>) -> CovarianceResult<f64> {
        let x_len = x.len();
        let y_len = y.len();

        if x_len != self.gamma.len() || y_len != self.gamma.len() {
            return Err(CovarianceParamLengthError {
                x_len,
                y_len,
                expected: self.gamma.len(),
            });
        }

        let diffs: f64 = self
            .gamma
            .iter()
            .enumerate()
            .map(|(index, g)| {
                let diff = x[index] - y[index];
                diff * diff * g
            })
            .sum();

        return Ok(diffs.exp());
    }
}
