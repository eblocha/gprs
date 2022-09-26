use super::errors::{CovarianceParamLengthError, InvalidKernelError};

pub type KernelResult<T> = Result<T, InvalidKernelError>;
pub type CovarianceResult<T> = Result<T, CovarianceParamLengthError>;

pub trait Kernel {
    /// Compute the covariance
    fn call<'x, 'y>(&self, x: &'x Vec<f64>, y: &'y Vec<f64>) -> CovarianceResult<f64>;
}
