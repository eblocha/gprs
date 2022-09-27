use super::errors::{CovarianceParamLengthError, InvalidKernelError};

pub type KernelResult<T> = Result<T, InvalidKernelError>;
pub type CovarianceResult<T> = Result<T, CovarianceParamLengthError>;

pub trait Kernel<P> {
    /// Compute the covariance
    fn call<'x, 'y>(&self, x: &'x Vec<f64>, y: &'y Vec<f64>) -> CovarianceResult<f64>;
    /// Get the kernel parameters
    fn get_params(&self) -> P;
    /// Set the kernel parameters
    fn set_params(&mut self, params: P);
    /// Create a new kernel from parameters
    fn from_params(params: P) -> Self;
}
