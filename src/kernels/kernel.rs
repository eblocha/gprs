use super::errors::{CovarianceParamLengthError, InvalidKernelError};

pub type KernelResult<T> = Result<T, InvalidKernelError>;
pub type CovarianceResult<T> = Result<T, CovarianceParamLengthError>;

pub trait Kernel<P> {
    /// Compute the covariance
    fn call<'x, 'y>(&self, x: &'x Vec<f64>, y: &'y Vec<f64>) -> CovarianceResult<f64>;
    fn get_params(&self) -> P;
    fn set_params(&mut self, params: P);
}
