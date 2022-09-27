use super::errors::InvalidKernelError;

pub type KernelResult<T> = Result<T, InvalidKernelError>;

pub trait Kernel<P, const DIMS: usize> {
    /// Compute the covariance
    fn call(&self, x: [f64; DIMS], y: [f64; DIMS]) -> f64;
    /// Get the kernel parameters
    fn get_params(&self) -> P;
    /// Set the kernel parameters
    fn set_params(&mut self, params: P);
    /// Create a new kernel from parameters
    fn from_params(params: P) -> Self;
}
