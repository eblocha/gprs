use ::nalgebra::DVector;

pub trait Kernel<P> {
    /// Compute the covariance
    fn call<'x, 'y>(&self, x: &'x DVector<f64>, y: &'y DVector<f64>) -> f64;
    /// Get the kernel parameters
    fn get_params(&self) -> &P;
    /// Set the kernel parameters
    fn set_params(&mut self, params: P);
    /// Create a new kernel from parameters
    fn from_params(params: P) -> Self;
}
