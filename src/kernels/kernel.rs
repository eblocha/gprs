use nalgebra::DMatrix;

use super::errors::IncompatibleShapeError;

pub trait Kernel<P> {
    /// Compute the covariance between sets of points
    ///
    /// Returns a MismatchedSizeError if the dimensions are not valid
    fn call<'x, 'y>(
        &self,
        x: &'x DMatrix<f64>,
        y: &'y DMatrix<f64>,
    ) -> Result<DMatrix<f64>, IncompatibleShapeError>;
    /// Get the kernel parameters
    fn get_params(&self) -> &P;
    /// Set the kernel parameters
    fn set_params(&mut self, params: P);
    /// Create a new kernel from parameters
    fn from_params(params: P) -> Self;
}
