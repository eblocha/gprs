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
    /// Compute the covariance and store the result in mutable matrix `into`
    ///
    /// This allows for re-use of memory
    fn call_into<'x, 'y>(
        &self,
        x: &'x DMatrix<f64>,
        y: &'y DMatrix<f64>,
        into: &mut DMatrix<f64>,
    ) -> Result<(), IncompatibleShapeError>;
}
