use nalgebra::DMatrix;

use crate::linalg::errors::IncompatibleShapeError;

pub enum TriangleSide {
    UPPER,
    LOWER,
}

pub trait Kernel {
    /// Compute the covariance between sets of points
    fn call(
        &self,
        x: &DMatrix<f64>,
        y: &DMatrix<f64>,
    ) -> Result<DMatrix<f64>, IncompatibleShapeError>;
    /// Compute the covariance and store the result in mutable matrix `into`
    ///
    /// This allows for re-use of memory
    fn call_inplace(
        &self,
        x: &DMatrix<f64>,
        y: &DMatrix<f64>,
        into: &mut DMatrix<f64>,
    ) -> Result<(), IncompatibleShapeError>;

    /// Compute the covariance only in one half of the triangular matrix
    ///
    /// This is used when compiling the GP. The uncomputed side is not used.
    fn call_triangular(
        &self,
        x: &DMatrix<f64>,
        side: TriangleSide,
    ) -> Result<DMatrix<f64>, IncompatibleShapeError>;

    /// Compute only the diagonal portion of the covariance matrix
    fn call_diagonal(&self, x: &DMatrix<f64>) -> Result<Vec<f64>, IncompatibleShapeError>;
}
