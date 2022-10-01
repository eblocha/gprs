use nalgebra::DMatrix;

use crate::linalg::errors::IncompatibleShapeError;

pub enum TriangleSide {
    UPPER,
    LOWER,
}

pub trait Kernel {
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
    fn call_inplace<'x, 'y>(
        &self,
        x: &'x DMatrix<f64>,
        y: &'y DMatrix<f64>,
        into: &mut DMatrix<f64>,
    ) -> Result<(), IncompatibleShapeError>;

    /// Compute the covariance only in one half of the triangular matrix
    ///
    /// This is used when compiling the GP. The uncomputed side is not used.
    fn call_triangular<'x>(
        &self,
        x: &'x DMatrix<f64>,
        side: TriangleSide,
    ) -> Result<DMatrix<f64>, IncompatibleShapeError>;
}
