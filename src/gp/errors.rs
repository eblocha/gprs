use crate::kernels::errors::IncompatibleShapeError;

#[derive(Debug)]
pub enum GPCompilationError {
    /// The kernel returned a non-positive-definite covariance matrix for the input data.
    ///
    /// This can either be caused by an invalid kernel formula, or duplicate points in the x data.
    NonPositiveDefiniteError,
    /// The input data shape is incompatible with itself or the kernel.
    IncompatibleShapeError(IncompatibleShapeError),
}
