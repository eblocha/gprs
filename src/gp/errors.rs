use crate::kernels::errors::IncompatibleShapeError;

#[derive(Debug)]
pub enum GPCompilationError {
    /// The kernel returned a non-positive-definite covariance matrix for the input data
    NonPositiveDefiniteError,
    /// The input data shape is incompatible with itself or the kernel
    IncompatibleShapeError(IncompatibleShapeError),
}
