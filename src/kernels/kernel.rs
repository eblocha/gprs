use super::errors::CovarianceParamLengthError;

pub trait Kernel {
    /// Compute the covariance
    fn call<'x, 'y>(&self, x: &'x [f64], y: &'y [f64]) -> Result<f64, CovarianceParamLengthError>;
}
