use nalgebra::DMatrix;

pub trait Parameterized<'a> {
    /// Get the model parameters
    fn get_params(&'a self) -> Vec<f64>;
    /// Set the model parameters
    fn set_params(&'a mut self, params: &[f64]);
    /// Create a new model from parameters
    fn from_params(params: &[f64]) -> Self;
}

pub trait Jacobian<'a> {
    /// Get the model derivative
    fn jacobian(&'a self, x: &DMatrix<f64>) -> DMatrix<f64>;
}

pub trait Hessian<'a> {
    /// Get the second derivative
    fn hessian(&'a self, x: &DMatrix<f64>) -> DMatrix<f64>;
}
