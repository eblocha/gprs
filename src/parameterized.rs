pub trait Parameterized<P> {
    /// Get the model parameters
    fn get_params(&self) -> &P;
    /// Set the model parameters
    fn set_params(&mut self, params: P);
    /// Create a new model from parameters
    fn from_params(params: P) -> Self;
}
