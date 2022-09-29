pub trait Parameterized<'a, P> {
    /// Get the model parameters
    fn get_params(&'a self) -> P;
    /// Set the model parameters
    fn set_params(&'a mut self, params: P);
    /// Create a new model from parameters
    fn from_params(params: P) -> Self;
}
