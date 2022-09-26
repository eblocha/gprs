pub enum InvalidKernelError {
    /// A negative or zero length scale was provided
    NonPositiveLengthScale,
    /// An empty vector was provided for the length scale
    EmptyLengthScale,
}

pub struct CovarianceParamLengthError {
    pub expected: usize,
    pub x_len: usize,
    pub y_len: usize,
}
