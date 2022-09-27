#[derive(Debug)]
pub enum InvalidKernelError {
    /// A negative or zero length scale was provided
    NonPositiveLengthScale,
    /// An invalid-length vector was provided for the length scale
    LengthScaleSizeInvalid,
}

#[derive(Debug)]
pub struct CovarianceParamLengthError {
    pub expected: usize,
    pub x_len: usize,
    pub y_len: usize,
}
