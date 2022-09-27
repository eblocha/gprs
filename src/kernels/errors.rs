#[derive(Debug)]
pub enum InvalidKernelError {
    /// A negative or zero length scale was provided
    NonPositiveLengthScale,
}

#[derive(Debug)]
pub struct MismatchedSizeError {
    pub shapes: Vec<(usize, usize)>,
}
