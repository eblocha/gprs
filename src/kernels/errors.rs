#[derive(Debug)]
pub enum InvalidKernelError {
    /// A negative or zero length scale was provided
    NonPositiveLengthScale,
}
