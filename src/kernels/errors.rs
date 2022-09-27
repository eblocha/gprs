#[derive(Debug)]
pub enum InvalidKernelError {
    /// A negative or zero length scale was provided
    NonPositiveLengthScale,
}

#[derive(Debug)]
pub struct IncompatibleShapeError {
    pub shapes: Vec<(usize, usize)>,
}
