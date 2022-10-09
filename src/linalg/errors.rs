#[derive(Debug, PartialEq, Eq)]
pub struct IncompatibleShapeError {
    pub shapes: Vec<(usize, usize)>,
}
