#[derive(Debug, PartialEq)]
pub struct IncompatibleShapeError {
    pub shapes: Vec<(usize, usize)>,
}
