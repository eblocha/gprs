#[derive(Debug)]
pub struct IncompatibleShapeError {
    pub shapes: Vec<(usize, usize)>,
}
