/// Given an index into a flattened 2-d matrix, find the 2-d coordinate of that index
///
/// `nmajor` is the length of the major axis of the matrix (i.e. ncols for column-major, nrows for row-major)
///
/// The returned index is in (minor, major) axis order
#[inline(always)]
pub fn index_to_2d(index: usize, nmajor: usize) -> (usize, usize) {
    let i = index / nmajor;
    let j = index - (i * nmajor);
    (i, j)
}

/// Given a major axis index and the number of dims,
/// return the start and end slice positions to slice along the minor axis at the index
#[inline(always)]
pub fn slice_indices(index: usize, dims: usize) -> (usize, usize) {
    let xs = index * dims;
    let xe = xs + dims;
    (xs, xe)
}
