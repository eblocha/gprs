use nalgebra::DMatrix;
use rayon::prelude::*;

/// Add a value to the matrix diagonal, in-place, in parallel
///
/// # Safety
/// unsafe if `mat` is not square
pub unsafe fn par_add_diagonal_mut_unchecked(mat: &mut DMatrix<f64>, f: &f64) {
    let size = mat.shape().0;
    if size == 0 {
        return;
    }

    mat.as_mut_slice()
        .par_chunks_exact_mut(size)
        .enumerate()
        .for_each(|(i, slice)| {
            *(slice.get_unchecked_mut(i)) += f;
        });
}
