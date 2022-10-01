use nalgebra::{DMatrix, RawStorage};
use rayon::prelude::*;

struct SyncMutPtr<T>(*mut T);

impl<T> SyncMutPtr<T> {
    pub unsafe fn offset(&self, offset: isize) -> *mut T {
        self.0.offset(offset)
    }
}

/// SAFETY: this is only used to add a value to the diagonal of a matrix in parallel.
///         It is safe because no two threads will be modifying data at the same index.
unsafe impl<T> Sync for SyncMutPtr<T> {}

/// Add a value to the matrix diagonal, in-place, in parallel
///
/// # Safety
/// unsafe if `mat` is not square
pub unsafe fn par_add_diagonal_mut_unchecked(mat: &mut DMatrix<f64>, f: &f64) {
    let mat_ptr = SyncMutPtr(mat.as_mut_ptr());

    (0..mat.shape().0).into_par_iter().for_each(|i| {
        // SAFETY: we can safely unwrap usize -> isize because a square matrix with size
        //         above the max isize would not fit in memory.
        let ii: isize = mat.data.linear_index(i, i).try_into().unwrap();
        *(mat_ptr.offset(ii)) += f;
    });
}
