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

#[cfg(test)]
mod tests {
    use nalgebra::DMatrix;

    use super::par_add_diagonal_mut_unchecked;

    #[test]
    fn test_empty() {
        let mut mat: DMatrix<f64> = DMatrix::from_vec(0, 0, vec![]);
        unsafe { par_add_diagonal_mut_unchecked(&mut mat, &10.0_f64) }
        assert_eq!(mat.as_slice(), vec![].as_slice());
    }

    #[test]
    #[rustfmt::skip]
    fn test_square() {
        let mut mat = DMatrix::from_vec(3, 3, vec![
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0]);
        let expected = vec![
            10.0,  0.0,  0.0,
             0.0, 10.0,  0.0,
             0.0,  0.0, 10.0];
        unsafe { par_add_diagonal_mut_unchecked(&mut mat, &10.0_f64) }
        assert_eq!(mat.as_slice(), expected.as_slice());
    }
}
