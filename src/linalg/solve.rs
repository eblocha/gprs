use nalgebra::DMatrix;
use rayon::prelude::*;

/// Solve a linear system of equations where the upper triangle of the coefficients matrix is assumed to be 0
///
/// This function will solve in parallel over the columns of `a`.
///
/// # Examples
///
/// ```
/// use gprs::linalg::par_solve_lower_triangular_unchecked;
/// use nalgebra::DMatrix;
///
/// let a = DMatrix::from_vec(2, 2, vec![
///     1.0, 0.0,
///     0.0, 1.0,
/// ]);
///
/// let b = DMatrix::from_vec(2, 1, vec![
///     1.0,
///     1.0,
/// ]);
///
/// assert_eq!(par_solve_lower_triangular_unchecked(&a, &b), b);
/// ```
///
/// ```
/// use gprs::linalg::par_solve_lower_triangular_unchecked;
/// use nalgebra::DMatrix;
///
/// let a = DMatrix::from_vec(2, 2, vec![
///     1.0, 2.0,
///     0.0, 1.0,
/// ]);
///
/// let b = DMatrix::from_vec(2, 1, vec![
///     1.0,
///     1.0,
/// ]);
///
/// let expect = DMatrix::from_vec(2, 1, vec![
///     1.0,
///     -1.0,
/// ]);
///
/// assert_eq!(par_solve_lower_triangular_unchecked(&a, &b), expect);
/// ```
///
pub fn par_solve_lower_triangular_unchecked(a: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
    let mut res = b.clone_owned();
    par_solve_lower_triangular_unchecked_mut(a, &mut res);
    res
}

fn par_solve_lower_triangular_unchecked_mut(a: &DMatrix<f64>, b: &mut DMatrix<f64>) {
    let nrows = b.nrows();

    b.as_mut_slice()
        .par_chunks_exact_mut(nrows)
        .for_each(|col| {
            solve_lower_triangular_vector_unchecked_mut(a, col);
        });
}

fn solve_lower_triangular_vector_unchecked_mut(a: &DMatrix<f64>, b: &mut [f64]) {
    let dim = a.nrows();

    for i in 0..dim {
        unsafe {
            let coeff = b.get_unchecked(i) / a.get_unchecked((i, i));
            *b.get_unchecked_mut(i) = coeff;

            b.get_unchecked_mut(i + 1..)
                .iter_mut()
                .zip(&a.slice_range(i + 1.., i))
                .for_each(|(l, r)| *l += r * -coeff);
        }
    }
}
