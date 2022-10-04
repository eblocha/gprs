use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;

/// Solve a linear system of equations where the upper triangle of the coefficients matrix is assumed to be 0
///
/// This function will solve in parallel over the columns of `a`.
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
            solve_lower_triangular_vector_unchecked_mut(a, &mut DVector::from_column_slice(col))
        });
}

fn solve_lower_triangular_vector_unchecked_mut(a: &DMatrix<f64>, b: &mut DVector<f64>) {
    let dim = a.nrows();

    for i in 0..dim {
        let coeff;

        unsafe {
            let diag = a.get_unchecked((i, i)).clone();
            coeff = b.vget_unchecked(i).clone() / diag;
            *b.vget_unchecked_mut(i) = coeff.clone();
        }

        b.rows_range_mut(i + 1..)
            .axpy(-coeff.clone(), &a.slice_range(i + 1.., i), 1.0);
    }
}
