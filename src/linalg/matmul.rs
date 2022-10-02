use std::iter::Sum;

use nalgebra::{Dim, Matrix, Storage};
use rayon::prelude::*;

use super::errors::IncompatibleShapeError;

/// Parallel matrix multiplication implementation with rayon
///
/// # Examples
/// ```rust
/// use nalgebra::DMatrix;
/// use gprs::linalg::par_matmul;
///
/// // these look transposed since they are stored column-major
///
/// let lhs = DMatrix::from_vec(2, 3, vec![
///     1.0, 4.0,
///     2.0, 5.0,
///     3.0, 6.0,
/// ]);
///
/// let rhs = DMatrix::from_vec(3, 2, vec![
///     7.0,  9.0, 11.0,
///     8.0, 10.0, 12.0,
/// ]);
///
///
/// let expected = vec![
///     58.0, 139.0,
///     64.0, 154.0,
/// ];
///
/// assert_eq!(par_matmul(&lhs, &rhs).unwrap(), expected);
/// ```
pub fn par_matmul<LI, LJ, RI, RJ, SL, SR>(
    lhs: &Matrix<f64, LI, LJ, SL>,
    rhs: &Matrix<f64, RI, RJ, SR>,
) -> Result<Vec<f64>, IncompatibleShapeError>
where
    LI: Dim,
    LJ: Dim,
    RI: Dim,
    RJ: Dim,
    SL: Storage<f64, LI, LJ> + Sync,
    SR: Storage<f64, RI, RJ> + Sync,
{
    let l_shape = lhs.shape();
    let r_shape = rhs.shape();

    // nrows of lhs must == ncols of rhs
    if l_shape.1 != r_shape.0 {
        return Err(IncompatibleShapeError {
            shapes: vec![l_shape, r_shape],
        });
    }

    Ok(matmul_wrapper(
        l_shape,
        r_shape,
        |(li, lj), (ri, rj)| unsafe {
            // SAFETY: indices are inherently valid since they come from the corresponding shapes
            lhs.get_unchecked((li, lj)) * rhs.get_unchecked((ri, rj))
        },
    ))
}

/// Parallel transpose matrix multiplication, equivalent to par_matmul(&lhs.transpose(), &rhs), but more efficient
///
/// # Examples
/// ```rust
/// use nalgebra::DMatrix;
/// use gprs::linalg::par_tr_matmul;
///
/// // these look transposed since they are stored column-major
///
/// let v = DMatrix::from_vec(3, 3, vec![
///     1.0, 4.0, 7.0,
///     2.0, 5.0, 8.0,
///     3.0, 6.0, 9.0,
/// ]);
///
///
/// let expected = vec![
///     66.0,  78.0,  90.0,
///     78.0,  93.0, 108.0,
///     90.0, 108.0, 126.0,
/// ];
///
/// assert_eq!(par_tr_matmul(&v, &v).unwrap(), expected);
/// ```
pub fn par_tr_matmul<LI, LJ, RI, RJ, SL, SR>(
    lhs: &Matrix<f64, LI, LJ, SL>,
    rhs: &Matrix<f64, RI, RJ, SR>,
) -> Result<Vec<f64>, IncompatibleShapeError>
where
    LI: Dim,
    LJ: Dim,
    RI: Dim,
    RJ: Dim,
    SL: Storage<f64, LI, LJ> + Sync,
    SR: Storage<f64, RI, RJ> + Sync,
{
    let l_shape = (lhs.shape().1, lhs.shape().0);
    let r_shape = rhs.shape();

    // nrows of lhs must == ncols of rhs
    if l_shape.1 != r_shape.0 {
        return Err(IncompatibleShapeError {
            shapes: vec![l_shape, r_shape],
        });
    }

    Ok(matmul_wrapper(
        l_shape,
        r_shape,
        |(li, lj), (ri, rj)| unsafe {
            // SAFETY: indices are inherently valid
            lhs.get_unchecked((lj, li)) * rhs.get_unchecked((ri, rj))
        },
    ))
}

/// Only compute the diagonal for transposed matrix multiplication
///
/// # Examples
/// ```rust
/// use nalgebra::DMatrix;
/// use gprs::linalg::par_tr_matmul_diag;
///
/// // these look transposed since they are stored column-major
///
/// let v = DMatrix::from_vec(3, 3, vec![
///     1.0, 4.0, 7.0,
///     2.0, 5.0, 8.0,
///     3.0, 6.0, 9.0,
/// ]);
///
///
/// let expected = vec![66.0, 93.0, 126.0];
///
/// assert_eq!(par_tr_matmul_diag(&v, &v).unwrap(), expected);
pub fn par_tr_matmul_diag<LI, LJ, RI, RJ, SL, SR>(
    lhs: &Matrix<f64, LI, LJ, SL>,
    rhs: &Matrix<f64, RI, RJ, SR>,
) -> Result<Vec<f64>, IncompatibleShapeError>
where
    LI: Dim,
    LJ: Dim,
    RI: Dim,
    RJ: Dim,
    SL: Storage<f64, LI, LJ> + Sync,
    SR: Storage<f64, RI, RJ> + Sync,
{
    let l_shape = (lhs.shape().1, lhs.shape().0);
    let r_shape = rhs.shape();

    // nrows of lhs must == ncols of rhs
    if l_shape.1 != r_shape.0 {
        return Err(IncompatibleShapeError {
            shapes: vec![l_shape, r_shape],
        });
    }

    Ok(matmul_wrapper_diag(
        l_shape,
        r_shape,
        |(li, lj), (ri, rj)| unsafe {
            // SAFETY: indices are inherently valid
            lhs.get_unchecked((lj, li)) * rhs.get_unchecked((ri, rj))
        },
    ))
}

/// Iteration wrapper for matrix multiplication. Applies `op` to each element pair (passes index), then sums the result
fn matmul_wrapper<O, R>(l_shape: (usize, usize), r_shape: (usize, usize), op: O) -> Vec<R>
where
    R: Sync + Send + Sum<R>,
    O: Fn((usize, usize), (usize, usize)) -> R + Sync + Send,
{
    let op_ref = &op;
    (0..r_shape.1)
        .into_par_iter()
        .flat_map(move |rj| {
            (0..l_shape.0).into_par_iter().map(move |li| {
                (0..r_shape.0)
                    .zip(0..l_shape.1)
                    .map(move |(ri, lj)| op_ref((li, lj), (ri, rj)))
                    .sum::<R>()
            })
        })
        .collect()
}

/// Iteration wrapper for diagonal-only matrix multiplication
fn matmul_wrapper_diag<O, R>(l_shape: (usize, usize), r_shape: (usize, usize), op: O) -> Vec<R>
where
    R: Sync + Send + Sum<R>,
    O: Fn((usize, usize), (usize, usize)) -> R + Sync + Send,
{
    let op_ref = &op;
    (0..r_shape.1)
        .into_par_iter()
        .map(move |li_rj| {
            (0..r_shape.0)
                .zip(0..l_shape.1)
                .map(move |(ri, lj)| op_ref((li_rj, lj), (ri, li_rj)))
                .sum::<R>()
        })
        .collect()
}
