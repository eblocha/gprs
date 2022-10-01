use nalgebra::{Dim, Matrix, Storage};
use rayon::prelude::*;

use crate::kernels::errors::IncompatibleShapeError;

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

    // iterate down cols of rhs, zipping with the rows of the lhs. Outer loop is rhs cols, first inner loop is lhs rows.
    // Second inner loop is zip of lhs cols and rhs rows.
    let vals: Vec<f64> = (0..r_shape.1)
        .into_par_iter()
        .flat_map(move |rj| {
            (0..l_shape.0).into_par_iter().map(move |li| {
                (0..r_shape.0)
                    .into_par_iter()
                    .zip((0..l_shape.1).into_par_iter())
                    // SAFETY: indices are inherently valid since they come from the corresponding shapes
                    .map(move |(ri, lj)| unsafe {
                        lhs.get_unchecked((li, lj)) * rhs.get_unchecked((ri, rj))
                    })
                    .sum::<f64>()
            })
        })
        .collect();

    Ok(vals)
}
