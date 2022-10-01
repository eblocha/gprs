use crate::{
    indexing::{index_to_2d, slice_indices},
    parameterized::Parameterized,
};

use super::{
    errors::IncompatibleShapeError,
    kernel::{Kernel, TriangleSide},
};
use nalgebra::DMatrix;
use rayon::prelude::*;

/// Radial Basis Function kernel
///
/// `K = exp(-||x - x'||^2 / (2 * l^2))`
///
/// where `||x - x'||` is the euclidean distance between vectors x and x',
/// and `l` is the length scale
///
/// # Examples
///
/// ```rust
/// use gprs::kernels::{RBF,Kernel};
/// use nalgebra::{DVector,DMatrix};
///
/// // create a 2-d RBF kernel
/// let kern = RBF::new(vec![1.0, 2.0].iter(), 1.0);
/// // estimate covariance between 2 sets of points
/// let x = DMatrix::from_vec(2, 3, vec![
///     1.8, 5.5,
///     1.5, 4.5,
///     2.3, 4.6
/// ]);
///
/// let y = DMatrix::from_vec(2, 4, vec![
///     2.2, 3.0,
///     1.8, 5.5,
///     1.5, 4.5,
///     2.3, 4.6
/// ]);
///
/// let k = kern.call(&x, &y).unwrap();
/// assert_eq!(k.shape(), (3, 4));
/// ```
///
/// Create a kernel using gamma directly:
///
/// ```rust
/// use gprs::kernels::{RBF,Kernel};
/// use gprs::parameterized::Parameterized;
/// use nalgebra::DVector;
///
/// let kern = RBF::from_params((&vec![-0.5, -0.125], 1.0));
/// ```
#[derive(Debug)]
pub struct RBF {
    gamma: Vec<f64>,
    amplitude: f64,
}

impl RBF {
    /// Create a new kernel from a length scale. Length scales are squared.
    pub fn new<'a, I>(length_scale: I, sigma: f64) -> Self
    where
        I: Iterator<Item = &'a f64>,
    {
        let gamma = Self::gamma(length_scale);
        RBF {
            gamma,
            amplitude: sigma * sigma,
        }
    }

    /// Compute the gamma property from a length scale vec
    ///
    /// This speeds up covariance computation by pre-computing `-1 / (2 * l^2)`
    fn gamma<'a, I>(length_scale: I) -> Vec<f64>
    where
        I: Iterator<Item = &'a f64>,
    {
        length_scale.map(|v| -0.5 / (v * v)).collect()
    }

    /// Compute the covariance between 2 points
    fn call_point(&self, x_point: &[f64], y_point: &[f64]) -> f64 {
        let unscaled = self
            .gamma
            .iter()
            .zip(x_point)
            .zip(y_point)
            .map(|((g, x), y)| {
                let diff = x - y;
                diff * diff * g
            })
            .sum::<f64>()
            .exp();

        unscaled * self.amplitude
    }

    fn check_shapes(
        &self,
        x_shape: (usize, usize),
        y_shape: (usize, usize),
        into_shape: (usize, usize),
    ) -> Result<(), IncompatibleShapeError> {
        if x_shape.0 != self.gamma.len()
            || y_shape.0 != self.gamma.len()
            || into_shape != (x_shape.1, y_shape.1)
        {
            return Err(IncompatibleShapeError {
                shapes: vec![x_shape, y_shape, (1, self.gamma.len()), into_shape],
            });
        }

        Ok(())
    }

    fn call_triangular_inplace<'x>(
        &self,
        x: &'x DMatrix<f64>,
        side: TriangleSide,
        into: &mut DMatrix<f64>,
    ) -> Result<(), IncompatibleShapeError> {
        let x_shape = x.shape();
        let into_shape = into.shape();

        self.check_shapes(x_shape, x_shape, into_shape)?;

        let dims = x_shape.0;
        let x_sl = x.as_slice();

        into.as_mut_slice()
            .into_par_iter()
            .enumerate()
            .map(|(index, v)| {
                let (i, j) = index_to_2d(index, x_shape.1);
                (i, j, v)
            })
            .filter(|(i, j, _v)| match side {
                TriangleSide::LOWER => i <= j,
                TriangleSide::UPPER => i >= j,
            })
            .for_each(|(i, j, v)| {
                let (xs, xe) = slice_indices(i, dims);
                let (ys, ye) = slice_indices(j, dims);

                // SAFETY: the indices are valid because we checked them at the beginning of the function
                unsafe {
                    let x_point = &x_sl.get_unchecked(xs..xe);
                    let y_point = &x_sl.get_unchecked(ys..ye);
                    *v = self.call_point(x_point, y_point);
                }
            });

        Ok(())
    }
}

impl Kernel for RBF {
    fn call_inplace<'x, 'y>(
        &self,
        x: &'x DMatrix<f64>,
        y: &'y DMatrix<f64>,
        into: &mut DMatrix<f64>,
    ) -> Result<(), IncompatibleShapeError> {
        let x_shape = x.shape();
        let y_shape = y.shape();
        let into_shape = into.shape();

        self.check_shapes(x_shape, y_shape, into_shape)?;

        let dims = x_shape.0;
        let x_sl = x.as_slice();
        let y_sl = y.as_slice();

        into.as_mut_slice()
            .into_par_iter()
            .enumerate()
            .for_each(|(index, v)| {
                let (i, j) = index_to_2d(index, y_shape.1);
                let (xs, xe) = slice_indices(i, dims);
                let (ys, ye) = slice_indices(j, dims);

                // SAFETY: the indices are valid because we checked them at the beginning of the function
                unsafe {
                    let x_point = &x_sl.get_unchecked(xs..xe);
                    let y_point = &y_sl.get_unchecked(ys..ye);
                    *v = self.call_point(x_point, y_point);
                }
            });

        Ok(())
    }

    fn call(
        &self,
        x: &DMatrix<f64>,
        y: &DMatrix<f64>,
    ) -> Result<DMatrix<f64>, IncompatibleShapeError> {
        let x_shape = x.shape();
        let y_shape = y.shape();
        let mut value = DMatrix::<f64>::zeros(x_shape.1, y_shape.1);

        self.call_inplace(x, y, &mut value)?;

        Ok(value)
    }

    fn call_triangular<'x>(
        &self,
        x: &'x DMatrix<f64>,
        side: TriangleSide,
    ) -> Result<DMatrix<f64>, IncompatibleShapeError> {
        let x_shape = x.shape();
        let mut value = DMatrix::<f64>::zeros(x_shape.1, x_shape.1);

        self.call_triangular_inplace(x, side, &mut value)?;

        Ok(value)
    }
}

/// Clone a vector with cloneable elements
fn clone_vec<T: Clone>(vec: &[T]) -> Vec<T> {
    vec.to_vec()
}

impl<'a> Parameterized<'a, (&'a Vec<f64>, f64)> for RBF {
    fn get_params(&'a self) -> (&'a Vec<f64>, f64) {
        (&self.gamma, self.amplitude)
    }

    fn set_params<'b>(&'a mut self, params: (&'b Vec<f64>, f64)) {
        self.gamma = clone_vec(params.0);
        self.amplitude = params.1
    }

    fn from_params(params: (&Vec<f64>, f64)) -> Self {
        RBF {
            gamma: clone_vec(params.0),
            amplitude: params.1,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::kernels::{Kernel, RBF};
    use nalgebra::DMatrix;

    fn create(v: Vec<f64>) -> RBF {
        RBF::new(v.iter(), 1.0)
    }

    /// Passing invalid data will return an error
    #[test]
    #[should_panic]
    fn test_mismatched() {
        let kern = create(vec![1.0]);

        let x = DMatrix::from_vec(2, 1, vec![1.0, 1.0]);
        let y = DMatrix::from_vec(2, 1, vec![1.0, 1.0]);
        kern.call(&x, &y).unwrap();
    }

    /// Passing a zero lengthscale will produce NaN
    #[test]
    fn test_zero_lengthscale() {
        let kern = create(vec![0.0]);
        let x = DMatrix::from_vec(1, 1, vec![1.0]);

        let y = DMatrix::from_vec(1, 1, vec![1.0]);
        let k = kern.call(&x, &y).unwrap();

        assert!(k[0].is_nan());
    }

    /// The covariance of a point to itself is 1.0
    #[test]
    fn test_1d_identity() {
        let kern = create(vec![1.0]);

        let x = DMatrix::from_vec(1, 1, vec![1.0]);
        let y = DMatrix::from_vec(1, 1, vec![1.0]);
        let k = kern.call(&x, &y).unwrap();

        assert_eq!(k[0], 1.0);
    }

    /// The covariance function is commutative
    #[test]
    fn test_1d_symmetry() {
        let kern = create(vec![1.0]);

        let x = DMatrix::from_vec(1, 1, vec![1.0]);
        let y = DMatrix::from_vec(1, 1, vec![2.0]);
        let k1 = kern.call(&x, &y).unwrap();
        let k2 = kern.call(&y, &x).unwrap();

        assert_eq!(k1, k2);
    }

    #[test]
    fn test_1d_correctness() {
        let kern = create(vec![0.5]);
        // gamma should be -0.5 / 0.25 = -2.0

        let x = DMatrix::from_vec(1, 1, vec![1.0]);
        let y = DMatrix::from_vec(1, 1, vec![3.0]);
        let k = kern.call(&x, &y).unwrap();

        assert_eq!(k[0], (-8.0_f64).exp());
    }

    #[test]
    fn test_2d_identity() {
        let kern = create(vec![1.0, 2.0]);

        let x = DMatrix::from_vec(2, 1, vec![1.0, 1.0]);
        let y = DMatrix::from_vec(2, 1, vec![1.0, 1.0]);
        let k = kern.call(&x, &y).unwrap();

        assert_eq!(k[0], 1.0);
    }

    #[test]
    fn test_2d_symmetry() {
        let kern = create(vec![1.0, 2.0]);

        let x = DMatrix::from_vec(2, 1, vec![1.0, 1.0]);
        let y = DMatrix::from_vec(2, 1, vec![2.0, 2.0]);
        let k1 = kern.call(&x, &y).unwrap();
        let k2 = kern.call(&y, &x).unwrap();

        assert_eq!(k1, k2);
    }

    #[test]
    fn test_2d_correctness() {
        let kern = create(vec![0.5, 2.0]);
        // gamma = -0.5 / [0.25, 4.0] = [-2.0, -0.125]

        let x = DMatrix::from_vec(2, 1, vec![1.0, 1.0]);
        let y = DMatrix::from_vec(2, 1, vec![3.0, 4.0]);
        let k = kern.call(&x, &y).unwrap();

        assert_eq!(k[0], (-9.125_f64).exp());
    }
}
