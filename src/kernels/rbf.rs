use crate::parameterized::Parameterized;

use super::{errors::IncompatibleShapeError, kernel::Kernel};
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
/// let kern = RBF::new(DVector::from_vec(vec![1.0, 2.0]));
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
/// use nalgebra::DVector;
///
/// let kern = RBF::from_params(DVector::from_vec(vec![-0.5, -0.125]));
/// ```
#[derive(Debug)]
pub struct RBF {
    gamma: Vec<f64>,
}

impl RBF {
    /// Create a new kernel from a length scale. Length scales are squared.
    pub fn new<'a, I>(length_scale: I) -> Self
    where
        I: Iterator<Item = &'a f64>,
    {
        let gamma = Self::gamma(length_scale);
        RBF { gamma }
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
    ///
    /// Unsafe to call if the dimensions of the x and y slices is not the same as the kernel length scale
    unsafe fn call_slice_unchecked(&self, x_slice: &[f64], y_slice: &[f64]) -> f64 {
        self.gamma
            .iter()
            .enumerate()
            .map(|(index, g)| {
                let diff = x_slice.get_unchecked(index) - y_slice.get_unchecked(index);
                diff * diff * g
            })
            .sum::<f64>()
            .exp()
    }
}

impl Kernel<Vec<f64>> for RBF {
    fn call_into<'x, 'y>(
        &self,
        x: &'x DMatrix<f64>,
        y: &'y DMatrix<f64>,
        into: &mut DMatrix<f64>,
    ) -> Result<(), IncompatibleShapeError> {
        let x_shape = x.shape();
        let y_shape = y.shape();
        let into_shape = into.shape();

        if x_shape.0 != self.gamma.len()
            || y_shape.0 != self.gamma.len()
            || into_shape != (x_shape.1, y_shape.1)
        {
            return Err(IncompatibleShapeError {
                shapes: vec![x_shape, y_shape, (1, self.gamma.len()), into_shape],
            });
        }

        into.as_mut_slice()
            .into_par_iter()
            .enumerate()
            .for_each(|(i, v)| {
                let xi = i % x_shape.1;
                let yi = i % y_shape.1;
                unsafe {
                    *v =
                        self.call_slice_unchecked(x.column(xi).as_slice(), y.column(yi).as_slice());
                }
            });

        return Ok(());
    }

    fn call(
        &self,
        x: &DMatrix<f64>,
        y: &DMatrix<f64>,
    ) -> Result<DMatrix<f64>, IncompatibleShapeError> {
        let x_shape = x.shape();
        let y_shape = y.shape();
        let mut value = DMatrix::<f64>::zeros(x_shape.1, y_shape.1);

        self.call_into(x, y, &mut value)?;

        return Ok(value);
    }

    fn call_symmetric_into(
        &self,
        x: &DMatrix<f64>,
        into: &mut DMatrix<f64>,
    ) -> Result<(), IncompatibleShapeError> {
        let x_shape = x.shape();
        let into_shape = into.shape();

        if x_shape.0 != self.gamma.len() || into_shape != (x_shape.1, x_shape.1) {
            return Err(IncompatibleShapeError {
                shapes: vec![x_shape, (1, self.gamma.len()), into_shape],
            });
        }

        for (i, x_slice) in x.column_iter().enumerate() {
            for j in i..x_shape.1 {
                let y_slice = x.column(j);

                let cell =
                    unsafe { self.call_slice_unchecked(&x_slice.as_slice(), &y_slice.as_slice()) };

                unsafe { *into.get_unchecked_mut((i, j)) = cell };

                if i != j {
                    unsafe { *into.get_unchecked_mut((j, i)) = cell };
                }
            }
        }

        return Ok(());
    }

    fn call_symmetric(&self, x: &DMatrix<f64>) -> Result<DMatrix<f64>, IncompatibleShapeError> {
        let x_shape = x.shape();

        let mut value = DMatrix::<f64>::zeros(x_shape.1, x_shape.1);

        self.call_symmetric_into(x, &mut value)?;

        return Ok(value);
    }
}

impl Parameterized<Vec<f64>> for RBF {
    fn get_params(&self) -> &Vec<f64> {
        &self.gamma
    }

    fn set_params(&mut self, params: Vec<f64>) {
        self.gamma = params;
    }

    fn from_params(params: Vec<f64>) -> Self {
        return RBF { gamma: params };
    }
}

#[cfg(test)]
mod tests {
    use crate::kernels::{Kernel, RBF};
    use nalgebra::DMatrix;

    fn create(v: Vec<f64>) -> RBF {
        RBF::new(v.iter())
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

        assert_eq!(k[0], (-8.0 as f64).exp());
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

        assert_eq!(k[0], (-9.125 as f64).exp());
    }
}
