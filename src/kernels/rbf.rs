use nalgebra::{DMatrix, DVector};

use super::{errors::IncompatibleShapeError, kernel::Kernel};

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
/// let x = DMatrix::from_vec(3, 2, vec![
///     1.8, 5.5,
///     1.5, 4.5,
///     2.3, 4.6
/// ]);
///
/// let y = DMatrix::from_vec(4, 2, vec![
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
    gamma: DVector<f64>,
}

impl RBF {
    /// Create a new kernel from a length scale. Length scales are squared.
    pub fn new(length_scale: DVector<f64>) -> Self {
        let gamma = Self::gamma(length_scale);
        RBF { gamma }
    }

    /// Compute the gamma property from a length scale vec
    ///
    /// This speeds up covariance computation by pre-computing `-1 / (2 * l^2)`
    fn gamma(length_scale: DVector<f64>) -> DVector<f64> {
        return length_scale.map(|v| -0.5 / (v * v));
    }
}

impl Kernel<DVector<f64>> for RBF {
    fn call(
        &self,
        x: &DMatrix<f64>,
        y: &DMatrix<f64>,
    ) -> Result<DMatrix<f64>, IncompatibleShapeError> {
        let x_shape = x.shape();
        let y_shape = y.shape();

        if x_shape.1 != self.gamma.len() || y_shape.1 != self.gamma.len() {
            return Err(IncompatibleShapeError {
                shapes: vec![x_shape, y_shape, self.gamma.shape()],
            });
        }

        // initialize memory
        let mut value = DMatrix::<f64>::zeros(x_shape.0, y_shape.0);

        for (i, x_slice) in x.row_iter().enumerate() {
            for (j, y_slice) in y.row_iter().enumerate() {
                *value.index_mut((i, j)) = self
                    .gamma
                    .iter()
                    .enumerate()
                    .map(|(index, g)| {
                        let diff = x_slice[index] - y_slice[index];
                        diff * diff * g
                    })
                    .sum::<f64>()
                    .exp();
            }
        }

        return Ok(value);
    }

    fn get_params(&self) -> &DVector<f64> {
        &self.gamma
    }

    fn set_params(&mut self, params: DVector<f64>) {
        self.gamma = params;
    }

    fn from_params(params: DVector<f64>) -> Self {
        return RBF { gamma: params };
    }
}

#[cfg(test)]
mod tests {
    use crate::kernels::{Kernel, RBF};
    use nalgebra::{DMatrix, DVector};

    fn create(v: Vec<f64>) -> RBF {
        let length_scale = DVector::from_vec(v);
        RBF::new(length_scale)
    }

    /// Passing invalid data will return an error
    #[test]
    #[should_panic]
    fn test_mismatched() {
        let kern = create(vec![1.0]);

        let x = DMatrix::from_vec(1, 2, vec![1.0, 1.0]);
        let y = DMatrix::from_vec(1, 2, vec![1.0, 1.0]);
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

        let x = DMatrix::from_vec(1, 2, vec![1.0, 1.0]);
        let y = DMatrix::from_vec(1, 2, vec![1.0, 1.0]);
        let k = kern.call(&x, &y).unwrap();

        assert_eq!(k[0], 1.0);
    }

    #[test]
    fn test_2d_symmetry() {
        let kern = create(vec![1.0, 2.0]);

        let x = DMatrix::from_vec(1, 2, vec![1.0, 1.0]);
        let y = DMatrix::from_vec(1, 2, vec![2.0, 2.0]);
        let k1 = kern.call(&x, &y).unwrap();
        let k2 = kern.call(&y, &x).unwrap();

        assert_eq!(k1, k2);
    }

    #[test]
    fn test_2d_correctness() {
        let kern = create(vec![0.5, 2.0]);
        // gamma = -0.5 / [0.25, 4.0] = [-2.0, -0.125]

        let x = DMatrix::from_vec(1, 2, vec![1.0, 1.0]);
        let y = DMatrix::from_vec(1, 2, vec![3.0, 4.0]);
        let k = kern.call(&x, &y).unwrap();

        assert_eq!(k[0], (-9.125 as f64).exp());
    }
}
