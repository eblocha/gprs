use nalgebra::DVector;

use super::kernel::Kernel;

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
/// use nalgebra::DVector;
///
/// // create a 2-d RBF kernel
/// let kern = RBF::new(DVector::from_vec(vec![1.0, 2.0]));
/// // estimate covariance between 2 points
/// let (x, y) = (DVector::from_vec(vec![1.8, 5.5]), DVector::from_vec(vec![2.2, 3.0]));
/// let k = kern.call(&x, &y);
/// ```
///
/// Create a kernel using gamma directly:
///
/// ```rust
/// use gprs::kernels::{RBF,Kernel};
/// use nalgebra::DVector;
///
/// let kern = RBF::from_params(DVector::from_vec(vec![1.0, 2.0]));
/// ```
#[derive(Debug)]
pub struct RBF {
    gamma: DVector<f64>,
}

impl RBF {
    /// Create a new kernel from a length scale. Length scales are squared.
    ///
    /// # Panics
    /// Panics if any length scale == 0.0
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
    fn call(&self, x: &DVector<f64>, y: &DVector<f64>) -> f64 {
        let k = self
            .gamma
            .iter()
            .enumerate()
            .map(|(index, g)| {
                let diff = x[index] - y[index];
                diff * diff * g
            })
            .sum::<f64>()
            .exp();

        return k;
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
    use nalgebra::DVector;

    use crate::kernels::{Kernel, RBF};

    fn create(v: Vec<f64>) -> RBF {
        let length_scale = DVector::from_vec(v);
        RBF::new(length_scale)
    }

    fn create_points(x: Vec<f64>, y: Vec<f64>) -> (DVector<f64>, DVector<f64>) {
        (DVector::from_vec(x), DVector::from_vec(y))
    }

    /// The covariance of a point to itself is 1.0
    #[test]
    fn test_1d_identity() {
        let kern = create(vec![1.0]);

        let (x, y) = create_points(vec![1.0], vec![1.0]);
        let k = kern.call(&x, &y);

        assert_eq!(k, 1.0);
    }

    /// The covariance function is commutative
    #[test]
    fn test_1d_symmetry() {
        let kern = create(vec![1.0]);

        let (x, y) = create_points(vec![1.0], vec![2.0]);
        let k1 = kern.call(&x, &y);
        let k2 = kern.call(&y, &x);

        assert_eq!(k1, k2);
    }

    #[test]
    fn test_1d_correctness() {
        let kern = create(vec![0.5]);
        // gamma should be -0.5 / 0.25 = -2.0

        let (x, y) = create_points(vec![1.0], vec![3.0]);
        let k = kern.call(&x, &y);

        assert_eq!(k, (-8.0 as f64).exp());
    }

    #[test]
    fn test_2d_identity() {
        let kern = create(vec![1.0, 2.0]);

        let (x, y) = create_points(vec![1.0, 1.0], vec![1.0, 1.0]);
        let k = kern.call(&x, &y);

        assert_eq!(k, 1.0);
    }

    #[test]
    fn test_2d_symmetry() {
        let kern = create(vec![1.0, 2.0]);

        let (x, y) = create_points(vec![1.0, 1.0], vec![2.0, 2.0]);
        let k1 = kern.call(&x, &y);
        let k2 = kern.call(&y, &x);

        assert_eq!(k1, k2);
    }

    #[test]
    fn test_2d_correctness() {
        let kern = create(vec![0.5, 2.0]);
        // gamma = -0.5 / [0.25, 4.0] = [-2.0, -0.125]

        let (x, y) = create_points(vec![1.0, 1.0], vec![3.0, 4.0]);
        let k = kern.call(&x, &y);

        assert_eq!(k, (-9.125 as f64).exp());
    }
}
