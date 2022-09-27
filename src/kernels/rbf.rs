use super::errors::InvalidKernelError;
use super::kernel::{Kernel, KernelResult};

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
/// // create a 2-d RBF kernel
/// let kern = RBF::new([1.0, 2.0]).unwrap();
/// // estimate covariance between 2 points
/// let (x, y) = ([1.8, 5.5], [2.2, 3.0]);
/// let k = kern.call(x, y);
/// ```
///
/// Create a kernel using gamma directly:
///
/// ```rust
/// use gprs::kernels::{RBF,Kernel};
/// let kern = RBF::from_params([1.0, 2.0]);
/// ```
#[derive(Debug)]
pub struct RBF<const DIMS: usize> {
    gamma: [f64; DIMS],
}

impl<const DIMS: usize> RBF<DIMS> {
    pub fn new(length_scale: [f64; DIMS]) -> KernelResult<Self> {
        let gamma = Self::gamma(length_scale)?;
        Ok(RBF { gamma })
    }

    /// Compute the gamma property from a length scale vec
    ///
    /// This speeds up covariance computation by pre-computing `-1 / (2 * l^2)`
    fn gamma(length_scale: [f64; DIMS]) -> KernelResult<[f64; DIMS]> {
        let all_positive = length_scale.iter().all(|l| *l > 0.0);

        if !all_positive {
            return Err(InvalidKernelError::NonPositiveLengthScale);
        }

        return Ok(length_scale.map(|v| -0.5 / (v * v)));
    }
}

impl<const DIMS: usize> Kernel<[f64; DIMS], DIMS> for RBF<DIMS> {
    fn call(&self, x: [f64; DIMS], y: [f64; DIMS]) -> f64 {
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

    fn get_params(&self) -> [f64; DIMS] {
        self.gamma
    }

    fn set_params(&mut self, params: [f64; DIMS]) {
        self.gamma = params;
    }

    fn from_params(params: [f64; DIMS]) -> Self {
        return RBF { gamma: params };
    }
}

#[cfg(test)]
mod tests {
    use crate::kernels::{Kernel, RBF};

    #[test]
    #[should_panic]
    fn test_zero_lengthscale() {
        RBF::<2>::new([1.0, 0.0]).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_neg_lengthscale() {
        RBF::<2>::new([1.0, -1.0]).unwrap();
    }

    /// The covariance of a point to itself is 1.0
    #[test]
    fn test_1d_identity() {
        let kern = RBF::new([1.0]).unwrap();

        let (x, y) = ([1.0], [1.0]);
        let k = kern.call(x, y);

        assert_eq!(k, 1.0);
    }

    /// The covariance function is commutative
    #[test]
    fn test_1d_symmetry() {
        let kern = RBF::new([1.0]).unwrap();

        let (x, y) = ([1.0], [2.0]);
        let k1 = kern.call(x, y);
        let k2 = kern.call(y, x);

        assert_eq!(k1, k2);
    }

    #[test]
    fn test_1d_correctness() {
        let kern = RBF::new([0.5]).unwrap();
        // gamma should be -0.5 / 0.25 = -2.0

        let (x, y) = ([1.0], [3.0]);
        let k = kern.call(x, y);

        assert_eq!(k, (-8.0 as f64).exp());
    }

    #[test]
    fn test_2d_identity() {
        let kern = RBF::new([1.0, 2.0]).unwrap();

        let (x, y) = ([1.0, 1.0], [1.0, 1.0]);
        let k = kern.call(x, y);

        assert_eq!(k, 1.0);
    }

    #[test]
    fn test_2d_symmetry() {
        let kern = RBF::new([1.0, 2.0]).unwrap();

        let (x, y) = ([1.0, 1.0], [2.0, 2.0]);
        let k1 = kern.call(x, y);
        let k2 = kern.call(y, x);

        assert_eq!(k1, k2);
    }

    #[test]
    fn test_2d_correctness() {
        let kern = RBF::new([0.5, 2.0]).unwrap();
        // gamma = -0.5 / [0.25, 4.0] = [-2.0, -0.125]

        let (x, y) = ([1.0, 1.0], [3.0, 4.0]);
        let k = kern.call(x, y);

        assert_eq!(k, (-9.125 as f64).exp());
    }
}
