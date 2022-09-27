use super::errors::{CovarianceParamLengthError, InvalidKernelError};
use super::kernel::{CovarianceResult, Kernel, KernelResult};

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
/// let length_scale = vec![1.0, 2.0];
/// let kern = RBF::<2>::new(&length_scale).unwrap();
/// // estimate covariance between 2 points
/// let (x, y) = (vec![1.8, 5.5], vec![2.2, 3.0]);
/// let k = kern.call(&x, &y).unwrap();
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
    pub fn new(length_scale: &Vec<f64>) -> KernelResult<Self> {
        let gamma = Self::gamma(length_scale)?;
        Ok(RBF { gamma })
    }

    /// Compute the gamma property from a length scale vec
    ///
    /// This speeds up covariance computation by pre-computing `-1 / (2 * l^2)`
    fn gamma(length_scale: &Vec<f64>) -> KernelResult<[f64; DIMS]> {
        if length_scale.len() != DIMS {
            return Err(InvalidKernelError::LengthScaleSizeInvalid);
        }

        let all_positive = length_scale.iter().all(|l| *l > 0.0);

        if !all_positive {
            return Err(InvalidKernelError::NonPositiveLengthScale);
        }

        let mut gamma: [f64; DIMS] = [1.0; DIMS];

        length_scale
            .iter()
            .enumerate()
            .for_each(|(index, v)| gamma[index] = -0.5 / (v * v));

        return Ok(gamma);
    }
}

impl<const DIMS: usize> Kernel<[f64; DIMS]> for RBF<DIMS> {
    fn call(&self, x: &Vec<f64>, y: &Vec<f64>) -> CovarianceResult<f64> {
        let x_len = x.len();
        let y_len = y.len();

        if x_len != DIMS || y_len != DIMS {
            return Err(CovarianceParamLengthError {
                x_len,
                y_len,
                expected: DIMS,
            });
        }

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

        return Ok(k);
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
    fn test_invalid_length() {
        RBF::<1>::new(&vec![1.0, 1.0]).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_zero_lengthscale() {
        RBF::<2>::new(&vec![1.0, 0.0]).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_neg_lengthscale() {
        RBF::<2>::new(&vec![1.0, -1.0]).unwrap();
    }

    /// The covariance of a point to itself is 1.0
    #[test]
    fn test_1d_identity() {
        let kern = RBF::<1>::new(&vec![1.0]).unwrap();

        let (x, y) = (vec![1.0], vec![1.0]);
        let k = kern.call(&x, &y).unwrap();

        assert_eq!(k, 1.0);
    }

    /// The covariance function is commutative
    #[test]
    fn test_1d_symmetry() {
        let kern = RBF::<1>::new(&vec![1.0]).unwrap();

        let (x, y) = (vec![1.0], vec![2.0]);
        let k1 = kern.call(&x, &y).unwrap();
        let k2 = kern.call(&y, &x).unwrap();

        assert_eq!(k1, k2);
    }

    #[test]
    fn test_1d_correctness() {
        let kern = RBF::<1>::new(&vec![0.5]).unwrap();
        // gamma should be -0.5 / 0.25 = -2.0

        let (x, y) = (vec![1.0], vec![3.0]);
        let k = kern.call(&x, &y).unwrap();

        assert_eq!(k, (-8.0 as f64).exp());
    }

    #[test]
    fn test_2d_identity() {
        let kern = RBF::<2>::new(&vec![1.0, 2.0]).unwrap();

        let (x, y) = (vec![1.0, 1.0], vec![1.0, 1.0]);
        let k = kern.call(&x, &y).unwrap();

        assert_eq!(k, 1.0);
    }

    #[test]
    fn test_2d_symmetry() {
        let kern = RBF::<2>::new(&vec![1.0, 2.0]).unwrap();

        let (x, y) = (vec![1.0, 1.0], vec![2.0, 2.0]);
        let k1 = kern.call(&x, &y).unwrap();
        let k2 = kern.call(&y, &x).unwrap();

        assert_eq!(k1, k2);
    }

    #[test]
    fn test_2d_correctness() {
        let kern = RBF::<2>::new(&vec![0.5, 2.0]).unwrap();
        // gamma = -0.5 / [0.25, 4.0] = [-2.0, -0.125]

        let (x, y) = (vec![1.0, 1.0], vec![3.0, 4.0]);
        let k = kern.call(&x, &y).unwrap();

        assert_eq!(k, (-9.125 as f64).exp());
    }
}
