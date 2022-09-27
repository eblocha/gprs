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
/// ```no_run
/// // create a 2-d RBF kernel
/// let length_scale = vec![1.0, 2.0]
/// let kern = RBF::new(&length_scale)?;
/// // estimate covariance between 2 points
/// let (x, y) = (vec![1.8, 5.5], vec![2.2, 3.0]);
/// let K = kern.call(&x, &y)?;
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
