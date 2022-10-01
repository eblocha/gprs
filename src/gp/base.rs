use nalgebra::{Cholesky, DMatrix, DVector, Dynamic};
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use crate::{
    // indexing::index_to_2d,
    kernels::{Kernel, TriangleSide},
    linalg::{errors::IncompatibleShapeError, par_matmul, par_tr_matmul, util::add_diagonal_mut},
};

use super::errors::GPCompilationError;

/// Standard Gaussian Process
///
/// Definition:
///
/// `f = K*T [K + sI]^-1 y`
///
/// `cov = K** - K*T [K + sI]^-1 K*`
pub struct GP<K: Kernel> {
    kernel: K,
    noise: f64,
}

impl<K: Kernel> GP<K> {
    pub fn new(kernel: K, noise: f64) -> Self {
        GP { kernel, noise }
    }

    /// Compile this GP for training or estimation
    ///
    /// # Examples
    /// ```rust
    /// use gprs::{gp::GP, kernels::{Kernel, RBF}};
    /// use nalgebra::{DVector, DMatrix};
    ///
    /// let kernel = RBF::new(vec![1.0, 2.0].iter(), 1.0);
    ///
    /// let gp = GP::new(
    ///     kernel,
    ///     1.0,
    /// );
    ///
    /// let x = DMatrix::from_vec(2, 3, vec![
    ///     1.8, 5.5,
    ///     1.5, 4.5,
    ///     2.3, 4.6
    /// ]);
    ///
    /// let y = DVector::from_vec(vec![
    ///     2.2,
    ///     1.8,
    ///     1.5,
    /// ]);
    ///
    /// let compiled = gp.compile(&x, &y).unwrap();
    /// ```
    pub fn compile<'a>(
        &'a self,
        x: &'a DMatrix<f64>,
        y: &DVector<f64>,
    ) -> Result<CompiledGP<K>, GPCompilationError> {
        if x.shape().1 != y.len() {
            return Err(GPCompilationError::IncompatibleShapeError(
                IncompatibleShapeError {
                    shapes: vec![x.shape(), y.shape()],
                },
            ));
        }

        let mut kxx = match self.kernel.call_triangular(x, TriangleSide::LOWER) {
            Err(e) => return Err(GPCompilationError::IncompatibleShapeError(e)),
            Ok(v) => v,
        };

        // SAFETY: kxx is guaranteed to be square
        unsafe {
            add_diagonal_mut(&mut kxx, &self.noise);
        }

        // TODO parallel cholesky decomp and solve
        let cholesky = match kxx.cholesky() {
            None => return Err(GPCompilationError::NonPositiveDefiniteError),
            Some(v) => v,
        };
        let alpha = cholesky.solve(y);

        Ok(CompiledGP {
            cholesky,
            alpha,
            kernel: &self.kernel,
            x,
        })
    }
}

pub struct CompiledGP<'kernel, K: Kernel> {
    /// The cholesky decomposition of (K + noise * I)
    cholesky: Cholesky<f64, Dynamic>,
    /// Factor to compute mean
    alpha: DVector<f64>,
    /// The original kernel
    kernel: &'kernel K,
    /// The input data set
    x: &'kernel DMatrix<f64>,
}

impl<'kernel, K: Kernel> CompiledGP<'kernel, K> {
    // /// Compute the mean and variance from input data
    // fn call(
    //     &self,
    //     x: &DMatrix<f64>,
    // ) -> Result<(DMatrix<f64>, DMatrix<f64>), IncompatibleShapeError> {

    // }

    /// Compute the mean from input data
    pub fn mean(&self, x: &DMatrix<f64>) -> Result<DVector<f64>, IncompatibleShapeError> {
        // compute K*T
        let k_x_xp = self.kernel.call(self.x, x)?;
        let res = par_matmul(&k_x_xp, &self.alpha)?;

        Ok(DVector::from_vec(res))
    }

    // /// Compute just the diagonal variance
    // pub fn var(&self, x: &DMatrix<f64>) -> Result<DVector<f64>, IncompatibleShapeError> {

    // }

    /// Compute the full variance matrix from input data
    pub fn var_full(&self, x: &DMatrix<f64>) -> Result<DMatrix<f64>, IncompatibleShapeError> {
        // compute K*
        let mut k_x_xp = self.kernel.call(x, self.x)?;

        // compute K**
        let mut k_xp_xp = self.kernel.call(x, x)?;

        // TODO parallel cholesky solve
        self.cholesky.solve_mut(&mut k_x_xp);

        let zipped = par_tr_matmul(&k_x_xp);

        k_xp_xp
            .as_mut_slice()
            .into_par_iter()
            .zip(zipped)
            .for_each(|(l, r)| *l -= r);

        Ok(k_xp_xp)
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{DMatrix, DVector};

    use crate::kernels::RBF;

    use super::GP;

    /// Predicting a noiseless GP on one of the input points returns the measured output
    #[test]
    fn test_mean_noiseless() {
        let kern = RBF::new(vec![1.0].iter(), 1.0);
        let gp = GP::new(kern, 0.0);

        let x = DMatrix::from_vec(1, 2, vec![0.0, 1.0]);
        let y = DVector::from_vec(vec![0.0, 1.0]);

        let compiled = gp.compile(&x, &y).unwrap();

        let xp = DMatrix::from_vec(1, 2, vec![0.0, 1.0]);
        let f = DVector::from_vec(vec![0.0, 1.0]);

        let res = compiled.mean(&xp).unwrap();

        assert_eq!(res.as_slice(), f.as_slice())
    }

    /// Predicting a noisy GP smooths the input data
    #[test]
    fn test_mean_noisy() {
        let kern = RBF::new(vec![1.0].iter(), 1.0);
        let gp = GP::new(kern, 1.2);

        let x = DMatrix::from_vec(1, 2, vec![0.0, 1.0]);
        let y = DVector::from_vec(vec![0.0, 1.0]);

        let compiled = gp.compile(&x, &y).unwrap();

        let xp = DMatrix::from_vec(1, 2, vec![0.0, 1.0]);
        let f = DVector::from_vec(vec![0.0, 1.0]);

        let res = compiled.mean(&xp).unwrap();

        // results should be somewhere in-between the measured data
        assert!(res[0] > f[0]);
        assert!(res[1] < f[1]);
    }

    /// Attempting to compile a GP with a non-positive-definite covariance matrix will return an Err
    #[test]
    #[should_panic]
    fn test_non_positive_definite() {
        let kern = RBF::new(vec![1.0].iter(), 1.0);
        let gp = GP::new(kern, 0.0);

        let x = DMatrix::from_vec(1, 2, vec![1.0, 1.0]);
        let y = DVector::from_vec(vec![0.0, 1.0]);

        gp.compile(&x, &y).unwrap();
    }
}
