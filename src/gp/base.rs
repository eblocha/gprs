use nalgebra::{Cholesky, DMatrix, DVector, Dynamic};
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use crate::{
    // indexing::index_to_2d,
    kernels::{Kernel, TriangleSide},
    linalg::{
        errors::IncompatibleShapeError, par_solve_lower_triangular_unchecked, par_tr_matmul,
        par_tr_matmul_diag, util::par_add_diagonal_mut_unchecked,
    },
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

    /// Compile this GP for training or estimation. Consumes `self` and `x`.
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
    /// let compiled = gp.compile(x, &y).unwrap();
    /// ```
    pub fn compile<'a>(
        self,
        x: DMatrix<f64>,
        y: &DVector<f64>,
    ) -> Result<CompiledGP<K>, GPCompilationError> {
        if x.shape().1 != y.len() {
            return Err(GPCompilationError::IncompatibleShapeError(
                IncompatibleShapeError {
                    shapes: vec![x.shape(), y.shape()],
                },
            ));
        }

        let mut kxx = match self.kernel.call_triangular(&x, TriangleSide::LOWER) {
            Err(e) => return Err(GPCompilationError::IncompatibleShapeError(e)),
            Ok(v) => v,
        };

        // SAFETY: kxx is guaranteed to be square
        unsafe {
            par_add_diagonal_mut_unchecked(&mut kxx, &self.noise);
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
            kernel: self.kernel,
            x,
        })
    }
}

pub type GPResult<T> = Result<T, IncompatibleShapeError>;

pub struct CompiledGP<K: Kernel> {
    /// The cholesky decomposition of (K + noise * I)
    cholesky: Cholesky<f64, Dynamic>,
    /// Factor to compute mean
    alpha: DVector<f64>,
    /// The original kernel
    kernel: K,
    /// The input data set
    x: DMatrix<f64>,
}

impl<K: Kernel> CompiledGP<K> {
    /// Compute the mean and variance from input data
    pub fn call(&self, x: &DMatrix<f64>) -> GPResult<(DVector<f64>, DVector<f64>)> {
        let k_xp_x = self.kernel.call(x, &self.x)?;

        let mean = self.mean_precomputed(&k_xp_x)?;
        let var = self.var_precomputed(x, &k_xp_x)?;

        Ok((mean, var))
    }

    /// Compute the mean from input data
    ///
    /// `f = K*' [K + sI]^-1 y`
    pub fn mean(&self, x: &DMatrix<f64>) -> GPResult<DVector<f64>> {
        // compute K*'
        let k_xp_x = self.kernel.call(x, &self.x)?;
        self.mean_precomputed(&k_xp_x)
    }

    /// Find the mean given a precomputed K*
    fn mean_precomputed(&self, k_xp_x: &DMatrix<f64>) -> GPResult<DVector<f64>> {
        let res = par_tr_matmul(&k_xp_x, &self.alpha)?;
        Ok(DVector::from_vec(res))
    }

    /// Compute just the diagonal variance
    pub fn var(&self, x: &DMatrix<f64>) -> GPResult<DVector<f64>> {
        let k_xp_x = self.kernel.call(x, &self.x)?;
        self.var_precomputed(x, &k_xp_x)
    }

    /// Find the variance given a precomputed K*
    fn var_precomputed(&self, x: &DMatrix<f64>, k_xp_x: &DMatrix<f64>) -> GPResult<DVector<f64>> {
        let mut k_xp_xp = self.kernel.call_diagonal(x)?;
        let fact = par_solve_lower_triangular_unchecked(self.cholesky.l_dirty(), &k_xp_x);
        let zipped = par_tr_matmul_diag(&fact, &fact)?;

        k_xp_xp
            .as_mut_slice()
            .into_par_iter()
            .zip(zipped)
            .for_each(|(l, r)| *l -= r);

        Ok(DVector::from_vec(k_xp_xp))
    }

    /// Compute the full covariance matrix from input data
    ///
    /// `V = K** - K*' [K + sI]^-1 K*`
    pub fn cov(&self, x: &DMatrix<f64>) -> GPResult<DMatrix<f64>> {
        // compute K*
        let k_xp_x = self.kernel.call(x, &self.x)?;
        self.cov_precomputed(x, &k_xp_x)
    }

    /// Find the covariance matrix given a precomputed K*
    fn cov_precomputed(&self, x: &DMatrix<f64>, k_xp_x: &DMatrix<f64>) -> GPResult<DMatrix<f64>> {
        // compute K**
        let mut k_xp_xp = self.kernel.call(x, x)?;
        let fact = par_solve_lower_triangular_unchecked(self.cholesky.l_dirty(), &k_xp_x);
        let zipped = par_tr_matmul(&fact, &fact)?;

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

        let compiled = gp.compile(x, &y).unwrap();

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

        let compiled = gp.compile(x, &y).unwrap();

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

        gp.compile(x, &y).unwrap();
    }
}
