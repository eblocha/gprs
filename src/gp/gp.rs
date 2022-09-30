use nalgebra::{Cholesky, DMatrix, DVector, Dynamic};

use crate::kernels::{
    errors::IncompatibleShapeError,
    {Kernel, TriangleSide},
};

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
    /// use gprs::{gp::{GP, CompiledGP}, kernels::{Kernel, RBF}};
    /// use nalgebra::{DVector,DMatrix};
    ///
    /// let kernel = RBF::new(vec![1.0].iter(), 1.0);
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
    /// let compiled = gp.compile(&x, &y);
    /// ```
    pub fn compile<'a>(
        &'a self,
        x: &'a DMatrix<f64>,
        y: &DVector<f64>,
    ) -> Result<CompiledGP<K>, IncompatibleShapeError> {
        let mut noise = DMatrix::identity(x.ncols(), x.ncols());
        // TODO parallel diagonal fill/subtraction
        noise.fill_diagonal(self.noise);

        let kxx = self.kernel.call_triangular(&x, TriangleSide::LOWER)?;

        // TODO parallel cholesky decomp and solve
        // TODO return Err if not positive definite
        let cholesky = (kxx + noise).cholesky().unwrap();
        let alpha = cholesky.solve(y);

        Ok(CompiledGP {
            cholesky,
            alpha,
            kernel: &self.kernel,
            x: &x,
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
    //     // compute K*
    //     let k_x_xp = self.kernel.call(&self.x, x)?;

    //     // TODO parallelize matmul
    //     return Ok(k_x_xp * self.alpha);
    // }

    /// Compute the mean from input data
    pub fn mean(&self, x: &DMatrix<f64>) -> Result<DVector<f64>, IncompatibleShapeError> {
        // compute K*T
        let k_x_xp = self.kernel.call(&self.x, x)?;

        // TODO parallelize matmul
        Ok(k_x_xp * &self.alpha)
    }

    // /// Compute just the diagonal variance
    // pub fn var(&self, x: &DMatrix<f64>) -> Result<DVector<f64>, IncompatibleShapeError> {

    // }

    /// Compute the full variance matrix from input data
    pub fn var_full(&self, x: &DMatrix<f64>) -> Result<DMatrix<f64>, IncompatibleShapeError> {
        // compute K*
        let mut k_x_xp = self.kernel.call(x, &self.x)?;

        // compute K**
        let k_xp_xp = self.kernel.call(x, x)?;

        // solve L \ K*
        self.cholesky.solve_mut(&mut k_x_xp);

        // TODO parallelize vTv
        let k_x_xp = k_x_xp.transpose() * k_x_xp;

        // TODO parallelize subtraction
        Ok(k_xp_xp - k_x_xp)
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{DMatrix, DVector};

    use crate::kernels::RBF;

    use super::GP;

    /// Predicting a noiseless GP on one of the input points returns the measured output
    #[test]
    fn test_point() {
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
}
