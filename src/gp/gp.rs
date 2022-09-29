use nalgebra::{Cholesky, DMatrix, DVector, Dynamic};

use crate::kernels::{
    errors::IncompatibleShapeError,
    {Kernel, TriangleSide},
};

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
    pub fn compile(
        &self,
        x: &DMatrix<f64>,
        y: &DVector<f64>,
    ) -> Result<CompiledGP, IncompatibleShapeError> {
        let mut noise = DMatrix::identity(x.nrows(), x.nrows());
        // TODO parallel diagonal fill
        noise.fill_diagonal(self.noise);

        let kxx = self.kernel.call_triangular(&x, TriangleSide::LOWER)?;

        // TODO parallel cholesky decomp
        let cholesky = (kxx + noise).cholesky().unwrap();
        let mut alpha = cholesky.solve(y);
        cholesky.solve_mut(&mut alpha);

        Ok(CompiledGP { cholesky, alpha })
    }
}

pub struct CompiledGP {
    /// The cholesky decomposition of (K + noise * I)
    cholesky: Cholesky<f64, Dynamic>,
    /// Factor to compute mean
    alpha: DVector<f64>,
}

impl CompiledGP {
    // /// Compute the mean and variance from input data
    // fn call(x: &DMatrix<f64>) -> (DMatrix<f64>, DMatrix<f64>) {}

    // /// Compute the mean from input data
    // fn mean(x: &DMatrix<f64>) -> DMatrix<f64> {}

    // /// Compute the variance from input data
    // fn var(x: &DMatrix<f64>) -> DMatrix<f64> {}
}
