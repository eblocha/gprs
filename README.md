# GPrs

Parallelized Gaussian Process Regression in Rust

This is my first serious rust library, to help me learn the language!

## Features

Currently, I have implemented the RBF kernel and the GP mean calculation.

```rs
use gprs::{kernels::{RBF,Kernel},gp::GP};
use nalgebra::{DVector,DMatrix};

fn main() {
    use gprs::{gp::GP, kernels::{Kernel, RBF}};
    use nalgebra::{DVector, DMatrix};

    // create a 2-d anisotropic RBF kernel with length scales of 1.0 and 2.0, and a sigma of 1.0
    let kernel = RBF::new(vec![1.0, 2.0].iter(), 1.0);

    // create a GP from this kernel with a noise value (sigma) of 1.0. Use 0.0 for noiseless GP
    let gp = GP::new(
        kernel,
        1.0,
    );

    let x = DMatrix::from_vec(2, 3, vec![
        1.8, 5.5,
        1.5, 4.5,
        2.3, 4.6
    ]);

    let y = DVector::from_vec(vec![
        2.2,
        1.8,
        1.5,
    ]);

    // May return a GPCompilationError
    let compiled = gp.compile(&x, &y).unwrap();

    // Predict on some new data
    let x_pred = DMatrix::from_vec(2, 3, vec![
        0.0, 5.0,
        1.0, 6.0,
        2.0, 7.0,
        3.0, 8.0,
    ]);

    // May return an IncompatibleShapeError
    let mean = compiled.mean(&x_pred).unwrap();

    // `mean` is a `nalgebra::DVector<f64>` with length 4

    // Can also compute the variance with the mean, or either independently
    let mean, var = compiled.call(&x_pred).unwrap();
    // `mean` and `var` are both `nalgebra::DVector<f64>` with length 4
    let var = compiled.var(&x_pred).unwrap();

    // Note that `call` will be more efficient than calling the `mean` and `var` functions independently.
    // Additionally, the variance computation is very expensive since it involves large matrix multiplications
    // If you only need to estimate the mean, use the `mean` method, which is much faster.

    // If you need the full covariance matrix, you can use the `cov` method:
    let cov = compiled.cov(&x_pred).unwrap();

    // Be aware that this is far slower than `mean` or `var`.
}
```

## Goals

- [x] Implement basic gaussian process regression with RBF
- [ ] Allow use of trained and untrained mean function (prior)
  - [ ] trained -> implements derivatives
  - [ ] untrained -> simple function
- [ ] Provide interface for computing derivatives w.r.t. kernel params
- [x] Learn multithreading to use when iterating over very large arrays
- [x] Add performance benchmarks
- [ ] Implement L-BFGS to optimize kernels
- [ ] Implement white noise, sum, product, and polynomial kernels
