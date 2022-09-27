# GPrs

Guassian Process Regression in Rust

This is my first serious rust library, to help me learn the language!

## Features

Currently, I have implemented the RBF kernel, nothing else yet.

```rs
use gprs::kernels::RBF;
use nalgebra::DVector;

fn main() {
    // create a 2-d anisotropic RBF kernel with length scales of 1.0 and 2.0
    // this will return an Err if any length scale is <= 0
    let kern = RBF::new(DVector::from_vec(vec![1.0, 2.0]));

    // compute the covariance using the kernel
    let (x, y) = (DVector::from_vec(vec![1.8, 5.5]), DVector::from_vec(vec![2.2, 3.0]));
    let k = kern.call(&x, &y);
}
```

## Goals

- Implement basic gaussian process regression with RBF
- Allow use of trained and untrained mean function (prior)
  - trained -> implements derivatives
  - untrained -> simple function
- Provide interface for computing derivatives w.r.t. kernel params
- Learn multithreading to use when iterating over very large arrays
- Add performance benchmarks
- Implement L-BFGS to optimize kernels
- Implement white noise, sum, product, and polynomial kernels
