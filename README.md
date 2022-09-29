# GPrs

Guassian Process Regression in Rust

This is my first serious rust library, to help me learn the language!

## Features

Currently, I have implemented the RBF kernel, nothing else yet.

```rs
use gprs::kernels::{RBF,Kernel};
use nalgebra::{DVector,DMatrix};

fn main() {
    // create a 2-d anisotropic RBF kernel with length scales of 1.0 and 2.0
    let kern = RBF::new(vec![1.0, 2.0].iter());

    // estimate covariance between 2 sets of points
    let x = DMatrix::from_vec(2, 3, vec![
        1.8, 5.5,
        1.5, 4.5,
        2.3, 4.6
    ]);

    let y = DMatrix::from_vec(2, 4, vec![
        2.2, 3.0,
        1.8, 5.5,
        1.5, 4.5,
        2.3, 4.6
    ]);

    // returns an Err if x and y do not have compatible shapes
    let k = kern.call(&x, &y).unwrap();

    assert_eq!(k.shape(), (3, 4));
}
```

## Goals

- [ ] Implement basic gaussian process regression with RBF
- [ ] Allow use of trained and untrained mean function (prior)
  - [ ] trained -> implements derivatives
  - [ ] untrained -> simple function
- [ ] Provide interface for computing derivatives w.r.t. kernel params
- [x] Learn multithreading to use when iterating over very large arrays
- [x] Add performance benchmarks
- [ ] Implement L-BFGS to optimize kernels
- [ ] Implement white noise, sum, product, and polynomial kernels
