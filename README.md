# GPrs

Guassian Process Regression in Rust

This is my first serious rust library, to help me learn the language!

## Features

Currently, I have implemented the RBF kernel, nothing else yet.

```rs
use gprs::kernels::RBF

fn main() {
  // create a 2-d anisotropic RBF kernel with length scales of 1.0 and 2.0
  let kern = RBF::<2>::new(&vec![1.0, 2.0]).unwrap();

  // compute the covariance using the kernel
  let (x, y) = (vec![1.8, 5.5], vec![2.2, 3.0]);
  let k = kern.call(&x, &y).unwrap();
}
```

The kernel parameters are statically allocated on the stack for maximum performance.
