use criterion::{black_box, criterion_group, criterion_main, Criterion};
use gprs::kernels::{Kernel, RBF};
use nalgebra::{DMatrix, DVector};

fn create_random(shape: (usize, usize)) -> DMatrix<f64> {
    DMatrix::<f64>::new_random(shape.0, shape.1)
}

fn criterion_benchmark(c: &mut Criterion) {
    let kern = RBF::new(DVector::from_vec(vec![1.0, 2.0]));
    let x = black_box(create_random((1000, 2)));

    c.bench_function("random mismatched", |b| {
        b.iter(|| kern.call(&x, &x).unwrap())
    });

    c.bench_function("random symmetric", |b| {
        b.iter(|| kern.call_symmetric(&x).unwrap())
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
