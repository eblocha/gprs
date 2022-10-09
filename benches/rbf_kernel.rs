use criterion::{black_box, criterion_group, criterion_main, Criterion};
use gprs::kernels::{Kernel, TriangleSide, RBF};
use nalgebra::DMatrix;

fn create_random(shape: (usize, usize)) -> DMatrix<f64> {
    DMatrix::<f64>::new_random(shape.0, shape.1)
}

pub fn bench_rbf_kernel(c: &mut Criterion) {
    let kern = RBF::new(vec![1.0, 2.0], 1.0);
    const SIZE: usize = 2000;

    rayon::ThreadPoolBuilder::new()
        .num_threads(12)
        .build_global()
        .unwrap();

    // This does allocations
    c.bench_function(format!("rbf-call-{}", SIZE).as_str(), |b| {
        let x = create_random((2, SIZE));
        b.iter(|| kern.call(black_box(&x), black_box(&x)).unwrap());
    });

    // This does not allocate
    c.bench_function(format!("rbf-call-inplace-{}", SIZE).as_str(), |b| {
        let x = create_random((2, SIZE));
        let mut raw = DMatrix::zeros(SIZE, SIZE);
        b.iter(|| {
            kern.call_inplace(black_box(&x), black_box(&x), black_box(&mut raw))
                .unwrap()
        })
    });

    // Only compute half the covariance matrix
    c.bench_function(format!("rbf-call-triangular-{}", SIZE).as_str(), |b| {
        let x = create_random((2, SIZE));
        b.iter(|| {
            kern.call_triangular(black_box(&x), TriangleSide::LOWER)
                .unwrap()
        });
    });

    // Only compute half the covariance matrix
    c.bench_function(format!("rbf-call-diagonal-{}", SIZE).as_str(), |b| {
        let x = create_random((2, SIZE));
        b.iter(|| kern.call_diagonal(black_box(&x)).unwrap());
    });
}

criterion_group!(benches, bench_rbf_kernel);
criterion_main!(benches);
