use criterion::{black_box, criterion_group, criterion_main, Criterion};
use gprs::kernels::{Kernel, RBF};
use nalgebra::{DMatrix, DVector};

fn create_random(shape: (usize, usize)) -> DMatrix<f64> {
    DMatrix::<f64>::new_random(shape.0, shape.1)
}
const SZ: usize = 2000;

fn criterion_benchmark(c: &mut Criterion) {
    let kern = RBF::new(DVector::from_vec(vec![1.0, 2.0]));

    let x = create_random((2, SZ));
    let mut raw = DMatrix::zeros(SZ, SZ);

    rayon::ThreadPoolBuilder::new()
        .num_threads(12)
        .build_global()
        .unwrap();

    c.bench_function("random mismatched", |b| {
        b.iter(|| kern.call(black_box(&x), black_box(&x)).unwrap())
    });

    c.bench_function("random symmetric", |b| {
        b.iter(|| {
            kern.call_symmetric_into(black_box(&x), black_box(&mut raw))
                .unwrap()
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
