use criterion::{black_box, criterion_group, criterion_main, Criterion};
use gprs::linalg::{par_matmul, par_tr_matmul};
use nalgebra::DMatrix;

fn create_random(shape: (usize, usize)) -> DMatrix<f64> {
    DMatrix::<f64>::new_random(shape.0, shape.1)
}

fn bench_matmul(c: &mut Criterion) {
    rayon::ThreadPoolBuilder::new()
        .num_threads(12)
        .build_global()
        .unwrap();

    const SZ: usize = 500;

    c.bench_function("matmul-different", |b| {
        let lhs = create_random((SZ, SZ));
        let rhs = create_random((SZ, SZ));

        b.iter(|| par_matmul(black_box(&lhs), black_box(&rhs)).unwrap());
    });

    c.bench_function("matmul-transpose", |b| {
        let lhs = create_random((SZ, SZ));

        b.iter(|| par_tr_matmul(black_box(&lhs)).unwrap());
    });
}

criterion_group!(benches, bench_matmul);
criterion_main!(benches);
