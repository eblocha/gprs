use criterion::{black_box, criterion_group, criterion_main, Criterion};
use gprs::{gp::GP, kernels::RBF};
use nalgebra::{DMatrix, DVector};

fn create_random_data(shape: (usize, usize)) -> (DMatrix<f64>, DVector<f64>) {
    let sz = shape.0 * shape.1;
    // x data cannot have any duplicates, or the variance matrix will not be positive-definite
    (
        DMatrix::<f64>::from_iterator(shape.0, shape.1, (0..sz).into_iter().map(|v| v as f64)),
        DVector::<f64>::new_random(shape.1),
    )
}

fn criterion_benchmark(c: &mut Criterion) {
    rayon::ThreadPoolBuilder::new()
        .num_threads(12)
        .build_global()
        .unwrap();

    c.bench_function("compile-gp", |b| {
        let kern = RBF::new(vec![1.0].iter(), 1.0);
        let gp = GP::new(kern, 0.0);
        let (x, y) = create_random_data((1, 1000));
        b.iter(|| gp.compile(black_box(&x), black_box(&y)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
