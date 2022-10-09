use criterion::{black_box, criterion_group, criterion_main, Criterion};
use gprs::{
    gp::{CompiledGP, GP},
    kernels::RBF,
};
use nalgebra::{DMatrix, DVector};

fn create_random(shape: (usize, usize)) -> DMatrix<f64> {
    DMatrix::<f64>::new_random(shape.0, shape.1)
}

fn create_random_data(shape: (usize, usize)) -> (DMatrix<f64>, DVector<f64>) {
    let sz = shape.0 * shape.1;
    // x data cannot have any duplicates, or the variance matrix will not be positive-definite
    (
        DMatrix::<f64>::from_iterator(shape.0, shape.1, (0..sz).into_iter().map(|v| v as f64)),
        DVector::<f64>::new_random(shape.1),
    )
}

const SZ: usize = 1000;
const NOISE: f64 = 1.2;

fn setup() -> (GP<RBF>, DMatrix<f64>, DVector<f64>) {
    let kernel = RBF::new(vec![1.0], 1.0);
    let gp = GP::new(kernel, NOISE);
    let (x, y) = create_random_data((1, SZ));

    (gp, x, y)
}

fn setup_compile() -> (CompiledGP<RBF>, DMatrix<f64>) {
    let (gp, x, y) = setup();
    let compiled = gp.compile(x, &y).unwrap();

    let xp = create_random((1, SZ));

    (compiled, xp)
}

fn criterion_benchmark(c: &mut Criterion) {
    rayon::ThreadPoolBuilder::new()
        .num_threads(12)
        .build_global()
        .unwrap();

    c.bench_function(format!("gp-compile-{}", SZ).as_str(), |b| {
        b.iter(move || {
            let (gp, x, y) = setup();
            gp.compile(black_box(x), black_box(&y)).unwrap();
        })
    });

    c.bench_function(format!("gp-mean-{}", SZ).as_str(), |b| {
        let (gp, xp) = setup_compile();
        b.iter(|| gp.mean(black_box(&xp)).unwrap());
    });

    c.bench_function(format!("gp-var-{}", SZ).as_str(), |b| {
        let (gp, xp) = setup_compile();
        b.iter(|| gp.var(black_box(&xp)).unwrap());
    });

    c.bench_function(format!("gp-cov-{}", SZ).as_str(), |b| {
        let (gp, xp) = setup_compile();
        b.iter(|| gp.cov(black_box(&xp)).unwrap());
    });

    c.bench_function(format!("gp-dual-{}", SZ).as_str(), |b| {
        let (gp, xp) = setup_compile();
        b.iter(|| gp.call(black_box(&xp)).unwrap());
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
