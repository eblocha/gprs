use criterion::{black_box, criterion_group, criterion_main, Criterion};
use gprs::kernels::{Kernel, RBF};
use nalgebra::DMatrix;

fn create_random(shape: (usize, usize)) -> DMatrix<f64> {
    DMatrix::<f64>::new_random(shape.0, shape.1)
}

fn criterion_benchmark(c: &mut Criterion) {
    let kern = RBF::new(vec![1.0, 2.0].iter());

    rayon::ThreadPoolBuilder::new()
        .num_threads(12)
        .build_global()
        .unwrap();

    // This does allocations
    let mut group_call = c.benchmark_group("rbf-call");

    for i in [1000 as usize, 1500, 2000, 2500, 3000].iter() {
        group_call.throughput(criterion::Throughput::Elements(*i as u64));
        group_call.bench_with_input(format!("{}x{}", i, i), i, |b, n| {
            let x = create_random((2, *n));

            b.iter(|| kern.call(black_box(&x), black_box(&x)).unwrap())
        });
    }
    group_call.finish();

    // This does not allocate
    let mut group_call_into = c.benchmark_group("rbf-call-into");

    for i in [1000 as usize, 1500, 2000, 2500, 3000].iter() {
        group_call_into.throughput(criterion::Throughput::Elements(*i as u64));
        group_call_into.bench_with_input(format!("{}x{}", i, i), i, |b, n| {
            let mut raw = DMatrix::zeros(*n, *n);
            let x = create_random((2, *n));

            b.iter(|| {
                kern.call_into(black_box(&x), black_box(&x), black_box(&mut raw))
                    .unwrap()
            })
        });
    }
    group_call_into.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
