use sa_placer_lib::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn sa_placer_small_benchmark(c: &mut Criterion) {
    let layout = black_box(build_simple_fpga_layout(64, 64));
    let netlist: NetlistGraph = black_box(build_simple_netlist(300, 30, 100));
    let initial_solution = black_box(gen_random_placement(&layout, &netlist));

    c.bench_function("fast_sa_placer_small", |b| {
        b.iter(|| fast_sa_placer(initial_solution.clone(), 500, 16, false, false))
    });
}

fn sa_placer_large_benchmark(c: &mut Criterion) {
    let layout = build_simple_fpga_layout(200, 200);
    let netlist = build_simple_netlist(1000, 50, 200);
    let initial_solution = black_box(gen_random_placement(&layout, &netlist));

    c.bench_function("fast_sa_placer_large", |b| {
        b.iter(|| {
            fast_sa_placer(
                black_box(initial_solution.clone()),
                black_box(500),
                black_box(16),
                false,
                false,
            )
        })
    });
}

fn greedy_vs_sa_benchmark(c: &mut Criterion) {
    let layout = build_simple_fpga_layout(32, 32);
    let netlist = build_simple_netlist(100, 10, 20);
    let initial = gen_random_placement(&layout, &netlist);

    let mut group = c.benchmark_group("greedy_vs_sa");

    group.bench_function("greedy", |b| {
        b.iter(|| fast_sa_placer(initial.clone(), 500, 3, false, false))
    });

    group.bench_function("true_sa_geometric", |b| {
        b.iter(|| {
            true_sa_placer(initial.clone(), 500, 1000.0, CoolingSchedule::geometric(0.99), false, false)
        })
    });

    group.finish();
}

fn cost_function_benchmark(c: &mut Criterion) {
    let layout = build_simple_fpga_layout(64, 64);
    let netlist = build_simple_netlist(300, 30, 100);
    let solution = gen_random_placement(&layout, &netlist);

    let mut group = c.benchmark_group("cost_functions");
    group.bench_function("cost_bb", |b| b.iter(|| solution.cost_bb()));
    group.bench_function("cost_hpwl", |b| b.iter(|| solution.cost_hpwl()));
    group.finish();
}

criterion_group!(
    benches,
    sa_placer_small_benchmark,
    sa_placer_large_benchmark,
    greedy_vs_sa_benchmark,
    cost_function_benchmark
);
criterion_main!(benches);
