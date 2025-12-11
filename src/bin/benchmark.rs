//! Benchmark Comparison: Greedy vs True SA on Real Circuits
//!
//! Tests the greedy vs SA comparison on real circuit benchmarks
//! from the Bookshelf format (ISPD/IBM benchmarks).
//!
//! Usage: cargo run --release --bin benchmark

use sa_placer_lib::*;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

const N_SEEDS: usize = 5;

// Experiment configuration (same as synthetic comparison)
const GREEDY_STEPS: u32 = 1000;
const N_NEIGHBORS: usize = 16;
const SA_STEPS: u32 = GREEDY_STEPS * N_NEIGHBORS as u32;

fn main() {
    println!("=======================================================");
    println!("  Real Benchmark Comparison: Greedy vs True SA");
    println!("=======================================================\n");

    // Check for benchmark files
    let data_dir = Path::new("./data");
    if !data_dir.exists() {
        println!("Error: data/ directory not found.");
        println!("Please download benchmarks first. See README for instructions.");
        return;
    }

    // Try to find benchmark files
    let primary1_nodes = data_dir.join("p1UnitWDims.nodes");
    let primary1_nets = data_dir.join("p1UnitWDims.nets");

    let ibm05_nodes = data_dir.join("ibm05WDims.nodes");
    let ibm05_nets = data_dir.join("ibm05WDims.nets");

    // Create output directory
    let output_dir = Path::new("./output_data/benchmark");
    if output_dir.exists() {
        std::fs::remove_dir_all(output_dir).expect("Failed to remove old benchmark directory");
    }
    std::fs::create_dir_all(output_dir).expect("Failed to create benchmark directory");

    let mut all_results = Vec::new();

    // Run primary1 if available
    if primary1_nodes.exists() && primary1_nets.exists() {
        println!("Loading primary1 benchmark...");
        match load_bookshelf_benchmark(&primary1_nodes, &primary1_nets) {
            Ok((layout, netlist, stats)) => {
                println!("  Loaded: {} nodes ({} movable, {} terminals), {} nets",
                         stats.num_nodes, stats.num_movable, stats.num_terminals, stats.num_nets);
                println!("  Grid size: {}x{}", stats.grid_size, stats.grid_size);

                let results = run_benchmark_comparison(&layout, &netlist, &stats.name, output_dir);
                all_results.push((stats, results));
            }
            Err(e) => println!("  Error loading primary1: {}", e),
        }
    } else {
        println!("primary1 benchmark not found (p1UnitWDims.nodes/.nets)");
    }

    println!();

    // Run IBM05 if available (larger benchmark)
    if ibm05_nodes.exists() && ibm05_nets.exists() {
        println!("Loading IBM05 benchmark...");
        match load_bookshelf_benchmark(&ibm05_nodes, &ibm05_nets) {
            Ok((layout, netlist, stats)) => {
                println!("  Loaded: {} nodes ({} movable, {} terminals), {} nets",
                         stats.num_nodes, stats.num_movable, stats.num_terminals, stats.num_nets);
                println!("  Grid size: {}x{}", stats.grid_size, stats.grid_size);

                // For large benchmarks, use fewer steps to keep runtime reasonable
                let results = run_benchmark_comparison(&layout, &netlist, &stats.name, output_dir);
                all_results.push((stats, results));
            }
            Err(e) => println!("  Error loading IBM05: {}", e),
        }
    } else {
        println!("IBM05 benchmark not found (ibm05WDims.nodes/.nets)");
    }

    // Write combined summary
    if !all_results.is_empty() {
        write_combined_summary(output_dir, &all_results);
    }

    println!("\n=======================================================");
    println!("  Benchmark complete!");
    println!("=======================================================");
    println!("\nOutput written to: {}/", output_dir.display());
}

#[derive(Clone)]
struct BenchmarkResult {
    seed: usize,
    initial_cost: f32,
    greedy_final: f32,
    sa_best: f32,
    uphill_accepted: u32,
    greedy_time_ms: u64,
    sa_time_ms: u64,
}

fn run_benchmark_comparison(
    layout: &FPGALayout,
    netlist: &NetlistGraph,
    name: &str,
    output_dir: &Path,
) -> Vec<BenchmarkResult> {
    println!("\nRunning fair budget comparison ({} seeds)...", N_SEEDS);
    println!("  Greedy: {} steps x {} neighbours = {} candidate evaluations",
             GREEDY_STEPS, N_NEIGHBORS, GREEDY_STEPS as usize * N_NEIGHBORS);
    println!("  SA:     {} steps x 1 neighbour  = {} candidate evaluations\n",
             SA_STEPS, SA_STEPS);

    let mut results = Vec::new();
    let mut greedy_convergence: Vec<Vec<f32>> = Vec::new();
    let mut sa_convergence: Vec<Vec<f32>> = Vec::new();

    for seed in 0..N_SEEDS {
        print!("  Seed {}/{}... ", seed + 1, N_SEEDS);
        std::io::stdout().flush().unwrap();

        // Generate random initial placement
        let initial = gen_random_placement(layout, netlist);
        let initial_cost = initial.cost_bb();

        // Run greedy descent
        let greedy_start = Instant::now();
        let greedy_output = fast_sa_placer(
            initial.clone(),
            GREEDY_STEPS,
            N_NEIGHBORS,
            false,
            false,
        );
        let greedy_time = greedy_start.elapsed().as_millis() as u64;
        let greedy_final = greedy_output.final_solution.cost_bb();

        greedy_convergence.push(greedy_output.y_cost.clone());

        // Run true SA with equal computational budget
        let temp = estimate_initial_temperature(&initial, 0.8, 200);
        let sa_start = Instant::now();
        let sa_output = true_sa_placer(
            initial.clone(),
            SA_STEPS,
            temp,
            CoolingSchedule::geometric(0.9995),
            false,
            false,
        );
        let sa_time = sa_start.elapsed().as_millis() as u64;
        let sa_best = sa_output.best_solution.cost_bb();

        // Sample SA convergence at aligned intervals
        let sa_sampled: Vec<f32> = sa_output.y_cost
            .iter()
            .enumerate()
            .filter(|(i, _)| i % N_NEIGHBORS == 0)
            .map(|(_, &cost)| cost)
            .collect();
        sa_convergence.push(sa_sampled);

        println!(
            "initial: {:.0}, greedy: {:.0} ({} ms), SA best: {:.0} ({} ms, {} uphill)",
            initial_cost, greedy_final, greedy_time, sa_best, sa_time, sa_output.uphill_accepted
        );

        results.push(BenchmarkResult {
            seed,
            initial_cost,
            greedy_final,
            sa_best,
            uphill_accepted: sa_output.uphill_accepted,
            greedy_time_ms: greedy_time,
            sa_time_ms: sa_time,
        });
    }

    // Write convergence data
    write_convergence_csv(output_dir, &format!("{}_greedy.csv", name), &greedy_convergence);
    write_convergence_csv(output_dir, &format!("{}_sa.csv", name), &sa_convergence);

    // Write and print summary
    write_summary_csv(output_dir, &format!("{}_summary.csv", name), &results);
    print_summary(name, &results);

    results
}

fn write_convergence_csv(output_dir: &Path, filename: &str, data: &[Vec<f32>]) {
    let mut file = File::create(output_dir.join(filename)).unwrap();

    write!(file, "candidate_evaluations").unwrap();
    for seed in 0..data.len() {
        write!(file, ",seed_{}", seed).unwrap();
    }
    writeln!(file).unwrap();

    let n_rows = data[0].len();
    for row in 0..n_rows {
        let candidate_evals = row * N_NEIGHBORS;
        write!(file, "{}", candidate_evals).unwrap();
        for seed_data in data {
            write!(file, ",{}", seed_data[row]).unwrap();
        }
        writeln!(file).unwrap();
    }
}

fn write_summary_csv(output_dir: &Path, filename: &str, results: &[BenchmarkResult]) {
    let mut file = File::create(output_dir.join(filename)).unwrap();

    writeln!(file, "seed,initial,greedy_final,sa_best,uphill,greedy_ms,sa_ms").unwrap();
    for r in results {
        writeln!(
            file,
            "{},{},{},{},{},{},{}",
            r.seed, r.initial_cost, r.greedy_final, r.sa_best,
            r.uphill_accepted, r.greedy_time_ms, r.sa_time_ms
        ).unwrap();
    }
}

fn print_summary(name: &str, results: &[BenchmarkResult]) {
    let n = results.len() as f32;

    let avg_initial: f32 = results.iter().map(|r| r.initial_cost).sum::<f32>() / n;
    let avg_greedy: f32 = results.iter().map(|r| r.greedy_final).sum::<f32>() / n;
    let avg_sa_best: f32 = results.iter().map(|r| r.sa_best).sum::<f32>() / n;
    let avg_uphill: f32 = results.iter().map(|r| r.uphill_accepted as f32).sum::<f32>() / n;

    let greedy_reduction = (1.0 - avg_greedy / avg_initial) * 100.0;
    let sa_reduction = (1.0 - avg_sa_best / avg_initial) * 100.0;

    println!("\n-------------------------------------------------------");
    println!("  {} RESULTS (averaged over {} seeds)", name.to_uppercase(), results.len());
    println!("-------------------------------------------------------");
    println!("  Initial cost:         {:.0}", avg_initial);
    println!();
    println!("  Greedy final cost:    {:.0} ({:.1}% reduction)", avg_greedy, greedy_reduction);
    println!("  SA best cost:         {:.0} ({:.1}% reduction)", avg_sa_best, sa_reduction);
    println!("  SA uphill moves:      {:.0}", avg_uphill);
    println!();

    if avg_greedy < avg_sa_best {
        let winner_pct = (1.0 - avg_greedy / avg_sa_best) * 100.0;
        println!("  WINNER: Greedy descent (by {:.1}%)", winner_pct);
    } else {
        let winner_pct = (1.0 - avg_sa_best / avg_greedy) * 100.0;
        println!("  WINNER: True SA (by {:.1}%)", winner_pct);
    }
    println!("-------------------------------------------------------");
}

fn write_combined_summary(output_dir: &Path, all_results: &[(BenchmarkStats, Vec<BenchmarkResult>)]) {
    let mut file = File::create(output_dir.join("combined_summary.csv")).unwrap();

    writeln!(file, "benchmark,nodes,movable,terminals,nets,grid_size,avg_initial,avg_greedy,avg_sa_best,greedy_reduction_pct,sa_reduction_pct,winner,margin_pct").unwrap();

    for (stats, results) in all_results {
        let n = results.len() as f32;
        let avg_initial: f32 = results.iter().map(|r| r.initial_cost).sum::<f32>() / n;
        let avg_greedy: f32 = results.iter().map(|r| r.greedy_final).sum::<f32>() / n;
        let avg_sa_best: f32 = results.iter().map(|r| r.sa_best).sum::<f32>() / n;

        let greedy_reduction = (1.0 - avg_greedy / avg_initial) * 100.0;
        let sa_reduction = (1.0 - avg_sa_best / avg_initial) * 100.0;

        let (winner, margin) = if avg_greedy < avg_sa_best {
            ("greedy", (1.0 - avg_greedy / avg_sa_best) * 100.0)
        } else {
            ("sa", (1.0 - avg_sa_best / avg_greedy) * 100.0)
        };

        writeln!(
            file,
            "{},{},{},{},{},{},{:.0},{:.0},{:.0},{:.1},{:.1},{},{:.1}",
            stats.name, stats.num_nodes, stats.num_movable, stats.num_terminals,
            stats.num_nets, stats.grid_size, avg_initial, avg_greedy, avg_sa_best,
            greedy_reduction, sa_reduction, winner, margin
        ).unwrap();
    }

    println!("\nCombined summary written to: {}/combined_summary.csv", output_dir.display());
}
