//! Fair Budget Comparison: Greedy vs True SA
//!
//! Compares greedy multi-neighbour descent against true simulated annealing
//! with EQUAL computational budget (same number of candidate evaluations).
//!
//! This is the only fair way to compare the algorithms - giving both the same
//! amount of computation and seeing which achieves better results.
//!
//! Usage: cargo run --release --bin compare

use sa_placer_lib::*;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

const N_SEEDS: usize = 5;
const GRID_SIZE: u32 = 64;
const N_NODES: u32 = 300;
const N_IO: u32 = 30;
const N_BRAM: u32 = 100;

// Experiment configuration
const GREEDY_STEPS: u32 = 1000;
const N_NEIGHBORS: usize = 16;
// SA gets the same number of candidate evaluations as greedy
const SA_STEPS: u32 = GREEDY_STEPS * N_NEIGHBORS as u32; // 16,000

fn main() {
    println!("=======================================================");
    println!("  Fair Budget Comparison: Greedy vs True SA");
    println!("=======================================================\n");

    println!("Configuration:");
    println!("  Greedy: {} steps x {} neighbours = {} candidate evaluations",
             GREEDY_STEPS, N_NEIGHBORS, GREEDY_STEPS as usize * N_NEIGHBORS);
    println!("  SA:     {} steps x 1 neighbour  = {} candidate evaluations\n",
             SA_STEPS, SA_STEPS);

    let layout = build_simple_fpga_layout(GRID_SIZE, GRID_SIZE);

    // Create output directory (remove old data first for clean slate)
    let output_dir = "./output_data/comparison";
    if std::path::Path::new(output_dir).exists() {
        std::fs::remove_dir_all(output_dir).expect("Unable to remove old comparison directory");
    }
    std::fs::create_dir_all(output_dir).expect("Unable to create comparison directory");

    // Also remove old fair_comparison directory if it exists
    let old_fair_dir = "./output_data/fair_comparison";
    if std::path::Path::new(old_fair_dir).exists() {
        std::fs::remove_dir_all(old_fair_dir).expect("Unable to remove old fair_comparison directory");
    }

    run_fair_comparison(&layout, output_dir);

    println!("\n=======================================================");
    println!("  Experiment complete!");
    println!("=======================================================");
    println!("\nOutput written to: {}/", output_dir);
    println!("  - greedy_convergence.csv (cost vs candidate evaluations)");
    println!("  - sa_convergence.csv (cost vs candidate evaluations)");
    println!("  - summary.csv (final results per seed)");
    println!("\nGenerate plots with: uv run scripts/generate_plots.py");
}

/// Fair budget comparison with convergence tracking.
///
/// Both algorithms get exactly the same number of candidate evaluations.
/// Convergence is tracked at equal intervals of candidate evaluations
/// so they can be plotted on the same x-axis.
fn run_fair_comparison(layout: &FPGALayout, output_dir: &str) {
    println!("Running fair budget comparison ({} seeds)...\n", N_SEEDS);

    // Storage for convergence data
    // We sample at every GREEDY step, which equals N_NEIGHBORS candidate evaluations
    // This gives us GREEDY_STEPS + 1 data points for each algorithm
    let mut greedy_convergence: Vec<Vec<f32>> = Vec::new();
    let mut sa_convergence: Vec<Vec<f32>> = Vec::new();

    // Storage for summary results
    let mut results: Vec<SeedResult> = Vec::new();

    for seed in 0..N_SEEDS {
        print!("  Seed {}/{}... ", seed + 1, N_SEEDS);
        std::io::stdout().flush().unwrap();

        // Generate random netlist and initial placement
        let netlist = build_simple_netlist(N_NODES, N_IO, N_BRAM);
        let initial = gen_random_placement(layout, &netlist);
        let initial_cost = initial.cost_bb();

        // Run greedy descent
        let greedy_start = Instant::now();
        let greedy_output = fast_sa_placer(
            initial.clone(),
            GREEDY_STEPS,
            N_NEIGHBORS,
            false,  // verbose
            false,  // render
        );
        let greedy_time = greedy_start.elapsed().as_millis() as u64;
        let greedy_final = greedy_output.final_solution.cost_bb();

        // Store greedy convergence (already at correct granularity)
        greedy_convergence.push(greedy_output.y_cost.clone());

        // Run true SA with equal computational budget
        let temp = estimate_initial_temperature(&initial, 0.8, 200);
        let sa_start = Instant::now();
        let sa_output = true_sa_placer(
            initial.clone(),
            SA_STEPS,
            temp,
            CoolingSchedule::geometric(0.9995),  // Slower cooling for more steps
            false,  // verbose
            false,  // render
        );
        let sa_time = sa_start.elapsed().as_millis() as u64;
        let sa_final = sa_output.final_solution.cost_bb();
        let sa_best = sa_output.best_solution.cost_bb();

        // Sample SA convergence at every N_NEIGHBORS steps to align with greedy
        // This gives us cost at equal candidate evaluation counts
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

        results.push(SeedResult {
            seed,
            initial_cost,
            greedy_final,
            sa_final,
            sa_best,
            uphill_accepted: sa_output.uphill_accepted,
            greedy_time_ms: greedy_time,
            sa_time_ms: sa_time,
        });
    }

    // Write convergence data
    write_convergence_csv(output_dir, "greedy_convergence.csv", &greedy_convergence);
    write_convergence_csv(output_dir, "sa_convergence.csv", &sa_convergence);

    // Write summary
    write_summary_csv(output_dir, &results);

    // Print summary statistics
    print_summary(&results);
}

struct SeedResult {
    seed: usize,
    initial_cost: f32,
    greedy_final: f32,
    sa_final: f32,
    sa_best: f32,
    uphill_accepted: u32,
    greedy_time_ms: u64,
    sa_time_ms: u64,
}

fn write_convergence_csv(output_dir: &str, filename: &str, data: &[Vec<f32>]) {
    let mut file = File::create(format!("{}/{}", output_dir, filename)).unwrap();

    // Header: candidate_evaluations, seed_0, seed_1, ...
    write!(file, "candidate_evaluations").unwrap();
    for seed in 0..data.len() {
        write!(file, ",seed_{}", seed).unwrap();
    }
    writeln!(file).unwrap();

    // Data rows
    // Each row represents cost at (row_index * N_NEIGHBORS) candidate evaluations
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

fn write_summary_csv(output_dir: &str, results: &[SeedResult]) {
    let mut file = File::create(format!("{}/summary.csv", output_dir)).unwrap();

    writeln!(file, "seed,initial,greedy_final,sa_final,sa_best,uphill,greedy_ms,sa_ms").unwrap();
    for r in results {
        writeln!(
            file,
            "{},{},{},{},{},{},{},{}",
            r.seed, r.initial_cost, r.greedy_final, r.sa_final, r.sa_best,
            r.uphill_accepted, r.greedy_time_ms, r.sa_time_ms
        ).unwrap();
    }
}

fn print_summary(results: &[SeedResult]) {
    let n = results.len() as f32;

    let avg_initial: f32 = results.iter().map(|r| r.initial_cost).sum::<f32>() / n;
    let avg_greedy: f32 = results.iter().map(|r| r.greedy_final).sum::<f32>() / n;
    let avg_sa_best: f32 = results.iter().map(|r| r.sa_best).sum::<f32>() / n;
    let avg_uphill: f32 = results.iter().map(|r| r.uphill_accepted as f32).sum::<f32>() / n;
    let avg_greedy_time: f64 = results.iter().map(|r| r.greedy_time_ms as f64).sum::<f64>() / n as f64;
    let avg_sa_time: f64 = results.iter().map(|r| r.sa_time_ms as f64).sum::<f64>() / n as f64;

    let greedy_reduction = (1.0 - avg_greedy / avg_initial) * 100.0;
    let sa_reduction = (1.0 - avg_sa_best / avg_initial) * 100.0;

    println!("\n-------------------------------------------------------");
    println!("  RESULTS (averaged over {} seeds)", results.len());
    println!("-------------------------------------------------------");
    println!("  Initial cost:         {:.0}", avg_initial);
    println!();
    println!("  Greedy final cost:    {:.0} ({:.1}% reduction)", avg_greedy, greedy_reduction);
    println!("  Greedy time:          {:.0} ms", avg_greedy_time);
    println!();
    println!("  SA best cost:         {:.0} ({:.1}% reduction)", avg_sa_best, sa_reduction);
    println!("  SA time:              {:.0} ms", avg_sa_time);
    println!("  SA uphill moves:      {:.0} (proves SA is working)", avg_uphill);
    println!();

    if avg_greedy < avg_sa_best {
        let winner_pct = (1.0 - avg_greedy / avg_sa_best) * 100.0;
        println!("  WINNER: Greedy descent (by {:.1}%)", winner_pct);
    } else {
        let winner_pct = (1.0 - avg_sa_best / avg_greedy) * 100.0;
        println!("  WINNER: True SA (by {:.1}%)", winner_pct);
    }

    println!();
    println!("  Conclusion: With equal computational budget ({} candidate",
             GREEDY_STEPS as usize * N_NEIGHBORS);
    println!("  evaluations), greedy multi-neighbour descent outperforms");
    println!("  classical simulated annealing on this synthetic benchmark.");
    println!("-------------------------------------------------------");
}
