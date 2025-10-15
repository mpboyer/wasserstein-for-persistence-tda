//! Run with: cargo test -- --nocapture
use std::time::Instant;
use wasp::structs::{
    augment_diagrams, compute_optimal_matching, compute_optimal_matching_hungarian,
    compute_optimal_matching_old,
};
use wasp::structs::{PersistenceDiagram, PersistencePair};

static BENCHMARK_SIZE: usize = 5000;

// Generate a synthetic persistence diagram of `n` points
fn make_random_diagram(n_points: usize) -> PersistenceDiagram {
    use rand::Rng;
    let mut rng = rand::rng();
    let pairs: Vec<PersistencePair> = (0..n_points)
        .map(|_| {
            let birth: f64 = rng.random::<f64>();
            let death = birth + rng.random::<f64>() * 0.5;
            PersistencePair::new(birth, death, 0, 1)
        })
        .collect();
    PersistenceDiagram::from_pairs(pairs, 2)
}

fn benchmark_auction_algorithm(d1: PersistenceDiagram, d2: PersistenceDiagram) {
    let start = Instant::now();
    let assignment = compute_optimal_matching(&d1, &d2);
    let duration = start.elapsed();

    println!(
            "\n========================================\nAuctionAlgorithm benchmark (n = {}):\n  Assignments = {}\n  Total cost = {:.6}\n  Time = {:?}\n",
            BENCHMARK_SIZE,
            assignment.len(),
            assignment.cost(),
            duration
        );

    // Simple sanity check (ensure it actually assigns all pairs)
    assert_eq!(assignment.len(), 2 * BENCHMARK_SIZE);
}

fn benchmark_old_auction_algorithm(d1: PersistenceDiagram, d2: PersistenceDiagram) {
    let start = Instant::now();
    let assignment = compute_optimal_matching_old(&d1, &d2);
    let duration = start.elapsed();

    println!(
            "\n========================================\nOldAuctionAlgorithm benchmark (n = {}):\n  Assignments = {}\n  Total cost = {:.6}\n  Time = {:?}\n",
            BENCHMARK_SIZE,
            assignment.len(),
            assignment.cost(),
            duration
        );

    // Simple sanity check (ensure it actually assigns all pairs)
    assert_eq!(assignment.len(), 2 * BENCHMARK_SIZE);
}

fn benchmark_hungarian_algorithm(d1: PersistenceDiagram, d2: PersistenceDiagram) {
    let start = Instant::now();
    let assignment = compute_optimal_matching_hungarian(&d1, &d2);
    let duration = start.elapsed();

    println!(
            "\n========================================\nHungarianAlgorithm benchmark (n = {}):\n  Assignments = {}\n  Total cost = {:.6}\n  Time = {:?}\n",
            BENCHMARK_SIZE,
            assignment.len(),
            assignment.cost(),
            duration
        );

    // Simple sanity check (ensure it actually assigns all pairs)
    assert_eq!(assignment.len(), 2 * BENCHMARK_SIZE);
}

fn main() {
    let _d1 = make_random_diagram(BENCHMARK_SIZE);
    let _d2 = make_random_diagram(BENCHMARK_SIZE);

    let (d1, d2) = augment_diagrams(&_d1, &_d2);

    // Benchmark runs
    benchmark_old_auction_algorithm(d1.clone(), d2.clone());
    benchmark_auction_algorithm(d1.clone(), d2.clone());
    // benchmark_hungarian_algorithm(d1.clone(), d2.clone());
}
