//! Run with: cargo test -- --nocapture
#[cfg(test)]
mod matching {
    use std::time::Instant;
    use wasp::structs::{compute_optimal_matching, compute_optimal_matching_hungarian};
    use wasp::structs::{PersistenceDiagram, PersistencePair};

    static BENCHMARK_SIZE: usize = 1000;

    /// Generate a synthetic persistence diagram of `n` points
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

    #[test]
    fn benchmark_auction_algorithm() {
        // Create two random persistence diagrams
        let d1 = make_random_diagram(BENCHMARK_SIZE);
        let d2 = make_random_diagram(BENCHMARK_SIZE);

        // Benchmark run
        let start = Instant::now();
        let assignment = compute_optimal_matching(&d1, &d2);
        let duration = start.elapsed();

        println!(
            "AuctionAlgorithm benchmark (n = {}):\n  Assignments = {}\n  Total cost = {:.6}\n  Time = {:?}",
            BENCHMARK_SIZE,
            assignment.len(),
            assignment.cost(),
            duration
        );

        // Simple sanity check (ensure it actually assigns all pairs)
        assert_eq!(assignment.len(), BENCHMARK_SIZE);
    }

    #[test]
    fn benchmark_hungarian_algorithm() {
        // Create two random persistence diagrams
        let d1 = make_random_diagram(BENCHMARK_SIZE);
        let d2 = make_random_diagram(BENCHMARK_SIZE);

        // Benchmark run
        let start = Instant::now();
        let assignment = compute_optimal_matching_hungarian(&d1, &d2);
        let duration = start.elapsed();

        println!(
            "HungarianAlgorithm benchmark (n = {}):\n  Assignments = {}\n  Total cost = {:.6}\n  Time = {:?}",
            BENCHMARK_SIZE,
            assignment.len(),
            assignment.cost(),
            duration
        );

        // Simple sanity check (ensure it actually assigns all pairs)
        assert_eq!(assignment.len(), BENCHMARK_SIZE);
    }
}
