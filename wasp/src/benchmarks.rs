//! Run with: cargo test -- --nocapture

#[cfg(test)]
mod tests {
    use std::time::Instant;
    use crate::auction::AuctionAlgorithm;
    use crate::structs::{PersistenceDiagram, PersistencePair};

    /// Generate a synthetic persistence diagram of `n` points
    fn make_random_diagram(n: usize) -> PersistenceDiagram {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let pairs: Vec<PersistencePair> = (0..n)
            .map(|_| {
                let birth: f64 = rng.gen::<f64>();
                let death = birth + rng.gen::<f64>() * 0.5;
                PersistencePair::new(birth, death, 0, 1)
            })
            .collect();
        PersistenceDiagram::from_pairs(pairs, 2)
    }

    #[test]
    fn benchmark_auction_algorithm() {
        // Adjust n to control benchmark size
        let n = 1000;

        // Create two random persistence diagrams
        let d1 = make_random_diagram(n);
        let d2 = make_random_diagram(n);

        // Initialize auction solver
        let mut auction = AuctionAlgorithm::new(0.0, 0.01);

        // Warmup run
        let _ = auction.run(&d1, &d2);

        // Benchmark run
        let start = Instant::now();
        let (assignment, total_cost) = auction.run(&d1, &d2);
        let duration = start.elapsed();

        println!(
            "AuctionAlgorithm benchmark (n = {}):\n  Assignments = {}\n  Total cost = {:.6}\n  Time = {:?}",
            n,
            assignment.len(),
            total_cost,
            duration
        );

        // Simple sanity check (ensure it actually assigns all pairs)
        assert_eq!(assignment.len(), n);
    }
}

