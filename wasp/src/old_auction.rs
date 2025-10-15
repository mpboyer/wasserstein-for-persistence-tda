use crate::structs::{wasserstein_cost, PersistenceDiagram};
use std::collections::HashMap;

/// Auction algorithm for computing optimal assignment between persistence diagrams
/// Based on Bertsekas (1981) and adapted for persistence diagrams [Kerber et al.
/// 2016](../../geometry_helps_to_compare_persistence_diagrams.pdf)
pub struct OldAuctionAlgorithm {
    epsilon: f64,
    prices: Vec<f64>,
    gamma: f64,
}

impl OldAuctionAlgorithm {
    /// Create a new auction algorithm with initial epsilon
    pub fn new(initial_epsilon: f64, gamma: f64) -> Self {
        Self {
            epsilon: initial_epsilon,
            prices: Vec::new(),
            gamma,
        }
    }

    pub fn terminate(&self, d: f64, n: f64) -> bool {
        if self.epsilon < 1e-15 {
            return true;
        }

        if d < 0.0 {
            return false;
        }

        let d = d.powf(2.);
        let rhs = d - n * self.epsilon;
        if rhs <= 0.0 {
            return false;
        }

        let thresh = (1.0 + self.gamma).powf(2.) * rhs;
        d <= thresh
    }

    /// Run auction algorithm to find optimal assignment
    /// Returns (assignment map, total cost)
    pub fn run(
        &mut self,
        diagram1: &PersistenceDiagram,
        diagram2: &PersistenceDiagram,
    ) -> (HashMap<usize, usize>, f64) {
        let n = diagram1.size();
        assert_eq!(
            n,
            diagram2.size(),
            "Diagrams must be augmented to the same size"
        );
        if n == 0 {
            return (HashMap::new(), 0.0);
        }

        self.prices = vec![0.0; n];

        let mut costs = vec![0.0; n * n];
        let mut max_cost = 0.0;
        for i in 0..n {
            let p1 = &diagram1.pairs()[i];
            for j in 0..n {
                let p2 = &diagram2.pairs()[j];
                let c = wasserstein_cost(p1, p2);
                costs[i * n + j] = c;
                max_cost = if c >= max_cost { c } else { max_cost };
            }
        }

        if self.epsilon <= 0.0 {
            self.epsilon = 5.0 * max_cost / 4.0;
        }

        let mut d = 0.0;
        let mut assignment = HashMap::new();

        while !self.terminate(d, n as f64) {
            self.epsilon /= 5.0;

            if self.epsilon < 1e-15 {
                break;
            }

            let (new_assignment, matching_cost) = self.auction_round(&costs, n);

            assignment = new_assignment;
            d = matching_cost;
        }
        (assignment, d)
    }

    fn auction_round(&mut self, costs: &[f64], n: usize) -> (HashMap<usize, usize>, f64) {
        let mut assignment: Vec<Option<usize>> = vec![None; n];
        let mut reverse: Vec<Option<usize>> = vec![None; n];

        while let Some(b) = (0..n).find(|&i| assignment[i].is_none()) {
            let bidder = b;
            let base = bidder * n;
            let mut best_j = 0;
            let mut best_value = f64::NEG_INFINITY;
            let mut snd_value = f64::NEG_INFINITY;

            for j in 0..n {
                // Value = benefit - price = -cost - price
                let value = -costs[base + j] - self.prices[j];

                if value > best_value {
                    snd_value = best_value;
                    best_value = value;
                    best_j = j;
                } else if value > snd_value {
                    snd_value = value;
                }
            }

            let price_increment = (best_value - snd_value) + self.epsilon;

            if let Some(prev_owner) = reverse[best_j] {
                assignment[prev_owner] = None;
            }

            assignment[bidder] = Some(best_j);
            reverse[best_j] = Some(bidder);

            self.prices[best_j] += price_increment;
        }

        let mut result = HashMap::new();
        let mut total_cost = 0.0;

        for (i, maybe_j) in assignment.iter().enumerate() {
            if let Some(j) = maybe_j {
                result.insert(i, *j);
                total_cost += costs[i * n + *j];
            }
        }

        (result, total_cost)
    }
}
