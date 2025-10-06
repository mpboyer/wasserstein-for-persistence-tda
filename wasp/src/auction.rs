use crate::structs::{wasserstein_cost, PersistenceDiagram};
use std::collections::{HashMap, VecDeque};

/// Auction algorithm for computing optimal assignment between persistence diagrams
/// Based on Bertsekas (1981) and adapted for persistence diagrams (Kerber et al. 2016)
pub struct AuctionAlgorithm {
    epsilon: f64,
    prices: Vec<f64>,
    gamma: f64,
}

impl AuctionAlgorithm {
    /// Create a new auction algorithm with initial epsilon
    pub fn new(initial_epsilon: f64, gamma: f64) -> Self {
        Self {
            epsilon: initial_epsilon,
            prices: Vec::new(),
            gamma,
        }
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

        // initialize prices if needed
        if self.prices.len() != n {
            self.prices = vec![0.0; n];
        }

        // initialize epsilon if not provided
        if self.epsilon <= 0.0 {
            self.epsilon = self.compute_initial_epsilon(diagram1, diagram2);
        }

        // ---- precompute cost matrix ----
        let mut costs = vec![0.0; n * n];
        for i in 0..n {
            let p1 = &diagram1.pairs()[i];
            for j in 0..n {
                let p2 = &diagram2.pairs()[j];
                costs[i * n + j] = wasserstein_cost(p1, p2);
            }
        }

        // ---- assignments ----
        let mut assignment: Vec<Option<usize>> = vec![None; n];
        let mut reverse: Vec<Option<usize>> = vec![None; n];

        // ---- epsilon scaling ----
        loop {
            self.auction_round(&costs, &mut assignment, &mut reverse);

            // check if all assigned
            if assignment.iter().all(|x| x.is_some()) && self.epsilon < 1e-12 {
                break;
            }

            self.epsilon /= 5.0;
            if self.epsilon < 1e-12 {
                break;
            }
        }

        // ---- build result ----
        let mut result = HashMap::new();
        for (i, maybe_j) in assignment.iter().enumerate() {
            if let Some(j) = maybe_j {
                result.insert(i, *j);
            }
        }

        // ---- compute total cost ----
        let mut total_cost = 0.0;
        for (i, j) in &result {
            total_cost += costs[i * n + j];
        }

        (result, total_cost)
    }

    fn auction_round(
        &mut self,
        costs: &[f64],
        assignment: &mut Vec<Option<usize>>,
        reverse: &mut Vec<Option<usize>>,
    ) {
        let n = assignment.len();
        let mut queue: VecDeque<usize> = (0..n).filter(|i| assignment[*i].is_none()).collect();

        while let Some(bidder) = queue.pop_front() {
            // find best and second best objects
            let base = bidder * n;
            let mut best_j = 0;
            let mut best_val = f64::NEG_INFINITY;
            let mut second_val = f64::NEG_INFINITY;

            for j in 0..n {
                let val = -costs[base + j] - self.prices[j]; // benefit - price
                if val > best_val {
                    second_val = best_val;
                    best_val = val;
                    best_j = j;
                } else if val > second_val {
                    second_val = val;
                }
            }

            let bid_increase = (best_val - second_val) + self.epsilon;
            let price_update = self.prices[best_j] + bid_increase;

            // if best_j already assigned, unassign old owner
            if let Some(prev_owner) = reverse[best_j] {
                assignment[prev_owner] = None;
                queue.push_back(prev_owner);
            }

            // assign
            assignment[bidder] = Some(best_j);
            reverse[best_j] = Some(bidder);
            self.prices[best_j] = price_update;
        }
    }

    fn compute_initial_epsilon(&self, d1: &PersistenceDiagram, d2: &PersistenceDiagram) -> f64 {
        let mut max_cost: f64 = 0.0;
        for p1 in d1.pairs() {
            for p2 in d2.pairs() {
                max_cost = max_cost.max(wasserstein_cost(p1, p2));
            }
        }
        max_cost / 4.0
    }
}
