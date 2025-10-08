use core::f64;
use std::collections::HashMap;

use pathfinding::kuhn_munkres::kuhn_munkres_min;
use pathfinding::matrix::Matrix;

/// Computes arithmetic mean of points in birth/death space (Section 2.3)
/// Used in the Update step of barycenter computation
pub fn arithmetic_mean(points: &[(f64, f64)], weights: &[f64]) -> (f64, f64) {
    assert_eq!(points.len(), weights.len());
    assert!(
        (weights.iter().sum::<f64>() - 1.0).abs() < 1e-6,
        "Weights must sum to 1"
    );

    let mut sum_b = 0.0;
    let mut sum_d = 0.0;

    for (point, weight) in points.iter().zip(weights.iter()) {
        sum_b += weight * point.0;
        sum_d += weight * point.1;
    }

    (sum_b, sum_d)
}

/// Projection onto probability simplex in (dlog d) based on <https://arxiv.org/pdf/1309.1541>
pub fn project_onto_simplex(weights: &[f64]) -> Vec<f64> {
    let d = weights.len();
    if d == 0 {
        return vec![];
    }

    let mut sorted: Vec<f64> = weights.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());

    let mut theta = 0.0;
    let mut rho = 0;

    for j in 0..d {
        theta += sorted[j];
        let candidate_theta = sorted[j] + (1.0 - theta) / (j + 1) as f64;

        if candidate_theta > 0.0 {
            rho = j + 1;
        } else {
            break;
        }
    }

    let sum_rho: f64 = sorted[..rho].iter().sum();
    let lambda = (1.0 - sum_rho) / rho as f64;

    weights.iter().map(|&w| (w + lambda).max(0.0)).collect()
}

/// Helper functions for computing l2 distance and squared l2 distance.
pub fn l2_distance(p1: (f64, f64), p2: (f64, f64)) -> f64 {
    let dx = p1.0 - p2.0;
    let dy = p1.1 - p2.1;
    (dx * dx + dy * dy).sqrt()
}

pub fn sql2_cost(p1: (f64, f64), p2: (f64, f64)) -> f64 {
    let dx = p1.0 - p2.0;
    let dy = p1.1 - p2.1;
    dx * dx + dy * dy
}

/// Computes wasserstein_cost for two points in augmented diagrams (based on 2.2)
pub fn wasserstein_cost(p1: &PersistencePair, p2: &PersistencePair) -> f64 {
    if p1.is_diagonal() && p2.is_diagonal() {
        0.0
    } else {
        sql2_cost(p1.to_point(), p2.to_point())
    }
}

/// Helper function to augment diagrams to have the same cardinal
pub fn augment_diagrams(
    diagram1: &PersistenceDiagram,
    diagram2: &PersistenceDiagram,
) -> (PersistenceDiagram, PersistenceDiagram) {
    let mut aug1_pairs = diagram1.pairs().to_vec();
    let mut aug2_pairs = diagram2.pairs().to_vec();

    for pair in diagram2.pairs() {
        if !pair.is_diagonal() {
            let (b, d) = pair.diagonal_projection();
            aug1_pairs.push(PersistencePair::new(
                b,
                d,
                pair.birth_index,
                pair.death_index,
            ));
        }
    }

    for pair in diagram1.pairs() {
        if !pair.is_diagonal() {
            let (b, d) = pair.diagonal_projection();
            aug2_pairs.push(PersistencePair::new(
                b,
                d,
                pair.birth_index,
                pair.death_index,
            ));
        }
    }

    (
        PersistenceDiagram::from_pairs(aug1_pairs, diagram1.manifold_dimension),
        PersistenceDiagram::from_pairs(aug2_pairs, diagram2.manifold_dimension),
    )
}

pub fn augment_diagram_set(diagrams: &[PersistenceDiagram]) -> Vec<PersistenceDiagram> {
    if diagrams.is_empty() {
        return vec![];
    }

    let mut augmented = Vec::new();

    for (i, diagram) in diagrams.iter().enumerate() {
        let mut aug_pairs = diagram.pairs().to_vec();

        for (j, other) in diagrams.iter().enumerate() {
            if i != j {
                for pair in other.pairs() {
                    if !pair.is_diagonal() {
                        let (b, d) = pair.diagonal_projection();
                        aug_pairs.push(PersistencePair::new(
                            b,
                            d,
                            pair.birth_index,
                            pair.death_index,
                        ));
                    }
                }
            }
        }
        augmented.push(PersistenceDiagram::from_pairs(
            aug_pairs,
            diagram.manifold_dimension(),
        ));
    }
    augmented
}

/// Represents an optimal matching between two persistence diagrams
#[derive(Debug, Clone)]
pub struct Matching {
    /// Maps indices from first diagram to second diagram
    assignments: HashMap<usize, usize>,
    /// Total cost of the matching (sum of squared distances)
    cost: f64,
}

impl Matching {
    /// Creates a new matching with assignments and cost
    pub fn new(assignments: HashMap<usize, usize>, cost: f64) -> Self {
        Matching { assignments, cost }
    }

    /// Returns the assignment for a given index
    pub fn get(&self, index: &usize) -> Option<&usize> {
        self.assignments.get(index)
    }

    /// Returns the total cost of the matching
    pub fn cost(&self) -> f64 {
        self.cost
    }

    /// Returns a reference to the assignments map
    pub fn assignments(&self) -> &HashMap<usize, usize> {
        &self.assignments
    }

    pub fn len(&self) -> usize {
        self.assignments.len()
    }
}

/// Compute optimal matching/transport map for assignment between two persistence diagrams
use crate::auction::AuctionAlgorithm;

/// Compute optimal matching/transport map for assignment between two persistence diagrams
/// using the Auction algorithm (more efficient than Munkres)
pub fn compute_optimal_matching(
    diagram1: &PersistenceDiagram,
    diagram2: &PersistenceDiagram,
) -> Matching {
    let n = diagram1.size();
    assert_eq!(
        n,
        diagram2.size(),
        "Diagrams must be augmented to the same size"
    );

    if n == 0 {
        return Matching::new(HashMap::new(), 0.);
    }

    // Use auction algorithm with default parameters
    // gamma = 0.01 as suggested in the paper
    let mut auction = AuctionAlgorithm::new(-1.0, 0.01); // -1.0 means auto-compute initial epsilon
    let (assignment, cost) = auction.run(diagram1, diagram2);

    Matching::new(assignment, cost)
}

/// Compute optimal matching/transport map for assignment between two persistence diagrams
/// USED ONLY FOR BENCHAMRK PURPOSES, SUCKS
pub fn compute_optimal_matching_hungarian(
    diagram1: &PersistenceDiagram,
    diagram2: &PersistenceDiagram,
) -> Matching {
    let n = diagram1.size();
    assert_eq!(
        n,
        diagram2.size(),
        "Diagrams must be augmented to the same size"
    );

    if n == 0 {
        return Matching::new(HashMap::new(), 0.);
    }

    // Scale floats to integers for `pathfinding` crate
    const SCALE_FACTOR: f64 = 1_000_000.0;

    let mut costs = Vec::with_capacity(n * n);
    (0..n).for_each(|i| {
        (0..n).for_each(|j| {
            let cost = wasserstein_cost(&diagram1.pairs()[i], &diagram2.pairs()[j]);
            costs.push((cost * SCALE_FACTOR).round() as i64);
        });
    });

    let cost_matrix = Matrix::from_vec(n, n, costs).unwrap();

    let (total_cost, assignments) = kuhn_munkres_min(&cost_matrix);

    let mut matching = HashMap::new();
    for (i, &j) in assignments.iter().enumerate() {
        matching.insert(i, j);
    }
    let cost = (total_cost as f64) / SCALE_FACTOR;

    Matching::new(matching, cost)
}

/// Compute Fréchet energy of a barycenter
pub fn frechet_energy(
    barycenter: &PersistenceDiagram,
    atoms: &[PersistenceDiagram],
    weights: &[f64],
    matchings: &[Matching],
) -> f64 {
    assert_eq!(atoms.len(), weights.len());
    assert_eq!(atoms.len(), matchings.len());

    let mut energy = 0.0;
    atoms
        .iter()
        .zip(weights.iter())
        .zip(matchings.iter())
        .for_each(|((atom, &weight), matching)| {
            let dist_sq = matching.cost;
            energy += weight * dist_sq;
        });
    energy
}

/// Represents a persistence pair (birth, death) of a certain topological feature
#[derive(Debug, Clone, Copy)]
pub struct PersistencePair {
    /// Birth value: isovalue (scalar) where feature appears
    pub birth: f64,
    /// Death value: isovalue (scalar) where feature disappears
    pub death: f64,

    /// Index of the birth critical point (integer number of negative eigenvalues of hessian at
    /// critical point)
    pub birth_index: usize,
    /// Index of the death critical point (integer number of negative eigenvalues of hessian at
    /// critical point)
    pub death_index: usize,
}

impl PersistencePair {
    /// Creates a new persistence pair and checks for the Elder rule
    pub fn new(birth: f64, death: f64, birth_index: usize, death_index: usize) -> Self {
        assert!(death >= birth);
        assert_eq!(death_index, birth_index + 1);
        PersistencePair {
            birth,
            death,
            birth_index,
            death_index,
        }
    }

    /// Returns the persistence or lifespan of the feature represented by the persistence pair
    pub fn persistence(&self) -> f64 {
        self.death - self.birth
    }

    /// Boolean check if the point is on the diagonal
    pub fn is_diagonal(&self) -> bool {
        (self.death - self.birth) < f64::EPSILON
    }

    pub fn diagonal_projection(&self) -> (f64, f64) {
        let midpoint = (self.birth + self.death) / 2.0;
        (midpoint, midpoint)
    }

    pub fn to_point(&self) -> (f64, f64) {
        (self.birth, self.death)
    }
}

#[derive(Debug, Clone)]
pub struct PersistenceDiagram {
    /// Set of persistence pairs
    pairs: Vec<PersistencePair>,
    manifold_dimension: usize,
}

impl PersistenceDiagram {
    /// Initializes a new PersistenceDiagram given a manifold_dimension
    pub fn new(manifold_dimension: usize) -> Self {
        PersistenceDiagram {
            pairs: Vec::new(),
            manifold_dimension,
        }
    }

    /// Creates a persistence diagram from a vector of pairs
    pub fn from_pairs(pairs: Vec<PersistencePair>, manifold_dimension: usize) -> Self {
        PersistenceDiagram {
            pairs,
            manifold_dimension,
        }
    }

    /// Adds a persistence pair to the diagram
    pub fn add_pair(&mut self, pair: PersistencePair) {
        self.pairs.push(pair);
    }

    /// Returns the number of points (pairs) in the diagram
    pub fn size(&self) -> usize {
        self.pairs.len()
    }

    /// Returns a reference to the pairs
    pub fn pairs(&self) -> &[PersistencePair] {
        &self.pairs
    }

    /// Filters pairs by minimum persistence threshold
    pub fn filter_by_persistence(&self, threshold: f64) -> PersistenceDiagram {
        let filtered_pairs: Vec<PersistencePair> = self
            .pairs
            .iter()
            .filter(|p| p.persistence() >= threshold)
            .copied()
            .collect();
        PersistenceDiagram::from_pairs(filtered_pairs, self.manifold_dimension)
    }

    pub fn filter_by_relative_persistence(
        &self,
        tau: f64,
        scalar_range: f64,
    ) -> PersistenceDiagram {
        let threshold = tau * scalar_range;
        self.filter_by_persistence(threshold)
    }

    /// Helper to get manifold dimension
    pub fn manifold_dimension(&self) -> usize {
        self.manifold_dimension
    }

    /// Returns only non-diagonal points
    pub fn off_diagonal_pairs(&self) -> Vec<&PersistencePair> {
        self.pairs.iter().filter(|p| !p.is_diagonal()).collect()
    }

    /// Number of non-diagonal points
    pub fn off_diagonal_count(&self) -> usize {
        self.pairs.iter().filter(|p| !p.is_diagonal()).count()
    }
}

/// Computes the Wasserstein barycenter of a set of persistence diagrams
/// given weights, using the progressive algorithm approach of cited paper in section 2.3
pub fn compute_wasserstein_barycenter(
    diagrams: &[PersistenceDiagram],
    weights: &[f64],
) -> PersistenceDiagram {
    assert_eq!(diagrams.len(), weights.len());
    assert!(!diagrams.is_empty(), "Need at least one diagram");
    assert!(
        (weights.iter().sum::<f64>() - 1.0).abs() < 1e-6,
        "Weights must sum to 1"
    );

    // Augment all diagrams to have the same cardinality
    let augmented = augment_diagram_set(diagrams);

    // Initialize barycenter with a random diagram from the set
    // (in practice, choose the one that minimizes initial Frechet energy)
    let mut barycenter = augmented[0].clone();

    // Store matchings for each diagram
    let mut matchings: Vec<Matching> = vec![];

    let max_iterations = 100;
    let mut prev_matchings: Vec<HashMap<usize, usize>> = vec![];

    for iteration in 0..max_iterations {
        // Assignment step: compute optimal matching between barycenter and each diagram
        matchings.clear();
        for diagram in augmented.iter() {
            let matching = compute_optimal_matching(&barycenter, diagram);
            matchings.push(matching);
        }

        // Check convergence: if assignments haven't changed, stop
        let current_assignments: Vec<HashMap<usize, usize>> =
            matchings.iter().map(|m| m.assignments().clone()).collect();

        if !prev_matchings.is_empty() && assignments_equal(&prev_matchings, &current_assignments) {
            break;
        }

        prev_matchings = current_assignments;

        barycenter = update_barycenter(&augmented, &matchings, weights);
    }

    barycenter
}

/// BENCHAMRK ONLY
pub fn compute_wasserstein_barycenter_no_progressive(
    diagrams: &[PersistenceDiagram],
    weights: &[f64],
) -> PersistenceDiagram {
    assert_eq!(diagrams.len(), weights.len());
    assert!(!diagrams.is_empty(), "Need at least one diagram");
    assert!(
        (weights.iter().sum::<f64>() - 1.0).abs() < 1e-6,
        "Weights must sum to 1"
    );

    // Augment all diagrams to have the same cardinality
    let augmented = augment_diagram_set(diagrams);

    // Initialize barycenter with a random diagram from the set
    // (in practice, choose the one that minimizes initial Frechet energy)
    let mut barycenter = augmented[0].clone();

    // Store matchings for each diagram
    let mut matchings: Vec<Matching> = vec![];

    let max_iterations = 100;
    let mut prev_matchings: Vec<HashMap<usize, usize>> = vec![];

    for iteration in 0..max_iterations {
        // Assignment step: compute optimal matching between barycenter and each diagram
        matchings.clear();
        for diagram in augmented.iter() {
            let matching = compute_optimal_matching(&barycenter, diagram);
            matchings.push(matching);
        }

        // Check convergence: if assignments haven't changed, stop
        let current_assignments: Vec<HashMap<usize, usize>> =
            matchings.iter().map(|m| m.assignments().clone()).collect();

        if !prev_matchings.is_empty() && assignments_equal(&prev_matchings, &current_assignments) {
            break;
        }

        prev_matchings = current_assignments;

        barycenter = update_barycenter(&augmented, &matchings, weights);
    }

    barycenter
}

/// Helper function to check if two sets of assignments are equal
fn assignments_equal(prev: &[HashMap<usize, usize>], current: &[HashMap<usize, usize>]) -> bool {
    if prev.len() != current.len() {
        return false;
    }

    for (p, c) in prev.iter().zip(current.iter()) {
        if p != c {
            return false;
        }
    }

    true
}

/// Update step: compute new barycenter positions as arithmetic means
fn update_barycenter(
    diagrams: &[PersistenceDiagram],
    matchings: &[Matching],
    weights: &[f64],
) -> PersistenceDiagram {
    assert_eq!(diagrams.len(), matchings.len());
    assert_eq!(diagrams.len(), weights.len());

    if diagrams.is_empty() {
        return PersistenceDiagram::new(0);
    }

    let n = diagrams[0].size();
    let mut new_pairs = Vec::new();

    // For each point in the barycenter
    for j in 0..n {
        // Collect matched points from all diagrams
        let mut matched_points = Vec::new();
        let mut point_weights = Vec::new();

        for (i, matching) in matchings.iter().enumerate() {
            if let Some(&matched_idx) = matching.get(&j) {
                let point = diagrams[i].pairs()[matched_idx].to_point();
                matched_points.push(point);
                point_weights.push(weights[i]);
            }
        }

        // Compute weighted arithmetic mean
        if !matched_points.is_empty() {
            let mean_point = arithmetic_mean(&matched_points, &point_weights);

            // Create a persistence pair from the mean point
            // Use dummy indices for birth/death
            let pair = PersistencePair::new(mean_point.0, mean_point.1, 0, 1);
            new_pairs.push(pair);
        }
    }

    // Return new barycenter with same manifold dimension as inputs
    PersistenceDiagram::from_pairs(new_pairs, diagrams[0].manifold_dimension())
}
