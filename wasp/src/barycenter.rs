use crate::structs::{PersistencePair, PersistenceDiagram, Matching, augment_diagram_set, compute_optimal_matching, arithmetic_mean};
use std::collections::HashMap;

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

    for _iteration in 0..max_iterations {
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

/// BENCHMARK ONLY
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

    for _iteration in 0..max_iterations {
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
