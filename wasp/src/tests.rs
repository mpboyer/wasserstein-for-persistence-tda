use crate::structs::*;
#[cfg(test)]
mod tests {
    use super::*;

    // ===== Basic PersistencePair Tests =====

    #[test]
    fn test_persistence_pair_creation() {
        let pair = PersistencePair::new(0.0, 1.0, 0, 1);
        assert_eq!(pair.birth, 0.0);
        assert_eq!(pair.death, 1.0);
        assert_eq!(pair.persistence(), 1.0);
    }

    #[test]
    #[should_panic]
    fn test_persistence_pair_invalid_elder_rule() {
        // Should panic because death_index != birth_index + 1
        PersistencePair::new(0.0, 1.0, 0, 2);
    }

    #[test]
    #[should_panic]
    fn test_persistence_pair_invalid_birth_death() {
        // Should panic because death < birth
        PersistencePair::new(1.0, 0.0, 0, 1);
    }

    #[test]
    fn test_diagonal_detection() {
        let diagonal = PersistencePair::new(0.5, 0.5, 0, 1);
        assert!(diagonal.is_diagonal());

        let off_diagonal = PersistencePair::new(0.0, 1.0, 0, 1);
        assert!(!off_diagonal.is_diagonal());
    }

    #[test]
    fn test_diagonal_projection() {
        let pair = PersistencePair::new(0.0, 2.0, 0, 1);
        let (b, d) = pair.diagonal_projection();
        assert_eq!(b, 1.0);
        assert_eq!(d, 1.0);
    }

    // ===== PersistenceDiagram Tests =====

    #[test]
    fn test_diagram_creation() {
        let mut diagram = PersistenceDiagram::new(2);
        assert_eq!(diagram.size(), 0);

        diagram.add_pair(PersistencePair::new(0.0, 1.0, 0, 1));
        assert_eq!(diagram.size(), 1);
    }

    #[test]
    fn test_filter_by_persistence() {
        let mut diagram = PersistenceDiagram::new(2);
        diagram.add_pair(PersistencePair::new(0.0, 0.1, 0, 1)); // persistence 0.1
        diagram.add_pair(PersistencePair::new(0.0, 0.5, 0, 1)); // persistence 0.5
        diagram.add_pair(PersistencePair::new(0.0, 1.0, 0, 1)); // persistence 1.0

        let filtered = diagram.filter_by_persistence(0.3);
        assert_eq!(filtered.size(), 2); // Should keep 0.5 and 1.0
    }

    #[test]
    fn test_off_diagonal_count() {
        let mut diagram = PersistenceDiagram::new(2);
        diagram.add_pair(PersistencePair::new(0.0, 1.0, 0, 1));
        diagram.add_pair(PersistencePair::new(0.5, 0.5, 0, 1)); // diagonal
        diagram.add_pair(PersistencePair::new(1.0, 2.0, 0, 1));

        assert_eq!(diagram.off_diagonal_count(), 2);
    }

    // ===== Augmentation Tests =====

    #[test]
    fn test_augment_diagrams_same_size() {
        let mut d1 = PersistenceDiagram::new(2);
        d1.add_pair(PersistencePair::new(0.0, 1.0, 0, 1));

        let mut d2 = PersistenceDiagram::new(2);
        d2.add_pair(PersistencePair::new(0.5, 1.5, 0, 1));

        let (aug1, aug2) = augment_diagrams(&d1, &d2);

        // Each should have 2 points: 1 original + 1 diagonal projection
        assert_eq!(aug1.size(), 2);
        assert_eq!(aug2.size(), 2);
    }

    #[test]
    fn test_augment_diagrams_different_sizes() {
        let mut d1 = PersistenceDiagram::new(2);
        d1.add_pair(PersistencePair::new(0.0, 1.0, 0, 1));

        let mut d2 = PersistenceDiagram::new(2);
        d2.add_pair(PersistencePair::new(0.5, 1.5, 0, 1));
        d2.add_pair(PersistencePair::new(1.0, 2.0, 0, 1));

        let (aug1, aug2) = augment_diagrams(&d1, &d2);

        // d1 gets 2 diagonal projections from d2
        // d2 gets 1 diagonal projection from d1
        assert_eq!(aug1.size(), 3); // 1 + 2
        assert_eq!(aug2.size(), 3); // 2 + 1
    }

    #[test]
    fn test_augment_diagram_set() {
        let mut d1 = PersistenceDiagram::new(2);
        d1.add_pair(PersistencePair::new(0.0, 1.0, 0, 1));

        let mut d2 = PersistenceDiagram::new(2);
        d2.add_pair(PersistencePair::new(0.5, 1.5, 0, 1));

        let mut d3 = PersistenceDiagram::new(2);
        d3.add_pair(PersistencePair::new(1.0, 2.0, 0, 1));

        let augmented = augment_diagram_set(&[d1, d2, d3]);

        // Each diagram should have 3 points (1 original + 2 from others)
        assert_eq!(augmented.len(), 3);
        assert_eq!(augmented[0].size(), 3);
        assert_eq!(augmented[1].size(), 3);
        assert_eq!(augmented[2].size(), 3);
    }

    // ===== Cost and Distance Tests =====

    #[test]
    fn test_wasserstein_cost_diagonal() {
        let p1 = PersistencePair::new(0.5, 0.5, 0, 1);
        let p2 = PersistencePair::new(1.0, 1.0, 0, 1);

        // Both diagonal, cost should be 0
        assert_eq!(wasserstein_cost(&p1, &p2), 0.0);
    }

    #[test]
    fn test_wasserstein_cost_off_diagonal() {
        let p1 = PersistencePair::new(0.0, 1.0, 0, 1);
        let p2 = PersistencePair::new(0.0, 2.0, 0, 1);

        // Squared distance: (0-0)^2 + (1-2)^2 = 1
        assert_eq!(wasserstein_cost(&p1, &p2), 1.0);
    }

    #[test]
    fn test_l2_distance() {
        let p1 = (0.0, 0.0);
        let p2 = (3.0, 4.0);

        assert_eq!(l2_distance(p1, p2), 5.0);
    }

    #[test]
    fn test_sql2_cost() {
        let p1 = (0.0, 0.0);
        let p2 = (3.0, 4.0);

        assert_eq!(sql2_cost(p1, p2), 25.0);
    }

    // ===== Matching Tests =====

    #[test]
    fn test_optimal_matching_identical() {
        let mut d1 = PersistenceDiagram::new(2);
        d1.add_pair(PersistencePair::new(0.0, 1.0, 0, 1));

        let d2 = d1.clone();

        let (aug1, aug2) = augment_diagrams(&d1, &d2);
        let matching = compute_optimal_matching(&aug1, &aug2);

        assert_eq!(matching.len(), aug1.size());
    }

    #[test]
    fn test_wasserstein_distance_identical() {
        let mut d1 = PersistenceDiagram::new(2);
        d1.add_pair(PersistencePair::new(0.0, 1.0, 0, 1));

        let d2 = d1.clone();

        let (aug1, aug2) = augment_diagrams(&d1, &d2);
        let matching = compute_optimal_matching(&aug1, &aug2);
        let dist_sq = compute_wasserstein_distance_squared(&aug1, &aug2, &matching);

        // Distance between identical diagrams should be 0
        assert!(dist_sq < 1e-6);
    }

    // ===== Simplex Projection Tests =====

    #[test]
    fn test_simplex_projection_already_valid() {
        let weights = vec![0.2, 0.3, 0.5];
        let projected = project_onto_simplex(&weights);

        // Should be approximately unchanged
        for i in 0..3 {
            assert!((projected[i] - weights[i]).abs() < 1e-6);
        }

        // Should sum to 1
        let sum: f64 = projected.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_simplex_projection_needs_adjustment() {
        let weights = vec![1.0, 2.0, 3.0];
        let projected = project_onto_simplex(&weights);

        // All should be non-negative
        assert!(projected.iter().all(|&w| w >= -1e-10));

        // Should sum to 1
        let sum: f64 = projected.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Should preserve order
        assert!(projected[0] <= projected[1]);
        assert!(projected[1] <= projected[2]);
    }

    #[test]
    fn test_simplex_projection_with_negatives() {
        let weights = vec![-1.0, 2.0, 1.0];
        let projected = project_onto_simplex(&weights);

        assert!(projected.iter().all(|&w| w >= -1e-10));
        let sum: f64 = projected.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_simplex_projection_uniform() {
        let weights = vec![0.0, 0.0, 0.0];
        let projected = project_onto_simplex(&weights);

        // Should become uniform
        for &w in &projected {
            assert!((w - 1.0 / 3.0).abs() < 1e-6);
        }
    }

    // ===== Arithmetic Mean Tests =====

    #[test]
    fn test_arithmetic_mean_uniform_weights() {
        let points = vec![(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)];
        let weights = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];

        let mean = arithmetic_mean(&points, &weights);
        assert!((mean.0 - 1.0).abs() < 1e-6);
        assert!((mean.1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_arithmetic_mean_weighted() {
        let points = vec![(0.0, 0.0), (2.0, 2.0)];
        let weights = vec![0.75, 0.25];

        let mean = arithmetic_mean(&points, &weights);
        assert!((mean.0 - 0.5).abs() < 1e-6);
        assert!((mean.1 - 0.5).abs() < 1e-6);
    }

    #[test]
    #[should_panic]
    fn test_arithmetic_mean_invalid_weights() {
        let points = vec![(0.0, 0.0), (1.0, 1.0)];
        let weights = vec![0.3, 0.5]; // Don't sum to 1

        arithmetic_mean(&points, &weights);
    }

    // ===== Integration Tests =====

    #[test]
    fn test_frechet_energy() {
        let mut barycenter = PersistenceDiagram::new(2);
        barycenter.add_pair(PersistencePair::new(0.5, 1.5, 0, 1));

        let mut atom1 = PersistenceDiagram::new(2);
        atom1.add_pair(PersistencePair::new(0.0, 1.0, 0, 1));

        let mut atom2 = PersistenceDiagram::new(2);
        atom2.add_pair(PersistencePair::new(1.0, 2.0, 0, 1));

        let atoms = vec![atom1, atom2];
        let weights = vec![0.5, 0.5];

        // Augment all diagrams
        let mut all_diagrams = vec![barycenter.clone()];
        all_diagrams.extend(atoms.clone());
        let augmented = augment_diagram_set(&all_diagrams);

        let aug_barycenter = &augmented[0];
        let aug_atoms = &augmented[1..];

        // Compute matchings
        let matchings: Vec<_> = aug_atoms
            .iter()
            .map(|atom| compute_optimal_matching(aug_barycenter, atom))
            .collect();

        let energy = frechet_energy(aug_barycenter, aug_atoms, &weights, &matchings);

        // Energy should be positive for non-identical diagrams
        assert!(energy > 0.0);
    }

    #[test]
    fn test_filter_by_relative_persistence() {
        let mut diagram = PersistenceDiagram::new(2);
        diagram.add_pair(PersistencePair::new(0.0, 0.1, 0, 1));
        diagram.add_pair(PersistencePair::new(0.0, 0.5, 0, 1));
        diagram.add_pair(PersistencePair::new(0.0, 1.0, 0, 1));

        let scalar_range = 10.0;
        let tau = 0.04; // 4% of range = 0.4

        let filtered = diagram.filter_by_relative_persistence(tau, scalar_range);

        // Should keep only pairs with persistence >= 0.4
        assert_eq!(filtered.size(), 2);
    }
}
