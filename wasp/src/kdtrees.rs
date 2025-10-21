use pathfinding::matrix::directions::S;

use crate::structs;
use crate::structs::{sql2_cost, PersistenceDiagram, PersistencePair};
use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::iter::Rev;

#[derive(Debug, Clone, Copy)]
struct BoundingBox {
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
}

impl BoundingBox {
    fn new(min_x: f64, max_x: f64, min_y: f64, max_y: f64) -> Self {
        BoundingBox {
            min_x,
            max_x,
            min_y,
            max_y,
        }
    }

    fn from_point(x: f64, y: f64) -> Self {
        BoundingBox::new(x, x, y, y)
    }

    fn merge(&self, other: &BoundingBox) -> Self {
        BoundingBox::new(
            self.min_x.min(other.min_x),
            self.max_x.max(other.max_x),
            self.min_y.min(other.min_y),
            self.max_y.max(other.max_y),
        )
    }

    fn sql2_point(&self, query: (f64, f64)) -> f64 {
        let dx = if query.0 < self.min_x {
            self.min_x - query.0
        } else if query.0 > self.max_x {
            query.0 - self.max_x
        } else {
            0.0
        };

        let dy = if query.1 < self.min_y {
            self.min_y - query.1
        } else if query.1 > self.max_y {
            query.1 - self.max_y
        } else {
            0.0
        };

        dx * dx + dy * dy
    }
}

#[derive(Debug, Clone)]
struct KdNode {
    point_idx: usize,  // ref to diagram 2
    point: (f64, f64), // (birth, death) coordinates
    weight: f64,       // current price
    min_subtree_weight: f64,
    left: Option<Box<KdNode>>,
    right: Option<Box<KdNode>>,
    bbox: BoundingBox,
    subtree_size: usize,
    axis: u8,
}

impl KdNode {
    fn new(point_idx: usize, point: (f64, f64), weight: f64, axis: u8) -> Self {
        KdNode {
            point_idx,
            point,
            weight,
            min_subtree_weight: weight,
            left: None,
            right: None,
            bbox: BoundingBox::from_point(point.0, point.1),
            subtree_size: 1,
            axis,
        }
    }

    fn update_bbox(&mut self) {
        let mut bbox = BoundingBox::from_point(self.point.0, self.point.1);

        if let Some(ref left) = self.left {
            bbox = bbox.merge(&left.bbox);
        }

        if let Some(ref right) = self.right {
            bbox = bbox.merge(&right.bbox);
        }

        self.bbox = bbox;
    }

    fn update_min_weight(&mut self) {
        let mut min_weight = self.weight;

        if let Some(ref left) = self.left {
            min_weight = min_weight.min(left.min_subtree_weight);
        }

        if let Some(ref right) = self.right {
            min_weight = min_weight.min(right.min_subtree_weight);
        }

        self.min_subtree_weight = min_weight;
    }

    fn update_size(&mut self) {
        let mut size = 1;

        if let Some(ref left) = self.left {
            size += left.subtree_size;
        }
        if let Some(ref right) = self.right {
            size += right.subtree_size;
        }

        self.subtree_size = size;
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SearchCandidate {
    pub point_idx: usize,
    pub value: f64, // benefit - price = -dist^q - price
}

impl SearchCandidate {
    fn new(point_idx: usize, value: f64) -> Self {
        SearchCandidate { point_idx, value }
    }

    fn neg_infinity() -> Self {
        SearchCandidate {
            point_idx: usize::MAX,
            value: f64::NEG_INFINITY,
        }
    }

    fn is_valid(&self) -> bool {
        self.point_idx != usize::MAX
    }
}

#[derive(Debug, Clone)]
pub struct KdTree {
    root: Option<Box<KdNode>>,
    points: Vec<(f64, f64)>,
    weights: Vec<f64>,
}

impl KdTree {
    pub fn new(points: Vec<(f64, f64)>) -> Self {
        let n = points.len();
        let weights = vec![0.0; n];

        if n == 0 {
            return KdTree {
                root: None,
                points,
                weights,
            };
        }

        let mut indices: Vec<usize> = (0..n).collect();
        let root = Self::build_tree(&points, &weights, &mut indices, 0);

        KdTree {
            root,
            points,
            weights,
        }
    }

    fn build_tree(
        points: &[(f64, f64)],
        weights: &[f64],
        indices: &mut [usize],
        depth: usize,
    ) -> Option<Box<KdNode>> {
        if indices.is_empty() {
            return None;
        }

        let axis = (depth % 2) as u8;

        let mid = indices.len() / 2;
        indices.select_nth_unstable_by(mid, |&a, &b| {
            let val_a = if axis == 0 { points[a].0 } else { points[a].1 };
            let val_b = if axis == 0 { points[b].0 } else { points[b].1 };
            val_a.partial_cmp(&val_b).unwrap_or(Ordering::Equal)
        });

        let median_idx = indices[mid];
        let mut node = KdNode::new(median_idx, points[median_idx], weights[median_idx], axis);

        node.left = Self::build_tree(points, weights, &mut indices[..mid], depth + 1);
        node.right = Self::build_tree(points, weights, &mut indices[mid + 1..], depth + 1);

        node.update_bbox();
        node.update_min_weight();
        node.update_size();

        Some(Box::new(node))
    }

    pub fn get_weight(&self, point_idx: usize) -> f64 {
        self.weights[point_idx]
    }

    pub fn update_weight(&mut self, point_idx: usize, new_weight: f64) {
        self.weights[point_idx] = new_weight;
        let target_point = self.points[point_idx];

        if let Some(ref mut root) = self.root {
            Self::update_weight_recursive(root, point_idx, new_weight, target_point);
        }
    }

    fn update_weight_recursive(
        node: &mut Box<KdNode>,
        point_idx: usize,
        new_weight: f64,
        target_point: (f64, f64),
    ) -> bool {
        if node.point_idx == point_idx {
            node.weight = new_weight;
            let old_min = node.min_subtree_weight;
            node.update_min_weight();
            return old_min != node.min_subtree_weight;
        }

        let mut changed = false;
        let axis = node.axis;
        let split_val = if axis == 0 {
            node.point.0
        } else {
            node.point.1
        };

        let target_val = if axis == 0 {
            target_point.0
        } else {
            target_point.1
        };

        if target_val < split_val {
            if let Some(ref mut left) = node.left {
                changed = Self::update_weight_recursive(left, point_idx, new_weight, target_point);
            }
        } else {
            if let Some(ref mut right) = node.right {
                changed = Self::update_weight_recursive(right, point_idx, new_weight, target_point);
            }
        }

        if changed {
            let old_min = node.min_subtree_weight;
            node.update_min_weight();

            old_min != node.min_subtree_weight
        } else {
            false
        }
    }

    pub fn find_best_two(&self, query_point: (f64, f64)) -> (SearchCandidate, SearchCandidate) {
        let mut best = SearchCandidate::neg_infinity();
        let mut snd_best = SearchCandidate::neg_infinity();

        if let Some(ref root) = self.root {
            self.search_best_two(root, query_point, &mut best, &mut snd_best);
        }

        (best, snd_best)
    }

    fn search_best_two(
        &self,
        node: &KdNode,
        query: (f64, f64),
        best: &mut SearchCandidate,
        snd_best: &mut SearchCandidate,
    ) {
        let dist = structs::sql2_cost(query, node.point);
        let value = -dist - node.weight;

        if value > best.value {
            *snd_best = *best;
            *best = SearchCandidate::new(node.point_idx, value);
        } else if value > snd_best.value {
            *snd_best = SearchCandidate::new(node.point_idx, value);
        }

        let bbox_dist = node.bbox.sql2_point(query);
        let lower_bound = -bbox_dist - node.min_subtree_weight;

        if lower_bound <= snd_best.value {
            return;
        }

        let axis = node.axis;
        let query_val = if axis == 0 { query.0 } else { query.1 };
        let split_val = if axis == 0 {
            node.point.0
        } else {
            node.point.1
        };

        let (closer, farther) = if query_val < split_val {
            (&node.left, &node.right)
        } else {
            (&node.right, &node.left)
        };

        if let Some(ref child) = closer {
            self.search_best_two(child, query, best, snd_best);
        }

        if let Some(ref child) = farther {
            let farther_bbox_dist = child.bbox.sql2_point(query);
            let farther_lower_bound = -farther_bbox_dist - child.min_subtree_weight;
            if farther_lower_bound > snd_best.value {
                self.search_best_two(child, query, best, snd_best);
            }
        }
    }
}

/// Heap element for diagonal objects in auction algorithm
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeapElement {
    pub index: usize,
    pub price: f64,
}

impl Eq for HeapElement {}

impl PartialOrd for HeapElement {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapElement {
    fn cmp(&self, other: &Self) -> Ordering {
        // For use with Reverse to create min-heap
        self.price
            .partial_cmp(&other.price)
            .unwrap_or(Ordering::Equal)
    }
}

/// Min-heap wrapper for tracking diagonal object prices
/// Uses Rust's BinaryHeap with Reverse for min-heap behavior
pub struct PriceHeap {
    heap: BinaryHeap<Reverse<HeapElement>>,
}

impl PriceHeap {
    pub fn new() -> Self {
        PriceHeap {
            heap: BinaryHeap::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        PriceHeap {
            heap: BinaryHeap::with_capacity(capacity),
        }
    }

    pub fn push(&mut self, elem: HeapElement) {
        self.heap.push(Reverse(elem));
    }

    pub fn pop(&mut self) -> Option<HeapElement> {
        self.heap.pop().map(|Reverse(elem)| elem)
    }

    pub fn peek(&self) -> Option<&HeapElement> {
        self.heap.peek().map(|Reverse(elem)| elem)
    }

    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    pub fn len(&self) -> usize {
        self.heap.len()
    }

    pub fn peek_two(&self) -> Option<(HeapElement, HeapElement)> {
        if self.heap.len() < 2 {
            return None;
        }

        let min1 = self.heap.peek().map(|Reverse(e)| *e)?;
        let slice = self.heap.as_slice();
        if slice.len() == 2 {
            let Reverse(min2) = slice[1];
            return Some((min1, min2));
        }

        let Reverse(l_child) = slice[1];
        let Reverse(r_child) = slice[2];

        let min2 = if l_child.price < r_child.price {
            l_child
        } else {
            r_child
        };

        Some((min1, min2))
    }

    // Expensive O(nlog n) but suggested in the paper.
    pub fn update_price(&mut self, point_idx: usize, new_price: f64) {
        let mut elements: Vec<HeapElement> = self.heap.drain().map(|Reverse(e)| e).collect();

        for element in &mut elements {
            if element.index == point_idx {
                element.price = new_price;
            }
        }

        for element in elements {
            self.heap.push(Reverse(element));
        }
    }
}

impl Default for PriceHeap {
    fn default() -> Self {
        Self::new()
    }
}
