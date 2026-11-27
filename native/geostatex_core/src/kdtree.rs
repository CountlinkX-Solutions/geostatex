//! A k-d tree for the nearest points of a moving window: the same neighbours, in the same order,
//! as computing the distance to every point and sorting by (distance, index).

/// A point in up to three dimensions (unused coordinates are 0).
pub type Point = [f64; 3];

/// The Euclidean distance, computed as kriging computes it (`(a - b)` per coordinate, squares
/// summed in coordinate order).
pub fn distance(a: Point, b: Point) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Points per leaf: below this a node compares its points directly.
const LEAF: usize = 12;

enum Node {
    Leaf {
        lo: usize,
        hi: usize,
    },
    /// Points `[lo, mid)` of the order lie at or below `at` along `axis`, the rest at or above.
    Split {
        axis: usize,
        at: f64,
        left: usize,
        right: usize,
    },
}

/// A candidate neighbour, ordered by distance and then index (a total order: no two candidates
/// are equal).
struct Candidate {
    d: f64,
    i: usize,
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.d.total_cmp(&other.d).then(self.i.cmp(&other.i))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl Eq for Candidate {}

/// A static k-d tree over a set of points.
pub struct KdTree {
    points: Vec<Point>,
    order: Vec<usize>,
    nodes: Vec<Node>,
    root: usize,
}

impl KdTree {
    /// The tree over `points`; `None` when a coordinate is not finite (the caller then compares
    /// every point).
    pub fn new(points: Vec<Point>) -> Option<Self> {
        if points.iter().any(|p| p.iter().any(|c| !c.is_finite())) {
            return None;
        }
        let mut order: Vec<usize> = (0..points.len()).collect();
        let mut nodes = Vec::new();
        let root = build(&points, &mut order, 0, points.len(), &mut nodes);
        Some(KdTree {
            points,
            order,
            nodes,
            root,
        })
    }

    /// The `k` points nearest to `p` as `(distance, index)`, ordered by distance and then index:
    /// the first `k` of every point sorted that way.
    pub fn nearest(&self, p: Point, k: usize) -> Vec<(f64, usize)> {
        if k == 0 || self.points.is_empty() {
            return Vec::new();
        }
        let mut heap = std::collections::BinaryHeap::with_capacity(k + 1);
        self.search(self.root, p, k, &mut heap);
        let mut out: Vec<(f64, usize)> = heap.into_iter().map(|c| (c.d, c.i)).collect();
        out.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        out
    }

    fn search(
        &self,
        node: usize,
        p: Point,
        k: usize,
        heap: &mut std::collections::BinaryHeap<Candidate>,
    ) {
        match self.nodes[node] {
            Node::Leaf { lo, hi } => {
                for &i in &self.order[lo..hi] {
                    let c = Candidate {
                        d: distance(p, self.points[i]),
                        i,
                    };
                    if heap.len() < k {
                        heap.push(c);
                    } else if heap.peek().is_some_and(|worst| c < *worst) {
                        heap.pop();
                        heap.push(c);
                    }
                }
            }
            Node::Split {
                axis,
                at,
                left,
                right,
            } => {
                let diff = p[axis] - at;
                let (near, far) = if diff < 0.0 {
                    (left, right)
                } else {
                    (right, left)
                };
                self.search(near, p, k, heap);
                // Every point beyond the split differs from p by at least |diff| along the axis
                // (rounded subtraction is monotone), and its computed distance is then at least
                // |diff| (1 - 2 eps): the squares and sums only grow. The far side is skipped only
                // when that bound exceeds the k-th distance, so no tie is ever skipped; a gap
                // whose square could underflow is never trusted.
                let gap = diff.abs();
                let skip = heap.len() == k
                    && gap > 1e-150
                    && heap
                        .peek()
                        .is_some_and(|worst| gap * (1.0 - 4.0 * f64::EPSILON) > worst.d);
                if !skip {
                    self.search(far, p, k, heap);
                }
            }
        }
    }
}

/// Builds the subtree over `order[lo..hi]` and returns its node index: a leaf for a few points,
/// else a split at the median of the axis with the largest spread.
fn build(
    points: &[Point],
    order: &mut [usize],
    lo: usize,
    hi: usize,
    nodes: &mut Vec<Node>,
) -> usize {
    if hi - lo <= LEAF {
        nodes.push(Node::Leaf { lo, hi });
        return nodes.len() - 1;
    }
    let spread = |a: usize| {
        let (mn, mx) = order[lo..hi]
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), &i| {
                (mn.min(points[i][a]), mx.max(points[i][a]))
            });
        mx - mn
    };
    // The first axis with the largest spread.
    let axis = (1..3).fold(0, |best, a| if spread(a) > spread(best) { a } else { best });
    let mid = lo + (hi - lo) / 2;
    order[lo..hi].select_nth_unstable_by(mid - lo, |&a, &b| {
        points[a][axis].total_cmp(&points[b][axis])
    });
    let at = points[order[mid]][axis];
    let left = build(points, order, lo, mid, nodes);
    let right = build(points, order, mid, hi, nodes);
    nodes.push(Node::Split {
        axis,
        at,
        left,
        right,
    });
    nodes.len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform(state: &mut u64) -> f64 {
        *state ^= *state >> 12;
        *state ^= *state << 25;
        *state ^= *state >> 27;
        (state.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11) as f64 / (1u64 << 53) as f64
    }

    fn brute(points: &[Point], p: Point, k: usize) -> Vec<(u64, usize)> {
        let mut all: Vec<(f64, usize)> = points
            .iter()
            .enumerate()
            .map(|(i, &q)| (distance(p, q), i))
            .collect();
        all.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        all.into_iter()
            .take(k)
            .map(|(d, i)| (d.to_bits(), i))
            .collect()
    }

    #[test]
    fn nearest_matches_sorting_every_point() {
        let mut s = 0x9e37_79b9_7f4a_7c15;
        let scattered: Vec<Point> = (0..500)
            .map(|_| {
                [
                    uniform(&mut s) * 100.0,
                    uniform(&mut s) * 100.0,
                    uniform(&mut s),
                ]
            })
            .collect();
        // A grid with duplicates: many exact ties in distance.
        let mut grid: Vec<Point> = (0..15)
            .flat_map(|i| (0..15).map(move |j| [i as f64, j as f64, 0.0]))
            .collect();
        grid.extend_from_within(..20);
        let line: Vec<Point> = (0..60).map(|i| [i as f64 * 0.5, 0.0, 0.0]).collect();
        for points in [scattered, grid, line] {
            let tree = KdTree::new(points.clone()).unwrap();
            let n = points.len();
            let mut targets: Vec<Point> = (0..100)
                .map(|_| {
                    [
                        uniform(&mut s) * 120.0 - 10.0,
                        uniform(&mut s) * 120.0 - 10.0,
                        0.0,
                    ]
                })
                .collect();
            targets.extend(points.iter().take(10).copied());
            targets.push([7.0, 7.0, 0.0]);
            targets.push([7.5, 7.5, 0.0]);
            for p in targets {
                for k in [0, 1, 4, 13, n, n + 3] {
                    let got: Vec<(u64, usize)> = tree
                        .nearest(p, k)
                        .into_iter()
                        .map(|(d, i)| (d.to_bits(), i))
                        .collect();
                    assert_eq!(got, brute(&points, p, k), "target {p:?}, k {k}");
                }
            }
        }
    }

    #[test]
    fn a_non_finite_coordinate_gives_no_tree() {
        assert!(KdTree::new(vec![[0.0, 0.0, 0.0], [f64::NAN, 1.0, 0.0]]).is_none());
        assert!(KdTree::new(vec![[0.0, f64::INFINITY, 0.0]]).is_none());
        assert!(KdTree::new(Vec::new()).is_some());
    }
}
