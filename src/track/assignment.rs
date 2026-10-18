//! Assignment: pair tracks with detections so the total cost is as small as possible.
//!
//! **The problem.** We have `n` tracks, `m` detections and a cost for every pair (in
//! the tracker, `1 − IoU`: 0 when a track's predicted box and a detection overlap
//! perfectly, 1 when they don't touch). We want to pick pairs so that each track and
//! each detection is used at most once and the **sum** of the chosen costs is the
//! smallest possible.
//!
//! **Why not greedy.** Greedy takes the cheapest pair, then the cheapest of what's
//! left, and so on. With tracks `A, B` and detections `1, 2`:
//!
//! ```text
//!        1     2
//!   A  0.10  0.20
//!   B  0.15  0.90
//! ```
//!
//! greedy takes `A1` (0.10) and is then forced into `B2` (0.90): total 1.00. The best is
//! `A2 + B1` = 0.35. In tracking, greedy's mistake is an **ID switch**: track B's object
//! gets matched to a detection far away, and the IDs of two objects get mixed up.
//!
//! **How the Hungarian method avoids it, intuitively.** Every row and every column
//! carries a *price* (a "potential"). Subtracting a row's price from all of that row's
//! costs (or a column's from that column's) never changes *which* assignment is best,
//! because every complete assignment uses each row and each column exactly once, so
//! every total drops by the same amount. But the right prices make some pairs cost
//! exactly 0 ("tight"), and an assignment made only of tight pairs is optimal: nothing
//! is cheaper than 0. The algorithm adds rows one at a time. For each new row it searches
//! for the cheapest way to fit it in, possibly moving earlier rows to other columns along
//! an **augmenting path** (new row → a column → that column's current owner → another
//! column → … → a free column), raising and lowering prices just enough to make each
//! step of the path tight. That costs O(n²·m), trivial for the ≤ 300 × 300 tables we
//! solve.
//!
//! **Rectangular tables.** More tracks than detections (or the reverse) is normal. The
//! solver below needs rows ≤ cols (every row gets a column). When there are more rows,
//! it solves the *transposed* problem, columns as rows, and maps the answer back.

/// A rows × cols table of costs, reused between calls (no allocation once grown).
pub struct CostMatrix {
    rows: usize,
    cols: usize,
    /// Row-major: the cost of row `r`, column `c` is at `r * cols + c`.
    data: Vec<f32>,
}

impl CostMatrix {
    pub fn new() -> Self {
        Self {
            rows: 0,
            cols: 0,
            data: Vec::new(),
        }
    }

    /// Resizes to `rows × cols` (reusing the capacity) and sets every cost to `fill`.
    pub fn reset(&mut self, rows: usize, cols: usize, fill: f32) {
        self.rows = rows;
        self.cols = cols;
        reset(&mut self.data, rows * cols, fill);
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn get(&self, r: usize, c: usize) -> f32 {
        debug_assert!(r < self.rows && c < self.cols, "({r}, {c}) out of bounds");
        self.data[r * self.cols + c]
    }

    pub fn set(&mut self, r: usize, c: usize, v: f32) {
        debug_assert!(r < self.rows && c < self.cols, "({r}, {c}) out of bounds");
        self.data[r * self.cols + c] = v;
    }
}

/// The Hungarian solver and its scratch buffers.
///
/// The buffers are indexed **from 1**: index 0 of every column buffer is a dummy column
/// that holds the row currently being added, and a `col_owner` of 0 means "no row".
/// That keeps the algorithm free of special cases for its first step.
pub struct Assignment {
    /// Price of each row (`u` in the textbook), index 1..=n.
    row_price: Vec<f64>,
    /// Price of each column (`v`), index 0..=m.
    col_price: Vec<f64>,
    /// The row that owns each column (`p`), 0 = free.
    col_owner: Vec<usize>,
    /// For each column, the column before it on the current augmenting path (`way`).
    prev_col: Vec<usize>,
    /// For each column not yet in the search, the smallest reduced cost to reach it (`minv`).
    min_slack: Vec<f64>,
    /// Columns already reached by the current search (`used`).
    visited: Vec<bool>,
}

impl Assignment {
    pub fn new() -> Self {
        Self {
            row_price: Vec::new(),
            col_price: Vec::new(),
            col_owner: Vec::new(),
            prev_col: Vec::new(),
            min_slack: Vec::new(),
            visited: Vec::new(),
        }
    }

    /// Minimum-total-cost one-to-one assignment. Clears `row_to_col`, then fills it with
    /// one entry per row: `Some(col)` or `None`. Exactly `min(rows, cols)` rows get a column.
    pub fn solve(&mut self, cost: &CostMatrix, row_to_col: &mut Vec<Option<usize>>) {
        reset(row_to_col, cost.rows(), None);
        if cost.rows() == 0 || cost.cols() == 0 {
            return;
        }
        // The method needs rows ≤ cols; otherwise solve the transposed table.
        let transposed = cost.rows() > cost.cols();
        self.solve_wide(cost, transposed);

        // Read the answer off the column owners, undoing the transposition.
        for (col, &owner) in self.col_owner.iter().enumerate().skip(1) {
            if owner == 0 {
                continue;
            }
            let (r, c) = if transposed {
                (col - 1, owner - 1)
            } else {
                (owner - 1, col - 1)
            };
            row_to_col[r] = Some(c);
        }
    }

    /// The Hungarian method (shortest augmenting paths with prices, as in e-maxx and
    /// SciPy's `linear_sum_assignment`) for a table with `n ≤ m`, 1-indexed.
    ///
    /// Computes in `f64`: prices add up many costs, and the tracker uses large
    /// "forbidden" costs (1e4) next to small IoU ones, which `f32` would blur.
    #[allow(clippy::needless_range_loop)] // the method is defined over indices
    fn solve_wide(&mut self, cost: &CostMatrix, transposed: bool) {
        let (n, m) = if transposed {
            (cost.cols(), cost.rows())
        } else {
            (cost.rows(), cost.cols())
        };
        reset(&mut self.row_price, n + 1, 0.0);
        reset(&mut self.col_price, m + 1, 0.0);
        reset(&mut self.col_owner, m + 1, 0);
        reset(&mut self.prev_col, m + 1, 0);
        reset(&mut self.min_slack, m + 1, f64::INFINITY);
        reset(&mut self.visited, m + 1, false);

        for row in 1..=n {
            // Start the search with the new row parked on the dummy column 0.
            self.col_owner[0] = row;
            let mut col = 0;
            self.min_slack.fill(f64::INFINITY);
            self.visited.fill(false);

            // Grow the search one column at a time until it reaches a free column.
            loop {
                self.visited[col] = true;
                let owner = self.col_owner[col];
                let mut delta = f64::INFINITY;
                let mut next_col = 0;

                // From `owner`, update the cheapest known way into every unvisited
                // column, and pick the column that is cheapest to reach overall.
                for j in 1..=m {
                    if self.visited[j] {
                        continue;
                    }
                    let reduced = cost_at(cost, transposed, owner - 1, j - 1)
                        - self.row_price[owner]
                        - self.col_price[j];
                    if reduced < self.min_slack[j] {
                        self.min_slack[j] = reduced;
                        self.prev_col[j] = col;
                    }
                    if self.min_slack[j] < delta {
                        delta = self.min_slack[j];
                        next_col = j;
                    }
                }

                // Shift the prices by `delta`: pairs already in the search stay tight,
                // and the pair into `next_col` becomes tight.
                for j in 0..=m {
                    if self.visited[j] {
                        self.row_price[self.col_owner[j]] += delta;
                        self.col_price[j] -= delta;
                    } else {
                        self.min_slack[j] -= delta;
                    }
                }

                col = next_col;
                if self.col_owner[col] == 0 {
                    break; // a free column: the augmenting path is complete
                }
            }

            // Flip the augmenting path: walking back, every column on it passes to the
            // row that reached it, which seats the new row and frees the dummy column.
            loop {
                let prev = self.prev_col[col];
                self.col_owner[col] = self.col_owner[prev];
                col = prev;
                if col == 0 {
                    break;
                }
            }
        }
    }
}

/// The cost of `(row, col)` of the problem being solved, 0-indexed. When `transposed`,
/// the solver's rows are the matrix's columns, so the indices swap: this reads the
/// table transposed without copying it.
fn cost_at(cost: &CostMatrix, transposed: bool, row: usize, col: usize) -> f64 {
    let c = if transposed {
        cost.get(col, row)
    } else {
        cost.get(row, col)
    };
    f64::from(c)
}

/// Sets `buf` to `len` copies of `value`, reusing its capacity.
fn reset<T: Copy>(buf: &mut Vec<T>, len: usize, value: T) {
    buf.clear();
    buf.resize(len, value);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(rows: &[&[f32]]) -> CostMatrix {
        let mut m = CostMatrix::new();
        m.reset(rows.len(), rows.first().map_or(0, |r| r.len()), 0.0);
        for (r, row) in rows.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                m.set(r, c, v);
            }
        }
        m
    }

    fn solve(cost: &CostMatrix) -> Vec<Option<usize>> {
        let mut out = Vec::new();
        Assignment::new().solve(cost, &mut out);
        out
    }

    fn total(cost: &CostMatrix, assignment: &[Option<usize>]) -> f32 {
        assignment
            .iter()
            .enumerate()
            .filter_map(|(r, c)| c.map(|c| cost.get(r, c)))
            .sum()
    }

    /// Exactly `min(rows, cols)` rows get a column, and no column is used twice.
    fn assert_valid(cost: &CostMatrix, assignment: &[Option<usize>]) {
        assert_eq!(assignment.len(), cost.rows());
        let mut used = vec![false; cost.cols()];
        for &c in assignment.iter().flatten() {
            assert!(!used[c], "column {c} used twice: {assignment:?}");
            used[c] = true;
        }
        let matched = assignment.iter().flatten().count();
        assert_eq!(matched, cost.rows().min(cost.cols()), "{assignment:?}");
    }

    /// The smallest total over every assignment that matches `min(rows, cols)` rows.
    fn brute_force(cost: &CostMatrix) -> f32 {
        fn best_from(cost: &CostMatrix, row: usize, used: &mut [bool], left: usize) -> f32 {
            if left == 0 {
                return 0.0;
            }
            if cost.rows() - row < left {
                return f32::INFINITY; // not enough rows left to match `left` more
            }
            // Either this row stays unmatched…
            let mut best = best_from(cost, row + 1, used, left);
            // …or it takes a free column.
            for c in 0..cost.cols() {
                if !used[c] {
                    used[c] = true;
                    let with_c = cost.get(row, c) + best_from(cost, row + 1, used, left - 1);
                    best = best.min(with_c);
                    used[c] = false;
                }
            }
            best
        }
        let mut used = vec![false; cost.cols()];
        best_from(cost, 0, &mut used, cost.rows().min(cost.cols()))
    }

    /// A tiny linear congruential generator (Knuth's MMIX constants), so the test is
    /// deterministic without a `rand` crate.
    struct Lcg(u64);

    impl Lcg {
        /// A value in `[0, 1)` from the top 24 bits, the best-mixed ones (and exactly
        /// what an `f32` mantissa holds).
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 40) as f32 / (1u64 << 24) as f32
        }
    }

    #[test]
    fn assignment_beats_greedy() {
        let cost = matrix(&[&[0.10, 0.20], &[0.15, 0.90]]);
        let a = solve(&cost);
        assert_eq!(a, vec![Some(1), Some(0)]);
        assert!((total(&cost, &a) - 0.35).abs() < 1e-6);
    }

    #[test]
    fn assignment_rectangular_more_rows() {
        let cost = matrix(&[&[0.9, 0.1], &[0.2, 0.8], &[0.3, 0.4]]);
        let a = solve(&cost);
        assert_valid(&cost, &a);
        assert_eq!(a, vec![Some(1), Some(0), None]);
        assert!((total(&cost, &a) - brute_force(&cost)).abs() < 1e-6);
    }

    #[test]
    fn assignment_rectangular_more_cols() {
        let cost = matrix(&[&[0.5, 0.2, 0.1], &[0.3, 0.1, 0.9]]);
        let a = solve(&cost);
        assert_valid(&cost, &a);
        assert_eq!(a, vec![Some(2), Some(1)]);
        assert!((total(&cost, &a) - brute_force(&cost)).abs() < 1e-6);
    }

    #[test]
    fn assignment_empty() {
        let mut cost = CostMatrix::new();
        cost.reset(0, 3, 0.0);
        assert_eq!(solve(&cost), vec![]);
        cost.reset(3, 0, 0.0);
        assert_eq!(solve(&cost), vec![None, None, None]);
    }

    #[test]
    fn assignment_matches_brute_force() {
        let mut rng = Lcg(42);
        let mut solver = Assignment::new();
        let mut cost = CostMatrix::new();
        let mut out = Vec::new();
        for rows in 1..=6 {
            for cols in 1..=6 {
                for _ in 0..50 {
                    cost.reset(rows, cols, 0.0);
                    for r in 0..rows {
                        for c in 0..cols {
                            cost.set(r, c, rng.next_f32());
                        }
                    }
                    solver.solve(&cost, &mut out);
                    assert_valid(&cost, &out);
                    let (got, want) = (total(&cost, &out), brute_force(&cost));
                    assert!((got - want).abs() < 1e-5, "{rows}×{cols}: {got} vs {want}");
                }
            }
        }
    }

    #[test]
    fn assignment_reuses_buffers() {
        let mut solver = Assignment::new();
        let mut out = Vec::new();

        let big = matrix(&[&[4.0, 1.0, 3.0], &[2.0, 0.0, 5.0], &[3.0, 2.0, 2.0]]);
        solver.solve(&big, &mut out);
        assert_eq!(out, vec![Some(1), Some(0), Some(2)]);

        let small = matrix(&[&[0.10, 0.20], &[0.15, 0.90]]);
        solver.solve(&small, &mut out);
        assert_eq!(out, vec![Some(1), Some(0)]);
    }
}
