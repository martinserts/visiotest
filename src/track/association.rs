//! Association: which detection belongs to which track, judged by box overlap.
//!
//! **The question.** After the Kalman filter has moved every track to the new frame's
//! time, each track has a *predicted* box: where its object should be now. The detector
//! gives a set of *detected* boxes. A detection that sits right on top of a prediction
//! is very likely the same object; one on the other side of the frame is not.
//!
//! **IoU, intersection over union.** The overlap of two boxes, divided by the area they
//! cover together. 1 means identical, 0 means they don't touch. Two 10 × 10 boxes, the
//! second shifted 5 px to the right:
//!
//! ```text
//!   ┌─────┬─────┬─────┐
//!   │  A  │ A∩B │  B  │    A∩B = 5 × 10 = 50
//!   │     │     │     │    A∪B = 100 + 100 − 50 = 150
//!   └─────┴─────┴─────┘    IoU = 50 / 150 = ⅓
//!   0     5     10    15
//! ```
//!
//! IoU doesn't care about the boxes' absolute size: a 5 px shift of a 10 px box (⅓)
//! is a much worse fit than a 5 px shift of a 400 px box (≈ 0.98), which is what we
//! want, because a big near object moves more pixels per frame than a small far one.
//!
//! **Cost = 1 − IoU.** The assignment solver ([`super::assignment`]) *minimises* a total
//! cost, so we turn "more overlap is better" into "less cost is better": a perfect fit
//! costs 0, two boxes that don't touch cost 1.
//!
//! **The gate.** The solver gives every row a column if it can, so on its own it would
//! happily pair a track on the left edge with a detection on the right edge just because
//! nothing else was left. That pair would teleport the track and steal another object's
//! identity. So a pair whose IoU is below `min_iou` gets the cost [`FORBIDDEN`], and
//! any such pair the solver still picks is thrown away afterwards: the track stays
//! unmatched, and the detection is free for a later round or a new track.
//!
//! **Later: appearance.** IoU only knows *where* things are. When two people cross,
//! their boxes overlap and IoU can't tell them apart. A ReID model would give every box
//! an appearance vector, and the cost would become
//! `(1 − IoU) + λ · (1 − cosine similarity)`. That change would live entirely in this
//! file: the tracker only sees the resulting [`Match`]es, so nothing else would change.

use super::assignment::{Assignment, CostMatrix};
use crate::detect::BBox;
use crate::detect::postprocess::iou;

/// Cost of a pair the IoU gate rules out. Not `f32::INFINITY`: the Hungarian adds and
/// subtracts costs, and ∞ − ∞ is NaN. 1e4 is more than any total of real costs
/// (each ≤ 1, at most 300 pairs), so the solver only uses a forbidden pair when it has
/// no other way to fill a row, and we throw those pairs away.
pub const FORBIDDEN: f32 = 1.0e4;

/// One accepted pair: `tracks[track]` and `dets[det]` are the same object.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Match {
    /// Index into the `tracks` slice given to [`Matcher::match_by_iou`].
    pub track: usize,
    /// Index into the `dets` slice.
    pub det: usize,
    /// Overlap of the predicted and the detected box, ≥ the gate.
    pub iou: f32,
}

/// IoU matching with reusable buffers.
pub struct Matcher {
    cost: CostMatrix,
    solver: Assignment,
    /// The solver's answer: the column (detection) of each row (track), if any.
    row_to_col: Vec<Option<usize>>,
}

impl Matcher {
    pub fn new() -> Self {
        Self {
            cost: CostMatrix::new(),
            solver: Assignment::new(),
            row_to_col: Vec::new(),
        }
    }

    /// Best one-to-one pairs between `tracks` (predicted boxes) and `dets`, keeping only
    /// pairs with IoU ≥ `min_iou`. Indices are into the two slices. Clears `out` first.
    pub fn match_by_iou(
        &mut self,
        tracks: &[BBox],
        dets: &[BBox],
        min_iou: f32,
        out: &mut Vec<Match>,
    ) {
        out.clear();
        self.fill_costs(tracks, dets, min_iou);
        self.solver.solve(&self.cost, &mut self.row_to_col);

        for (track, col) in self.row_to_col.iter().enumerate() {
            let Some(det) = *col else { continue };
            // The solver may have used a forbidden pair to fill a row: drop it.
            let overlap = iou(tracks[track], dets[det]);
            if overlap >= min_iou {
                out.push(Match {
                    track,
                    det,
                    iou: overlap,
                });
            }
        }
    }

    /// Fills the cost table: `1 − IoU` for pairs that pass the gate, else [`FORBIDDEN`].
    fn fill_costs(&mut self, tracks: &[BBox], dets: &[BBox], min_iou: f32) {
        self.cost.reset(tracks.len(), dets.len(), FORBIDDEN);
        for (r, &track) in tracks.iter().enumerate() {
            for (c, &det) in dets.iter().enumerate() {
                let overlap = iou(track, det);
                if overlap >= min_iou {
                    self.cost.set(r, c, 1.0 - overlap);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A box spanning `x0..x0 + 100` horizontally and `0..100` vertically, so the IoU
    /// of two of them only depends on their horizontal shift.
    fn strip(x0: f32) -> BBox {
        BBox::new(x0, 0.0, x0 + 100.0, 100.0)
    }

    fn run(tracks: &[BBox], dets: &[BBox], min_iou: f32) -> Vec<Match> {
        let mut out = Vec::new();
        Matcher::new().match_by_iou(tracks, dets, min_iou, &mut out);
        out
    }

    #[test]
    fn association_pairs_overlapping_boxes() {
        let tracks = [
            BBox::new(0.0, 0.0, 10.0, 10.0),
            BBox::new(100.0, 0.0, 110.0, 10.0),
        ];
        // Swapped order: detection 0 sits on track 1, detection 1 next to track 0.
        let dets = [
            BBox::new(100.0, 0.0, 110.0, 10.0),
            BBox::new(5.0, 0.0, 15.0, 10.0),
        ];
        let matches = run(&tracks, &dets, 0.2);
        assert_eq!(matches.len(), 2);
        assert_eq!((matches[0].track, matches[0].det), (0, 1));
        assert_eq!((matches[1].track, matches[1].det), (1, 0));
        // The module doc's example: a 10 px box shifted by 5 px.
        assert!((matches[0].iou - 1.0 / 3.0).abs() < 1e-6);
        assert!((matches[1].iou - 1.0).abs() < 1e-6);
    }

    #[test]
    fn association_gate_rejects_far_pairs() {
        // The only track and the only detection, but nowhere near each other.
        let matches = run(&[strip(0.0)], &[strip(500.0)], 0.2);
        assert!(matches.is_empty());
    }

    #[test]
    fn association_empty_inputs() {
        assert!(run(&[], &[], 0.2).is_empty());
        assert!(run(&[strip(0.0)], &[], 0.2).is_empty());
        assert!(run(&[], &[strip(0.0)], 0.2).is_empty());
    }

    #[test]
    fn association_prefers_global_best() {
        // Track A overlaps detection 1 best of all (IoU 0.905), so greedy would take
        // A–1 and leave B–2 (0.379): total cost 0.095 + 0.621 = 0.716. A–2 + B–1 is
        // 0.667 + 0.667 IoU, cost 0.333 + 0.333 = 0.667, cheaper overall.
        // (Exact 0.9 / 0.8 / 0.85 / 0.1 IoUs can't be built from boxes: 1 − IoU is a
        // distance, so the four can't be that far apart.)
        let (track_a, track_b) = (strip(25.0), strip(0.0));
        let (det_1, det_2) = (strip(20.0), strip(45.0));
        assert!(iou(track_a, det_1) > iou(track_a, det_2));
        assert!(iou(track_a, det_1) > iou(track_b, det_1));

        let matches = run(&[track_a, track_b], &[det_1, det_2], 0.2);
        assert_eq!(matches.len(), 2);
        assert_eq!((matches[0].track, matches[0].det), (0, 1));
        assert_eq!((matches[1].track, matches[1].det), (1, 0));
        for m in &matches {
            assert!((m.iou - 2.0 / 3.0).abs() < 1e-5, "iou {}", m.iou);
        }
    }
}
