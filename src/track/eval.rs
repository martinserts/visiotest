//! Evaluation: how good is a tracker, in numbers? The CLEAR-MOT basics.
//!
//! To score a tracker we need **ground truth (GT)**: the true objects in every frame,
//! each with its true identity ("walker 1", "walker 2"). A synthetic scene
//! ([`super::sim`]) knows these exactly; a real video would need them labelled by hand.
//!
//! **Each frame**, the GT boxes are matched one-to-one to the boxes the tracker shows
//! (its *Tracked* views: Lost tracks are hidden guesses, never drawn, so they don't
//! count). A pair needs IoU ≥ 0.5, the standard MOT threshold, and among the pairs that
//! pass, the solver picks the set with the smallest total cost `1 − IoU`, using the
//! same IoU matcher the tracker uses ([`super::association::Matcher`]). Then every
//! frame adds to four counts:
//!
//! - **Match**: a GT object paired with a shown track.
//! - **FN, false negative (miss)**: a GT object with no track on it.
//! - **FP, false positive**: a shown track with no GT object under it.
//! - **IDSW, ID switch**: a GT object matched to a *different* track id than the last
//!   time it was matched. That is the number that matters most: keeping each
//!   object's ID is the tracker's whole job.
//!
//! **A timeline example.** Walker 1 crosses the frame and hides behind a wall:
//!
//! ```text
//! frame:       1    2    3    4    5    6    7
//! walker 1:   #3   #3    –    –   #5   #5   #5     (– = no track on it)
//! counts:      ✓    ✓   FN   FN  IDSW   ✓    ✓
//! ```
//!
//! Frames 3–4 are two misses. Frame 5 is a match *and* an ID switch, because walker 1
//! was last `#3` and is now `#5`. A gap alone is not a switch: had it come back as
//! `#3`, frame 5 would be a plain match.
//!
//! **MOTA** (multi-object tracking accuracy) folds the errors into one number:
//!
//! ```text
//! MOTA = 1 − (FN + FP + IDSW) / total GT
//! ```
//!
//! In the example, 7 GT boxes and 3 errors: MOTA = 1 − 3/7 ≈ 0.57. 1 is perfect, and
//! it can go below 0 when the tracker shows many boxes on nothing.
//!
//! Two more numbers:
//! - **IDs per object**: how many distinct track ids each GT object was ever matched
//!   to, averaged over the objects. 1 is perfect; walker 1 above has 2.
//! - **Mean IoU** of the matched pairs: how tightly the shown boxes fit.

use std::collections::HashSet;

use super::TrackId;
use super::association::{Match, Matcher};
use crate::detect::BBox;

/// The standard MOT gate: below half overlap, a track isn't "on" the object.
pub const MIN_MATCH_IOU: f32 = 0.5;

/// One true object in one frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Truth {
    /// The object's true identity, `0..n` within a scene.
    pub object: u32,
    pub bbox: BBox,
}

/// The counts summed over every scored frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MotScore {
    /// Frames scored.
    pub frames: u32,
    /// GT boxes over all frames.
    pub truths: u32,
    /// GT boxes matched to a shown track.
    pub matches: u32,
    /// GT boxes with no track on them (FN).
    pub misses: u32,
    /// Shown tracks with no GT under them (FP).
    pub false_positives: u32,
    /// Matches whose track id differs from the object's previous match (IDSW).
    pub id_switches: u32,
    /// Sum of the matched pairs' IoU, for [`MotScore::mean_iou`].
    pub iou_sum: f32,
}

impl MotScore {
    /// `1 − (FN + FP + IDSW) / total GT`. With no GT at all, every shown box is an
    /// error against a total of 1, so an empty scene with no boxes scores 1.
    pub fn mota(&self) -> f32 {
        let errors = self.misses + self.false_positives + self.id_switches;
        1.0 - errors as f32 / self.truths.max(1) as f32
    }

    /// Mean IoU of the matched pairs, `None` before the first match.
    pub fn mean_iou(&self) -> Option<f32> {
        (self.matches > 0).then(|| self.iou_sum / self.matches as f32)
    }
}

/// Scores a tracker frame by frame against ground truth.
pub struct MotEvaluator {
    score: MotScore,
    /// Per object (indexed by `Truth::object`): the track id of its latest match.
    last_id: Vec<Option<TrackId>>,
    /// Per object: every track id it was ever matched to.
    ids_seen: Vec<HashSet<TrackId>>,
    matcher: Matcher,
    /// Scratch for one frame: the GT boxes, the shown boxes and their matches.
    truth_boxes: Vec<BBox>,
    shown_boxes: Vec<BBox>,
    matches: Vec<Match>,
}

impl MotEvaluator {
    pub fn new() -> Self {
        Self {
            score: MotScore::default(),
            last_id: Vec::new(),
            ids_seen: Vec::new(),
            matcher: Matcher::new(),
            truth_boxes: Vec::new(),
            shown_boxes: Vec::new(),
            matches: Vec::new(),
        }
    }

    /// Scores one frame: `truths` vs the boxes the user would see (`(id, bbox)`).
    pub fn frame(&mut self, truths: &[Truth], shown: &[(TrackId, BBox)]) {
        self.truth_boxes.clear();
        self.truth_boxes.extend(truths.iter().map(|t| t.bbox));
        self.shown_boxes.clear();
        self.shown_boxes.extend(shown.iter().map(|&(_, b)| b));
        // The matcher's "tracks" are the GT rows and its "detections" the shown
        // columns; it only sees boxes, so the names don't matter here.
        self.matcher.match_by_iou(
            &self.truth_boxes,
            &self.shown_boxes,
            MIN_MATCH_IOU,
            &mut self.matches,
        );

        // Taken out for the loop, because `record_match` needs `&mut self`; put back
        // after, so its capacity is reused next frame.
        let matches = std::mem::take(&mut self.matches);
        for m in &matches {
            let (id, _) = shown[m.det];
            self.record_match(truths[m.track].object, id, m.iou);
        }
        let matched = matches.len() as u32;
        self.matches = matches;
        self.score.frames += 1;
        self.score.truths += truths.len() as u32;
        self.score.misses += truths.len() as u32 - matched;
        self.score.false_positives += shown.len() as u32 - matched;
    }

    pub fn score(&self) -> MotScore {
        self.score
    }

    /// Mean distinct track IDs per object that was matched at least once, or 0 when no
    /// object ever was.
    pub fn ids_per_object(&self) -> f32 {
        let matched: Vec<usize> = self
            .ids_seen
            .iter()
            .map(HashSet::len)
            .filter(|&n| n > 0)
            .collect();
        if matched.is_empty() {
            return 0.0;
        }
        matched.iter().sum::<usize>() as f32 / matched.len() as f32
    }

    /// Counts one GT ↔ track pair, and an ID switch if the object had another id before.
    fn record_match(&mut self, object: u32, id: TrackId, iou: f32) {
        let o = object as usize;
        // Scenes number their objects 0..n, so a Vec indexed by object is enough; grow
        // it the first time a higher number shows up.
        if o >= self.last_id.len() {
            self.last_id.resize(o + 1, None);
            self.ids_seen.resize_with(o + 1, HashSet::new);
        }
        if self.last_id[o].is_some_and(|last| last != id) {
            self.score.id_switches += 1;
        }
        self.last_id[o] = Some(id);
        self.ids_seen[o].insert(id);
        self.score.matches += 1;
        self.score.iou_sum += iou;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(x0: f32) -> BBox {
        BBox::new(x0, 0.0, x0 + 100.0, 100.0)
    }

    fn truth(object: u32, x0: f32) -> Truth {
        Truth {
            object,
            bbox: square(x0),
        }
    }

    #[test]
    fn eval_perfect_tracking_scores_one() {
        let mut eval = MotEvaluator::new();
        for i in 0..10 {
            let x = 10.0 * i as f32;
            eval.frame(
                &[truth(0, x), truth(1, x + 500.0)],
                &[(1, square(x)), (2, square(x + 500.0))],
            );
        }
        let s = eval.score();
        assert_eq!((s.frames, s.truths, s.matches), (10, 20, 20));
        assert_eq!((s.misses, s.false_positives, s.id_switches), (0, 0, 0));
        assert_eq!(s.mota(), 1.0);
        assert_eq!(s.mean_iou(), Some(1.0));
        assert_eq!(eval.ids_per_object(), 1.0);
    }

    #[test]
    fn eval_counts_an_id_switch() {
        let mut eval = MotEvaluator::new();
        for i in 0..10 {
            let id = if i < 5 { 1 } else { 2 };
            eval.frame(&[truth(0, 0.0)], &[(id, square(0.0))]);
        }
        let s = eval.score();
        assert_eq!(s.id_switches, 1);
        assert_eq!(eval.ids_per_object(), 2.0);
        assert!((s.mota() - 0.9).abs() < 1e-6, "1 error in 10 GT");
    }

    #[test]
    fn eval_gap_then_same_id_is_not_a_switch() {
        let mut eval = MotEvaluator::new();
        eval.frame(&[truth(0, 0.0)], &[(1, square(0.0))]);
        for _ in 0..3 {
            eval.frame(&[truth(0, 0.0)], &[]);
        }
        eval.frame(&[truth(0, 0.0)], &[(1, square(0.0))]);
        let s = eval.score();
        assert_eq!((s.misses, s.id_switches), (3, 0));
        assert_eq!(eval.ids_per_object(), 1.0);
    }

    #[test]
    fn eval_false_positive_and_miss() {
        let mut eval = MotEvaluator::new();
        // Object 0 is tracked, object 1 isn't, and track 9 sits on nothing.
        eval.frame(
            &[truth(0, 0.0), truth(1, 300.0)],
            &[(1, square(0.0)), (9, square(800.0))],
        );
        let s = eval.score();
        assert_eq!((s.matches, s.misses, s.false_positives), (1, 1, 1));
        assert_eq!(s.mota(), 0.0, "2 errors in 2 GT");
    }

    #[test]
    fn eval_iou_gate() {
        // Two 100 px boxes shifted by d overlap (100 − d) / (100 + d); d = 300/7 gives 0.4.
        let shift = 300.0 / 7.0;
        assert!((crate::detect::postprocess::iou(square(0.0), square(shift)) - 0.4).abs() < 1e-4);
        let mut eval = MotEvaluator::new();
        eval.frame(&[truth(0, 0.0)], &[(1, square(shift))]);
        let s = eval.score();
        assert_eq!((s.matches, s.misses, s.false_positives), (0, 1, 1));
        assert_eq!(s.mean_iou(), None);
        assert_eq!(eval.ids_per_object(), 0.0);
    }
}
