//! Postprocessing: the model's raw `[84 × 8400]` output → a short list of boxes.
//!
//! YOLO11 makes one prediction per *anchor* (a grid cell on one of three scales, 8400
//! in total). Each prediction is 84 numbers: the box `cx, cy, w, h` in 640×640 input
//! pixels, then one score per COCO class. Most anchors see nothing, and an object is
//! usually seen by several neighbouring anchors at once. So we:
//!
//! 1. **Decode:** keep each anchor's best class if its score clears the threshold.
//! 2. **Sort** the survivors, best score first.
//! 3. **NMS** (non-maximum suppression): drop a box if a better box of the same class
//!    already covers the same object.
//! 4. **Un-letterbox:** map the kept boxes from input space back to frame pixels.

use super::preprocess::Letterbox;
use super::{BBox, Detection, MAX_DETECTIONS, NUM_ANCHORS, NUM_CLASSES};

/// Box attributes per anchor before the class scores: `cx, cy, w, h`.
const BOX_ATTRS: usize = 4;
/// Values per anchor in the output: 4 box attributes + 80 class scores.
const ATTRS: usize = BOX_ATTRS + NUM_CLASSES;

/// Two boxes of the same class that overlap by more than this IoU are taken to be the
/// same object, and only the higher-scoring one is kept. IoU 0.5 means the shared area
/// is half of the area the two boxes cover together. Ultralytics' `predict` default is
/// 0.7, which keeps more near-duplicates; 0.5 merges them more eagerly, a common
/// choice for a live display.
pub const IOU_THRESHOLD: f32 = 0.5;

/// A decoded prediction that passed the confidence threshold, still in input space.
#[derive(Clone, Copy, Debug)]
struct Candidate {
    bbox: BBox,
    class: u8,
    score: f32,
}

/// Decodes model output into detections. Owns its scratch buffer, so `run` never allocates.
pub struct Postprocessor {
    candidates: Vec<Candidate>,
}

impl Default for Postprocessor {
    fn default() -> Self {
        Self::new()
    }
}

impl Postprocessor {
    pub fn new() -> Self {
        // At most one candidate per anchor, so this capacity is never exceeded.
        Self {
            candidates: Vec::with_capacity(NUM_ANCHORS),
        }
    }

    /// `raw` = the model output, `[84 × 8400]` row-major (attribute-major).
    /// Clears `out`, then fills it, best first, at most MAX_DETECTIONS.
    pub fn run(&mut self, raw: &[f32], conf: f32, lb: &Letterbox, out: &mut Vec<Detection>) {
        assert_eq!(raw.len(), ATTRS * NUM_ANCHORS, "output must be [84 × 8400]");
        self.decode(raw, conf);
        // `total_cmp` gives a total order even for NaN, so a bad value can't panic here.
        self.candidates
            .sort_unstable_by(|a, b| b.score.total_cmp(&a.score));
        out.clear();
        self.suppress_into(out);
        un_letterbox(out, lb);
    }

    /// Step 1: one candidate per anchor whose best class score is at least `conf`.
    fn decode(&mut self, raw: &[f32], conf: f32) {
        self.candidates.clear();
        for anchor in 0..NUM_ANCHORS {
            let (class, score) = best_class(raw, anchor);
            if score >= conf {
                let [cx, cy, w, h] = [0, 1, 2, 3].map(|a| at(raw, a, anchor));
                self.candidates.push(Candidate {
                    bbox: BBox::from_center(cx, cy, w, h),
                    class,
                    score,
                });
            }
        }
    }

    /// Step 3, class-aware greedy NMS over the sorted candidates.
    ///
    /// Walking best-first, a candidate is kept unless an already-kept box **of the same
    /// class** overlaps it by more than [`IOU_THRESHOLD`]. Class-aware because different
    /// objects of different classes overlap all the time (a person holding a cup must keep
    /// both boxes). It also means the class filter can be applied before or after NMS
    /// with the same result, so the UI filters at draw time.
    fn suppress_into(&self, kept: &mut Vec<Detection>) {
        for c in &self.candidates {
            if kept.len() == MAX_DETECTIONS {
                break;
            }
            let duplicate = kept
                .iter()
                .any(|k| k.class == c.class && iou(k.bbox, c.bbox) > IOU_THRESHOLD);
            if !duplicate {
                kept.push(Detection {
                    bbox: c.bbox,
                    class: c.class,
                    score: c.score,
                });
            }
        }
    }
}

/// Value of attribute `attr` for anchor `anchor`.
///
/// The output has shape `[1, 84, 8400]`, so in memory it is 84 rows of 8400: first every
/// anchor's `cx`, then every anchor's `cy`, and so on. One anchor's 84 values are
/// therefore 8400 apart, not next to each other (it's `raw[attr·8400 + anchor]`, never
/// `raw[anchor·84 + attr]`).
fn at(raw: &[f32], attr: usize, anchor: usize) -> f32 {
    raw[attr * NUM_ANCHORS + anchor]
}

/// The highest-scoring class of one anchor. The scores are already sigmoided by the
/// model, and YOLO11 has no separate objectness score, so the class score *is* the
/// confidence.
fn best_class(raw: &[f32], anchor: usize) -> (u8, f32) {
    let mut best = (0, f32::NEG_INFINITY);
    for class in 0..NUM_CLASSES {
        let score = at(raw, BOX_ATTRS + class, anchor);
        if score > best.1 {
            best = (class as u8, score);
        }
    }
    best
}

/// Step 4: maps boxes to frame pixels and drops any that end up empty after clamping
/// (a box predicted entirely inside a letterbox bar).
fn un_letterbox(dets: &mut Vec<Detection>, lb: &Letterbox) {
    for d in dets.iter_mut() {
        d.bbox = lb.to_frame(d.bbox);
    }
    dets.retain(|d| d.bbox.area() > 0.0);
}

/// Intersection over union: overlap area / combined area, from 0 (disjoint) to 1 (equal).
pub fn iou(a: BBox, b: BBox) -> f32 {
    let overlap = BBox {
        x0: a.x0.max(b.x0),
        y0: a.y0.max(b.y0),
        x1: a.x1.min(b.x1),
        y1: a.y1.min(b.y1),
    };
    let inter = overlap.area();
    let union = a.area() + b.area() - inter;
    if union > 0.0 { inter / union } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Writes one anchor's prediction into a synthetic output tensor.
    #[allow(clippy::too_many_arguments)] // mirrors the 4 box values + class + score
    fn put(
        raw: &mut [f32],
        anchor: usize,
        cx: f32,
        cy: f32,
        w: f32,
        h: f32,
        class: usize,
        score: f32,
    ) {
        for (attr, v) in [cx, cy, w, h].into_iter().enumerate() {
            raw[attr * NUM_ANCHORS + anchor] = v;
        }
        raw[(BOX_ATTRS + class) * NUM_ANCHORS + anchor] = score;
    }

    fn run(raw: &[f32], lb: &Letterbox) -> Vec<Detection> {
        let mut out = Vec::new();
        Postprocessor::new().run(raw, 0.25, lb, &mut out);
        out
    }

    fn empty() -> Vec<f32> {
        vec![0.0; ATTRS * NUM_ANCHORS]
    }

    /// 640×640 frame: input space = frame space.
    fn identity() -> Letterbox {
        Letterbox::new(640, 640)
    }

    #[test]
    fn postprocess_below_conf_is_empty() {
        let mut raw = empty();
        put(&mut raw, 7, 100.0, 100.0, 50.0, 50.0, 0, 0.2);
        assert!(run(&raw, &identity()).is_empty());
    }

    #[test]
    fn postprocess_nms_same_class_keeps_best() {
        let mut raw = empty();
        put(&mut raw, 10, 100.0, 100.0, 50.0, 50.0, 0, 0.6);
        put(&mut raw, 11, 102.0, 101.0, 50.0, 50.0, 0, 0.9);
        let out = run(&raw, &identity());
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].score, 0.9);
    }

    #[test]
    fn postprocess_nms_different_classes_both_kept() {
        let mut raw = empty();
        put(&mut raw, 10, 100.0, 100.0, 50.0, 50.0, 0, 0.6); // person
        put(&mut raw, 11, 102.0, 101.0, 50.0, 50.0, 41, 0.9); // cup
        let out = run(&raw, &identity());
        assert_eq!(out.iter().map(|d| d.class).collect::<Vec<_>>(), [41, 0]);
    }

    #[test]
    fn postprocess_maps_to_frame_pixels() {
        let mut raw = empty();
        put(&mut raw, 0, 320.0, 320.0, 30.0, 60.0, 2, 0.8);
        let out = run(&raw, &Letterbox::new(1920, 1080));
        let b = out[0].bbox;
        let near = |a: f32, e: f32| (a - e).abs() < 1e-2;
        assert!(
            near((b.x0 + b.x1) / 2.0, 960.0) && near((b.y0 + b.y1) / 2.0, 540.0),
            "{b:?}"
        );
        assert!(near(b.width(), 90.0) && near(b.height(), 180.0), "{b:?}");
    }

    #[test]
    fn postprocess_box_in_letterbox_bar_is_dropped() {
        let mut raw = empty();
        put(&mut raw, 0, 320.0, 60.0, 40.0, 40.0, 0, 0.8); // inside the top bar (y < 140)
        assert!(run(&raw, &Letterbox::new(1920, 1080)).is_empty());
    }

    #[test]
    fn postprocess_caps_and_sorts() {
        let mut raw = empty();
        // 400 small, non-overlapping boxes on a 20×20 grid with distinct scores.
        for i in 0..400 {
            let (cx, cy) = (16.0 + 32.0 * (i % 20) as f32, 16.0 + 32.0 * (i / 20) as f32);
            put(&mut raw, i, cx, cy, 10.0, 10.0, 0, 0.3 + i as f32 / 1000.0);
        }
        let out = run(&raw, &identity());
        assert_eq!(out.len(), MAX_DETECTIONS);
        assert!(
            out.windows(2).all(|p| p[0].score >= p[1].score),
            "not best-first"
        );
        assert!((out[0].score - 0.699).abs() < 1e-6);
    }

    #[test]
    fn postprocess_iou() {
        let a = BBox::new(0.0, 0.0, 2.0, 2.0);
        let disjoint = BBox::new(5.0, 5.0, 6.0, 6.0);
        let half = BBox::new(1.0, 0.0, 3.0, 2.0);
        assert_eq!(iou(a, a), 1.0);
        assert_eq!(iou(a, disjoint), 0.0);
        assert!((iou(a, half) - 1.0 / 3.0).abs() < 1e-6);
        let point = BBox::new(1.0, 1.0, 1.0, 1.0);
        assert_eq!(iou(point, point), 0.0, "zero union");
    }
}
