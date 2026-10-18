//! Synthetic scenes: objects with known identities, played through the real tracker.
//!
//! **Why synthetic.** Counting ID switches needs to know who is who in every frame.
//! On a real video somebody would have to label that by hand (and there is no test
//! video yet). Here the scene *is* the truth: walker 1 is walker 1 by construction.
//! Every run is identical (a fixed seed per scene, see [`Rng`]), so a number that
//! changes means the code changed. And each scene isolates **one** difficulty, a pole,
//! a wall, a crossing, so a failure points at its cause.
//!
//! **What the "detector" sees.** A real detector boxes what it can see, and is less
//! sure the less it sees. For a walker partly behind an [`Occluder`]:
//!
//! ```text
//! visible = 1 − hidden width / box width   (hidden: behind an occluder or off-frame)
//!
//!   visible ≥ 0.6          a box of the visible part, score 0.8 ± 0.05
//!   0.25 ≤ visible < 0.6   a box of the visible part, score 0.2 ± 0.05  (ByteTrack's low round)
//!   visible < 0.25         nothing
//! ```
//!
//! Example: a 100 px wide walker with 50 px behind a pole is half visible, so it gets
//! a 50 px wide box of the uncovered half with a score near 0.2. The box's centre moves
//! as the walker slides behind the pole: that is what real detectors do, and what the
//! tracker has to cope with. Every box is jittered by a little Gaussian noise, and a
//! scene can add the odd false positive, a confident box on nothing.
//!
//! **Ground truth** is the same visible-part box, for every object at least 0.25
//! visible: what a viewer (and a detector) could point at. A fully hidden walker is not
//! in the truth, so a box drawn on it counts as a false positive and no box isn't a miss.
//!
//! **The runner** ([`run`]) plays a scene at 30 fps through a fresh [`Tracker`], paced
//! by the real [`IntervalController`], exactly like the detector thread, and scores
//! *every* frame with [`MotEvaluator`], including the frames the detector skipped: there
//! the user sees predicted boxes, and those are what gets scored.
//!
//! **What it leaves out:** real detector errors (boxes that swap between two people,
//! missed detections in plain view, class confusion), lighting, motion blur, people
//! occluding each other and the camera moving. That is why the live check with a real
//! camera still matters.

mod rng;
mod scenes;
#[cfg(test)]
mod tests;

use std::time::{Duration, Instant};

use super::eval::{MotEvaluator, MotScore, Truth};
use super::tracker::{LOW_CONF, TrackState, Tracker, TrackerConfig};
use crate::detect::interval::{DetectInterval, IntervalController};
use crate::detect::{BBox, Detection};

pub use rng::Rng;
pub use scenes::scenes;

/// The camera's frame size (the FaceTime HD camera's 1080p).
pub const FRAME_W: f32 = 1920.0;
pub const FRAME_H: f32 = 1080.0;
/// The camera's frame rate.
pub const FPS: f64 = 30.0;
/// Corner jitter of a detected box, as a fraction of its size (3 px on a 100 px box).
pub const DEFAULT_BOX_NOISE: f32 = 0.03;

/// Below this visible fraction the detector sees nothing (and it isn't ground truth).
const MIN_VISIBLE: f32 = 0.25;
/// From this visible fraction on, the detector is confident.
const HIGH_VISIBLE: f32 = 0.6;
/// Score of a mostly visible object: YOLO11n gives a clear person ~0.6–0.9.
const HIGH_SCORE: f32 = 0.8;
/// Score of a half-hidden object: below the 0.35 slider default, above `LOW_CONF`.
const LOW_SCORE: f32 = 0.2;
/// Scores are spread uniformly ± this around `HIGH_SCORE` / `LOW_SCORE`.
const SCORE_SPREAD: f32 = 0.05;

/// One object moving in a straight line at constant speed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Walker {
    /// True identity, `0..n` within a scene.
    pub object: u32,
    /// Box centre at scene time 0 (it may be off-frame then), px.
    pub start: (f32, f32),
    /// px/s.
    pub velocity: (f32, f32),
    /// Full box width and height, px.
    pub size: (f32, f32),
    /// The object exists only from `enter_s` to `leave_s` (scene seconds).
    pub enter_s: f32,
    pub leave_s: f32,
    /// COCO class id.
    pub class: u8,
}

impl Walker {
    /// The full box at scene time `s`, or `None` outside its time window.
    pub fn full_box(&self, s: f32) -> Option<BBox> {
        if s < self.enter_s || s >= self.leave_s {
            return None;
        }
        let cx = self.start.0 + self.velocity.0 * s;
        let cy = self.start.1 + self.velocity.1 * s;
        Some(BBox::from_center(cx, cy, self.size.0, self.size.1))
    }
}

/// A vertical band `x0..x1` (a pole, a wall) that hides whatever is behind it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Occluder {
    pub x0: f32,
    pub x1: f32,
}

/// One synthetic scene.
#[derive(Clone, Debug, PartialEq)]
pub struct Scene {
    pub name: &'static str,
    pub duration_s: f32,
    pub walkers: Vec<Walker>,
    pub occluders: Vec<Occluder>,
    /// Seed of the detector noise, so every run of the scene is identical.
    pub seed: u64,
    /// Std of the corner jitter as a fraction of the box size.
    pub box_noise: f32,
    /// Expected confident boxes on nothing, per second of 30 fps detection.
    pub false_positives_per_s: f32,
}

/// What can be seen of an object.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sighting {
    /// The box around the visible part (its height clipped to the frame; the width that shows).
    pub bbox: BBox,
    /// The visible fraction of the full box's width, `0..=1`.
    pub visible: f32,
}

/// Which part of `full` is visible inside the frame and outside every occluder.
/// `None` when nothing shows.
pub fn sighting(full: BBox, occluders: &[Occluder]) -> Option<Sighting> {
    let x0 = full.x0.max(0.0);
    let x1 = full.x1.min(FRAME_W);
    if x1 <= x0 {
        return None;
    }
    let pieces = visible_pieces(x0, x1, occluders);
    let shown: f32 = pieces.iter().map(|&(a, b)| b - a).sum();
    if shown <= 0.0 {
        return None;
    }
    // A thin pole in front of a wide person leaves two pieces; a detector would still
    // box the whole person, so the box spans from the leftmost to the rightmost piece.
    let left = pieces.iter().map(|&(a, _)| a).fold(f32::INFINITY, f32::min);
    let right = pieces
        .iter()
        .map(|&(_, b)| b)
        .fold(f32::NEG_INFINITY, f32::max);
    Some(Sighting {
        bbox: BBox::new(left, full.y0.max(0.0), right, full.y1.min(FRAME_H)),
        visible: shown / full.width(),
    })
}

/// The parts of `x0..x1` that no occluder covers, as `(start, end)` pairs.
fn visible_pieces(x0: f32, x1: f32, occluders: &[Occluder]) -> Vec<(f32, f32)> {
    let mut pieces = vec![(x0, x1)];
    for occ in occluders {
        let mut rest = Vec::with_capacity(pieces.len() + 1);
        for (a, b) in pieces {
            // What is left of `a..b` on each side of the occluder (maybe nothing).
            if a < occ.x0 {
                rest.push((a, b.min(occ.x0)));
            }
            if b > occ.x1 {
                rest.push((a.max(occ.x1), b));
            }
        }
        pieces = rest;
    }
    pieces
}

/// What the detector reports for one sighting: a jittered box and a score that falls
/// with visibility, or nothing when too little shows.
pub fn detect(seen: Sighting, class: u8, box_noise: f32, rng: &mut Rng) -> Option<Detection> {
    let base = if seen.visible >= HIGH_VISIBLE {
        HIGH_SCORE
    } else if seen.visible >= MIN_VISIBLE {
        LOW_SCORE
    } else {
        return None;
    };
    let score = base + SCORE_SPREAD * (2.0 * rng.next_f32() - 1.0);
    let b = seen.bbox;
    let (jx, jy) = (box_noise * b.width(), box_noise * b.height());
    let bbox = BBox::new(
        b.x0 + jx * rng.normal(),
        b.y0 + jy * rng.normal(),
        b.x1 + jx * rng.normal(),
        b.y1 + jy * rng.normal(),
    );
    Some(Detection { bbox, class, score })
}

impl Scene {
    /// Camera frames in the scene.
    pub fn frames(&self) -> u32 {
        (self.duration_s as f64 * FPS).round() as u32
    }

    /// Clears `out`, then pushes the ground truth at scene time `s`.
    pub fn truths(&self, s: f32, out: &mut Vec<Truth>) {
        out.clear();
        for w in &self.walkers {
            let Some(seen) = w.full_box(s).and_then(|b| sighting(b, &self.occluders)) else {
                continue;
            };
            if seen.visible >= MIN_VISIBLE {
                out.push(Truth {
                    object: w.object,
                    bbox: seen.bbox,
                });
            }
        }
    }

    /// Clears `out`, then pushes what one detector run sees at scene time `s`.
    pub fn detections(&self, s: f32, rng: &mut Rng, out: &mut Vec<Detection>) {
        out.clear();
        for w in &self.walkers {
            let Some(seen) = w.full_box(s).and_then(|b| sighting(b, &self.occluders)) else {
                continue;
            };
            if let Some(det) = detect(seen, w.class, self.box_noise, rng) {
                out.push(det);
            }
        }
        // A false positive is a property of one inference, so it's drawn per run:
        // at 30 runs/s this gives `false_positives_per_s` on average.
        let chance = self.false_positives_per_s / FPS as f32;
        if rng.next_f32() < chance {
            out.push(ghost(rng));
        }
        // The worker's postprocess drops everything below this floor.
        out.retain(|d| d.score >= LOW_CONF);
    }
}

/// A confident box on nothing, somewhere in the frame.
fn ghost(rng: &mut Rng) -> Detection {
    let (w, h) = (rng.range(60.0, 200.0), rng.range(80.0, 300.0));
    let cx = rng.range(w / 2.0, FRAME_W - w / 2.0);
    let cy = rng.range(h / 2.0, FRAME_H - h / 2.0);
    Detection {
        bbox: BBox::from_center(cx, cy, w, h),
        class: 0,
        score: rng.range(0.4, 0.6),
    }
}

/// The result of playing one scene in one detect-interval mode.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RunReport {
    pub scene: &'static str,
    pub mode: DetectInterval,
    /// Camera frames played (and scored).
    pub frames: u32,
    /// Frames the detector (and tracker) ran on.
    pub detector_runs: u32,
    pub score: MotScore,
    pub ids_per_object: f32,
}

/// Plays `scene` at 30 fps through a fresh `Tracker` paced by `mode`.
///
/// Detector latency is zero here: the detections of frame `t` are ready at `t`. The
/// real pipeline adds ~16–33 ms (Frame → boxes on the Latency page), which the UI
/// hides by predicting.
pub fn run(scene: &Scene, mode: DetectInterval, high_conf: f32) -> RunReport {
    run_with(scene, mode, high_conf, TrackerConfig::default())
}

/// [`run`] with a tracker configured by `cfg`, for trying out other settings.
pub fn run_with(
    scene: &Scene,
    mode: DetectInterval,
    high_conf: f32,
    cfg: TrackerConfig,
) -> RunReport {
    let t0 = Instant::now();
    let mut rng = Rng::new(scene.seed);
    let mut tracker = Tracker::new(cfg);
    let mut controller = IntervalController::new(mode);
    let mut eval = MotEvaluator::new();
    let (mut truths, mut dets, mut views, mut shown) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut detector_runs = 0;

    for i in 0..scene.frames() {
        // From the index, not by adding 33 ms per frame, so no rounding drift adds up.
        let secs = i as f64 / FPS;
        let t = t0 + Duration::from_secs_f64(secs);
        let s = secs as f32;
        if controller.is_due(t) {
            controller.ran(t);
            scene.detections(s, &mut rng, &mut dets);
            let step = tracker.update(&dets, high_conf, t);
            controller.on_result(step.interval_signal());
            tracker.views(&mut views);
            detector_runs += 1;
        }
        // Every frame, run or not: score what the UI would draw, the Tracked views
        // predicted to this frame's time. Lost tracks aren't drawn as tracks.
        shown.clear();
        shown.extend(
            views
                .iter()
                .filter(|v| v.state == TrackState::Tracked)
                .map(|v| (v.id, v.bbox_at(t))),
        );
        scene.truths(s, &mut truths);
        eval.frame(&truths, &shown);
    }

    RunReport {
        scene: scene.name,
        mode,
        frames: scene.frames(),
        detector_runs,
        score: eval.score(),
        ids_per_object: eval.ids_per_object(),
    }
}
