//! The ByteTrack tracker: one `update` per detector run turns detections into tracks
//! with stable IDs.
//!
//! **The idea of ByteTrack** (Zhang et al., 2022, "ByteTrack: Multi-Object Tracking by
//! Associating Every Detection Box"). Most trackers throw away low-score detections.
//! But when a person walks behind a chair, the detector doesn't stop seeing them at
//! once: the score drops from 0.8 to 0.3 to 0.2 as less of them shows. Those weak boxes
//! are mostly right *if* they sit where a known track is predicted to be. So ByteTrack
//! matches in two rounds: first the confident boxes against every track, then the weak
//! boxes, only against tracks that are still unmatched and were seen on the last run.
//! A weak box can keep a track alive, but never starts one.
//!
//! **One update** (the numbered list in [`super`] in code form):
//!
//! ```text
//! predict all tracks to t ─► split dets: high (≥ slider), low (≥ 0.10)
//!   round 1: Tracked + Lost   × high   (IoU ≥ 0.2)
//!   round 2: Tracked leftovers × low   (IoU ≥ 0.5)   → unmatched Tracked become Lost
//!   round 3: Tentative        × high leftovers (IoU ≥ 0.3) → 2nd hit confirms, else delete
//!   start a Tentative track per high detection still left
//!   delete Lost tracks missing ≥ 1 s and ≥ 3 runs
//! ```
//!
//! **IDs at confirmation.** A new track has no id until it is matched on a second run.
//! A one-frame false positive then never uses up an id, and `#7` on screen really is
//! the 7th object the tracker believed in. Ultralytics' ByteTrack does the same.
//!
//! **Class-agnostic.** Matching looks only at boxes, never at classes: YOLO often
//! flickers between car and truck, or dog and cat, and matching by class would break
//! the track at every flicker. A track's class is that of its latest *high* detection.
//!
//! **Time, not frames.** The detector may be paced (one run every 500 ms) or skip
//! frames when busy, so every prediction uses the real time between frames, and a lost
//! track's expiry is measured in seconds *and* in missed runs.
//!
//! **Later upgrades**, not built here: appearance features (ReID) for crossings, see
//! [`super::association`]; camera-motion compensation (BoT-SORT), which shifts every
//! prediction when the camera itself moves; and OC-SORT's re-update, which, when a lost
//! track is found again, replays its hidden stretch as a straight line to fix the
//! velocity the long coast has drifted.

use std::time::{Duration, Instant};

use super::TrackId;
use super::association::{Match, Matcher};
use super::kalman::{BoxFilter, KalmanConfig};
use crate::detect::{BBox, Detection};

pub use super::view::{TrackState, TrackStep, TrackView};

/// Below this, a detection is noise even for ByteTrack's second round. The worker
/// postprocesses at this floor. ByteTrack and Ultralytics (`track_low_thresh`)
/// use 0.1 too.
pub const LOW_CONF: f32 = 0.10;

/// Matches on consecutive runs a tentative track needs before it is confirmed. ByteTrack
/// confirms on the second frame: one frame is too easily a false positive.
const HITS_TO_CONFIRM: u32 = 2;

/// The tracker's thresholds and filter settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackerConfig {
    /// Detections below this score are ignored (see [`LOW_CONF`]).
    pub low_conf: f32,
    /// Round 1 gate: confirmed tracks × high detections.
    pub min_iou_high: f32,
    /// Round 2 gate: still-unmatched Tracked tracks × low detections.
    pub min_iou_low: f32,
    /// Round 3 gate: tentative tracks × leftover high detections.
    pub min_iou_tentative: f32,
    /// A lost track is deleted once it has been lost this long…
    pub lost_for: Duration,
    /// …and has missed at least this many detector runs.
    pub lost_min_misses: u32,
    pub kalman: KalmanConfig,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            low_conf: LOW_CONF,
            // ByteTrack's `match_thresh = 0.8` is a limit on the *cost* 1 − IoU, so
            // IoU ≥ 0.2. Generous, because a fast object's prediction can be off.
            min_iou_high: 0.2,
            // ByteTrack's round-2 threshold. Low boxes are less reliable, so they must
            // fit the prediction well.
            min_iou_low: 0.5,
            // ByteTrack matches unconfirmed tracks with cost ≤ 0.7, so IoU ≥ 0.3.
            min_iou_tentative: 0.3,
            // ByteTrack keeps lost tracks 30 frames = 1 s at 30 fps. In seconds,
            // because the detector may be paced…
            lost_for: Duration::from_secs(1),
            // …plus a run count, so a 500 ms interval still gives a hidden object
            // 3 chances to reappear before its id is gone.
            lost_min_misses: 3,
            kalman: KalmanConfig::default(),
        }
    }
}

/// One object the tracker follows (private: the UI gets [`TrackView`] copies).
struct Track {
    /// `None` while Tentative; the id is given at confirmation.
    id: Option<TrackId>,
    state: TrackState,
    filter: BoxFilter,
    /// Class of the last matched *high* detection.
    class: u8,
    /// Score of the last matched detection.
    score: f32,
    /// The frame time `filter` is predicted/updated to.
    at: Instant,
    /// Matches while Tentative.
    hits: u32,
    /// Consecutive detector runs without a match.
    misses: u32,
    /// When the track went Lost; `None` while it is not.
    lost_at: Option<Instant>,
    /// Matched in the current update; reset by `predict_all`.
    matched: bool,
}

impl Track {
    /// A tentative track from one high detection, seen at `t`.
    fn tentative(det: &Detection, t: Instant, cfg: KalmanConfig) -> Self {
        Self {
            id: None,
            state: TrackState::Tentative,
            filter: BoxFilter::new(det.bbox, cfg),
            class: det.class,
            score: det.score,
            at: t,
            hits: 1,
            misses: 0,
            lost_at: None,
            matched: true,
        }
    }

    /// Blends a matched detection into the track. Only a `high` detection sets the
    /// class: a 0.2 box is often a wrong guess about a partly hidden object.
    fn hit(&mut self, det: &Detection, high: bool) {
        self.filter.update(det.bbox);
        self.score = det.score;
        if high {
            self.class = det.class;
        }
        self.misses = 0;
        self.lost_at = None;
        self.matched = true;
        if self.id.is_some() {
            self.state = TrackState::Tracked;
        }
    }

    fn view(&self, id: TrackId) -> TrackView {
        TrackView {
            id,
            class: self.class,
            score: self.score,
            state: self.state,
            bbox: self.filter.bbox(),
            velocity: self.filter.velocity(),
            at: self.at,
        }
    }
}

/// Buffers reused by every `update`.
///
/// Allocated as they grow, then reused; `update` doesn't allocate in steady state.
/// Each matching round works on *subsets* of the tracks and detections, so `pool` and
/// `candidates` map the matcher's row and column numbers back to indices into
/// `Tracker::tracks` and the `dets` slice.
#[derive(Default)]
struct Scratch {
    /// Indices of the high-score detections.
    high: Vec<usize>,
    /// Indices of the low-score detections.
    low: Vec<usize>,
    /// Per detection: already matched (or used to start a track) in this update.
    det_used: Vec<bool>,
    /// Track indices taking part in the current round (matcher rows).
    pool: Vec<usize>,
    /// Detection indices taking part in the current round (matcher columns).
    candidates: Vec<usize>,
    /// Predicted boxes of the `pool` tracks, for the matcher.
    pool_boxes: Vec<BBox>,
    /// Boxes of the `candidates`, for the matcher.
    candidate_boxes: Vec<BBox>,
    /// The current round's matches, mapped back to track and detection indices.
    matches: Vec<Match>,
}

/// ByteTrack: turns each detector run's boxes into tracks with stable IDs.
pub struct Tracker {
    cfg: TrackerConfig,
    tracks: Vec<Track>,
    /// The id the next confirmed track gets. Starts at 1, so 0 never appears.
    next_id: TrackId,
    matcher: Matcher,
    scratch: Scratch,
}

impl Tracker {
    pub fn new(cfg: TrackerConfig) -> Self {
        Self {
            cfg,
            tracks: Vec::new(),
            next_id: 1,
            matcher: Matcher::new(),
            scratch: Scratch::default(),
        }
    }

    /// One ByteTrack step for the detections of a frame received at `t`.
    /// `high_conf` = the confidence slider (detections ≥ it are "high").
    pub fn update(&mut self, dets: &[Detection], high_conf: f32, t: Instant) -> TrackStep {
        let mut step = TrackStep::default();
        self.predict_all(t);
        self.split_detections(dets, high_conf);
        self.round_confirmed(dets, &mut step);
        self.round_low(dets);
        self.mark_missed(t, &mut step);
        self.round_tentative(dets, &mut step);
        self.start_tracks(dets, t, &mut step);
        self.expire_lost(t, &mut step);
        self.fill_counts(&mut step);
        step
    }

    /// Clears `out`, then pushes every confirmed track (Tracked and Lost), by id.
    pub fn views(&self, out: &mut Vec<TrackView>) {
        out.clear();
        for track in &self.tracks {
            if let Some(id) = track.id {
                out.push(track.view(id));
            }
        }
        // Vector order is creation order, which today matches confirmation order; the
        // sort keeps "by id" true even if that changes. It doesn't allocate.
        out.sort_unstable_by_key(|v| v.id);
    }

    /// IDs handed out so far (= the highest TrackId).
    pub fn ids_issued(&self) -> u32 {
        self.next_id - 1
    }

    /// Step 1: moves every track's filter to the new frame's time `t`.
    fn predict_all(&mut self, t: Instant) {
        for track in &mut self.tracks {
            // Saturating: a frame older than the track's time predicts by 0, not < 0.
            let dt = t.saturating_duration_since(track.at).as_secs_f32();
            if track.state == TrackState::Lost {
                track.filter.stop_resizing();
            }
            track.filter.predict(dt);
            track.at = t;
            track.matched = false;
        }
    }

    /// Step 2: sorts detection indices into high (≥ `high_conf`) and low
    /// (`low_conf ≤ score < high_conf`).
    fn split_detections(&mut self, dets: &[Detection], high_conf: f32) {
        let s = &mut self.scratch;
        s.high.clear();
        s.low.clear();
        s.det_used.clear();
        s.det_used.resize(dets.len(), false);
        for (i, det) in dets.iter().enumerate() {
            // If the slider is set below `low_conf`, everything ≥ `high_conf` is high,
            // and a score below `high_conf` is then below `low_conf` too: low stays empty.
            if det.score >= high_conf {
                s.high.push(i);
            } else if det.score >= self.cfg.low_conf {
                s.low.push(i);
            }
        }
    }

    /// Step 3, round 1: confirmed tracks (Tracked + Lost) vs the high detections.
    fn round_confirmed(&mut self, dets: &[Detection], step: &mut TrackStep) {
        let s = &mut self.scratch;
        s.pool.clear();
        s.pool.extend(
            (0..self.tracks.len()).filter(|&i| self.tracks[i].state != TrackState::Tentative),
        );
        s.candidates.clear();
        s.candidates.extend_from_slice(&s.high);
        self.match_round(dets, self.cfg.min_iou_high);

        let (mut iou_sum, mut iou_count) = (0.0, 0);
        for m in &self.scratch.matches {
            let track = &mut self.tracks[m.track];
            if track.state == TrackState::Lost {
                step.recovered += 1;
            } else {
                // Only tracks seen last run say how good a one-step prediction is; a
                // long-lost track's coasted box would drag the mean down.
                iou_sum += m.iou;
                iou_count += 1;
            }
            track.hit(&dets[m.det], true);
            self.scratch.det_used[m.det] = true;
        }
        step.mean_iou = (iou_count > 0).then(|| iou_sum / iou_count as f32);
    }

    /// Step 4, round 2: still-unmatched Tracked tracks vs the low detections. Not Lost
    /// tracks: a weak box near a long-hidden track is too little evidence.
    fn round_low(&mut self, dets: &[Detection]) {
        let s = &mut self.scratch;
        s.pool.clear();
        s.pool.extend((0..self.tracks.len()).filter(|&i| {
            let track = &self.tracks[i];
            track.state == TrackState::Tracked && !track.matched
        }));
        s.candidates.clear();
        s.candidates.extend_from_slice(&s.low);
        self.match_round(dets, self.cfg.min_iou_low);

        for m in &self.scratch.matches {
            self.tracks[m.track].hit(&dets[m.det], false);
            self.scratch.det_used[m.det] = true;
        }
    }

    /// After round 2 (not round 1, or round 2 would have nothing to rescue): a Tracked
    /// track nobody matched goes Lost, and a Lost one counts another miss.
    fn mark_missed(&mut self, t: Instant, step: &mut TrackStep) {
        for track in &mut self.tracks {
            if track.matched {
                continue;
            }
            match track.state {
                TrackState::Tracked => {
                    track.state = TrackState::Lost;
                    track.lost_at = Some(t);
                    track.misses = 1;
                    step.newly_lost += 1;
                }
                TrackState::Lost => track.misses += 1,
                TrackState::Tentative => {}
            }
        }
    }

    /// Step 5, round 3: tentative tracks vs the high detections nobody took. A second
    /// hit confirms a track and gives it an id; an unmatched tentative track is deleted
    /// (it never had an id, so it doesn't count as `removed`).
    fn round_tentative(&mut self, dets: &[Detection], step: &mut TrackStep) {
        let s = &mut self.scratch;
        s.pool.clear();
        s.pool.extend(
            (0..self.tracks.len()).filter(|&i| self.tracks[i].state == TrackState::Tentative),
        );
        s.candidates.clear();
        s.candidates
            .extend(s.high.iter().copied().filter(|&d| !s.det_used[d]));
        self.match_round(dets, self.cfg.min_iou_tentative);

        for m in &self.scratch.matches {
            let track = &mut self.tracks[m.track];
            track.hit(&dets[m.det], true);
            track.hits += 1;
            if track.hits >= HITS_TO_CONFIRM {
                track.id = Some(self.next_id);
                self.next_id += 1;
                track.state = TrackState::Tracked;
                step.confirmed += 1;
            }
            self.scratch.det_used[m.det] = true;
        }
        // Safe here: no index map is used after this point.
        self.tracks
            .retain(|track| track.state != TrackState::Tentative || track.matched);
    }

    /// Step 6: a tentative track for each high detection still unmatched. Low ones
    /// never start a track.
    fn start_tracks(&mut self, dets: &[Detection], t: Instant, step: &mut TrackStep) {
        for &d in &self.scratch.high {
            if !self.scratch.det_used[d] {
                self.tracks
                    .push(Track::tentative(&dets[d], t, self.cfg.kalman));
                step.born += 1;
            }
        }
    }

    /// Step 6, continued: deletes Lost tracks missing for `lost_for` *and* `lost_min_misses` runs.
    fn expire_lost(&mut self, t: Instant, step: &mut TrackStep) {
        let (lost_for, min_misses) = (self.cfg.lost_for, self.cfg.lost_min_misses);
        let before = self.tracks.len();
        self.tracks.retain(|track| {
            let Some(lost_at) = track.lost_at else {
                return true;
            };
            let expired =
                t.saturating_duration_since(lost_at) >= lost_for && track.misses >= min_misses;
            !expired
        });
        step.removed = before - self.tracks.len();
    }

    /// Last: how many confirmed tracks are Tracked and Lost after the update.
    fn fill_counts(&self, step: &mut TrackStep) {
        for track in &self.tracks {
            match track.state {
                TrackState::Tracked => step.tracked += 1,
                TrackState::Lost => step.lost += 1,
                TrackState::Tentative => {}
            }
        }
    }

    /// Matches the `pool` tracks' predicted boxes against the `candidates`' boxes and
    /// leaves the result in `scratch.matches`, with indices mapped back to
    /// `self.tracks` and `dets` (never use a matcher row as a track index directly).
    fn match_round(&mut self, dets: &[Detection], min_iou: f32) {
        let s = &mut self.scratch;
        s.pool_boxes.clear();
        s.pool_boxes
            .extend(s.pool.iter().map(|&i| self.tracks[i].filter.bbox()));
        s.candidate_boxes.clear();
        s.candidate_boxes
            .extend(s.candidates.iter().map(|&d| dets[d].bbox));
        self.matcher
            .match_by_iou(&s.pool_boxes, &s.candidate_boxes, min_iou, &mut s.matches);
        for m in &mut s.matches {
            m.track = s.pool[m.track];
            m.det = s.candidates[m.det];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The default high threshold: the confidence slider's default.
    const HIGH: f32 = 0.35;
    /// One run's gap at 30 fps.
    const FRAME: Duration = Duration::from_millis(33);

    /// A class-0 detection centred at `(cx, cy)`.
    fn det(cx: f32, cy: f32, w: f32, h: f32, score: f32) -> Detection {
        Detection {
            bbox: BBox::from_center(cx, cy, w, h),
            class: 0,
            score,
        }
    }

    /// A fixed start time, far enough from `now` that tests can go back in time.
    fn t0() -> Instant {
        Instant::now() + Duration::from_secs(100)
    }

    /// The time of run `i` at 30 fps.
    fn t(start: Instant, i: u32) -> Instant {
        start + FRAME * i
    }

    /// A 60 × 120 person walking right at 6 px per run.
    fn walker(i: u32) -> Detection {
        det(100.0 + 6.0 * i as f32, 300.0, 60.0, 120.0, 0.9)
    }

    fn views(tracker: &Tracker) -> Vec<TrackView> {
        let mut out = Vec::new();
        tracker.views(&mut out);
        out
    }

    fn center_y(b: BBox) -> f32 {
        (b.y0 + b.y1) / 2.0
    }

    #[test]
    fn tracker_confirms_on_second_hit() {
        let start = t0();
        let mut tracker = Tracker::new(TrackerConfig::default());
        let step = tracker.update(&[walker(0)], HIGH, t(start, 0));
        assert_eq!(step.born, 1);
        assert!(views(&tracker).is_empty());

        let step = tracker.update(&[walker(1)], HIGH, t(start, 1));
        assert_eq!((step.born, step.confirmed, step.tracked), (0, 1, 1));
        let v = views(&tracker);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].id, 1);
        assert_eq!(v[0].state, TrackState::Tracked);
        assert_eq!(tracker.ids_issued(), 1);
    }

    #[test]
    fn tracker_follows_a_walker() {
        let start = t0();
        let mut tracker = Tracker::new(TrackerConfig::default());
        for i in 0..60 {
            tracker.update(&[walker(i)], HIGH, t(start, i));
            if i >= 1 {
                let v = views(&tracker);
                assert_eq!(v.len(), 1);
                assert_eq!(v[0].id, 1, "run {i}");
            }
        }
        assert_eq!(tracker.ids_issued(), 1);
        let expected = 6.0 / FRAME.as_secs_f32();
        let vx = views(&tracker)[0].velocity.cx;
        assert!(
            (vx - expected).abs() < 0.15 * expected,
            "vx {vx} vs {expected}"
        );
    }

    #[test]
    fn tracker_survives_a_short_occlusion() {
        let start = t0();
        let mut tracker = Tracker::new(TrackerConfig::default());
        for i in 0..30 {
            tracker.update(&[walker(i)], HIGH, t(start, i));
        }
        // 10 runs (0.33 s) with nothing detected.
        for i in 30..40 {
            let step = tracker.update(&[], HIGH, t(start, i));
            assert_eq!(step.newly_lost, usize::from(i == 30), "run {i}");
            assert_eq!(views(&tracker)[0].state, TrackState::Lost);
        }
        // The walker reappears where it would have been.
        let step = tracker.update(&[walker(40)], HIGH, t(start, 40));
        assert_eq!(step.recovered, 1);
        let v = views(&tracker);
        assert_eq!(v.len(), 1);
        assert_eq!((v[0].id, v[0].state), (1, TrackState::Tracked));
        assert_eq!(tracker.ids_issued(), 1);
    }

    #[test]
    fn tracker_low_score_keeps_track_alive() {
        let start = t0();
        let mut tracker = Tracker::new(TrackerConfig::default());
        for i in 0..30 {
            tracker.update(&[walker(i)], HIGH, t(start, i));
        }
        // Half hidden: a weak box of another class, between LOW_CONF and HIGH.
        for i in 30..40 {
            let weak = Detection {
                class: 5,
                score: 0.2,
                ..walker(i)
            };
            let step = tracker.update(&[weak], HIGH, t(start, i));
            assert_eq!((step.newly_lost, step.born), (0, 0), "run {i}");
            let v = views(&tracker);
            assert_eq!(v[0].state, TrackState::Tracked);
            assert_eq!(v[0].class, 0);
            assert_eq!(v[0].score, 0.2);
        }
        assert_eq!(tracker.ids_issued(), 1);
    }

    #[test]
    fn tracker_long_occlusion_gets_new_id() {
        let start = t0();
        let mut tracker = Tracker::new(TrackerConfig::default());
        for i in 0..30 {
            tracker.update(&[walker(i)], HIGH, t(start, i));
        }
        // Lost at run 30. Removal needs ≥ 1 s since then (run 61: 31 × 33 ms) and
        // ≥ 3 misses (long passed).
        let mut removed_at = None;
        for i in 30..76 {
            let step = tracker.update(&[], HIGH, t(start, i));
            if step.removed == 1 {
                removed_at = Some(i);
            }
        }
        assert_eq!(removed_at, Some(61));
        assert!(views(&tracker).is_empty());

        tracker.update(&[walker(76)], HIGH, t(start, 76));
        tracker.update(&[walker(77)], HIGH, t(start, 77));
        let v = views(&tracker);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].id, 2);
    }

    #[test]
    fn tracker_paced_runs_need_three_misses() {
        let start = t0();
        let gap = Duration::from_millis(500);
        // A person standing still, seen on two runs 500 ms apart: confirmed.
        let person = det(300.0, 300.0, 60.0, 120.0, 0.9);
        let mut tracker = Tracker::new(TrackerConfig::default());
        tracker.update(&[person], HIGH, start);
        tracker.update(&[person], HIGH, start + gap);

        // Misses 1 (lost) and 2 (0.5 s lost): kept.
        for run in 2..4 {
            let step = tracker.update(&[], HIGH, start + gap * run);
            assert_eq!((step.lost, step.removed), (1, 0), "run {run}");
        }
        // Miss 3, 1.0 s after it went lost: both conditions hold.
        let step = tracker.update(&[], HIGH, start + gap * 4);
        assert_eq!((step.lost, step.removed), (0, 1));
    }

    #[test]
    fn tracker_long_gaps_still_need_three_misses() {
        let start = t0();
        // Runs 1.2 s apart (a stalled detector): the time limit passes on miss 2.
        let gap = Duration::from_millis(1200);
        let person = det(300.0, 300.0, 60.0, 120.0, 0.9);
        let mut tracker = Tracker::new(TrackerConfig::default());
        tracker.update(&[person], HIGH, start);
        tracker.update(&[person], HIGH, start + gap);
        tracker.update(&[], HIGH, start + gap * 2);
        let step = tracker.update(&[], HIGH, start + gap * 3);
        assert_eq!((step.lost, step.removed), (1, 0), "1.2 s lost, 2 misses");
        let step = tracker.update(&[], HIGH, start + gap * 4);
        assert_eq!(step.removed, 1);
    }

    #[test]
    fn tracker_ignores_one_frame_noise() {
        let start = t0();
        let mut tracker = Tracker::new(TrackerConfig::default());
        let step = tracker.update(&[walker(0)], HIGH, t(start, 0));
        assert_eq!(step.born, 1);
        let step = tracker.update(&[], HIGH, t(start, 1));
        // Deleted silently: it never had an id.
        assert_eq!(step, TrackStep::default());
        assert!(views(&tracker).is_empty());
        assert_eq!(tracker.ids_issued(), 0);
    }

    #[test]
    fn tracker_keeps_ids_through_a_crossing() {
        let start = t0();
        let mut tracker = Tracker::new(TrackerConfig::default());
        // A walks right at y 300 (120 px tall), B walks left at y 380 (200 px tall).
        // Their boxes overlap in y, and in x while they pass each other.
        let a = |i: u32| det(100.0 + 8.0 * i as f32, 300.0, 60.0, 120.0, 0.9);
        let b = |i: u32| det(600.0 - 8.0 * i as f32, 380.0, 100.0, 200.0, 0.9);
        for i in 0..60 {
            tracker.update(&[a(i), b(i)], HIGH, t(start, i));
            if i >= 1 {
                let v = views(&tracker);
                assert_eq!(v.len(), 2, "run {i}");
                assert!((center_y(v[0].bbox) - 300.0).abs() < 10.0, "run {i}: {v:?}");
                assert!((center_y(v[1].bbox) - 380.0).abs() < 10.0, "run {i}: {v:?}");
            }
        }
        assert_eq!(tracker.ids_issued(), 2);
    }

    #[test]
    fn tracker_class_flicker_keeps_id() {
        let start = t0();
        let mut tracker = Tracker::new(TrackerConfig::default());
        for i in 0..20 {
            // A car that YOLO alternately calls car (2) and truck (7).
            let class = if i % 2 == 0 { 2 } else { 7 };
            let d = Detection { class, ..walker(i) };
            tracker.update(&[d], HIGH, t(start, i));
            if i >= 1 {
                let v = views(&tracker);
                assert_eq!(v.len(), 1);
                assert_eq!((v[0].id, v[0].class), (1, class));
            }
        }
        assert_eq!(tracker.ids_issued(), 1);
    }

    #[test]
    fn tracker_views_sorted_by_id_and_skip_tentative() {
        let start = t0();
        let mut tracker = Tracker::new(TrackerConfig::default());
        let left = |i: u32| det(100.0, 100.0 + 2.0 * i as f32, 50.0, 50.0, 0.9);
        let right = |i: u32| det(500.0, 100.0 + 2.0 * i as f32, 50.0, 50.0, 0.9);
        let newcomer = det(900.0, 400.0, 50.0, 50.0, 0.9);
        tracker.update(&[right(0), left(0)], HIGH, t(start, 0));
        tracker.update(&[right(1), left(1)], HIGH, t(start, 1));
        // The newcomer is tentative after this run, so it has no view.
        let step = tracker.update(&[newcomer, left(2), right(2)], HIGH, t(start, 2));
        assert_eq!(step.born, 1);
        let v = views(&tracker);
        let ids: Vec<TrackId> = v.iter().map(|v| v.id).collect();
        assert_eq!(ids, [1, 2]);
        assert!(v.iter().all(|v| v.state == TrackState::Tracked));
        assert!((v[0].bbox.x0 - 475.0).abs() < 1.0, "id 1 is the right box");
    }

    #[test]
    fn tracker_same_time_update_is_harmless() {
        let start = t0();
        let mut tracker = Tracker::new(TrackerConfig::default());
        tracker.update(&[walker(0)], HIGH, start);
        // dt = 0, and a frame older than the tracks (dt saturates to 0): no panic.
        let step = tracker.update(&[walker(0)], HIGH, start);
        assert_eq!(step.confirmed, 1);
        tracker.update(&[walker(0)], HIGH, start - FRAME);
        assert_eq!(views(&tracker).len(), 1);
    }

    #[test]
    fn tracker_mean_iou_over_tracked_matches() {
        let start = t0();
        let mut tracker = Tracker::new(TrackerConfig::default());
        let step = tracker.update(&[walker(0)], HIGH, t(start, 0));
        assert_eq!(step.mean_iou, None);
        tracker.update(&[walker(1)], HIGH, t(start, 1));
        let step = tracker.update(&[walker(2)], HIGH, t(start, 2));
        let mean = step.mean_iou.expect("one Tracked match");
        assert!(mean > 0.8 && mean <= 1.0, "mean {mean}");
    }

    #[test]
    fn tracker_high_below_low_conf() {
        let start = t0();
        let mut tracker = Tracker::new(TrackerConfig::default());
        let dets = [
            det(100.0, 100.0, 50.0, 50.0, 0.07),
            det(300.0, 100.0, 50.0, 50.0, 0.2),
            det(500.0, 100.0, 50.0, 50.0, 0.03),
        ];
        let step = tracker.update(&dets, 0.05, t(start, 0));
        assert_eq!(step.born, 2);
        assert_eq!(tracker.scratch.high, [0, 1]);
        assert!(tracker.scratch.low.is_empty());
    }
}
