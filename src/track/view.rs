//! What the tracker reports: the state of a track, a per-update summary, and the
//! copied-out view of a confirmed track that the UI draws.
//!
//! These types are re-exported from [`super::tracker`], which is where the rest of the
//! crate names them (`track::tracker::TrackView`). They live in their own file only to
//! keep `tracker.rs` about the update loop.
//!
//! **Why the UI extrapolates.** The tracker runs on the detector thread, maybe every
//! 500 ms when detection is paced, while the UI shows a new camera frame every 33 ms. A
//! box drawn where the detector last saw the object would trail the video. A
//! [`TrackView`] therefore carries the Kalman velocity and the time it is valid at, and
//! [`TrackView::bbox_at`] moves the box to the time of the frame on screen.

use std::time::{Duration, Instant};

use super::TrackId;
use super::kalman::{BoxVelocity, MIN_SIZE};
use crate::detect::BBox;
use crate::detect::interval::IntervalSignal;

/// How far [`TrackView::bbox_at`] extrapolates at most, either way. Constant velocity
/// is a fair guess for a fraction of a second; beyond a second, people stop, turn and
/// sit down, and the guess is fiction.
pub const MAX_EXTRAPOLATION: Duration = Duration::from_secs(1);

/// Where a track is in its life.
///
/// ```text
///   new high detection
///          │
///          ▼
///     Tentative ──(matched again on the next run)──► Tracked ◄──(matched)──┐
///          │                                           │                  │
///   (not matched: deleted,                    (not matched)               │
///    it never got an id)                               ▼                  │
///                                                    Lost ────────────────┘
///                                                      │
///                                     (missing ≥ 1 s and ≥ 3 runs: deleted)
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrackState {
    /// Seen once; it may be a one-frame false positive, so it has no id yet.
    Tentative,
    /// Confirmed and matched on the latest detector run.
    Tracked,
    /// Confirmed, but not matched on the latest run: hidden, or missed by the detector.
    Lost,
}

/// What one `update` did. Counts are for this update unless named as totals.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TrackStep {
    /// Confirmed tracks matched this update (state Tracked).
    pub tracked: usize,
    /// Confirmed tracks currently Lost.
    pub lost: usize,
    /// New Tentative tracks.
    pub born: usize,
    /// Tentative → Tracked (each got a new `TrackId`).
    pub confirmed: usize,
    /// Tracked → Lost.
    pub newly_lost: usize,
    /// Lost → Tracked: the occlusion was survived.
    pub recovered: usize,
    /// Lost for too long, deleted.
    pub removed: usize,
    /// Mean IoU of predicted vs detected box, over round-1 matches of tracks that were
    /// Tracked: how well the Kalman predictions held up. `None` with no such match.
    pub mean_iou: Option<f32>,
}

impl TrackStep {
    /// The part of this step that the detect-interval controller learns from.
    pub fn interval_signal(&self) -> IntervalSignal {
        IntervalSignal {
            tracks: self.tracked + self.lost,
            mean_iou: self.mean_iou,
            born: self.born,
            lost: self.newly_lost,
        }
    }
}

/// A confirmed track as the UI sees it (copied out; no reference into the tracker).
///
/// It is `Copy` and holds no references, so a `Vec<TrackView>` can be sent from the
/// detector thread to the UI.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackView {
    pub id: TrackId,
    /// Class of the last matched high-score detection.
    pub class: u8,
    /// Score of the last matched detection.
    pub score: f32,
    /// Tracked or Lost; views never include Tentative.
    pub state: TrackState,
    /// Kalman estimate at `at`, frame pixels.
    pub bbox: BBox,
    /// Kalman velocity of `(cx, cy, w, h)`, px/s.
    pub velocity: BoxVelocity,
    /// The frame time of the update that produced this view.
    pub at: Instant,
}

impl TrackView {
    /// Constant-velocity extrapolation to `t` (earlier or later than `at`),
    /// with |t − at| capped at [`MAX_EXTRAPOLATION`] (1 s) and sizes ≥ [`MIN_SIZE`].
    ///
    /// `t` can be *earlier* than `at`: the detector may have finished a newer frame than
    /// the one the UI is still showing, and then the box moves back to that frame.
    pub fn bbox_at(&self, t: Instant) -> BBox {
        let cap = MAX_EXTRAPOLATION.as_secs_f32();
        let dt = signed_secs(self.at, t).clamp(-cap, cap);
        let b = self.bbox;
        let v = self.velocity;
        let cx = (b.x0 + b.x1) / 2.0 + v.cx * dt;
        let cy = (b.y0 + b.y1) / 2.0 + v.cy * dt;
        let w = (b.width() + v.w * dt).max(MIN_SIZE);
        let h = (b.height() + v.h * dt).max(MIN_SIZE);
        BBox::from_center(cx, cy, w, h)
    }
}

/// `to − from` in seconds, negative when `to` is earlier. `Instant` subtraction panics
/// when it would go negative, so try both directions with `checked_duration_since`.
fn signed_secs(from: Instant, to: Instant) -> f32 {
    if let Some(ahead) = to.checked_duration_since(from) {
        ahead.as_secs_f32()
    } else if let Some(behind) = from.checked_duration_since(to) {
        -behind.as_secs_f32()
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view(at: Instant) -> TrackView {
        TrackView {
            id: 1,
            class: 0,
            score: 0.9,
            state: TrackState::Tracked,
            bbox: BBox::from_center(100.0, 200.0, 50.0, 80.0),
            velocity: BoxVelocity {
                cx: 100.0,
                cy: -20.0,
                w: 10.0,
                h: -100.0,
            },
            at,
        }
    }

    fn assert_box(b: BBox, cx: f32, cy: f32, w: f32, h: f32) {
        let eps = 1e-3;
        assert!(((b.x0 + b.x1) / 2.0 - cx).abs() < eps, "cx of {b:?}");
        assert!(((b.y0 + b.y1) / 2.0 - cy).abs() < eps, "cy of {b:?}");
        assert!((b.width() - w).abs() < eps, "w of {b:?}");
        assert!((b.height() - h).abs() < eps, "h of {b:?}");
    }

    #[test]
    fn tracker_bbox_at_extrapolates_and_caps() {
        let t0 = Instant::now() + Duration::from_secs(10);
        let v = view(t0);
        assert_eq!(v.bbox_at(t0), v.bbox);
        // Forward 0.5 s: every coordinate moves by velocity × 0.5.
        let forward = v.bbox_at(t0 + Duration::from_millis(500));
        assert_box(forward, 150.0, 190.0, 55.0, 30.0);
        // Backward 0.1 s: the shown frame is older than the detection.
        let backward = v.bbox_at(t0 - Duration::from_millis(100));
        assert_box(backward, 90.0, 202.0, 49.0, 90.0);
        // 3 s ahead is capped at 1 s; the height would go negative and stops at MIN_SIZE.
        let far = v.bbox_at(t0 + Duration::from_secs(3));
        assert_box(far, 200.0, 180.0, 60.0, MIN_SIZE);
    }

    #[test]
    fn tracker_signed_secs_both_ways() {
        let t0 = Instant::now() + Duration::from_secs(10);
        let later = t0 + Duration::from_millis(250);
        assert!((signed_secs(t0, later) - 0.25).abs() < 1e-6);
        assert!((signed_secs(later, t0) + 0.25).abs() < 1e-6);
        assert_eq!(signed_secs(t0, t0), 0.0);
    }

    #[test]
    fn tracker_step_to_interval_signal() {
        let step = TrackStep {
            tracked: 3,
            lost: 2,
            born: 4,
            confirmed: 1,
            newly_lost: 5,
            recovered: 6,
            removed: 7,
            mean_iou: Some(0.8),
        };
        let signal = step.interval_signal();
        assert_eq!(
            signal,
            IntervalSignal {
                tracks: 5,
                mean_iou: Some(0.8),
                born: 4,
                lost: 5,
            }
        );
    }
}
