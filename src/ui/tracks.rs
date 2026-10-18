//! Track boxes over the video: the tracker's objects, each with its stable id
//! (`#7 person`), in its group colour.
//!
//! **What a track box means.** A detection box says "the detector saw a person here, in
//! one frame". A track box says "this is object #7, the same one as before". The id
//! survives from frame to frame, and through short occlusions.
//!
//! **Why it is predicted to the shown frame.** The tracker's views come from an older
//! frame than the one on screen: the detector needs 15–40 ms per frame, and when it is
//! paced it may have last run 500 ms ago. Drawn where the detector saw it, a walking
//! person's box would trail behind them. Each [`TrackView`] carries its Kalman velocity
//! and the time `at` it is valid for, so [`TrackView::bbox_at`] slides it to the time
//! the frame on screen arrived:
//!
//! ```text
//! box at the shown frame = box at `at` + velocity × (shown_at − at)
//!
//!   a walker at 300 px/s, view 100 ms older than the frame on screen:
//!   the box moves 300 × 0.1 = 30 frame px to the right
//! ```
//!
//! It is the time of the **shown** frame, not "now": the image on screen is already
//! ~70–80 ms old when it is drawn (sensor → app), and the box must match the image.
//!
//! **What a dim box is.** A *lost* track: the tracker didn't find the object on the
//! latest detector run (it walked behind something, or the detector missed it). It
//! keeps coasting on its prediction, drawn thin and dim with a `#7 lost` label, until
//! it is found again (same id, a survived occlusion) or has been lost for too long and
//! is deleted. Watching one is the best way to see the tracker work.

use std::time::Instant;

use eframe::egui::{Painter, Rect, Stroke, StrokeKind, Vec2};

use super::detections::{group_color, label, to_screen};
use crate::detect::classes::{CLASS_NAMES, ClassFilter, Group};
use crate::track::tracker::{TrackState, TrackView};

/// A live track: the same 2 px stroke and full colour as a detection box.
const TRACKED_STROKE: f32 = 2.0;
/// A lost track: thinner, so it reads as "a guess", not "seen".
const LOST_STROKE: f32 = 1.0;
/// A lost track's colour factor: dim enough to tell apart at a glance, bright enough
/// to still follow on live video.
const LOST_DIM: f32 = 0.45;

/// `#7 person` for a tracked object, `#7 lost` while it's hidden.
pub fn track_label(v: &TrackView) -> String {
    match v.state {
        TrackState::Lost => format!("#{} lost", v.id),
        TrackState::Tracked | TrackState::Tentative => {
            format!("#{} {}", v.id, CLASS_NAMES[usize::from(v.class)])
        }
    }
}

/// Stroke width and colour factor: tracked 2 px at full colour, lost 1 px at 45 %.
pub fn track_style(state: TrackState) -> (f32, f32) {
    match state {
        TrackState::Lost => (LOST_STROKE, LOST_DIM),
        // Views never hold Tentative tracks; if one ever did, it would look live.
        TrackState::Tracked | TrackState::Tentative => (TRACKED_STROKE, 1.0),
    }
}

/// Where the track goes on screen for a frame received at `shown_at`: predicted to
/// that time, then mapped from frame pixels into the video rect.
pub fn screen_box(v: &TrackView, shown_at: Instant, frame: Vec2, video: Rect) -> Rect {
    to_screen(v.bbox_at(shown_at), frame, video)
}

/// Draws the tracks whose class `filter` lets through, predicted to `shown_at` (the
/// shown frame's `received`). Lost tracks are drawn only with `show_lost`.
///
/// Lost tracks go first and Tracked ones after, so a live box and its label end up on
/// top where they overlap a coasting one. Within each pass the views keep their order,
/// which is by id.
pub fn draw(
    painter: &Painter,
    video: Rect,
    frame: Vec2,
    tracks: &[TrackView],
    filter: ClassFilter,
    show_lost: bool,
    shown_at: Instant,
) {
    let shown = |v: &&TrackView| filter.contains(v.class);
    if show_lost {
        let lost = tracks.iter().filter(|v| v.state == TrackState::Lost);
        for v in lost.filter(shown) {
            draw_one(painter, video, frame, v, shown_at);
        }
    }
    let tracked = tracks.iter().filter(|v| v.state == TrackState::Tracked);
    for v in tracked.filter(shown) {
        draw_one(painter, video, frame, v, shown_at);
    }
}

fn draw_one(painter: &Painter, video: Rect, frame: Vec2, v: &TrackView, shown_at: Instant) {
    let (width, dim) = track_style(v.state);
    let color = group_color(Group::of(v.class)).gamma_multiply(dim);
    let boxr = screen_box(v, shown_at, frame, video);
    painter.rect_stroke(boxr, 0.0, Stroke::new(width, color), StrokeKind::Middle);
    label(painter, boxr, track_label(v), color, video);
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use eframe::egui::{pos2, vec2};

    use super::*;
    use crate::detect::BBox;
    use crate::track::kalman::BoxVelocity;

    fn close(a: Rect, b: Rect) -> bool {
        (a.min - b.min).length() < 1e-3 && (a.max - b.max).length() < 1e-3
    }

    /// A 1920×1080 frame shown in a 960×540 video rect at (100, 50): half scale.
    const FRAME: Vec2 = vec2(1920.0, 1080.0);
    fn video() -> Rect {
        Rect::from_min_size(pos2(100.0, 50.0), vec2(960.0, 540.0))
    }

    /// A person (class 0) at x 200..400, y 100..500, walking right at 300 px/s.
    fn walker(at: Instant, state: TrackState) -> TrackView {
        TrackView {
            id: 7,
            class: 0,
            score: 0.8,
            state,
            bbox: BBox::new(200.0, 100.0, 400.0, 500.0),
            velocity: BoxVelocity {
                cx: 300.0,
                ..BoxVelocity::default()
            },
            at,
        }
    }

    #[test]
    fn tracks_label_tracked_and_lost() {
        let at = Instant::now();
        assert_eq!(track_label(&walker(at, TrackState::Tracked)), "#7 person");
        assert_eq!(track_label(&walker(at, TrackState::Lost)), "#7 lost");
    }

    #[test]
    fn tracks_style_dims_lost() {
        assert_eq!(track_style(TrackState::Tracked), (2.0, 1.0));
        let (width, dim) = track_style(TrackState::Lost);
        assert!(width < 2.0, "{width}");
        assert!(dim < 1.0, "{dim}");
    }

    #[test]
    fn tracks_screen_box_predicts_to_shown_frame() {
        let at = Instant::now();
        let v = walker(at, TrackState::Tracked);
        // 100 ms later at 300 px/s: 30 frame px right, 15 screen px at half scale.
        let r = screen_box(&v, at + Duration::from_millis(100), FRAME, video());
        let want = Rect::from_min_max(pos2(215.0, 100.0), pos2(315.0, 300.0));
        assert!(close(r, want), "{r:?}");
    }

    #[test]
    fn tracks_screen_box_backwards() {
        let at = Instant::now() + Duration::from_secs(1);
        let v = walker(at, TrackState::Tracked);
        // A frame on screen 100 ms *older* than the view: the box moves back.
        let r = screen_box(&v, at - Duration::from_millis(100), FRAME, video());
        let want = Rect::from_min_max(pos2(185.0, 100.0), pos2(285.0, 300.0));
        assert!(close(r, want), "{r:?}");
    }
}
