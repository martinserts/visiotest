//! The HUD inside the video rect: clock, frame chip, waiting and stalled states.
//! Everything is drawn with a painter clipped to the video rect, so it never spills
//! under the panels. The detection boxes (`ui::detections`) go into the same painter.

use std::time::{Duration, Instant};

use eframe::egui::{self, Align2, Color32, FontId, Painter, Pos2, Rect, Vec2, pos2, vec2};

use super::AMBER;

/// With no new frame for this long, the HUD flags the camera as stalled.
pub const STALL_AFTER: Duration = Duration::from_secs(1);
const INSET: f32 = 24.0;
/// The HUD's translucent black, also behind the rule overlay's letters and hints.
pub const BOX_FILL: Color32 = Color32::from_black_alpha(190);

/// Clock font size: readable to the camera at any window size.
pub fn clock_size(video_height: f32) -> f32 {
    (video_height * 0.07).clamp(32.0, 96.0)
}

/// How long the camera has been silent, if that is at least [`STALL_AFTER`].
pub fn stalled_for(received: Instant, now: Instant) -> Option<Duration> {
    let age = now.saturating_duration_since(received);
    (age >= STALL_AFTER).then_some(age)
}

/// The wall clock, bottom-left. Drawn every frame, never throttled.
pub fn clock(painter: &Painter, video: Rect, text: String) {
    let size = clock_size(video.height());
    let at = pos2(video.left() + INSET, video.bottom() - INSET);
    boxed(
        painter,
        at,
        Align2::LEFT_BOTTOM,
        text,
        FontId::monospace(size),
        vec2(12.0, 6.0),
        8.0,
    );
}

/// `#seq`, top-left.
pub fn frame_chip(painter: &Painter, video: Rect, seq: u64) {
    let at = pos2(video.left() + INSET, video.top() + INSET);
    boxed(
        painter,
        at,
        Align2::LEFT_TOP,
        format!("#{seq}"),
        FontId::monospace(12.0),
        vec2(8.0, 4.0),
        6.0,
    );
}

/// No frame yet: a spinner and a caption, centred in `area`.
pub fn waiting(ui: &mut egui::Ui, area: Rect) {
    let c = area.center();
    ui.put(
        Rect::from_center_size(c - vec2(0.0, 24.0), Vec2::splat(32.0)),
        egui::Spinner::new().size(32.0),
    );
    ui.painter_at(area).text(
        c + vec2(0.0, 8.0),
        Align2::CENTER_TOP,
        "Waiting for camera…",
        FontId::proportional(18.0),
        Color32::GRAY,
    );
}

/// An amber banner over the last image.
pub fn stalled(painter: &Painter, video: Rect, age: Duration) {
    let galley = painter.layout_no_wrap(
        format!("No frames from camera for {:.1} s", age.as_secs_f64()),
        FontId::proportional(20.0),
        Color32::BLACK,
    );
    let bg = Rect::from_center_size(video.center(), galley.size() + vec2(32.0, 16.0));
    painter.rect_filled(bg, 8.0, AMBER);
    painter.galley(bg.center() - galley.size() / 2.0, galley, Color32::BLACK);
}

/// White text on a translucent black rounded box anchored at `at`.
pub fn boxed(
    painter: &Painter,
    at: Pos2,
    align: Align2,
    text: String,
    font: FontId,
    pad: Vec2,
    radius: f32,
) {
    let galley = painter.layout_no_wrap(text, font, Color32::WHITE);
    let bg = align.anchor_size(at, galley.size() + 2.0 * pad);
    painter.rect_filled(bg, radius, BOX_FILL);
    painter.galley(bg.min + pad, galley, Color32::WHITE);
}

/// The largest rect with `size`'s aspect ratio that fits in `avail`, centred: where the
/// video goes, and so the rect the whole HUD (and every box) is drawn in.
pub fn fit_rect(avail: Rect, size: Vec2) -> Rect {
    if size.x <= 0.0 || size.y <= 0.0 {
        return avail;
    }
    let scale = (avail.width() / size.x).min(avail.height() / size.y);
    Rect::from_center_size(avail.center(), size * scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: Rect, b: Rect) -> bool {
        (a.min - b.min).length() < 1e-3 && (a.max - b.max).length() < 1e-3
    }

    #[test]
    fn fit_rect_letterboxes_in_4_3() {
        let avail = Rect::from_min_size(pos2(0.0, 0.0), vec2(1600.0, 1200.0));
        let r = fit_rect(avail, vec2(1920.0, 1080.0));
        // Full width, 900 tall, centred vertically.
        let want = Rect::from_min_max(pos2(0.0, 150.0), pos2(1600.0, 1050.0));
        assert!(close(r, want), "{r:?}");
    }

    #[test]
    fn fit_rect_pillarboxes_in_21_9() {
        let avail = Rect::from_min_size(pos2(10.0, 20.0), vec2(2100.0, 900.0));
        let r = fit_rect(avail, vec2(1920.0, 1080.0));
        // Full height, 1600 wide, centred horizontally.
        let want = Rect::from_min_max(pos2(260.0, 20.0), pos2(1860.0, 920.0));
        assert!(close(r, want), "{r:?}");
    }

    #[test]
    fn clock_size_clamps() {
        assert_eq!(clock_size(100.0), 32.0);
        assert!((clock_size(1000.0) - 70.0).abs() < 1e-3);
        assert_eq!(clock_size(5000.0), 96.0);
    }

    #[test]
    fn stall_after_one_second() {
        let t0 = Instant::now();
        assert_eq!(stalled_for(t0, t0 + Duration::from_millis(999)), None);
        let age = Duration::from_millis(1500);
        assert_eq!(stalled_for(t0, t0 + age), Some(age));
    }
}
