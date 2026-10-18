//! Detection boxes over the video: a 2 px rectangle in the box's group colour plus a
//! label chip (`person 0.87`).
//!
//! Boxes arrive in **camera frame pixels** (the postprocessor has already undone the
//! letterbox), so the only mapping left is frame → the aspect-fit video rect on screen.
//! The [`ClassFilter`] and the confidence slider are applied here, at draw time: the
//! detector always reports all 80 classes, and every box down to `LOW_CONF` (0.10)
//! because the tracker's second round needs the weak ones. So toggling a class or
//! moving the slider is instant, and the panel can still count a hidden class.

use eframe::egui::{Color32, FontId, Painter, Rect, Stroke, StrokeKind, Vec2, pos2, vec2};

use crate::detect::classes::{CLASS_NAMES, ClassFilter, Group};
use crate::detect::{BBox, Detection};

/// One colour per group, in [`Group::ALL`] order. Bright, mid-saturation tones that
/// stay readable on top of live video and differ from each other at a glance.
pub const GROUP_COLORS: [Color32; 12] = [
    Color32::from_rgb(0x4F, 0xC3, 0xF7), // People
    Color32::from_rgb(0xFF, 0xB7, 0x4D), // Vehicles
    Color32::from_rgb(0xA1, 0x88, 0x7F), // Outdoor
    Color32::from_rgb(0x81, 0xC7, 0x84), // Animals
    Color32::from_rgb(0xBA, 0x68, 0xC8), // Accessories
    Color32::from_rgb(0xF0, 0x62, 0x92), // Sports
    Color32::from_rgb(0x4D, 0xB6, 0xAC), // Kitchen
    Color32::from_rgb(0xFF, 0xF1, 0x76), // Food
    Color32::from_rgb(0x90, 0xA4, 0xAE), // Furniture
    Color32::from_rgb(0x79, 0x86, 0xCB), // Electronics
    Color32::from_rgb(0xE5, 0x73, 0x73), // Appliances
    Color32::from_rgb(0xDC, 0xE7, 0x75), // Indoor
];

const BOX_STROKE: f32 = 2.0;
const LABEL_FONT: f32 = 12.0;
/// Near-black text: every group colour is light, so dark text reads best on it.
const LABEL_TEXT: Color32 = Color32::from_rgb(0x11, 0x11, 0x11);
const LABEL_PAD: Vec2 = vec2(4.0, 1.0);
const LABEL_RADIUS: f32 = 3.0;

pub fn group_color(g: Group) -> Color32 {
    // `Group::ALL` lists the variants in declaration order, so `g as usize` is its index.
    GROUP_COLORS[g as usize]
}

/// Maps a box from frame pixels into the on-screen video rect. The video rect has the
/// frame's aspect ratio (`hud::fit_rect`), so one scale per axis is all it takes.
pub fn to_screen(b: BBox, frame: Vec2, video: Rect) -> Rect {
    let scale = video.size() / frame;
    Rect::from_min_max(
        video.min + vec2(b.x0, b.y0) * scale,
        video.min + vec2(b.x1, b.y1) * scale,
    )
}

/// Where a label of size `label` goes: just above the box's top-left corner; inside
/// the box instead when that would cross the video's top edge; pushed left when it
/// would cross the right edge. The painter clips whatever still sticks out.
pub fn label_rect(boxr: Rect, label: Vec2, video: Rect) -> Rect {
    let mut y = boxr.top() - label.y;
    if y < video.top() {
        y = boxr.top();
    }
    let x = boxr.left().min(video.right() - label.x).max(video.left());
    Rect::from_min_size(pos2(x, y), label)
}

/// Draws every detection whose class `filter` lets through and whose score is at least
/// `min_score` (the confidence slider).
///
/// `dets` is best first, so we draw it in reverse: the best box (and its label) is
/// painted last and ends up on top where boxes overlap. Formatting ≤ 300 short labels
/// per redraw is cheap and nothing here scales with the frame size.
pub fn draw(
    painter: &Painter,
    video: Rect,
    frame: Vec2,
    dets: &[Detection],
    filter: ClassFilter,
    min_score: f32,
) {
    let shown = |d: &&Detection| filter.contains(d.class) && d.score >= min_score;
    for d in dets.iter().rev().filter(shown) {
        let color = group_color(Group::of(d.class));
        let boxr = to_screen(d.bbox, frame, video);
        painter.rect_stroke(
            boxr,
            0.0,
            Stroke::new(BOX_STROKE, color),
            StrokeKind::Middle,
        );

        let name = CLASS_NAMES[usize::from(d.class)];
        label(
            painter,
            boxr,
            format!("{name} {:.2}", d.score),
            color,
            video,
        );
    }
}

/// A label chip for the box `boxr`: `text` in near-black on a `color` pill, placed by
/// [`label_rect`]. Shared by the detection boxes and the track boxes (`ui::tracks`).
pub fn label(painter: &Painter, boxr: Rect, text: String, color: Color32, video: Rect) {
    chip(painter, text, color, |size| label_rect(boxr, size, video));
}

/// `text` in near-black on a `color` pill, in the rect `place` picks for the chip's
/// size. The one chip style for box labels and the rule overlay's dwell timers.
pub fn chip(painter: &Painter, text: String, color: Color32, place: impl FnOnce(Vec2) -> Rect) {
    let galley = painter.layout_no_wrap(text, FontId::proportional(LABEL_FONT), LABEL_TEXT);
    let rect = place(galley.size() + 2.0 * LABEL_PAD);
    painter.rect_filled(rect, LABEL_RADIUS, color);
    painter.galley(rect.min + LABEL_PAD, galley, LABEL_TEXT);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: Rect, b: Rect) -> bool {
        (a.min - b.min).length() < 1e-3 && (a.max - b.max).length() < 1e-3
    }

    /// A 960×540 video rect at (100, 50).
    fn video() -> Rect {
        Rect::from_min_size(pos2(100.0, 50.0), vec2(960.0, 540.0))
    }

    #[test]
    fn detections_to_screen_halves_a_1080p_box() {
        let frame = vec2(1920.0, 1080.0);
        let r = to_screen(BBox::new(200.0, 100.0, 600.0, 500.0), frame, video());
        let want = Rect::from_min_max(pos2(200.0, 100.0), pos2(400.0, 300.0));
        assert!(close(r, want), "{r:?}");
        // The whole frame maps onto the whole video rect.
        let all = to_screen(BBox::new(0.0, 0.0, 1920.0, 1080.0), frame, video());
        assert!(close(all, video()), "{all:?}");
    }

    #[test]
    fn detections_label_sits_above_the_box() {
        let boxr = Rect::from_min_max(pos2(300.0, 200.0), pos2(500.0, 400.0));
        let r = label_rect(boxr, vec2(80.0, 16.0), video());
        let want = Rect::from_min_size(pos2(300.0, 184.0), vec2(80.0, 16.0));
        assert!(close(r, want), "{r:?}");
    }

    #[test]
    fn detections_label_flips_inside_at_the_top() {
        let boxr = Rect::from_min_max(pos2(300.0, 55.0), pos2(500.0, 400.0));
        let r = label_rect(boxr, vec2(80.0, 16.0), video());
        let want = Rect::from_min_size(pos2(300.0, 55.0), vec2(80.0, 16.0));
        assert!(close(r, want), "{r:?}");
    }

    #[test]
    fn detections_label_shifts_left_at_the_right_edge() {
        let boxr = Rect::from_min_max(pos2(1030.0, 200.0), pos2(1060.0, 300.0));
        let r = label_rect(boxr, vec2(80.0, 16.0), video());
        let want = Rect::from_min_size(pos2(980.0, 184.0), vec2(80.0, 16.0));
        assert!(close(r, want), "{r:?}");
    }

    #[test]
    fn detections_group_colors_follow_group_all() {
        assert_eq!(group_color(Group::People), GROUP_COLORS[0]);
        assert_eq!(group_color(Group::Indoor), GROUP_COLORS[11]);
        for (i, g) in Group::ALL.into_iter().enumerate() {
            assert_eq!(g as usize, i, "{g:?}");
        }
    }
}
