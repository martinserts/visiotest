//! The rules on the video: lines and zones drawn over the image, and the edit mode's
//! mouse.
//!
//! **What the marks mean.**
//!
//! ```text
//!        A                    a line, drawn from `a` to `b`, with its two sides:
//!   ●━━━━━━━━━━━●  door  A→B 5 · B→A 3
//!        B  ↓             A and B sit either side of the middle, and the arrow
//!                         points from A to B: an object walking that way counts A→B
//!   ╭─────────╮
//!   │  desk   │ 2 inside · 7 visits     a zone: outline, name and how many are inside
//!   ╰─────────╯
//!     [#7] 0:42                         a dwell timer under each object inside a zone
//! ```
//!
//! Side A is the one [`signed_distance`] calls positive: on your right when you stand
//! on `a` and face `b` (screen y points down). The labels are placed with the same
//! normal the engine uses, so the A on screen is always the A in the counts.
//!
//! A rule **flashes** green for [`FLASH`] after it fires. The flash reads
//! `RuleTotals::last_event`, which travels with the cumulative totals, so it shows even
//! when the result carrying the event itself was overwritten before the UI saw it.
//!
//! Everything is drawn into the video painter, after the boxes and before the HUD (frame
//! chip, stall banner, clock), so the HUD stays readable on top. In edit mode the
//! handles appear, the selected rule turns amber and a half-drawn shape is dashed.
//!
//! [`interact`] (in `input.rs`) is the small adapter between egui and the pure
//! [`Editor`](super::rule_editor::Editor): it turns pointer events into
//! [`EditorInput`]s; the keys reach the editor from `App::keys`. All the editing logic
//! lives in the editor and is tested there.

use std::time::{Duration, Instant};

use eframe::egui::{Align2, Color32, FontId, Painter, Pos2, Rect, Shape, Stroke, Vec2, pos2, vec2};

mod input;

pub use input::interact;

use super::detections::{chip, label};
use super::hud::{self, BOX_FILL};
use super::rule_editor::{Draft, Editor, HANDLE_RADIUS, Tool, Viewport};
use super::tracks::screen_box;
use super::{AMBER, GREEN};
use crate::events::event::{EventTotals, RuleTotals};
use crate::events::geometry::{Point, is_convex};
use crate::events::rules::{MIN_ZONE_POINTS, RuleId, Rules};
use crate::track::tracker::{TrackState, TrackView};

/// The rules' colour: a bright cyan that is none of the 12 group colours, so a rule
/// never looks like a detection box.
pub const RULE: Color32 = Color32::from_rgb(0x00, 0xE5, 0xFF);
/// How long a rule stays green after it fires: long enough to catch the eye, short
/// enough that two crossings a second apart flash twice.
pub const FLASH: Duration = Duration::from_millis(400);
/// How far the A and B letters sit from the line, in screen points.
pub const SIDE_OFFSET: f32 = 18.0;
/// Half the A→B arrow's length, in screen points.
const ARROW_HALF: f32 = 12.0;
/// How far along the line the arrow is moved from the middle, so it doesn't cover the
/// A and B letters.
const ARROW_SHIFT: f32 = 16.0;
/// Thicker than a box's 2 px, so a line reads as a rule, not as part of a box.
const LINE_STROKE: f32 = 3.0;
const ZONE_STROKE: f32 = 2.0;
/// The zone's fill alpha factor: enough to see the area, little enough to see through.
const ZONE_FILL: f32 = 0.12;
const HANDLE_DRAWN: f32 = 5.0;
const HANDLE_STROKE: f32 = 1.5;
const DRAFT_STROKE: f32 = 2.0;
const DASH: f32 = 8.0;
const GAP: f32 = 6.0;
/// The A/B letters sit on a dark disc so they read on any video.
const SIDE_DISC: f32 = 8.0;
const SIDE_FONT: f32 = 13.0;
/// The hint bar's bottom edge sits this far above the video's bottom. The clock's box
/// ends 24 points above the bottom, so the bar (about 20 points tall) never overlaps it.
const HINT_LIFT: f32 = 4.0;
/// The hint bar's text size: the panel's small text, readable over video.
const HINT_FONT: f32 = 12.0;
/// Padding around the hint text, in screen points: a little wider than tall, like a chip.
const HINT_PAD: Vec2 = vec2(8.0, 3.0);
/// The hint bar's corner radius.
const HINT_RADIUS: f32 = 6.0;

/// The video rect as the editor's [`Viewport`].
pub fn viewport(video: Rect) -> Viewport {
    Viewport {
        origin: Point::new(video.min.x, video.min.y),
        width: video.width(),
        height: video.height(),
    }
}

fn to_pos(p: Point) -> Pos2 {
    pos2(p.x, p.y)
}

fn to_point(p: Pos2) -> Point {
    Point::new(p.x, p.y)
}

/// Where a line's side letters and A→B arrow go, in screen points.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SideMarkers {
    pub a_label: Pos2,
    pub b_label: Pos2,
    /// The arrow starts on side A and ends on side B, crossing the line.
    pub arrow_from: Pos2,
    pub arrow_to: Pos2,
}

/// The side markers of the line `a` → `b` (screen points).
///
/// The normal that points to side A is `n = (−dy, dx) / length`, the direction `a → b`
/// turned a quarter. Check it with a line drawn top to bottom, `(dx, dy) = (0, 1)`:
/// `n = (−1, 0)` points screen-left, and [`signed_distance`] calls screen-left side A.
/// A zero-length line has no sides, so everything sits on its one point.
pub fn side_markers(a: Pos2, b: Pos2) -> SideMarkers {
    let d = b - a;
    let length = d.length();
    let (along, normal) = if length > 0.0 {
        (d / length, vec2(-d.y, d.x) / length)
    } else {
        (Vec2::ZERO, Vec2::ZERO)
    };
    let mid = a + d / 2.0;
    let arrow_mid = mid + along * ARROW_SHIFT;
    SideMarkers {
        a_label: mid + normal * SIDE_OFFSET,
        b_label: mid - normal * SIDE_OFFSET,
        arrow_from: arrow_mid + normal * ARROW_HALF,
        arrow_to: arrow_mid - normal * ARROW_HALF,
    }
}

/// A line's counts: `A→B 5 · B→A 3`.
pub fn line_count_text(t: &RuleTotals) -> String {
    format!("A→B {} · B→A {}", t.a_to_b, t.b_to_a)
}

/// A zone's occupancy: `2 inside · 7 visits` (`1 visit`).
pub fn zone_text(t: &RuleTotals) -> String {
    let visits = if t.visits == 1 { "visit" } else { "visits" };
    format!("{} inside · {} {visits}", t.inside.len(), t.visits)
}

/// A dwell time as a clock: `0:42`, `12:05`, `1:02:09`.
pub fn dwell_text(d: Duration) -> String {
    let secs = d.as_secs();
    let (hours, minutes, seconds) = (secs / 3600, secs % 3600 / 60, secs % 60);
    if hours > 0 {
        format!("{hours}:{minutes:02}:{seconds:02}")
    } else {
        format!("{minutes}:{seconds:02}")
    }
}

/// Whether a rule that last fired at `last_event` is still flashing at `now`. Both are
/// `Instant`s on the same clock (`last_event` is a frame's arrival time), so comparing
/// them is fine; `saturating_duration_since` covers a `last_event` after `now`.
pub fn flashing(last_event: Option<Instant>, now: Instant) -> bool {
    last_event.is_some_and(|t| now.saturating_duration_since(t) < FLASH)
}

/// The hint bar's text for each tool.
pub fn hint_text(tool: Tool) -> &'static str {
    match tool {
        Tool::Select => "Drag handles or shapes · Delete removes · Esc leaves edit",
        Tool::Line => "Drag to draw a line · Shift snaps · Esc cancels",
        Tool::Zone => "Click corners · click the first or Enter to close · Esc cancels",
    }
}

/// What [`draw`] needs besides the painter and the video rect.
pub struct Overlay<'a> {
    pub rules: &'a Rules,
    /// The newest fresh totals; `None` before the first result or once it is stale.
    pub totals: Option<&'a EventTotals>,
    /// The tracks to hang dwell timers on (by id, as the detector publishes them);
    /// empty when the boxes are hidden.
    pub tracks: &'a [TrackView],
    /// `Some` in edit mode: handles, selection and the draft are drawn.
    pub editor: Option<&'a Editor>,
    /// The shown frame's arrival time, which the track boxes are predicted to.
    pub shown_at: Instant,
    pub now: Instant,
}

impl Overlay<'_> {
    fn totals_of(&self, id: RuleId) -> Option<&RuleTotals> {
        self.totals.and_then(|t| t.rule(id))
    }

    /// Amber when selected in edit mode, green while flashing, cyan otherwise.
    fn color_of(&self, id: RuleId) -> Color32 {
        if self.selected(id) {
            AMBER
        } else if flashing(self.totals_of(id).and_then(|t| t.last_event), self.now) {
            GREEN
        } else {
            RULE
        }
    }

    /// Selected in the editor; only in edit mode.
    fn selected(&self, id: RuleId) -> bool {
        self.editor.is_some_and(|e| e.selected() == Some(id))
    }
}

/// Draws every rule into `painter`: zones first, lines over them (the editor's hit test
/// grabs lines before zones, so the top shape is the one the pointer gets), then the
/// dwell timers, then the draft and the hint bar in edit mode. `frame` is the camera
/// frame size, for placing the dwell timers at the tracks' boxes.
pub fn draw(painter: &Painter, video: Rect, frame: Vec2, o: &Overlay) {
    let view = viewport(video);
    for zone in &o.rules.zones {
        let corners: Vec<Pos2> = zone.points.iter().map(|&p| screen(view, p)).collect();
        let color = o.color_of(zone.id);
        // egui fills a path correctly only when it is convex; a concave zone keeps
        // just its outline rather than a wrongly filled shape.
        if is_convex(&zone.points) {
            let fill = color.gamma_multiply(ZONE_FILL);
            painter.add(Shape::convex_polygon(corners.clone(), fill, Stroke::NONE));
        }
        painter.add(Shape::closed_line(
            corners.clone(),
            Stroke::new(ZONE_STROKE, color),
        ));
        let text = match o.totals_of(zone.id) {
            Some(t) => format!("{}  {}", zone.name, zone_text(t)),
            None => zone.name.clone(),
        };
        label(painter, Rect::from_points(&corners), text, color, video);
        if o.editor.is_some() {
            handles(painter, &corners, color, o.selected(zone.id));
        }
    }
    for line in &o.rules.lines {
        let (a, b) = (screen(view, line.a), screen(view, line.b));
        let color = o.color_of(line.id);
        painter.line_segment([a, b], Stroke::new(LINE_STROKE, color));
        sides(painter, side_markers(a, b), color);
        let text = match o.totals_of(line.id) {
            Some(t) => format!("{}  {}", line.name, line_count_text(t)),
            None => line.name.clone(),
        };
        // The chip goes at the end `b`, where the line was drawn to.
        label(
            painter,
            Rect::from_center_size(b, Vec2::ZERO),
            text,
            color,
            video,
        );
        if o.editor.is_some() {
            handles(painter, &[a, b], color, o.selected(line.id));
        }
    }
    dwell_timers(painter, video, frame, o);
    if let Some(editor) = o.editor {
        if let Some(d) = editor.draft() {
            draft(painter, view, d);
        }
        hint_bar(painter, video, hint_text(editor.tool()));
    }
}

fn screen(view: Viewport, p: Point) -> Pos2 {
    to_pos(view.to_screen(p))
}

/// The A and B letters on dark discs, and the A→B arrow.
fn sides(painter: &Painter, m: SideMarkers, color: Color32) {
    for (at, letter) in [(m.a_label, "A"), (m.b_label, "B")] {
        painter.circle_filled(at, SIDE_DISC, BOX_FILL);
        let font = FontId::proportional(SIDE_FONT);
        painter.text(at, Align2::CENTER_CENTER, letter, font, color);
    }
    painter.arrow(
        m.arrow_from,
        m.arrow_to - m.arrow_from,
        Stroke::new(HANDLE_STROKE, color),
    );
}

/// Round handles on a rule's points: filled for the selected rule, rings otherwise.
fn handles(painter: &Painter, points: &[Pos2], color: Color32, selected: bool) {
    for &p in points {
        if selected {
            painter.circle_filled(p, HANDLE_DRAWN, color);
        } else {
            painter.circle_filled(p, HANDLE_DRAWN, BOX_FILL);
            painter.circle_stroke(p, HANDLE_DRAWN, Stroke::new(HANDLE_STROKE, color));
        }
    }
}

/// A `0:42` chip under each tracked object inside a zone. Lost tracks are skipped:
/// their box is a guess, and is hidden unless "Show lost" is on.
fn dwell_timers(painter: &Painter, video: Rect, frame: Vec2, o: &Overlay) {
    let Some(totals) = o.totals else {
        return;
    };
    for zone in &o.rules.zones {
        let Some(t) = totals.rule(zone.id) else {
            continue;
        };
        for dweller in &t.inside {
            // Both lists are sorted by track id.
            let Ok(i) = o.tracks.binary_search_by_key(&dweller.track, |v| v.id) else {
                continue;
            };
            let view = &o.tracks[i];
            if view.state != TrackState::Tracked {
                continue;
            }
            let boxr = screen_box(view, o.shown_at, frame, video);
            let time = dwell_text(o.shown_at.saturating_duration_since(dweller.since));
            chip_below(painter, boxr, time, o.color_of(zone.id), video);
        }
    }
}

/// A label chip under the box's bottom-left corner (the track's own `#7 person` label
/// sits above it), kept inside the video.
fn chip_below(painter: &Painter, boxr: Rect, text: String, color: Color32, video: Rect) {
    chip(painter, text, color, |size| {
        let x = boxr.left().min(video.right() - size.x).max(video.left());
        let y = boxr.bottom().min(video.bottom() - size.y);
        Rect::from_min_size(pos2(x, y), size)
    });
}

/// The shape being drawn, dashed. A zone draft also shows its corners, with a ring
/// around the first one once it can be closed there.
fn draft(painter: &Painter, view: Viewport, d: &Draft) {
    let stroke = Stroke::new(DRAFT_STROKE, RULE);
    match d {
        Draft::Line { a, b } => {
            let path = [screen(view, *a), screen(view, *b)];
            painter.extend(Shape::dashed_line(&path, stroke, DASH, GAP));
        }
        Draft::Zone { points, hover } => {
            let mut path: Vec<Pos2> = points.iter().map(|&p| screen(view, p)).collect();
            let corners = path.len();
            if let Some(h) = hover {
                path.push(screen(view, *h));
            }
            painter.extend(Shape::dashed_line(&path, stroke, DASH, GAP));
            for &p in &path[..corners] {
                painter.circle_filled(p, HANDLE_DRAWN - 1.0, RULE);
            }
            if corners >= MIN_ZONE_POINTS {
                painter.circle_stroke(path[0], HANDLE_RADIUS, stroke);
            }
        }
    }
}

/// White text on a dark chip, bottom-centre of the video.
fn hint_bar(painter: &Painter, video: Rect, text: &str) {
    let at = pos2(video.center().x, video.bottom() - HINT_LIFT);
    let font = FontId::proportional(HINT_FONT);
    hud::boxed(
        painter,
        at,
        Align2::CENTER_BOTTOM,
        text.to_owned(),
        font,
        HINT_PAD,
        HINT_RADIUS,
    );
}

#[cfg(test)]
mod tests;
