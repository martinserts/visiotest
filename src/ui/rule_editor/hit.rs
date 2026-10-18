//! Hit testing: which rule, and which part of it, is under the pointer.
//!
//! Everything is measured in **screen points**: each rule's normalised points are
//! converted with the [`Viewport`] first, so the grab radii mean the same on both axes
//! and at every window size.

use super::{HANDLE_RADIUS, Hit, LINE_GRAB, Viewport, distance};
use crate::events::geometry::{Point, point_in_polygon};
use crate::events::rules::{RuleId, Rules};

/// What is under `screen`, or `None`.
///
/// The order matters when things overlap:
/// 1. **Handles before bodies.** Handles are small and drawn on top of the shapes, and a
///    corner inside another zone must still be grabbable.
/// 2. **Lines before zones.** A line is a thin stroke, a zone a large area; a line drawn
///    inside a zone could otherwise never be grabbed by its body. (The UI draws zones
///    first and lines over them, so this is also the drawing order, top first.)
/// 3. **Newest first** within each kind: rules are drawn in the order they were added,
///    so the most recent one is on top, and it is the one the user sees under the pointer.
pub fn hit_test(rules: &Rules, screen: Point, view: &Viewport) -> Option<Hit> {
    hit_handle(rules, screen, view).or_else(|| hit_body(rules, screen, view))
}

/// A line end or zone corner within [`HANDLE_RADIUS`] of `screen`.
fn hit_handle(rules: &Rules, screen: Point, view: &Viewport) -> Option<Hit> {
    let near = |p: Point| distance(view.to_screen(p), screen) <= HANDLE_RADIUS;
    for line in rules.lines.iter().rev() {
        // Index 0 is `a`, 1 is `b`.
        if let Some(index) = [line.a, line.b].into_iter().position(near) {
            return Some(Hit::Handle {
                rule: line.id,
                index,
            });
        }
    }
    for zone in rules.zones.iter().rev() {
        if let Some(index) = zone.points.iter().position(|&p| near(p)) {
            return Some(Hit::Handle {
                rule: zone.id,
                index,
            });
        }
    }
    None
}

/// A line within [`LINE_GRAB`] of `screen`, or a zone containing it.
fn hit_body(rules: &Rules, screen: Point, view: &Viewport) -> Option<Hit> {
    for line in rules.lines.iter().rev() {
        let (a, b) = (view.to_screen(line.a), view.to_screen(line.b));
        if distance_to_segment(screen, a, b) <= LINE_GRAB {
            return Some(Hit::Body { rule: line.id });
        }
    }
    for zone in rules.zones.iter().rev() {
        let corners: Vec<Point> = zone.points.iter().map(|&p| view.to_screen(p)).collect();
        if point_in_polygon(screen, &corners) {
            return Some(Hit::Body { rule: zone.id });
        }
    }
    None
}

/// The shortest distance from `p` to the segment `a`–`b` (not the infinite line, which
/// [`crate::events::geometry::signed_distance`] measures: a point far past a line's end
/// is not "on" it).
///
/// Project `p` onto the line: `t = (p − a)·(b − a) / |b − a|²` is how far along the
/// segment the foot of the perpendicular lands, 0 at `a` and 1 at `b`. Clamping `t` to
/// `0..=1` picks the nearest end when the foot falls outside the segment.
fn distance_to_segment(p: Point, a: Point, b: Point) -> f32 {
    let (dx, dy) = (b.x - a.x, b.y - a.y);
    let length_sq = dx * dx + dy * dy;
    if length_sq == 0.0 {
        // A zero-length segment is just the point `a`.
        return distance(p, a);
    }
    let t = (((p.x - a.x) * dx + (p.y - a.y) * dy) / length_sq).clamp(0.0, 1.0);
    distance(p, Point::new(a.x + t * dx, a.y + t * dy))
}

/// The points of rule `rule`, for a body drag to start from: `[a, b]` for a line, the
/// corners for a zone, empty if there is no such rule.
pub fn shape_points(rules: &Rules, rule: RuleId) -> Vec<Point> {
    if let Some(line) = rules.line(rule) {
        return vec![line.a, line.b];
    }
    rules
        .zone(rule)
        .map(|zone| zone.points.clone())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rule_editor_distance_to_segment() {
        let (a, b) = (Point::new(0.0, 0.0), Point::new(10.0, 0.0));
        // Beside the middle: the perpendicular distance.
        assert_eq!(distance_to_segment(Point::new(5.0, 3.0), a, b), 3.0);
        // Past the end `b`: the distance to `b`, not to the infinite line.
        assert_eq!(distance_to_segment(Point::new(13.0, 4.0), a, b), 5.0);
        // A zero-length segment.
        assert_eq!(distance_to_segment(Point::new(3.0, 4.0), a, a), 5.0);
    }
}
