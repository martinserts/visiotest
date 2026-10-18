//! Moving what the Select tool grabbed: one handle, or a whole shape.
//!
//! Each function looks the rule up by id on every call (the card can delete it between
//! two redraws) and reports [`Outcome::changed`] only when a point really moved.

use super::Outcome;
use crate::events::geometry::Point;
use crate::events::rules::{RuleId, Rules};

/// Puts point `index` of rule `rule` at `to` (normalised). A line refuses a move that
/// would put both ends on the same point, which would leave it with no direction.
pub fn move_handle(rules: &mut Rules, rule: RuleId, index: usize, to: Point) -> Outcome {
    if let Some(line) = rules.line_mut(rule) {
        let (point, other) = if index == 0 {
            (&mut line.a, line.b)
        } else {
            (&mut line.b, line.a)
        };
        if to == other || *point == to {
            return Outcome::default();
        }
        *point = to;
        return Outcome::changed();
    }
    if let Some(zone) = rules.zone_mut(rule)
        && let Some(point) = zone.points.get_mut(index)
        && *point != to
    {
        *point = to;
        return Outcome::changed();
    }
    Outcome::default()
}

/// Puts every point of rule `rule` at `start + offset`. `start` must be the rule's
/// points when the drag began; a zone whose corner count changed since is left alone.
pub fn move_body(rules: &mut Rules, rule: RuleId, start: &[Point], offset: Point) -> Outcome {
    let shifted = |p: Point| Point::new(p.x + offset.x, p.y + offset.y);
    if let Some(line) = rules.line_mut(rule)
        && let [a, b] = start
    {
        let (a, b) = (shifted(*a), shifted(*b));
        if (line.a, line.b) == (a, b) {
            return Outcome::default();
        }
        (line.a, line.b) = (a, b);
        return Outcome::changed();
    }
    if let Some(zone) = rules.zone_mut(rule)
        && zone.points.len() == start.len()
    {
        let mut changed = false;
        for (point, &from) in zone.points.iter_mut().zip(start) {
            let to = shifted(from);
            changed |= *point != to;
            *point = to;
        }
        return Outcome { changed };
    }
    Outcome::default()
}

/// How far to move a shape whose points were `start`: the pointer's move `now − grab`,
/// limited so that no point leaves `0..=1`. Limiting the offset (not each point) keeps
/// the shape intact when it is pushed against an edge, instead of squashing it.
///
/// The limits come from the shape's bounding box: its left edge may move left by at
/// most `min_x` and its right edge right by at most `1 − max_x`, and likewise in y.
pub fn clamped_offset(start: &[Point], now: Point, grab: Point) -> Point {
    let (mut min_x, mut min_y) = (f32::INFINITY, f32::INFINITY);
    let (mut max_x, mut max_y) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for p in start {
        min_x = min_x.min(p.x);
        min_y = min_y.min(p.y);
        max_x = max_x.max(p.x);
        max_y = max_y.max(p.y);
    }
    // `max` then `min` rather than `clamp`, which panics if rounding ever made the
    // lower limit exceed the upper one.
    Point::new(
        (now.x - grab.x).max(-min_x).min(1.0 - max_x),
        (now.y - grab.y).max(-min_y).min(1.0 - max_y),
    )
}
